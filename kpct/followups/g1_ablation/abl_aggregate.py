#!/usr/bin/env python3
"""abl_aggregate.py -- G_1 ablation: per-variant metrics, paired contrasts (Poisson(1) bootstrap B = 2000, seed 0;
strata B = 1000) and the existing-prediction analysis -> multitq_g1_ablation_results.json + multitq_g1_ablation_table.csv.
Descriptive only (rule multitq_g1_ablation_rule.json registers no threshold)."""
import json, sys
import numpy as np
import pandas as pd

VARIANTS = ["A_full", "A_no_trace", "A_facts_cand", "A_facts", "A_GUfacts", "A_facts_ord"]
CHAIN = [("A_facts_cand", "A_facts", "candidate strings (unranked) added to shuffled facts"),
         ("A_no_trace", "A_facts_cand", "ranker outputs added: S order, RULE rank, S score, operator flag, fact links, frozen fact order"),
         ("A_full", "A_no_trace", "parse + operator trace added"),
         ("A_facts_ord", "A_facts", "frozen (S seed-0 provenance) fact order instead of a seeded shuffle"),
         ("A_full", "A_GUfacts", "structural provenance facts instead of GU dense-retrieved facts (same count, full format)")]


def boot(a, b, B=2000, seed=0):
    """paired Poisson(1) bootstrap of mean(a) - mean(b) (tkg-claims-protocol paired_poisson_bootstrap)"""
    a = np.asarray(a, float); b = np.asarray(b, float); d = a - b
    rng = np.random.default_rng(seed)
    bs = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0]); s = w.sum()
        bs[i] = (w * d).sum() / s if s > 0 else 0.0
    lo, hi = np.quantile(bs, [0.025, 0.975])
    return dict(delta=float(d.mean()), ci=[float(lo), float(hi)], B=B)


def variant_metrics(g):
    n = len(g)
    fx = int(((g.hit1 == 1) & (g.s0_hit1 == 0)).sum()); ls = int(((g.hit1 == 0) & (g.s0_hit1 == 1)).sum())
    h1 = boot(g.hit1.values, np.zeros(n)); h10 = boot(g.hit10.values, np.zeros(n))
    return dict(n=n, hits1=h1["delta"], hits1_ci=h1["ci"], hits10=h10["delta"], hits10_ci=h10["ci"],
                invalid_rate=float(1 - g.valid.mean()), answer_outside_candidates=float(1 - g.in_cands.mean()),
                answer_outside_shown_facts=float(1 - g.in_shown_facts.mean()),
                gold_coverage_shown_facts=float(g.gold_in_shown_facts.mean()), gold_coverage_candidates=float(g.gold_in_cands.mean()),
                truncation_rate=float((g.groups_dropped > 0).mean()), prompt_tokens_mean=float(g.prompt_len.mean()),
                fact_groups_shown_mean=float(g.groups_shown.mean()),
                copy_S0=float(g.copy_s0.mean()), fixes_vs_S0=fx, losses_vs_S0=ls, net_vs_S0=fx - ls,
                hits1_when_gold_in_shown_facts=float(g.hit1[g.gold_in_shown_facts == 1].mean()) if (g.gold_in_shown_facts == 1).any() else None,
                hits1_when_gold_not_in_shown_facts=float(g.hit1[g.gold_in_shown_facts == 0].mean()) if (g.gold_in_shown_facts == 0).any() else None)


def main(perq, part4, freeze, out_json, out_csv):
    df = pd.read_csv(perq)
    fr = json.load(open(freeze)); p4 = json.load(open(part4))
    res = dict(name="multitq_g1_ablation_results", rule="multitq_g1_ablation_rule.json (sha256 e32304a1...)",
               status="POST-HOC, DESCRIPTIVE; dev only; no threshold, no decision", sets={})
    rows_csv = []
    for sname in ["D_main", "dev_sel2000"]:
        d = df[df.set == sname]
        if d.empty:
            continue
        variants = [v for v in VARIANTS if v in set(d.variant)]
        piv = {v: d[d.variant == v].sort_values("quid").reset_index(drop=True) for v in variants}
        qs = piv["A_full"].quid.values
        for v in variants:
            assert (piv[v].quid.values == qs).all()
        S = dict(n=int(len(qs)), quid_list_sha256=fr["sets"][sname]["quid_list_sha256"], variants=variants,
                 S0_hits1=float(piv["A_full"].s0_hit1.mean()), metrics={}, contrasts_vs_A_full={}, chain={}, strata={})
        for v in variants:
            S["metrics"][v] = variant_metrics(piv[v])
        for v in variants:
            if v == "A_full":
                continue
            S["contrasts_vs_A_full"][v] = dict(hits1=boot(piv[v].hit1, piv["A_full"].hit1), hits10=boot(piv[v].hit10, piv["A_full"].hit10),
                                               discordant=dict(variant_only=int(((piv[v].hit1 == 1) & (piv["A_full"].hit1 == 0)).sum()),
                                                               A_full_only=int(((piv[v].hit1 == 0) & (piv["A_full"].hit1 == 1)).sum())),
                                               same_answer_as_A_full=None)
        for a_, b_, what in CHAIN:
            if a_ in piv and b_ in piv:
                S["chain"][f"{a_} - {b_}"] = dict(what=what, hits1=boot(piv[a_].hit1, piv[b_].hit1))
        S["contrasts_vs_S0"] = {v: boot(piv[v].hit1, piv[v].s0_hit1) for v in variants}
        for col in ("qtype", "answer_type"):
            S["strata"][col] = {}
            for lv in sorted(piv["A_full"][col].unique()):
                m = (piv["A_full"][col] == lv).values
                S["strata"][col][lv] = dict(n=int(m.sum()), S0_hits1=float(piv["A_full"].s0_hit1[m].mean()),
                                            hits1={v: float(piv[v].hit1[m].mean()) for v in variants},
                                            delta_vs_A_full={v: boot(piv[v].hit1[m], piv["A_full"].hit1[m], B=1000) for v in variants if v != "A_full"})
        res["sets"][sname] = S
        for v in variants:
            m = S["metrics"][v]; c = S["contrasts_vs_A_full"].get(v)
            rows_csv.append(dict(set=sname, variant=v, n=m["n"], hits1=round(m["hits1"], 4), hits1_lo=round(m["hits1_ci"][0], 4), hits1_hi=round(m["hits1_ci"][1], 4),
                                 hits10=round(m["hits10"], 4),
                                 delta_vs_A_full=round(c["hits1"]["delta"], 4) if c else 0.0, delta_lo=round(c["hits1"]["ci"][0], 4) if c else 0.0,
                                 delta_hi=round(c["hits1"]["ci"][1], 4) if c else 0.0,
                                 invalid_rate=round(m["invalid_rate"], 4), answer_outside_candidates=round(m["answer_outside_candidates"], 4),
                                 answer_outside_shown_facts=round(m["answer_outside_shown_facts"], 4),
                                 gold_coverage_shown_facts=round(m["gold_coverage_shown_facts"], 4), truncation_rate=round(m["truncation_rate"], 4),
                                 copy_S0=round(m["copy_S0"], 4), fixes_vs_S0=m["fixes_vs_S0"], losses_vs_S0=m["losses_vs_S0"],
                                 prompt_tokens_mean=round(m["prompt_tokens_mean"], 1)))
    # reference = Reader_1 mechanism values computed separately on the frozen test predictions (the quantities
    # produced by analysis/reader_mechanism.py). abl_part4.json has no such key, so these hard-coded constants are
    # what the reconciliation compares against (values unchanged from the original analysis code).
    aud = p4.get("reference_test") or dict(copy_rate=0.7443, fixes=5457, losses=1383, net=4074, fixes_in_top20=5348,
                                                 gold_coverage_structural_facts=0.8989, gold_coverage_GU_facts=0.7368)
    t = p4["splits"]["test"]
    res["existing_predictions"] = dict(
        note="frozen G_1 and S_1 seed-0 predictions; no new inference; test predictions were scored once in the k% pass (descriptive re-analysis)",
        splits={s: {k: v for k, v in x.items() if k != "inputs"} for s, x in p4["splits"].items()},
        inputs={s: x["inputs"] for s, x in p4["splits"].items()},
        reference_reconciliation_test=dict(
            copy_rate=dict(reference=aud["copy_rate"], ours=round(t["copy_rate"], 4)),
            fixes=dict(reference=aud["fixes"], ours=t["fixes"]), losses=dict(reference=aud["losses"], ours=t["losses"]), net=dict(reference=aud["net"], ours=t["net"]),
            fixes_in_top20=dict(reference=aud["fixes_in_top20"], ours_bundle_candidates=t["fixes_in_bundle_candidates"], ours_S0_list_first20=t["fixes_in_S0_top20_list"],
                                explanation="the reference 5,348 is the bundle reading (the <= 20 structural candidates); the S0 ranked list's first 20 entries include padding answers for small pools and give 5,383"),
            gold_coverage_structural_facts=dict(reference=aud["gold_coverage_structural_facts"], ours=round(t["gold_coverage"]["structural_facts"], 4)),
            gold_coverage_GU_facts=dict(reference=aud["gold_coverage_GU_facts"], ours_top20=round(t["gold_coverage"]["gu_facts_top20"], 4),
                                        ours_same_count=round(t["gold_coverage"]["gu_facts_same_count"], 4),
                                        explanation="the reference 73.68% uses all 20 GU facts; A_GUfacts shows the first n (= structural count) and covers 72.39% on test")))
    json.dump(res, open(out_json, "w"), indent=1)
    pd.DataFrame(rows_csv).to_csv(out_csv, index=False)
    return res


if __name__ == "__main__":
    main(*sys.argv[1:6])
