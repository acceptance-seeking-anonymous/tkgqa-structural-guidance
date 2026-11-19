#!/usr/bin/env python3
"""cs_compare.py -- dev_sel2000 comparison of the clean-silver P_1 follow-up (multitq_p1_clean_silver_rule.json).

Primary configuration of every variant = the frozen P_1 configuration (final epoch ep3, tau off), fixed a priori.
P hit per question = mean over the 3 S_k (k01_s0) seeds of the official hit (multitq_eval.hit_at_k, sha256 checked).
Contrasts (paired Poisson(1) bootstrap over the 2,000 dev_sel2000 questions, B = 2000, seed 0, two-sided 95% percentile
CI; tkg-claims-protocol paired_poisson_bootstrap):
  primary   D_clean  = P_1clean - P_1frozen
  noise     D_rerun  = P_1rerun - P_1frozen (same silver, same seed, same data order: GPU / decoding nondeterminism)
            D_order  = P_1order - P_1frozen (same silver and LoRA init, data-order seed 1)
  secondary P_1clean - P_1rerun, P_1clean - P_1order
Labels as registered. Secondary (descriptive): each variant's rule-selected configuration (pk_select outputs).
Reads answers ONLY for the 2,000 dev_sel2000 quids. No test file is read or written."""
import argparse, collections, gzip, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

K = "/kpct"; Q = f"{K}/parser_clean"; P = f"{K}/parser"
EXEC_DIR = f"{K}/executor"
SCORER_SHA = "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"
VARIANTS = ["clean", "rerun", "order"]


def paired_poisson_bootstrap(rr_a, rr_b, B=2000, seed=0, alpha=0.05):
    """verbatim tkg-claims-protocol helper (+ the opposite one-sided p)"""
    a = np.asarray(rr_a, dtype=np.float64); b = np.asarray(rr_b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError("rr_a and rr_b must be paired (same shape)")
    d = a - b
    rng = np.random.default_rng(seed)
    boots = np.empty(B)
    for i in range(B):
        w = rng.poisson(1.0, size=d.shape[0]); s = w.sum()
        boots[i] = (w * d).sum() / s if s > 0 else 0.0
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return {"delta": float(d.mean()), "ci": [float(lo), float(hi)], "p_one_sided": float((boots <= 0).mean()),
            "p_one_sided_lower": float((boots >= 0).mean()), "B": B}


def mean_ci(x, B=2000, seed=0):
    r = paired_poisson_bootstrap(x, np.zeros_like(np.asarray(x, float)), B=B, seed=seed)
    return [r["ci"][0], r["ci"][1]]


def load_preds(path):
    out = {}
    for r in PC.iter_jsonl(path):
        out[int(r["quid"])] = r["ranked"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--dev_json", default="/data/questions/dev.json")
    a = ap.parse_args()
    sys.path.insert(0, os.path.join(EXEC_DIR, "struct"))
    assert PC.sha256_file(os.path.join(EXEC_DIR, "struct", "multitq_eval.py")) == SCORER_SHA, "scorer sha256 mismatch"
    import multitq_eval as ME
    rule = json.load(open(f"{Q}/multitq_p1_clean_silver_rule.json")); assert rule["sha256"] == PC.sha_recipe(rule)
    fz = json.load(open(a.freeze)); assert fz["sha256"] == PC.sha_recipe(fz), "freeze modified"
    fz1 = json.load(open(f"{P}/freeze/freeze_P_k01_s0.json"))
    sel = PC.load_list(f"{P}/code/multitq_kpct_subsets.json", "dev_sel2000"); S = set(sel)
    D = {int(q["quid"]): q for q in json.load(open(a.dev_json)) if int(q["quid"]) in S}
    assert set(D) == S and len(sel) == 2000
    unseen = {int(r["quid"]): bool(r.get("template_unseen")) for r in PC.iter_jsonl(f"{K}/bundles/k01_s0/dev_sel2000.jsonl")}
    un = np.array([unseen[q] for q in sel])
    qt = np.array([D[q]["qtype"] for q in sel])

    def hits(files):
        H1, H10 = [], []
        for f in files:
            Pp = load_preds(f); assert set(Pp) == S, f
            H1.append(np.array([ME.hit_at_k(D[q], Pp[q], 1) for q in sel], float))
            H10.append(np.array([ME.hit_at_k(D[q], Pp[q], 10) for q in sel], float))
        return np.mean(H1, 0), np.mean(H10, 0), H1

    # frozen P_1 (hash-verified)
    ff = []
    for s in range(3):
        fn = f"pred_P_k01_s0_dev_sel2000_ep3_tauoff_seed{s}.jsonl"
        assert PC.sha256_file(f"{P}/exec/k01_s0/{fn}") == fz1["dev_sel2000"]["chosen_pred_files"][fn], "STOP_INTEGRITY frozen dev preds"
        ff.append(f"{P}/exec/k01_s0/{fn}")
    arms = {"frozen": hits(ff)}
    for v in VARIANTS:
        fs = []
        for s in range(3):
            rel = f"parser_clean/exec/pred_P_k01_s0_dev_sel2000_{v}_ep3_tauoff_seed{s}.jsonl"
            assert PC.sha256_file(f"{K}/{rel}") == fz["variants"][v]["predictions"][rel], f"freeze mismatch {rel}"
            fs.append(f"{K}/{rel}")
        arms[v] = hits(fs)
    est = json.load(open(f"{Q}/exec/exec_stats_P_k01_s0_dev_sel2000.json"))
    f_est = json.load(open(f"{P}/exec/k01_s0/exec_stats_P_k01_s0_dev_sel2000.json"))
    rate = {"frozen": dict(fallback=f_est["adapters"]["ep3"]["tau"]["off"]["fallback_rate"], invalid=f_est["adapters"]["ep3"]["invalid_rate"],
                           reasons=f_est["adapters"]["ep3"]["reasons"])}
    for v in VARIANTS:
        x = est["adapters"][f"{v}_ep3"]
        rate[v] = dict(fallback=x["tau"]["off"]["fallback_rate"], invalid=x["invalid_rate"], reasons=x["reasons"])
    res_arms = {}
    for n, (h1, h10, H1) in arms.items():
        res_arms[n] = dict(config="ep3 / tau off", hits1=float(h1.mean()), hits1_ci95=mean_ci(h1), hits1_per_seed=[float(x.mean()) for x in H1],
                           hits10=float(h10.mean()), hits10_ci95=mean_ci(h10),
                           unseen=dict(n=int(un.sum()), hits1=float(h1[un].mean()), hits10=float(h10[un].mean())),
                           seen=dict(n=int((~un).sum()), hits1=float(h1[~un].mean()), hits10=float(h10[~un].mean())),
                           by_qtype={t: float(h1[qt == t].mean()) for t in sorted(set(qt))},
                           fallback_rate=rate[n]["fallback"], invalid_rate=rate[n]["invalid"], fallback_reasons=rate[n]["reasons"],
                           STOP_DEV_FAILURE=bool(h1.mean() < 0.30 or rate[n]["invalid"] > 0.20))
    pairs = [("clean", "frozen"), ("rerun", "frozen"), ("order", "frozen"), ("clean", "rerun"), ("clean", "order")]
    contrasts = {}
    for x, y in pairs:
        c1 = paired_poisson_bootstrap(arms[x][0], arms[y][0]); c10 = paired_poisson_bootstrap(arms[x][1], arms[y][1])
        cu = paired_poisson_bootstrap(arms[x][0][un], arms[y][0][un])
        contrasts[f"{x}-{y}"] = dict(hits1=c1, hits10=c10, unseen_hits1=cu,
                                     questions_changed_3seed_mean=int((arms[x][0] != arms[y][0]).sum()),
                                     questions_changed_seed0=int((arms[x][2][0] != arms[y][2][0]).sum()))
    # generation-level reproducibility of the rerun (ep3 raw texts vs the frozen ep3 raw texts)
    def raw(p):
        with gzip.open(p, "rt") as f:
            return {int(json.loads(l)["quid"]): json.loads(l)["text"] for l in f if l.strip()}
    fr = raw(f"{P}/raw/k01_s0/raw_P_dev_sel2000_ep3.jsonl.gz")
    gen = {}
    for v in VARIANTS:
        rv = raw(f"{Q}/raw/raw_P_dev_sel2000_{v}_ep3.jsonl.gz")
        gen[v] = dict(identical_generations=int(sum(rv[q] == fr[q] for q in sel)), n=len(sel))
    fa = fz1["hashes_pre_test"]["adapter"]["parser/adapters/P_k01_s0_s0/ep3/adapter_model.safetensors"]
    ra = fz["variants"]["rerun"]["adapters"]["parser_clean/adapters/P_k01_s0rerun_s0/ep3/adapter_model.safetensors"]
    reproducibility = dict(rerun_adapter_identical_to_frozen=(fa == ra), rerun_data_order_identical=fz["variants"]["rerun"]["training"]["data_order_sha256"],
                           generations=gen)
    # labels (registered)
    d_c = contrasts["clean-frozen"]["hits1"]
    noise = max(abs(contrasts["rerun-frozen"]["hits1"]["delta"]), abs(contrasts["order-frozen"]["hits1"]["delta"]))
    if d_c["ci"][0] > 0 and d_c["delta"] > noise:
        label = "CLEAN_HIGHER"
    elif d_c["ci"][1] < 0 and -d_c["delta"] > noise:
        label = "CLEAN_LOWER"
    else:
        label = "NO_CLEAN_EFFECT_DETECTED"
    qualifiers = []
    if abs(d_c["delta"]) <= noise:
        qualifiers.append("WITHIN_RETRAIN_NOISE")
    qualifiers.append("RETRAIN_REPRODUCED" if contrasts["rerun-frozen"]["questions_changed_3seed_mean"] == 0 else "RETRAIN_NONDETERMINISTIC")
    if any(v["STOP_DEV_FAILURE"] for v in res_arms.values()):
        qualifiers.append("DEV_FAILURE")
    # secondary: rule-selected configurations (descriptive)
    selected = {}
    for v in VARIANTS:
        s_ = json.load(open(f"{Q}/select/{v}/select_P_k01_s0.json"))
        selected[v] = dict(checkpoint=s_["chosen"]["checkpoint"], tau=s_["chosen"]["tau"], hits1=s_["dev_sel2000_hits1"], hits10=s_["chosen"]["hits10"],
                           unseen_hits1=s_["unseen_hits1"], minus_frozen_selected=s_["dev_sel2000_hits1"] - res_arms["frozen"]["hits1"],
                           grid=[(g["checkpoint"], g["tau"], g["hits1"]) for g in s_["grid"]])
    out = dict(name="multitq_p1_clean_silver_results", rule_sha256=rule["sha256"], freeze_sha256=fz["sha256"], split="dev_sel2000 (n = 2000); no test",
               statistic="official hit@k averaged over the 3 S_k seeds; paired Poisson(1) bootstrap B = 2000 seed 0; two-sided 95% percentile CI",
               primary_configuration="ep3 / tau off for every variant (the frozen P_1 configuration, fixed a priori)",
               arms=res_arms, contrasts=contrasts, noise_band_hits1=noise, label=label, qualifiers=qualifiers,
               reproducibility=reproducibility, secondary_selected=selected, post_hoc=True,
               created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    out["sha256"] = PC.sha_recipe(out)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps({"label": label, "qualifiers": qualifiers, "D_clean": d_c, "noise": noise,
                      "hits1": {n: round(v["hits1"], 4) for n, v in res_arms.items()}}))


if __name__ == "__main__":
    main()
