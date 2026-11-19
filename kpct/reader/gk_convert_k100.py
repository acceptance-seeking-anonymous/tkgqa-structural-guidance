#!/usr/bin/env python3
"""gk_convert_k100.py -- convert the frozen k = 100 structural exports into the guidance-bundle JSONL schema.

Sources (all frozen, sha256 in multitq_structural_freeze.json / multitq_reader_freeze.json):
  exports/{train,dev,test}_parser_outputs.jsonl.gz   parse (train: 2-fold OOF classifier probabilities)
  rdata/sblock_{train60k,dev,test}.parquet           S_seed0 top-20 block over the full frozen pool (re-scored, reader track)
  exports/{train60k,dev,test}_top50_candidates.parquet  RULE rank + operator features (f_sat_top, f_q_ncand_sat)
  exports/{train60k,dev,test}_top20_facts.jsonl.gz   the 20 provenance facts (mtq_export.provenance_facts)
  questions/{train,dev,test}.json                     question text; answers for TRAIN rows only
Label columns of the dev/test parquet files are never loaded; dev/test bundles carry no gold field.

Output: <out>/{train,dev_sel2000,test}.jsonl.gz (train = reader_train60k, the registered k = 100 LLM list).
The operator trace is synthesised from the frozen parse and S features (the k100 exports carry no trace); the
structural k% track writes its own trace for k < 100.
"""
import argparse, gzip, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC

OPS = {"equal": "equal (same {g} as the time constraint)", "equal_multi": "equal (same {g}; several answers)",
       "before_after": "{ba} the time constraint", "first_last": "{fl} occurrence",
       "after_first": "first event after the anchor event", "before_last": "last event before the anchor event"}


def operator_of(p):
    qt = p.get("qtype_pred")
    s = OPS.get(qt, qt or "unknown")
    return s.format(g=p.get("granularity_pred") or "day", ba=p.get("kw_before_after") or "before/after",
                    fl=p.get("kw_first_last") or "first/last")


def parse_obj(p):
    ents = []
    for m in p.get("mentions") or []:
        alts = m.get("alternatives") or []
        if not alts:
            continue
        ents.append(dict(mention_tokens=m.get("tokens"), entity=alts[0][0], score=alts[0][1], alternatives=[a[0] for a in alts]))
    rels = p.get("relations_top5") or []
    t = p.get("time")
    return dict(entities=ents, relation=rels[0][0] if rels else None, direction=rels[0][1] if rels else None,
                relation_p=rels[0][2] if rels else None, relation_alternatives=rels[1:],
                qtype=p.get("qtype_pred"), qtype_p=p.get("qtype_p"), operator=operator_of(p),
                answer_type=p.get("answer_type_pred"), granularity=p.get("granularity_pred"),
                time=(t or {}).get("value") if isinstance(t, dict) else t, time_level=(t or {}).get("level") if isinstance(t, dict) else None,
                anchor=None, kw_before_after=p.get("kw_before_after"), kw_first_last=p.get("kw_first_last"))


def fact_support(cand, facts):
    if GC.is_time(cand):
        return [i for i, f in enumerate(facts) if f[3][:len(cand)] == cand]
    return [i for i, f in enumerate(facts) if cand in (f[0], f[2])]


def trace_text(parse, cands, n_sat, facts):
    s = f"operator {parse['operator']}"
    if parse.get("time"):
        s += f"; time constraint {parse['time']}"
    s += f"; {n_sat} candidate(s) of the pool satisfy the operator under the top relation hypothesis"
    if cands:
        c0 = cands[0]
        sat = c0["operator_satisfied"]
        verb = "satisfies" if sat else ("does not satisfy" if sat is not None else "has no operator check for")
        s += f"; top candidate {c0['candidate']} {verb} it"
        if c0["fact_ids"]:
            f2g = GC.group_facts([tuple(f) for f in facts])[1]
            s += " (supporting facts " + ", ".join(f"F{g}" for g in sorted({f2g[i] for i in c0["fact_ids"]})[:3]) + ")"
    return s


def convert(split_tag, qfile, parser_file, sblock, top50, facts_file, keep, with_gold, out):
    t0 = time.time()
    Q = {q["quid"]: q for q in json.load(open(qfile)) if q["quid"] in keep}
    P = {}
    with gzip.open(parser_file, "rt") as f:
        for l in f:
            r = json.loads(l)
            if r["quid"] in keep:
                P[r["quid"]] = r
    F = {}
    with gzip.open(facts_file, "rt") as f:
        for l in f:
            r = json.loads(l)
            if int(r["quid"]) in keep:
                F[int(r["quid"])] = [tuple(x) for x in r["facts"]]
    sb = pd.read_parquet(sblock, columns=["quid", "seed", "rank", "candidate", "S"])
    sb = sb[(sb.seed == 0) & sb.quid.isin(keep)].sort_values(["quid", "rank"])
    t5 = pd.read_parquet(top50, columns=["quid", "candidate", "rank_RULE", "f_sat_top", "f_q_ncand_sat"])
    t5 = t5[t5.quid.isin(keep)]
    rr = {(q, c): (int(a), float(s)) for q, c, a, s in zip(t5.quid, t5.candidate, t5.rank_RULE, t5.f_sat_top)}
    nsat = t5.groupby("quid").f_q_ncand_sat.first().map(lambda v: int(round(np.expm1(v)))).to_dict()
    blocks = {q: list(zip(g.candidate, g.S)) for q, g in sb.groupby("quid")}
    rows, n_no_parse, n_rule_missing = [], 0, 0
    for quid in sorted(keep):
        q = Q[quid]
        p = parse_obj(P[quid]) if quid in P else {}
        n_no_parse += quid not in P
        facts = F.get(quid, [])[:GC.N_FACTS]
        cands = []
        for i, (c, s) in enumerate(blocks.get(quid, [])[:GC.K_CANDS]):
            a = rr.get((quid, c))
            n_rule_missing += a is None
            cands.append(dict(candidate=c, s_rank=i, rule_rank=None if a is None else a[0], s_score=round(float(s), 4),
                              operator_satisfied=None if a is None else bool(a[1] > 0.5), fact_ids=fact_support(c, facts)))
        row = dict(quid=quid, question=q["question"], parse=p, candidates=cands,
                   facts=[list(f) for f in facts], operator_trace=trace_text(p, cands, nsat.get(quid, 0), facts) if p else "",
                   source="k100 frozen exports (gk_convert_k100.py)")
        if with_gold:
            row["gold"] = q["answers"]
        rows.append(row)
    GC.write_jsonl(out, rows)
    st = GC.check_bundles(GC.read_jsonl(out), "train" if with_gold else split_tag)
    st.update(split=split_tag, n_no_parse=n_no_parse, n_cand_without_rule_rank=n_rule_missing, seconds=round(time.time() - t0, 1),
              sha256=GC.sha256_file(out))
    return st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="MultiTQ data dir (questions/, kg/)")
    ap.add_argument("--exports", required=True); ap.add_argument("--rdata", required=True)
    ap.add_argument("--manifest", required=True); ap.add_argument("--train60k", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    tr60 = set(int(x) for x in json.load(open(a.train60k)))
    assert GC.list_sha256(tr60) == "0a1e618aefca3e3533d1752f6311257b7bd268b1ae3116654b21650a4b58c496"
    dsel = set(GC.load_list(a.manifest, "dev_sel2000"))
    E, R, Dq = a.exports, a.rdata, os.path.join(a.data, "questions")
    stats = {}
    stats["train"] = convert("train", f"{Dq}/train.json", f"{E}/train_parser_outputs.jsonl.gz", f"{R}/sblock_train60k.parquet",
                             f"{E}/train60k_top50_candidates.parquet", f"{E}/train60k_top20_facts.jsonl.gz", tr60, True,
                             os.path.join(a.out, "train.jsonl.gz"))
    stats["dev_sel2000"] = convert("dev_sel2000", f"{Dq}/dev.json", f"{E}/dev_parser_outputs.jsonl.gz", f"{R}/sblock_dev.parquet",
                                   f"{E}/dev_top50_candidates.parquet", f"{E}/dev_top20_facts.jsonl.gz", dsel, False,
                                   os.path.join(a.out, "dev_sel2000.jsonl.gz"))
    teq = set(q["quid"] for q in json.load(open(f"{Dq}/test.json")))
    stats["test"] = convert("test", f"{Dq}/test.json", f"{E}/test_parser_outputs.jsonl.gz", f"{R}/sblock_test.parquet",
                            f"{E}/test_top50_candidates.parquet", f"{E}/test_top20_facts.jsonl.gz", teq, False,
                            os.path.join(a.out, "test.jsonl.gz"))
    json.dump(stats, open(os.path.join(a.out, "convert_stats.json"), "w"), indent=1)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
