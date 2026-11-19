#!/usr/bin/env python3
"""gk_crosscheck_k100.py -- agreement between the structural track's k100 bundles (used by G_100) and the guided track's
conversion of the frozen k100 exports (gk_convert_k100.py; cross-check only). No answers of dev/test are read; train
gold fields are compared with each other only."""
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gk_common as GC


def top_rel(p):
    if p.get("relations"):
        r = p["relations"][0]
        return (r["relation"], r["direction"]) if isinstance(r, dict) else (r[0], r[1])
    return (p.get("relation"), p.get("direction"))


def first_ent(p):
    for m in p.get("mentions") or p.get("entities") or []:
        alts = m.get("alternatives") or []
        n = m.get("entity") or (GC._alt_name(alts[0]) if alts else None)
        if n:
            return n
    return None


def lab(x):
    return x.get("label") if isinstance(x, dict) else x


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--struct", required=True); ap.add_argument("--conv", required=True)
    ap.add_argument("--out", required=True); a = ap.parse_args()
    res = {}
    for split in ("train", "dev_sel2000", "test"):
        sp = [p for p in (os.path.join(a.struct, split + e) for e in (".jsonl", ".jsonl.gz")) if os.path.exists(p)][0]
        cp = [p for p in (os.path.join(a.conv, split + e) for e in (".jsonl", ".jsonl.gz")) if os.path.exists(p)][0]
        Sr = {int(r["quid"]): r for r in GC.read_jsonl(sp)}
        Cr = {int(r["quid"]): r for r in GC.read_jsonl(cp)}
        common = sorted(set(Sr) & set(Cr))
        st = dict(n_struct=len(Sr), n_conv=len(Cr), n_common=len(common))
        acc = {k: [] for k in ("question", "top_relation", "first_entity", "qtype", "answer_type", "granularity", "time",
                              "n_cands_equal", "top1_cand", "cand_list_equal", "cand_jaccard", "rule_rank_equal_on_shared",
                              "s_score_absdiff_on_shared", "facts_set_equal", "facts_list_equal", "facts_jaccard", "gold_equal")}
        for q in common:
            s_, c_ = Sr[q], Cr[q]
            bs, bc = GC.normalize_bundle(s_), GC.normalize_bundle(c_)
            ps, pc = s_["parse"], c_["parse"]
            acc["question"].append(s_["question"] == c_["question"])
            acc["top_relation"].append(top_rel(ps) == top_rel(pc))
            acc["first_entity"].append(first_ent(ps) == first_ent(pc))
            acc["qtype"].append(lab(ps.get("qtype")) == pc.get("qtype"))
            acc["answer_type"].append(lab(ps.get("answer_type")) == pc.get("answer_type"))
            acc["granularity"].append(lab(ps.get("granularity")) == pc.get("granularity"))
            ts = ps.get("time"); ts = ts.get("value") if isinstance(ts, dict) else ts
            acc["time"].append(ts == pc.get("time"))
            ls, lc = [x["candidate"] for x in bs["candidates"]], [x["candidate"] for x in bc["candidates"]]
            acc["n_cands_equal"].append(len(ls) == len(lc))
            acc["top1_cand"].append((ls[:1] == lc[:1]))
            acc["cand_list_equal"].append(ls == lc)
            u = set(ls) | set(lc)
            acc["cand_jaccard"].append(len(set(ls) & set(lc)) / len(u) if u else 1.0)
            ms = {x["candidate"]: x for x in bs["candidates"]}; mc = {x["candidate"]: x for x in bc["candidates"]}
            for k in set(ms) & set(mc):
                acc["rule_rank_equal_on_shared"].append(ms[k]["rule_rank"] == mc[k]["rule_rank"])
                if ms[k]["s_score"] is not None and mc[k]["s_score"] is not None:
                    acc["s_score_absdiff_on_shared"].append(abs(float(ms[k]["s_score"]) - float(mc[k]["s_score"])))
            fs, fc = [tuple(f) for f in bs["facts"]], [tuple(f) for f in bc["facts"]]
            acc["facts_set_equal"].append(set(fs) == set(fc)); acc["facts_list_equal"].append(fs == fc)
            uf = set(fs) | set(fc)
            acc["facts_jaccard"].append(len(set(fs) & set(fc)) / len(uf) if uf else 1.0)
            if split == "train":
                acc["gold_equal"].append(list(s_.get("gold") or []) == list(c_.get("gold") or []))
        for k, v in acc.items():
            if v:
                st[k] = float(np.median(v)) if k == "s_score_absdiff_on_shared" else float(np.mean(v))
        if acc["s_score_absdiff_on_shared"]:
            st["s_score_absdiff_max"] = float(np.max(acc["s_score_absdiff_on_shared"]))
        res[split] = st
        print(split, json.dumps(st), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
