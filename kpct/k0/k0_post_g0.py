#!/usr/bin/env python3
"""k0_post_g0.py -- G_0 generations -> prediction files (rule arms.G_0.output = as G_k; ranked list = [answer] + P_0 list).

Answer validity and canonicalisation exactly as G_k (gk_common.canon_answer: after stripping, whitespace collapsing,
case folding and _ -> space the output equals a KG entity name -> written as that KG name; or it is a time string
YYYY / YYYY-MM / YYYY-MM-DD within 2005-01-01..2015-12-31). An invalid answer stays at rank 1 (a miss); no fallback.
Ranked list = gk_common.g_ranked(answer, P_0 list) (deduplicated). Invalid rate reported. No answer field is read.
Output: pred_G0_<split>_<g>.jsonl ({"quid", "ranked"}), post_G0_<split>_<g>.jsonl.gz (raw + answer + valid),
post_stats_G0_<split>.json.
"""
import argparse, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gk_common as GC


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True)
    ap.add_argument("--raw", required=True, help="g=raw_G0_<split>_<g>.jsonl.gz[,...]")
    ap.add_argument("--p0_pred", required=True, help="P_0 prediction file of the split (selected design)")
    ap.add_argument("--kg", required=True, help="kg/full.txt (entity vocabulary for validity)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    kg = GC.load_kg(a.kg)
    enorm = GC.entity_norm_map(sorted({f[0] for f in kg} | {f[2] for f in kg}))
    P0 = {int(r["quid"]): r["ranked"] for r in GC.read_jsonl(a.p0_pred)}
    sf = os.path.join(a.out, f"post_stats_G0_{a.split}.json")
    st = json.load(open(sf)) if os.path.exists(sf) else dict(split=a.split, p0_pred=os.path.basename(a.p0_pred), out={})
    st["p0_pred_sha256"] = GC.sha256_file(a.p0_pred)
    for item in a.raw.split(","):
        d, path = item.split("=", 1)
        raw = GC.read_jsonl(path)
        assert set(int(r["quid"]) for r in raw) == set(P0), f"{d}: raw quids != P_0 prediction quids"
        post, preds, n_valid, n_override = [], [], 0, 0
        for r in raw:
            q = int(r["quid"])
            ans, ok = GC.canon_answer(r["text"], enorm)
            n_valid += ok
            p0 = P0[q]
            n_override += bool(ok and p0 and ans != p0[0])
            post.append(dict(r, answer=ans, valid=bool(ok)))
            preds.append(dict(quid=q, ranked=GC.g_ranked(ans, p0)))
        preds.sort(key=lambda x: x["quid"])
        pf = os.path.join(a.out, f"pred_G0_{a.split}_{d}.jsonl")
        GC.write_jsonl(pf, preds)
        xf = os.path.join(a.out, f"post_G0_{a.split}_{d}.jsonl.gz")
        GC.write_jsonl(xf, post)
        n = len(raw)
        st["out"][d] = dict(raw_file=os.path.basename(path), raw_sha256=GC.sha256_file(path), pred_file=os.path.basename(pf),
                            pred_sha256=GC.sha256_file(pf), post_sha256=GC.sha256_file(xf), n=n, invalid_rate=1 - n_valid / max(1, n),
                            override_rate_vs_p0_top1=n_override / max(1, n),
                            p0_empty_list_rate=sum(1 for r in raw if not P0[int(r["quid"])]) / max(1, n))
        print(d, json.dumps({k: st["out"][d][k] for k in ("n", "invalid_rate", "override_rate_vs_p0_top1")}), flush=True)
    st["code_sha256"] = {f: GC.sha256_file(os.path.join(HERE, f)) for f in ("k0_post_g0.py", "gk_common.py")}
    json.dump(st, open(sf, "w"), indent=1)
    os.system(f"chmod -R a+rX {a.out}")


if __name__ == "__main__":
    main()
