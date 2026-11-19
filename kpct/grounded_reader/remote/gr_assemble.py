#!/usr/bin/env python3
"""gr_assemble.py -- grounded_reader track: per-row prediction files of a Parser-grounded Reader variant (gold-free).

For level k, build split s and source m, every row of build/<k>/<s>/status_<m>.jsonl (optionally filtered by vidx) gets
  fallback    -> the frozen G_k ranked list of that row (Reader on the structural bundle; = P_k's fallback logic)
  same_prompt -> gk_common.g_ranked(frozen G_k answer, Parser-pool S seed-0 list)  (prompt-identity reuse)
  generate    -> the gr_infer.py prediction of the row (pred_G_<label>_G_<k>.jsonl files given with --gen)
A row to generate without a prediction is 'missing' (ranked = [], a miss) and counted; the caller decides (STOP rule).
Writes <out>.jsonl.gz ({"quid", "ranked"}), <out>.src.json (per-source counts, missing quids, sha256 of inputs/outputs).
"""
import argparse, glob, gzip, hashlib, json, os, sys
sys.path.insert(0, os.environ.get("GK_CODE", "/kpct/guided/code"))
import gk_common as GC          # frozen (g_ranked)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_ranked(p):
    d = {}
    op = gzip.open if p.endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        for l in f:
            if l.strip():
                x = json.loads(l); rk = x["ranked"]
                d[int(x["quid"])] = json.loads(rk.replace("'", '"')) if isinstance(rk, str) else rk
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, help="build/<k>/<split> dir")
    ap.add_argument("--mode", required=True)
    ap.add_argument("--frozen_g", required=True)
    ap.add_argument("--gen", default="", help="comma list of gr_infer prediction files (globs allowed)")
    ap.add_argument("--vidx", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep_v = set(int(x) for x in a.vidx.split(",")) if a.vidx else None
    S = [json.loads(l) for l in open(os.path.join(a.build, f"status_{a.mode}.jsonl")) if l.strip()]
    S = [s for s in S if keep_v is None or s["quid"] % 10 in keep_v]
    s0 = read_ranked(os.path.join(a.build, f"s0_{a.mode}.jsonl"))
    FG = read_ranked(a.frozen_g)
    gen_files = sorted({f for g in a.gen.split(",") if g for f in glob.glob(g)})
    GEN = {}
    for f in gen_files:
        for q, rk in read_ranked(f).items():
            assert q not in GEN, f"quid {q} generated twice"
            GEN[q] = rk
    cnt = dict(fallback=0, reused=0, generated=0, missing=0)
    missing, rows = [], []
    for s in sorted(S, key=lambda x: x["quid"]):
        q = s["quid"]
        if s["status"] != "ok":
            rk = FG[q]; cnt["fallback"] += 1
        elif s["same_prompt"]:
            rk = GC.g_ranked(FG[q][0], s0[q]); cnt["reused"] += 1
        elif q in GEN:
            rk = GEN[q]; cnt["generated"] += 1
        else:
            rk = []; cnt["missing"] += 1; missing.append(q)
        rows.append({"quid": q, "ranked": rk})
    unused = sorted(set(GEN) - {s["quid"] for s in S if s["status"] == "ok" and not s["same_prompt"]})
    with gzip.open(a.out + ".jsonl.gz", "wt", encoding="utf-8", compresslevel=6) as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = dict(build=a.build, mode=a.mode, vidx=a.vidx, n=len(rows), counts=cnt, missing_quids=missing[:200],
               unused_generated=len(unused), frozen_g=[a.frozen_g, sha256_file(a.frozen_g)],
               gen_files={f: sha256_file(f) for f in gen_files},
               status_sha256=sha256_file(os.path.join(a.build, f"status_{a.mode}.jsonl")),
               out=os.path.basename(a.out) + ".jsonl.gz", out_sha256=sha256_file(a.out + ".jsonl.gz"))
    json.dump(rep, open(a.out + ".src.json", "w"), indent=1)
    print(json.dumps({"out": rep["out"], "n": len(rows), **cnt, "unused": len(unused)}))


if __name__ == "__main__":
    main()
