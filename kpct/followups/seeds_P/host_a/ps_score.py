#!/usr/bin/env python3
"""ps_score.py -- parser LoRA-seed amendment: score the FROZEN test prediction files (sha256 re-verified before any
file is read) with the registered scorer (multitq_eval.hit_at_k, sha256 1f85bbdc...). Writes per-question hits
(one column per LoRA seed x S seed, @1 and @10) keyed by quid plus qlabel bucket and answer_type. Also re-scores the
already-registered seed-0 files (frozen parser freeze) as an identity check against the registered per-question hits.
"""
import argparse, csv, gzip, hashlib, json, os, sys, time
K = os.path.expanduser("$WORK_DIR"); PS = os.path.join(K, "parser_seeds")
def sh(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()
ap = argparse.ArgumentParser(); ap.add_argument("--test_json", required=True); ap.add_argument("--scorer", required=True)
ap.add_argument("--ref", required=True, help="json {col: {relpath: sha256}} for the registered seed-0 arms"); ap.add_argument("--out", required=True)
a = ap.parse_args()
assert sh(a.scorer) == "1f85bbdcd623f1810e9df2910e042c24d6f5c269d278e82b30ef05387a1e2d69"
assert sh(a.test_json) == "d9bf3bf23b489ecc76c7940510c718260a42350d934a6edc1c1c7efcd5238fee"
sys.path.insert(0, os.path.dirname(a.scorer)); import multitq_eval as ME
Q = json.load(open(a.test_json)); assert len(Q) == 54584
comb = json.load(open(os.path.join(PS, "multitq_kpct_seed_freeze_P.json")))
COL = {"P_k05_s0": "P_5", "P_k01_s0": "P_1"}
files = {}
for arm, d in comb["arms"].items():
    if not d["test_pred_files"]: continue
    base, lora = arm.rsplit("_lora", 1); col = f"{COL[base]}_lora{lora}"
    files[col] = d["test_pred_files"]
files.update(json.load(open(a.ref)))
ver = {}
for col, fs in files.items():
    for p, h in fs.items():
        got = sh(os.path.join(K, p)); assert got == h, f"sha256 mismatch {p}"; ver[p] = h
print(f"verified {len(ver)} files before scoring")
cols, hits = [], {}
for col, fs in sorted(files.items()):
    for p in sorted(fs):
        s = int(p.rsplit("_seed", 1)[1].split(".")[0]); P = ME.load_predictions(os.path.join(K, p))
        assert set(P) == {int(q["quid"]) for q in Q}, f"{p} coverage"
        for kk in (1, 10):
            c = f"{col}_S{s}@{kk}"; cols.append(c)
            hits[c] = [ME.hit_at_k(q, P.get(int(q["quid"])), kk) for q in Q]
with gzip.open(a.out, "wt", newline="") as f:
    w = csv.writer(f); w.writerow(["quid", "qlabel_bucket", "answer_type", "qtype"] + cols)
    for i, q in enumerate(Q):
        w.writerow([q["quid"], ME.qlabel_bucket(q), q["answer_type"], q.get("qtype")] + [hits[c][i] for c in cols])
summ = dict(scored_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n=len(Q), verified_files=ver, out=os.path.relpath(a.out, K), out_sha256=sh(a.out),
            means={c: sum(hits[c]) / len(Q) for c in cols})
json.dump(summ, open(a.out.replace(".csv.gz", "_summary.json"), "w"), indent=1)
print(json.dumps(summ["means"]))
