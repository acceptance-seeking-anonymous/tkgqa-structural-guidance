#!/bin/bash
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -e
K=$WORK_DIR; P=$K/parser
mkdir -p out
python3 - <<'PY'
import json, hashlib, os
K = os.path.expanduser("$WORK_DIR")
def qfile(src, dst):
    rows = []
    for l in open(src):
        r = json.loads(l); assert not r.get("gold"), "split bundle with gold"
        rows.append((int(r["quid"]), r["question"]))
    rows.sort()
    with open(dst, "w") as f:
        for q, t in rows: f.write(json.dumps({"quid": q, "question": t}, ensure_ascii=False) + "\n")
    return dict(rows)
info = {}
for split in ["dev_sel2000", "test"]:
    a = qfile(f"{K}/bundles/k05_s0/{split}.jsonl", f"out/q_{split}.jsonl")
    b = qfile(f"{K}/bundles/k01_s0/{split}.jsonl", f"out/ps_q_{split}_k01.jsonl")
    info[split] = dict(n=len(a), identical_k05_k01=(a == b), sha256=hashlib.sha256(open(f"out/q_{split}.jsonl", "rb").read()).hexdigest())
sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
def walk(d, out):
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str) and len(v) == 64 and "/" in k and os.path.exists(f"{K}/{k}"):
                out[k] = (sh(f"{K}/{k}") == v)
            else:
                walk(v, out)
    elif isinstance(d, list):
        for v in d: walk(v, out)
for L in ["k05_s0", "k01_s0"]:
    fz = json.load(open(f"{K}/parser/freeze/freeze_P_{L}.json"))
    out = {}; walk(fz, out)
    info[f"hashes_now_vs_freeze_{L}"] = out
    info[f"freeze_keys_{L}"] = list(fz.keys())
json.dump(info, open("out/prep_info.json", "w"), indent=1)
print(json.dumps({k: (v if not isinstance(v, dict) or len(v) < 6 else {"all_true": all(v.values()), "n": len(v)}) for k, v in info.items()}))
PY
for L in k05_s0 k01_s0; do
  cp $P/silver/silver_${L}_train.jsonl $P/silver/silver_${L}_manifest.json out/
  cp $P/adapters/P_${L}_s0/train_stats.json out/frozen_train_stats_${L}.json
  cp $P/adapters/P_${L}_s0/train_log.txt out/frozen_train_log_${L}.txt
  cp $P/freeze/freeze_P_${L}.json out/
  cp $P/raw/${L}/infer_stats_P_dev_sel2000.json out/frozen_infer_stats_dev_${L}.json
  cp $P/raw/${L}/infer_stats_P_test.json out/frozen_infer_stats_test_${L}.json
  cp $P/select/select_P_${L}.json out/frozen_select_${L}.json
done
cp $P/raw/k05_s0/raw_P_dev_sel2000_ep1.jsonl.gz out/frozen_raw_dev_k05_ep1.jsonl.gz
cp $P/raw/k05_s0/raw_P_dev_sel2000_ep2.jsonl.gz out/frozen_raw_dev_k05_ep2.jsonl.gz
mkdir -p out/frozen_P_k05_s0_ep1
cp $P/adapters/P_k05_s0_s0/ep1/adapter_config.json $P/adapters/P_k05_s0_s0/ep1/adapter_model.safetensors out/frozen_P_k05_s0_ep1/
cp $K/guided/code/gk_common.py out/
sha256sum out/frozen_P_k05_s0_ep1/* out/silver_*_train.jsonl out/q_*.jsonl
ls -la out
