#!/bin/bash
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
K=$WORK_DIR; Q=$K/parser_clean; C=$Q/code
mkdir -p $C $Q/logs
# run from a job directory holding this folder's files, the shared parser files (kpct/parser: pk_common.py, pk_infer.py,
# pk_execute.py, pk_select.py, pk_train.py, run_cpu_p.sh) and protocol/multitq_p1_clean_silver_rule.json
cp -L ./*.py ./*.sh $C/ 2>/dev/null
cp multitq_p1_clean_silver_rule.json $Q/multitq_p1_clean_silver_rule.json
python3 - <<'PY'
import json, hashlib, os
K = os.environ["WORK_DIR"]; C = K + "/parser_clean/code"
r = json.load(open(K + "/parser_clean/multitq_p1_clean_silver_rule.json"))
d = {k: v for k, v in r.items() if k != "sha256"}
if not (hashlib.sha256(json.dumps(d, indent=1, sort_keys=True, default=str).encode()).hexdigest() == r["sha256"] == "004023a3d1cf3bd32ed25d5a46b8cfffc7c0f365bc04921d4a62786d6286b5bb"):
    print("WARNING: rule file differs from the registered rule (the shipped protocol copy is anonymised)")
sh = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
bad = [f for f, h in r["code_sha256_at_registration"].items() if sh(f"{C}/{f}") != h]
if bad:
    print("WARNING: code differs from the hashes recorded at registration (anonymised release):", bad)
same = {f: sh(f"{C}/{f}") == sh(f"{K}/parser/code/{f}") for f in ("pk_common.py", "pk_infer.py", "pk_execute.py", "pk_select.py", "run_cpu_p.sh")}
assert all(same.values()), same
print("shared parser files equal the parser/code copies")
PY
set +e
for PH in $PHASES; do
  bash $C/cs_driver.sh $PH; rc=$?
  echo "phase $PH rc=$rc"
  [ $rc = 0 ] || break
done
mkdir -p out
cp $Q/logs/*.log out/ 2>/dev/null
cp $Q/silver/*_manifest.json out/ 2>/dev/null
[ -f $Q/freeze/multitq_p1_clean_silver_freeze.json ] && cp $Q/freeze/multitq_p1_clean_silver_freeze.json out/
[ -f $Q/multitq_p1_clean_silver_results.json ] && cp $Q/multitq_p1_clean_silver_results.json out/
for V in clean rerun order; do
  [ -f $Q/adapters/P_k01_s0${V}_s0/train_stats.json ] && cp $Q/adapters/P_k01_s0${V}_s0/train_stats.json out/train_stats_$V.json
  [ -f $Q/select/$V/select_P_k01_s0.json ] && cp $Q/select/$V/select_P_k01_s0.json out/select_$V.json
done
[ -f $Q/raw/infer_stats_P_dev_sel2000.json ] && cp $Q/raw/infer_stats_P_dev_sel2000.json out/
[ -f $Q/exec/exec_stats_P_k01_s0_dev_sel2000.json ] && cp $Q/exec/exec_stats_P_k01_s0_dev_sel2000.json out/
grep -P "^clean_silver_p1\t" $K/gpu_usage_kpct.tsv > out/usage_log_clean_silver_p1.tsv || true
docker ps --format '{{.Names}}' | grep '^cs_' > out/leftover_containers.txt || true
ls out
exit ${rc:-0}
