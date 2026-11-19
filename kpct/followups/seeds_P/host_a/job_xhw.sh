#!/bin/bash
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
K=$WORK_DIR; PS=$K/parser_seeds
mkdir -p $PS/code $PS/raw/P_k05_s0_lora0_4090 $PS/amendment
cp ps_dev.sh $PS/code/; cp multitq_kpct_seed_amendment_P.json $PS/amendment/
cp raw_P_dev_sel2000_ep1.jsonl.gz infer_stats_P_dev_sel2000.json $PS/raw/P_k05_s0_lora0_4090/
sha256sum $K/parser/code/pk_execute.py $K/parser/code/pk_select.py $K/parser/code/pk_common.py $K/parser/code/run_cpu_p.sh $PS/raw/P_k05_s0_lora0_4090/*
bash $PS/code/ps_dev.sh P_k05_s0_lora0_4090 k05_s0 k05_s0 ep1=raw_P_dev_sel2000_ep1.jsonl.gz ep1
mkdir -p out; cp $PS/select/P_k05_s0_lora0_4090/* $PS/exec/P_k05_s0_lora0_4090/exec_stats_* out/
python3 - <<'PY'
import json, glob
s = json.load(open(glob.glob("out/select_P_*.json")[0]))
print(json.dumps({k: s[k] for k in ["chosen","dev_sel2000_hits1","dev_sel2000_hits1_ci95","unseen_hits1","STOP_DEV_FAILURE"]})[:800])
for g in s["grid"]: print(g["checkpoint"], g["tau"], round(g["hits1"],4), g["hits1_per_seed"])
PY
tar czf out.tgz out
