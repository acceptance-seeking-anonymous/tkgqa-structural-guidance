#!/bin/bash
# reproduce_kpct_6_parser.sh -- Parser P_k: answer-consistent silver targets (CPU), LoRA training, dev_sel2000 decoding
# of every checkpoint, symbolic execution x confidence threshold tau, selection, pre-test freeze, test decoding +
# execution, test freeze. Card 1.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
K=$WORK_DIR; P=$K/parser; cd "$P/code"; mkdir -p "$P/silver"
for L in k01_s0 k05_s0 k10_s0 k100; do
  [ -f "$P/silver/silver_${L}_train.jsonl" ] && continue
  bash run_cpu_p.sh silver_$L "$P" 8 pylib "python3 /kpct/parser/code/pk_silver.py --subset $L --out /kpct/parser/silver --workers 8"
done
for A in "k01_s0 k01_s0" "k100 k100" "k10_s0 k10_s0" "k05_s0 k05_s0"; do   # registered run order
  set -- $A; bash pk_arm.sh $1 $2 1 all
done
