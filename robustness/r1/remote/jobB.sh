#!/bin/bash
# jobB.sh <card> -- G_k on the variant bundles with the frozen guided adapters and the frozen gk_infer.py
# (vLLM 0.10.2, greedy, max 48 new tokens, max_model_len 2304, gpu_mem 0.85: the frozen session defaults). Resume-safe.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=$1; J=$PWD; K=$WORK_DIR; R=$K/robustness
declare -A AD=( [k01_s0]=/kpct/guided/adapters/G_k01_s0_ordS_s0/ep2 [k05_s0]=/kpct/guided/adapters/G_k05_s0_ordS_s0/ep2 [k100]=/kpct/guided/adapters/G_k100_ordS_s0/ep1 )
rc_all=0
for k in k01_s0 k05_s0 k100; do
  if [ -s $R/G/$k/pred_G_variants_G_$k.jsonl ]; then echo "G $k exists"; continue; fi
  mkdir -p $R/G/$k
  bash $R/code/run_card_r.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} G_$k "python3 -u /kpct/guided/code/gk_infer.py --arm G --split variants --bundles /kpct/robustness/bundles/$k/variants.jsonl --adapters G_$k=${AD[$k]} --order S --s_list /kpct/robustness/preds/$k/variants_S_seed0.jsonl --kg /data/kg/full.txt --out /kpct/robustness/G/$k" > $R/logs/G_$k.log 2>&1
  rc=$?; echo "G $k rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
mkdir -p $J/out; cp $R/logs/G_*.log $J/out/ 2>/dev/null; cp $R/G/*/infer_stats_*.json $J/out/ 2>/dev/null
for k in k01_s0 k05_s0 k100; do cp $R/G/$k/infer_stats_G_variants.json $J/out/infer_stats_G_variants_$k.json 2>/dev/null; done
grep robustness $K/gpu_usage_kpct.tsv > $J/out/usage_log_robustness.tsv
exit $rc_all
