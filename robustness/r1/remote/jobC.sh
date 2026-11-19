#!/bin/bash
# jobC.sh <card> -- GU retrieval (frozen gu_retrieve.py) + GU_k (frozen adapters, frozen gk_infer.py) + P_k parse
# inference (frozen pk_infer.py, the three frozen P adapters in one engine session) + P execution (frozen pk_execute.py,
# tau off, fallback = the variant's own S_k lists). Resume-safe.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=$1; J=$PWD; K=$WORK_DIR; R=$K/robustness
rc_all=0
mkdir -p $R/gu $R/GU $R/P/raw
if [ ! -s $R/gu/gu_variants.jsonl.gz ]; then
  bash $R/code/run_card_r.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} GU_retrieve "python3 -u /kpct/guided/code/gu_retrieve.py --mode queries --kg /data/kg/full.txt --emb /kpct/gu_retrieval/fact_emb_f16.npy --qfile /kpct/robustness/inputs/variants.jsonl --name gu_variants --out /kpct/robustness/gu" > $R/logs/GU_retrieve.log 2>&1
  rc=$?; echo "GU retrieve rc=$rc"; [ $rc -ne 0 ] && exit $rc
fi
declare -A GUA=( [k01_s0]=/kpct/guided/adapters/GU_k01_s0_s0 [k100]=/kpct/guided/adapters/GU_k100_s0 )
declare -A GUE=( [k01_s0]=ep3 [k100]=ep1 )
for k in k01_s0 k100; do
  if [ -s $R/GU/$k/pred_GU_variants_GU_$k.jsonl ]; then echo "GU $k exists"; continue; fi
  mkdir -p $R/GU/$k
  bash $R/code/run_card_r.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} GU_$k "python3 -u /kpct/guided/code/gk_infer.py --arm GU --split variants --questions /kpct/robustness/inputs/variants_questions.json --gu_facts /kpct/robustness/gu/gu_variants.jsonl.gz --gu_pad ${GUA[$k]}/gu_pad.json --adapters GU_$k=${GUA[$k]}/${GUE[$k]} --kg /data/kg/full.txt --out /kpct/robustness/GU/$k" > $R/logs/GU_$k.log 2>&1
  rc=$?; echo "GU $k rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
if [ ! -s $R/P/raw/raw_P_variants_P_k100.jsonl.gz ]; then
  bash $R/code/run_card_r.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} P_infer "python3 -u /kpct/parser/code/pk_infer.py --split variants --bundles /kpct/robustness/bundles/k01_s0/variants.jsonl --adapters P_k01_s0=/kpct/parser/adapters/P_k01_s0_s0/ep3,P_k05_s0=/kpct/parser/adapters/P_k05_s0_s0/ep1,P_k100=/kpct/parser/adapters/P_k100_s0/ep0.5 --out /kpct/robustness/P/raw" > $R/logs/P_infer.log 2>&1
  rc=$?; echo "P infer rc=$rc"; [ $rc -ne 0 ] && exit $rc
fi
for k in k01_s0 k05_s0 k100; do
  if [ -s $R/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed2.jsonl ]; then echo "P exec $k exists"; continue; fi
  mkdir -p $R/P/$k
  bash $R/code/run_cpu_r.sh P_exec_$k 8 "python3 -u /kpct/parser/code/pk_execute.py --k $k --split variants --raw P_$k=/kpct/robustness/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles /kpct/robustness/bundles/$k/variants.jsonl --preds /kpct/robustness/preds --taus=off --out /kpct/robustness/P/$k --workers 8" > $R/logs/P_exec_$k.log 2>&1
  rc=$?; echo "P exec $k rc=$rc"; [ $rc -ne 0 ] && rc_all=$rc
done
mkdir -p $J/out; cp $R/logs/GU_*.log $R/logs/P_*.log $J/out/ 2>/dev/null
cp $R/gu/stats_queries_gu_variants.json $R/P/raw/infer_stats_P_variants.json $J/out/ 2>/dev/null
for k in k01_s0 k100; do cp $R/GU/$k/infer_stats_GU_variants.json $J/out/infer_stats_GU_variants_$k.json 2>/dev/null; done
for k in k01_s0 k05_s0 k100; do cp $R/P/$k/exec_stats_P_${k}_variants.json $J/out/ 2>/dev/null; done
grep robustness $K/gpu_usage_kpct.tsv > $J/out/usage_log_robustness.tsv
exit $rc_all
