#!/bin/bash
# jobBC2.sh -- R2 arms on the GPU host GPU 0 only, sequential, cut order G -> P -> GU (registered GPU-hour cap, cut points).
# Frozen gk_infer.py / pk_infer.py / pk_execute.py / gu_retrieve.py, R1 adapters; resume-safe.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; K=$WORK_DIR; R=$K/robustness_r2; L=$K/gpu_usage_kpct.tsv; CAP=2.0
spent() { awk -F'\t' '$1=="robustness_r2"{s+=$7} END{printf "%.4f", s+0}' $L; }
gpu_cap_ok() { awk -v s=$(spent) -v p=$1 -v c=$CAP 'BEGIN{exit !(s+p<=c)}'; }
mkdir -p $J/out $R/G $R/GU $R/P/raw $R/gu
declare -A AD=( [k01_s0]=/kpct/guided/adapters/G_k01_s0_ordS_s0/ep2 [k05_s0]=/kpct/guided/adapters/G_k05_s0_ordS_s0/ep2 [k100]=/kpct/guided/adapters/G_k100_ordS_s0/ep1 )
for k in k01_s0 k05_s0 k100; do
  [ -s $R/G/$k/pred_G_variants_G_$k.jsonl ] && continue
  gpu_cap_ok 0.10 || { echo "NOT_RUN G_$k (GPU-hour cap: spent $(spent))" | tee -a $J/out/not_run.txt; continue; }
  mkdir -p $R/G/$k
  bash $R/code/run_card_r2.sh 0 ${VLLM_IMAGE:-tkgqa-vllm:latest} G_$k "python3 -u /kpct/guided/code/gk_infer.py --arm G --split variants --bundles /kpct/robustness_r2/bundles/$k/variants.jsonl --adapters G_$k=${AD[$k]} --order S --s_list /kpct/robustness_r2/preds/$k/variants_S_seed0.jsonl --kg /data/kg/full.txt --out /kpct/robustness_r2/G/$k" > $R/logs/G_$k.log 2>&1
  echo "G $k rc=$?"
done
if [ ! -s $R/P/raw/raw_P_variants_P_k100.jsonl.gz ]; then
  if gpu_cap_ok 0.10; then
    bash $R/code/run_card_r2.sh 0 ${VLLM_IMAGE:-tkgqa-vllm:latest} P_infer "python3 -u /kpct/parser/code/pk_infer.py --split variants --bundles /kpct/robustness_r2/bundles/k01_s0/variants.jsonl --adapters P_k01_s0=/kpct/parser/adapters/P_k01_s0_s0/ep3,P_k05_s0=/kpct/parser/adapters/P_k05_s0_s0/ep1,P_k100=/kpct/parser/adapters/P_k100_s0/ep0.5 --out /kpct/robustness_r2/P/raw" > $R/logs/P_infer.log 2>&1
    echo "P infer rc=$?"
  else echo "NOT_RUN P_infer (GPU-hour cap: spent $(spent))" | tee -a $J/out/not_run.txt; fi
fi
for k in k01_s0 k05_s0 k100; do
  [ -s $R/P/raw/raw_P_variants_P_$k.jsonl.gz ] || continue
  [ -s $R/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed2.jsonl ] && continue
  mkdir -p $R/P/$k
  bash $R/code/run_cpu_r2.sh P_exec_$k 8 "python3 -u /kpct/parser/code/pk_execute.py --k $k --split variants --raw P_$k=/kpct/robustness_r2/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles /kpct/robustness_r2/bundles/$k/variants.jsonl --preds /kpct/robustness_r2/preds --taus=off --out /kpct/robustness_r2/P/$k --workers 8" > $R/logs/P_exec_$k.log 2>&1
  echo "P exec $k rc=$?"
done
if [ ! -s $R/gu/gu_variants.jsonl.gz ] && gpu_cap_ok 0.02; then
  bash $R/code/run_card_r2.sh 0 ${GPU_IMAGE:-tkgqa-gpu:latest} GU_retrieve "python3 -u /kpct/guided/code/gu_retrieve.py --mode queries --kg /data/kg/full.txt --emb /kpct/gu_retrieval/fact_emb_f16.npy --qfile /kpct/robustness_r2/inputs/variants.jsonl --name gu_variants --out /kpct/robustness_r2/gu" > $R/logs/GU_retrieve.log 2>&1
  echo "GU retrieve rc=$?"
fi
declare -A GUA=( [k01_s0]=/kpct/guided/adapters/GU_k01_s0_s0 [k100]=/kpct/guided/adapters/GU_k100_s0 )
declare -A GUE=( [k01_s0]=ep3 [k100]=ep1 )
for k in k01_s0 k100; do
  [ -s $R/gu/gu_variants.jsonl.gz ] || break
  [ -s $R/GU/$k/pred_GU_variants_GU_$k.jsonl ] && continue
  gpu_cap_ok 0.08 || { echo "NOT_RUN GU_$k (GPU-hour cap: spent $(spent))" | tee -a $J/out/not_run.txt; continue; }
  mkdir -p $R/GU/$k
  bash $R/code/run_card_r2.sh 0 ${VLLM_IMAGE:-tkgqa-vllm:latest} GU_$k "python3 -u /kpct/guided/code/gk_infer.py --arm GU --split variants --questions /kpct/robustness_r2/inputs/variants_questions.json --gu_facts /kpct/robustness_r2/gu/gu_variants.jsonl.gz --gu_pad ${GUA[$k]}/gu_pad.json --adapters GU_$k=${GUA[$k]}/${GUE[$k]} --kg /data/kg/full.txt --out /kpct/robustness_r2/GU/$k" > $R/logs/GU_$k.log 2>&1
  echo "GU $k rc=$?"
done
cp $R/logs/G_*.log $R/logs/P_*.log $R/logs/GU_*.log $J/out/ 2>/dev/null
for k in k01_s0 k05_s0 k100; do cp $R/G/$k/infer_stats_G_variants.json $J/out/infer_stats_G_variants_$k.json 2>/dev/null; cp $R/P/$k/exec_stats_P_${k}_variants.json $J/out/ 2>/dev/null; done
for k in k01_s0 k100; do cp $R/GU/$k/infer_stats_GU_variants.json $J/out/infer_stats_GU_variants_$k.json 2>/dev/null; done
cp $R/P/raw/infer_stats_P_variants.json $R/gu/stats_queries_gu_variants.json $J/out/ 2>/dev/null
grep robustness_r2 $L > $J/out/usage_log_robustness_r2.tsv; echo "spent $(spent)"
