#!/bin/bash
# jobBC3.sh <card> -- rule R3 frozen LLM arms on the REWRITE rows (R3 rewrites + dev paraphrases, root llm/) on ONE card of the GPU
# host, sequential, registered cut order: G_1, G_5, G_100 -> P (3 adapters, 1 session) + P exec (CPU) -> GU retrieval ->
# GU_1, GU_100, GU_5 -> GPLAIN_1. Frozen gk_infer.py / pk_infer.py / pk_execute.py / gu_retrieve.py (kpct) and
# gplain_infer.py (llm_controls), frozen adapters; vLLM gpu_memory_utilization 0.40 (as llm_controls amendment LC1-A1:
# other processes held ~48 GB of the card; GPU_MEM overrides it). Before every GPU item: spent(robustness_r3) + projection <= CAP, else NOT_RUN.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=${1:-0}
J=$PWD; K=$WORK_DIR; R3=$K/robustness_r3; L=$K/gpu_usage_kpct.tsv; CAP=1.2; GM=${GPU_MEM:-0.40}
O=/kpct/robustness_r3/llm
spent() { awk -F'\t' '$1=="robustness_r3"{s+=$7} END{printf "%.4f", s+0}' $L; }
gpu_cap_ok() { awk -v s=$(spent) -v p=$1 -v c=$CAP 'BEGIN{exit !(s+p<=c)}'; }
notrun() { echo "NOT_RUN $1 (GPU-hour cap: spent $(spent), projected $2, cap $CAP)" | tee -a $J/out/not_run.txt; }
RCG="bash $R3/code/run_card_r3.sh $CARD"; RCC="bash $R3/code/run_cpu_r3.sh"
mkdir -p $J/out $R3/llm/G $R3/llm/GU $R3/llm/P/raw $R3/llm/gu $R3/llm/GPLAIN
N=$(grep -c . $R3/llm/inputs/variants.jsonl)
PG=$(awk -v n=$N 'BEGIN{printf "%.3f", (70 + 0.075*n)/3600}')    # projection per G / GPLAIN container
PU=$(awk -v n=$N 'BEGIN{printf "%.3f", (70 + 0.055*n)/3600}')    # per GU container
PP=$(awk -v n=$N 'BEGIN{printf "%.3f", (70 + 0.060*n)/3600}')    # P session (3 adapters)
echo "n_rows $N projections G $PG GU $PU P $PP spent $(spent)" | tee $J/out/projections.txt
declare -A AD=( [k01_s0]=/kpct/guided/adapters/G_k01_s0_ordS_s0/ep2 [k05_s0]=/kpct/guided/adapters/G_k05_s0_ordS_s0/ep2 [k100]=/kpct/guided/adapters/G_k100_ordS_s0/ep1 )
for k in k01_s0 k05_s0 k100; do
  [ -s $R3/llm/G/$k/pred_G_variants_G_$k.jsonl ] && continue
  gpu_cap_ok $PG || { notrun G_$k $PG; continue; }
  mkdir -p $R3/llm/G/$k
  $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} G_$k 2400 "python3 -u /kpct/guided/code/gk_infer.py --arm G --split variants --bundles $O/bundles/$k/variants.jsonl --adapters G_$k=${AD[$k]} --order S --s_list $O/preds/$k/variants_S_seed0.jsonl --kg /data/kg/full.txt --gpu_mem $GM --out $O/G/$k" > $R3/logs/G_$k.log 2>&1
  echo "G $k rc=$?"
done
if [ ! -s $R3/llm/P/raw/raw_P_variants_P_k100.jsonl.gz ]; then
  if gpu_cap_ok $PP; then
    $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} P_infer 2400 "python3 -u /kpct/parser/code/pk_infer.py --split variants --bundles $O/bundles/k01_s0/variants.jsonl --adapters P_k01_s0=/kpct/parser/adapters/P_k01_s0_s0/ep3,P_k05_s0=/kpct/parser/adapters/P_k05_s0_s0/ep1,P_k100=/kpct/parser/adapters/P_k100_s0/ep0.5 --gpu_mem $GM --out $O/P/raw" > $R3/logs/P_infer.log 2>&1
    echo "P infer rc=$?"
  else notrun P_infer $PP; fi
fi
for k in k01_s0 k05_s0 k100; do
  [ -s $R3/llm/P/raw/raw_P_variants_P_$k.jsonl.gz ] || continue
  [ -s $R3/llm/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed2.jsonl ] && continue
  mkdir -p $R3/llm/P/$k
  $RCC P_exec_$k 8 "python3 -u /kpct/parser/code/pk_execute.py --k $k --split variants --raw P_$k=$O/P/raw/raw_P_variants_P_$k.jsonl.gz --bundles $O/bundles/$k/variants.jsonl --preds $O/preds --taus=off --out $O/P/$k --workers 8" > $R3/logs/P_exec_$k.log 2>&1
  echo "P exec $k rc=$?"
done
if [ ! -s $R3/llm/gu/gu_variants.jsonl.gz ]; then
  if gpu_cap_ok 0.02; then
    $RCG ${GPU_IMAGE:-tkgqa-gpu:latest} GU_retrieve 1200 "python3 -u /kpct/guided/code/gu_retrieve.py --mode queries --kg /data/kg/full.txt --emb /kpct/gu_retrieval/fact_emb_f16.npy --qfile $O/inputs/variants.jsonl --name gu_variants --out $O/gu" > $R3/logs/GU_retrieve.log 2>&1
    echo "GU retrieve rc=$?"
  else notrun GU_retrieve 0.02; fi
fi
declare -A GUA=( [k01_s0]=/kpct/guided/adapters/GU_k01_s0_s0 [k100]=/kpct/guided/adapters/GU_k100_s0 [k05_s0]=/kpct/llm_controls/adapters/GU_k05_s0_s0 )
declare -A GUE=( [k01_s0]=ep3 [k100]=ep1 [k05_s0]=ep2 )
for k in k01_s0 k100 k05_s0; do
  [ -s $R3/llm/gu/gu_variants.jsonl.gz ] || break
  [ -s $R3/llm/GU/$k/pred_GU_variants_GU_$k.jsonl ] && continue
  gpu_cap_ok $PU || { notrun GU_$k $PU; continue; }
  mkdir -p $R3/llm/GU/$k
  $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} GU_$k 2400 "python3 -u /kpct/guided/code/gk_infer.py --arm GU --split variants --questions $O/inputs/variants_questions.json --gu_facts $O/gu/gu_variants.jsonl.gz --gu_pad ${GUA[$k]}/gu_pad.json --adapters GU_$k=${GUA[$k]}/${GUE[$k]} --kg /data/kg/full.txt --gpu_mem $GM --out $O/GU/$k" > $R3/logs/GU_$k.log 2>&1
  echo "GU $k rc=$?"
done
if [ ! -s $R3/llm/GPLAIN/k01_s0/pred_G_variants_GPLAIN_k01_s0.jsonl ]; then
  if gpu_cap_ok $PG; then
    mkdir -p $R3/llm/GPLAIN/k01_s0
    $RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} GPLAIN_k01_s0 2400 "python3 -u /kpct/llm_controls/code/gplain_infer.py --arm G --split variants --bundles $O/bundles/k01_s0/variants.jsonl --adapters GPLAIN_k01_s0=/kpct/llm_controls/adapters/GPLAIN_k01_s0_s0/ep3 --order S --kg /data/kg/full.txt --gpu_mem $GM --out $O/GPLAIN/k01_s0" > $R3/logs/GPLAIN_k01_s0.log 2>&1
    echo "GPLAIN rc=$?"
  else notrun GPLAIN_1 $PG; fi
fi
cp $R3/logs/G_*.log $R3/logs/P_*.log $R3/logs/GU_*.log $R3/logs/GPLAIN_*.log $J/out/ 2>/dev/null
head -1 $L > $J/out/usage_log_robustness_r3.tsv; grep -P "^robustness_r3\t" $L >> $J/out/usage_log_robustness_r3.tsv; echo "spent $(spent)"
