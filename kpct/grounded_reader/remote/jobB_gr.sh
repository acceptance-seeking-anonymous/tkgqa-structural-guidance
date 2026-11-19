#!/bin/bash
# jobB_gr.sh <card> <levels> -- grounded_reader DEV session: plan (CPU) -> ONE vLLM container on ONE card of the GPU host (run_card_gr.sh) -> assembly (CPU) -> dev scoring + selection (CPU,
# dev gold only). GPU-h guard: budget_s = (CAP - spent(grounded_reader) - 0.02) h - 60 s; per-item NOT_RUN inside gr_infer.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=${1:-1}; LEVELS=${2:-k05_s0,k100}
J=$PWD; K=$WORK_DIR; GR=$K/grounded_reader; L=$K/gpu_usage_kpct.tsv; CAP=2.0
spent() { awk -F'\t' '$1=="grounded_reader"{s+=$7} END{printf "%.4f", s+0}' $L; }
BUD=$(awk -v s=$(spent) -v c=$CAP 'BEGIN{printf "%.0f", (c - s - 0.02)*3600 - 60}')
RC="bash $GR/code/run_cpu_gr.sh"; RCG="bash $GR/code/run_card_gr.sh $CARD"
D=/kpct/grounded_reader/gen/dev
mkdir -p $GR/gen/dev
echo "spent $(spent) budget_s $BUD" | tee $J/budget_dev.txt
$RC prep_dev 2 "python3 -u /kpct/grounded_reader/code/gr_prep.py dev --levels $LEVELS --out $D --budget_s $BUD --gpu_mem ${GPU_MEM:-0.40}" > $GR/logs/prep_dev.log 2>&1 || { cat $GR/logs/prep_dev.log; exit 4; }
cp $GR/gen/dev/counts_dev.json $GR/gen/dev/plan_dev.json $J/
$RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} DEV_$(echo $LEVELS | tr ',' '_') $((BUD + 300)) "python3 -u /kpct/grounded_reader/code/gr_infer.py --plan $D/plan_dev.json --stats $D/infer_stats_dev.json" > $GR/logs/infer_dev.log 2>&1
echo "infer rc=$?" | tee -a $J/budget_dev.txt
declare -A GT=( [k01_s0]=ordS_ep2 [k05_s0]=main_ep2 [k100]=main_ep1 )
for k in $(echo $LEVELS | tr ',' ' '); do for m in PALONE PK; do
  A=$D/assembled/$k/$m; mkdir -p $GR/gen/dev/assembled/$k/$m
  $RC asm_dev 2 "python3 -u /kpct/grounded_reader/code/gr_assemble.py --build /kpct/grounded_reader/build/$k/dev --mode $m --frozen_g /kpct/guided/preds/G_$k/pred_G_dev_sel2000_${GT[$k]}.jsonl --gen $D/$k/$m/pred_G_dev_G_$k.jsonl --out $A/dev && \
    python3 -u /kpct/grounded_reader/code/gr_assemble.py --build /kpct/grounded_reader/build/$k/llm --mode $m --vidx 5 --frozen_g /kpct/robustness_r3/llm/G/$k/pred_G_variants_G_$k.jsonl --gen $D/$k/$m/pred_G_devpara_G_$k.jsonl --out $A/devpara" >> $GR/logs/asm_dev.log 2>&1
done; done
$RC devscore 2 "python3 -u /kpct/grounded_reader/code/gr_dev_score.py --levels $LEVELS --out $D/selection_dev_$(echo $LEVELS | tr ',' '_').json" > $GR/logs/devscore.log 2>&1
cp $GR/logs/infer_dev.log $GR/logs/asm_dev.log $GR/logs/devscore.log $GR/gen/dev/infer_stats_dev.json $GR/gen/dev/selection_dev_*.json $J/ 2>/dev/null
mkdir -p $J/assembled; (cd $GR/gen/dev/assembled && tar czf $J/assembled/dev_assembled.tgz .)
(cd $GR/gen/dev && tar czf $J/dev_generated.tgz */*/pred_G_*.jsonl */*/raw_G_*.jsonl.gz */REUSECHECK/*.jsonl 2>/dev/null)
head -1 $L > $J/usage_log_grounded_reader.tsv; grep -P "^grounded_reader\t" $L >> $J/usage_log_grounded_reader.tsv; echo "spent $(spent)" | tee -a $J/budget_dev.txt
