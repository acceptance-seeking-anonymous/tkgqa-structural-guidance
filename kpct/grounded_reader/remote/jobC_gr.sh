#!/bin/bash
# jobC_gr.sh <card> <levels> -- grounded_reader FINAL session for the dev-selected sources (selection.json staged in the job
# cwd, registered before this job): plan (CPU) -> ONE vLLM container (cut order of gr_prep.py final; per-item GPU-h guard)
# -> assembly of test, R3 rewrites, R1 P3, R2 rewrites (CPU) -> sha256 freeze list. NO scoring here (no gold read).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=${1:-1}; LEVELS=${2:-k05_s0,k100}
J=$PWD; K=$WORK_DIR; GR=$K/grounded_reader; L=$K/gpu_usage_kpct.tsv; CAP=2.0
spent() { awk -F'\t' '$1=="grounded_reader"{s+=$7} END{printf "%.4f", s+0}' $L; }
BUD=$(awk -v s=$(spent) -v c=$CAP 'BEGIN{printf "%.0f", (c - s - 0.02)*3600 - 60}')
RC="bash $GR/code/run_cpu_gr.sh"; RCG="bash $GR/code/run_card_gr.sh $CARD"
TAG=$(echo $LEVELS | tr ',' '_'); D=/kpct/grounded_reader/gen/final_$TAG
mkdir -p $GR/gen/final_$TAG; cp $J/selection.json $GR/gen/final_$TAG/selection.json
echo "spent $(spent) budget_s $BUD" | tee $J/budget_final.txt
$RC prep_final 2 "python3 -u /kpct/grounded_reader/code/gr_prep.py final --levels $LEVELS --selection $D/selection.json --out $D --budget_s $BUD --gpu_mem ${GPU_MEM:-0.40} --gen_root $D" > $GR/logs/prep_final_$TAG.log 2>&1 || { cat $GR/logs/prep_final_$TAG.log; exit 4; }
cp $GR/gen/final_$TAG/counts_final.json $GR/gen/final_$TAG/plan_final.json $J/
$RCG ${VLLM_IMAGE:-tkgqa-vllm:latest} FINAL_$TAG $((BUD + 300)) "python3 -u /kpct/grounded_reader/code/gr_infer.py --plan $D/plan_final.json --stats $D/infer_stats_final.json" > $GR/logs/infer_final_$TAG.log 2>&1
echo "infer rc=$?" | tee -a $J/budget_final.txt
declare -A GT=( [k01_s0]=ordS_ep2 [k05_s0]=main_ep2 [k100]=main_ep1 )
for k in $(echo $LEVELS | tr ',' ' '); do
  m=$(python3 -c "import json; print(json.load(open('$J/selection.json')).get('$k',''))")
  [ "$m" = PALONE ] || [ "$m" = PK ] || continue
  A=/kpct/grounded_reader/final/$k/$m; mkdir -p $GR/final/$k/$m; G=$D/$k/$m
  $RC asm_final 2 "set -e; P=/kpct/grounded_reader/code/gr_assemble.py; \
    python3 -u \$P --build /kpct/grounded_reader/build/$k/test --mode $m --frozen_g /kpct/guided/preds/G_$k/pred_G_test_${GT[$k]}.jsonl --gen '$G/pred_G_testr3orig_G_$k.jsonl,$G/pred_G_testr12orig_G_$k.jsonl,$G/pred_G_testrest_G_$k.jsonl' --out $A/test; \
    python3 -u \$P --build /kpct/grounded_reader/build/$k/llm --mode $m --vidx 4 --frozen_g /kpct/robustness_r3/llm/G/$k/pred_G_variants_G_$k.jsonl --gen $G/pred_G_r3rw_G_$k.jsonl --out $A/r3rw; \
    python3 -u \$P --build /kpct/grounded_reader/build/$k/r1 --mode $m --frozen_g /kpct/robustness/G/$k/pred_G_variants_G_$k.jsonl --gen $G/pred_G_r1p3_G_$k.jsonl --out $A/r1p3; \
    python3 -u \$P --build /kpct/grounded_reader/build/$k/r2 --mode $m --frozen_g /kpct/robustness_r2/G/$k/pred_G_variants_G_$k.jsonl --gen $G/pred_G_r2rw_G_$k.jsonl --out $A/r2rw" >> $GR/logs/asm_final_$TAG.log 2>&1
  echo "asm $k $m rc=$?" | tee -a $J/budget_final.txt
done
(cd $GR && find final -name '*.jsonl.gz' -o -name '*.src.json' | sort | xargs sha256sum) > $J/final_sha256.txt
(cd $GR && find gen/final_$TAG -name 'pred_G_*.jsonl' -o -name 'raw_G_*.jsonl.gz' | sort | xargs sha256sum) > $J/final_gen_sha256.txt
cp $GR/logs/infer_final_$TAG.log $GR/logs/asm_final_$TAG.log $GR/gen/final_$TAG/infer_stats_final.json $J/ 2>/dev/null
mkdir -p $J/final; (cd $GR/final && tar czf $J/final/final_assembled_$TAG.tgz .)
(cd $GR/gen/final_$TAG && tar czf $J/final/final_generated_$TAG.tgz */*/pred_G_*.jsonl */*/raw_G_*.jsonl.gz 2>/dev/null)
head -1 $L > $J/usage_log_grounded_reader.tsv; grep -P "^grounded_reader\t" $L >> $J/usage_log_grounded_reader.tsv; echo "spent $(spent)" | tee -a $J/budget_final.txt
