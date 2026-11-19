#!/bin/bash
# gen_job_r3_completion.sh (amendment A3; = gen_job_r3.sh with ARMS_RESERVE 0) -- rule R3: DEV (dev_sel2000) + TRAINING (5% list k05_s0) paraphrases with Qwen2.5-72B-Instruct-AWQ on
# ONE card of the GPU host. GPU-hour cap of the track: CAP = 1.2 GPU-h (usage_log track
# robustness_r3); the training stage gets a wall-clock deadline = (CAP - spent - ARMS_RESERVE - MARGIN) h minus 90 s,
# so the arms reserve is never consumed by generation. Resume-safe (r3_qwen_gen.py skips written quids).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=${1:-0}
J=$PWD; K=$WORK_DIR; R3=$K/robustness_r3; L=$K/gpu_usage_kpct.tsv
CAP=1.2; ARMS_RESERVE=0.0; MARGIN=0.03   # completion session (amendment A3): every arm has run, so no arms reserve
mkdir -p $R3/code $R3/inputs $R3/gen $R3/logs $R3/checks
for f in rob_common.py r2_common.py r3_common.py r3_qwen_gen.py run_card_r3.sh run_cpu_r3.sh; do
  if [ -f $R3/code/$f ] && ! cmp -s $J/$f $R3/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code"; exit 2; fi
  cp -n $J/$f $R3/code/$f
done
cp -n $J/dev_qtext.json $J/train_order.json $R3/inputs/
(cd $R3 && sha256sum code/* inputs/*) > $J/code_inputs_sha_gen.txt
spent() { awk -F'\t' '$1=="robustness_r3"{s+=$7} END{printf "%.4f", s+0}' $L; }
S0=$(spent)
DL=$(awk -v c=$CAP -v s=$S0 -v a=$ARMS_RESERVE -v m=$MARGIN 'BEGIN{d=(c-s-a-m)*3600-90; if (d<0) d=0; printf "%d", d}')
echo "spent_before $S0 deadline_s $DL" | tee $J/budget_gen.txt
[ "$DL" -gt 600 ] || { echo "NOT_RUN generation: deadline $DL s < 600 s"; exit 3; }
MAXS=$((DL + 900))
bash $R3/code/run_card_r3.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} GEN_qwen_completion $MAXS \
  "GM=\$(python3 -c \"import os; print(max(0.30, int(100*(int(os.environ['CARD_FREE_MIB'])-1536)/97887)/100))\"); echo gpu_mem=\$GM; \
   python3 -u /kpct/robustness_r3/code/r3_qwen_gen.py --dev /kpct/robustness_r3/inputs/dev_qtext.json \
   --train_order /kpct/robustness_r3/inputs/train_order.json --train_q /data/questions/train.json \
   --out /kpct/robustness_r3/gen --deadline_s $DL --gpu_mem \$GM" > $R3/logs/GEN_qwen_completion.log 2>&1
rc=$?
cp $R3/gen/*.jsonl $R3/gen/gen_stats.json $J/ 2>/dev/null
tail -80 $R3/logs/GEN_qwen_completion.log > $J/GEN_qwen_log_tail.txt; grep -iE "KV cache|concurrency|cudagraph|graph|Free memory|awq|ENGINE_FAIL|Error" $R3/logs/GEN_qwen_completion.log | head -40 > $J/GEN_qwen_engine_lines.txt
head -1 $L > $J/usage_log_robustness_r3.tsv; grep -P "^robustness_r3\t" $L >> $J/usage_log_robustness_r3.tsv
echo "rc=$rc spent_after $(spent)"
exit $rc
