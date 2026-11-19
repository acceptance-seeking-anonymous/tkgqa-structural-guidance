#!/bin/bash
# gen_job.sh <phase: smoke | attempt1 | attempt2> -- R2 rewrite generation on the GPU host GPU 0 (pinned), Qwen2.5-72B-Instruct-AWQ.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
PH=$1; J=$PWD; K=$WORK_DIR; R=$K/robustness_r2
mkdir -p $R/code $R/gen $R/logs
cp $J/r2_gen.py $J/run_card_r2.sh $J/run_cpu_r2.sh $R/code/ 2>/dev/null
cp $J/${PH}_prompts.jsonl $R/gen/
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader > $J/gpu_state_before.txt; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}' >> $J/gpu_state_before.txt
bash $R/code/run_card_r2.sh 0 ${VLLM_IMAGE:-tkgqa-vllm:latest} GEN_$PH "python3 -u /kpct/robustness_r2/code/r2_gen.py --inp /kpct/robustness_r2/gen/${PH}_prompts.jsonl --out /kpct/robustness_r2/gen/${PH}_gen.jsonl" > $R/logs/GEN_$PH.log 2>&1
rc=$?
cp $R/gen/${PH}_gen.jsonl $R/gen/${PH}_gen_stats.json $J/ 2>/dev/null; tail -60 $R/logs/GEN_$PH.log > $J/GEN_${PH}_log_tail.txt
grep robustness_r2 $K/gpu_usage_kpct.tsv > $J/usage_log_robustness_r2.tsv
exit $rc
