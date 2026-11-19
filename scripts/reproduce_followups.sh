#!/bin/bash
# reproduce_followups.sh <g1_ablation | clean_silver_p1 | seeds_G | seeds_P_dev | seeds_P_test | seeds_P_score>
# Dev-only / replication follow-ups of the k% experiment. Each runs from a job directory ($JOBDIR) into which this
# script copies the track's code; the registered rules are protocol/multitq_g1_ablation_rule.json,
# protocol/multitq_p1_clean_silver_rule.json, protocol/multitq_kpct_seed_amendment_{G,P,P_A1}.json.
#   g1_ablation      G_1 input-intervention ablation on 46,326 dev questions: abl_build.py (CPU) -> abl_gpu_job.sh (GU
#                    retrieval + one vLLM session) -> abl_analyze.py -> abl_part4.py -> abl_aggregate.py (local)
#   clean_silver_p1  clean-silver P_1 retrain (dev only): PHASES='counts gpu post' cs_launch.sh
#   seeds_G          Reader_1 LoRA seeds 1, 2 (card 1): seeds_job.sh, then seeds_analyze.py locally
#   seeds_P_*        Parser_1 / Parser_5 LoRA seeds: training + decoding on a second GPU host (host_b/j2-j4.sh),
#                    execution / selection / freeze / scoring on the main GPU host (host_a/job_*.sh), ps_analyze.py
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
T=${1:?track}; JD=${JOBDIR:-$PWD/${T}_job}; mkdir -p "$JD"; F=$REPO/kpct/followups
case $T in
  g1_ablation)
    cp "$F"/g1_ablation/* "$REPO"/kpct/reader/{gk_common.py,gk_infer.py,gk_train.py,gu_retrieve.py} "$REPO/src/multitq_eval.py" "$JD/"
    mkdir -p "$WORK_DIR/ablation_g1/code"; cp "$JD"/*.py "$JD"/*.sh "$WORK_DIR/ablation_g1/code/"
    echo "run in order (see README): abl_build.py, abl_gpu_job.sh, abl_analyze.py, abl_part4.py, abl_aggregate.py" ;;
  clean_silver_p1)
    cp "$F"/clean_silver_p1/* "$REPO"/kpct/parser/{pk_common.py,pk_infer.py,pk_execute.py,pk_select.py,pk_train.py,run_cpu_p.sh} \
       "$REPO/protocol/multitq_p1_clean_silver_rule.json" "$JD/"
    (cd "$JD" && PHASES=${PHASES:-counts gpu post} bash cs_launch.sh) ;;
  seeds_G)
    cp "$F"/seeds_G/* "$JD/"; (cd "$JD" && bash seeds_job.sh) ;;
  seeds_P_dev)   cp "$F"/seeds_P/host_a/* "$JD/"; (cd "$JD" && bash job_dev.sh) ;;
  seeds_P_test)  cp "$F"/seeds_P/host_a/* "$JD/"; (cd "$JD" && bash job_test.sh) ;;
  seeds_P_score) cp "$F"/seeds_P/host_a/* "$JD/"; (cd "$JD" && bash job_score.sh) ;;
  *) echo "unknown track $T"; exit 2 ;;
esac
