#!/bin/bash
# GU_1 (k = 1 core) on card 1 while card 0 runs G_1: train -> session (dev ckpts -> select -> freeze -> test).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; H=$WORK_DIR; G=$H/guided
cp $J/gk_run_arm.py $G/code/gk_run_arm.py
cd $G/code && source ./stageB_plan.sh; CARD=1
echo "== card 1 GU_1 start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}'
U1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1 | tr -d ' ')
[ "$U1" -gt 1000 ] && { echo "card 1 busy ($U1 MiB) -- abort"; exit 3; }
train_GU k01_s0 || { echo "train GU failed"; tail -30 $G/logs/train_GU_k01_s0.log; exit 1; }
session_GU k01_s0 || { echo "GU session failed"; tail -40 $G/logs/sess_GU_k01_s0.log; exit 1; }
mkdir -p $J/out; cp $G/preds/GU_k01_s0/*.json $J/out/; cp $G/adapters/GU_k01_s0_s0/train_stats.json $J/out/train_stats_GU_k01_s0_s0.json
cp $H/gpu_usage_kpct.tsv $J/out/gpu_usage_snapshot.tsv; echo "== done $(date -u +%FT%TZ)"
