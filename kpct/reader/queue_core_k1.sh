#!/bin/bash
# k = 1 subset-seed-0 core of the guided track (registered run-order step 2), card 0, resume-safe.
# Order: bundle checks (CPU; k01_s0 must PASS) -> G_1 ordS -> G_1 ordrandom -> G_1 session (dev ckpts of both runs,
# order decision, freeze, test, SHUF_G1 / QONLY_G1 dev+test) -> GU_1 -> GU_1 session. Then copies summaries to the job cwd.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
J=$PWD; H=$WORK_DIR; G=$H/guided
cp $J/code/* $G/code/ 2>/dev/null
cd $G/code && source ./stageB_plan.sh
mkdir -p $G/logs $G/checks $G/preds $G/adapters
echo "== start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}'
check_bundles k01_s0 > $G/logs/check_k01_s0.log 2>&1; rc=$?
echo "check k01_s0 rc=$rc"; if [ $rc -ne 0 ]; then tail -40 $G/logs/check_k01_s0.log; exit 1; fi
# remaining bundle checks + k100 cross-check on CPU in the background (2 at a time; not counted as GPU time)
(
  for k in k100 k10_s0; do check_bundles $k > $G/logs/check_$k.log 2>&1 & done; wait
  for k in k05_s0 k01_s1 k01_s2; do check_bundles $k > $G/logs/check_$k.log 2>&1 & done; wait
  docker run --rm -v $H:/kpct -w /kpct/guided/code ${GPU_IMAGE:-tkgqa-gpu:latest} python3 gk_crosscheck_k100.py --struct /kpct/bundles/k100 \
    --conv /kpct/guided/k100_from_exports --out /kpct/guided/checks/crosscheck_k100.json > $G/logs/crosscheck_k100.log 2>&1
) &
CHK=$!
set -o pipefail
train_G k01_s0 k01_s0 S          || { echo "train ordS failed"; tail -30 $G/logs/train_G_k01_s0_ordS.log; exit 1; }
train_G k01_s0 k01_s0 random     || { echo "train ordrandom failed"; tail -30 $G/logs/train_G_k01_s0_ordrandom.log; exit 1; }
session_G k01_s0 k01_s0 ordS:S:/kpct/guided/adapters/G_k01_s0_ordS_s0,ordrandom:random:/kpct/guided/adapters/G_k01_s0_ordrandom_s0 SHUF,QONLY \
  || { echo "G session failed"; tail -40 $G/logs/sess_G_k01_s0.log; exit 1; }
train_GU k01_s0                  || { echo "train GU failed"; tail -30 $G/logs/train_GU_k01_s0.log; exit 1; }
session_GU k01_s0                || { echo "GU session failed"; tail -40 $G/logs/sess_GU_k01_s0.log; exit 1; }
wait $CHK
echo "== core done $(date -u +%FT%TZ)"
mkdir -p $J/out; cd $G
cp checks/*.json logs/*.log $J/out/ 2>/dev/null
for d in preds/G_k01_s0 preds/GU_k01_s0; do cp $d/*.json $J/out/ 2>/dev/null; done
for d in adapters/*k01_s0*; do cp $d/train_stats.json $J/out/train_stats_$(basename $d).json 2>/dev/null; cp $d/gu_pad.json $J/out/gu_pad_$(basename $d).json 2>/dev/null; done
cp multitq_kpct_guided_selection_log.jsonl $J/out/ 2>/dev/null; cp $H/gpu_usage_kpct.tsv $J/out/gpu_usage_snapshot.tsv
ls -la $J/out | head -60
