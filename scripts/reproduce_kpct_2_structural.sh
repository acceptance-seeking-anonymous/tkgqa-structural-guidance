#!/bin/bash
# reproduce_kpct_2_structural.sh -- retrain the structural stack (classifier, linker statistics, Ranker = LightGBM
# lambdarank x 3 seeds, out-of-fold stacks) on each k% list; write dev_sel2000 / test Rules (RULE_k) and Ranker (S_k)
# prediction files and the guidance bundles; then the k100 bundles from the frozen full-data exports; freeze record and
# GPU-host staging tree ($SWORK/stage/kpct). CPU only (about 6 min per 1% list with 8 processes).
# Determinism: mtq_clf.Vocab orders the n-gram vocabulary by iterating a Python set, so retraining is repeatable only
# with a fixed PYTHONHASHSEED (set below); the original runs did not fix it, so retrained models are new versions.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR SWORK PHASE1_STRUCT MTQ_DIR
SW=$SWORK; mkdir -p "$SW/struct" "$SW/rule"
cp "$REPO"/kpct/structural/* "$SW/"; cp "$REPO"/src/struct/*.py "$REPO/src/multitq_eval.py" "$SW/struct/"
cp "$PHASE1_STRUCT"/{alias_rules.json,ent_block.json} "$SW/struct/"
cp "$REPO/protocol/multitq_kpct_decision_rule.json" "$SW/rule/"
ln -sfn "$DATA_DIR" "$SW/mtqdata"; ln -sfn "$MTQ_DIR/phase1_structural" "$SW/../exp"
cd "$SW"
export PYTHONHASHSEED=${PYTHONHASHSEED:-0} OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES OMP_NUM_THREADS=1 KPCT_NPROC=${KPCT_NPROC:-8}
for k in ${KLIST:-k01_s0 k05_s0 k10_s0 k01_s1 k01_s2}; do
  mkdir -p out/$k; python kpct_run.py all $k > out/$k/run.log 2>&1
done
mkdir -p out/k100; KPCT_EXPORTS="$SW/../exp" python kpct_k100.py > out/k100/run.log 2>&1
ln -sfn "$SW" "$SW/../kp"; (cd "$SW/.." && python kp/check_k100.py)
python kpct_finalize.py
python kpct_stage.py k01_s0 k05_s0 k10_s0 k01_s1 k01_s2 k100 k0
