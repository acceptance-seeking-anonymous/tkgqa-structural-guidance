#!/bin/bash
# queue_runner.sh <card> [queue_file]  -- pops guided-track items from a shared queue file (flock) and runs them on <card>.
# Lines: item names (G_100 GU_100 G_10 GU_10 G_5 G_1s1 G_1s2) or "WAIT:<path>" (block, without holding a GPU, until
# <path> exists -- used to pause the queue at a registered checkpoint after k = 100). A second runner on the other card can be
# started at any time; each item runs exactly once. Resume-safe (gk_train / gk_run_arm skip completed work).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD=$1; H=$WORK_DIR; G=$H/guided; Q=${2:-$G/queue_main.txt}
cd $G/code && source ./stageB_plan.sh; CARD=$1
OD=$G/preds/G_k01_s0/order_decision_k01_s0.json
ORD=$(python3 -c "import json; print(json.load(open('$OD'))['chosen'])") || { echo "no order decision"; exit 1; }
run_G() { check_bundles $1 > $G/logs/check_$1.log 2>&1 || { echo "check $1 failed"; return 1; }
          train_G $1 $1 $ORD || return 1; session_G $1 $1 main:$ORD:/kpct/guided/adapters/G_$1_ord${ORD}_s0; }
run_GU() { train_GU $1 || return 1; session_GU $1; }
pop() { ( flock 8; L=$(head -1 $Q); [ -n "$L" ] && sed -i '1d' $Q; echo "$L" ) 8>$Q.lock; }
echo "== runner card $CARD order $ORD start $(date -u +%FT%TZ)"
while true; do
  it=$(pop); [ -z "$it" ] && break
  if [[ $it == WAIT:* ]]; then
    f=${it#WAIT:}; echo "== card $CARD waiting for $f $(date -u +%FT%TZ)"
    while [ ! -f $f ]; do sleep 60; done; continue
  fi
  U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $CARD | tr -d ' ')
  if [ "$U" -gt 1000 ]; then echo "card $CARD busy ($U MiB) -- re-queue $it and stop"; ( flock 8; sed -i "1i $it" $Q ) 8>$Q.lock; exit 3; fi
  t0=$(date +%s); echo "== card $CARD item $it start $(date -u +%FT%TZ)"
  case $it in
    G_100) run_G k100 ;; GU_100) run_GU k100 ;; G_10) run_G k10_s0 ;; GU_10) run_GU k10_s0 ;; G_5) run_G k05_s0 ;;
    G_1s1) run_G k01_s1 ;; G_1s2) run_G k01_s2 ;; *) echo "unknown $it"; false ;;
  esac
  rc=$?; echo "== card $CARD item $it rc=$rc $(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)" | tee -a $G/logs/queue_items.log
  [ $rc -ne 0 ] && echo "$it rc=$rc" >> $G/logs/queue_failed.txt
done
echo "== runner card $CARD done $(date -u +%FT%TZ)"
