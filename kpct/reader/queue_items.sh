#!/bin/bash
# queue_items.sh <card> <item> [<item> ...]   -- runs guided-track items sequentially on one card (resume-safe).
# Items: G_100 GU_100 G_10 GU_10 G_5 G_1s1 G_1s2. The candidate order is the one fixed at k = 1 subset seed 0
# (preds/G_k01_s0/order_decision_k01_s0.json). STOP_CORE_FIRST is checked by the caller (manual confirmation).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
CARD_ARG=$1; shift
J=$PWD; H=$WORK_DIR; G=$H/guided
cd $G/code && source ./stageB_plan.sh; CARD=$CARD_ARG
OD=$G/preds/G_k01_s0/order_decision_k01_s0.json
ORD=$(python3 -c "import json; print(json.load(open('$OD'))['chosen'])") || { echo "no order decision"; exit 1; }
echo "== card $CARD order $ORD items $* start $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
run_G() {  # <subset>
  check_bundles $1 > $G/logs/check_$1.log 2>&1 || { echo "check $1 failed"; return 1; }
  train_G $1 $1 $ORD || { echo "train G $1 failed"; return 1; }
  session_G $1 $1 main:$ORD:/kpct/guided/adapters/G_$1_ord${ORD}_s0 || { echo "session G $1 failed"; return 1; }
}
run_GU() { train_GU $1 || { echo "train GU $1 failed"; return 1; }; session_GU $1 || { echo "session GU $1 failed"; return 1; }; }
rc_all=0
for it in "$@"; do
  t0=$(date +%s)
  case $it in
    G_100) run_G k100 ;; GU_100) run_GU k100 ;; G_10) run_G k10_s0 ;; GU_10) run_GU k10_s0 ;; G_5) run_G k05_s0 ;;
    G_1s1) run_G k01_s1 ;; G_1s2) run_G k01_s2 ;; *) echo "unknown item $it"; false ;;
  esac
  rc=$?; echo "== item $it rc=$rc $(( $(date +%s) - t0 ))s $(date -u +%FT%TZ)"; [ $rc -ne 0 ] && rc_all=$rc
done
mkdir -p $J/out; cd $G
for d in preds/*; do cp $d/freeze_*.json $d/selection_*.json $d/session_*.json $J/out/ 2>/dev/null; done
for d in adapters/*; do cp $d/train_stats.json $J/out/train_stats_$(basename $d).json 2>/dev/null; done
cp checks/*.json $J/out/ 2>/dev/null; cp multitq_kpct_guided_selection_log.jsonl $J/out/ 2>/dev/null; cp $H/gpu_usage_kpct.tsv $J/out/gpu_usage_snapshot.tsv
exit $rc_all
