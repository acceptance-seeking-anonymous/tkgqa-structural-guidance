#!/bin/bash
# STAGE B driver for one P_k arm (rule arms.P_k), resumable (finished steps are skipped):
#   pk_arm.sh <llm_list> <executor_subset> <card> <phase: dev|test|all> [fallback preds dir, default /kpct/preds]
# dev : 1 LoRA training [GPU]  2 vLLM decoding of dev_sel2000, every checkpoint, one session [GPU]
#       3 execution + fallback, every checkpoint x tau [CPU]  4 selection on dev_sel2000 [CPU]
#       5 STOP_DEV_FAILURE check (halts before test)  6 pre-test freeze (sha256 of adapter, tau, prompt, models, code) [CPU]
# test: 7 vLLM decoding of test with the frozen checkpoint [GPU]  8 execution with the frozen tau [CPU]
#       9 test freeze (re-verifies every pre-test hash; adds the sha256 of the unscored test prediction files) [CPU]
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
LST=$1; SUB=$2; CARD=$3; PHASE=$4; FBP=${5:-/kpct/preds}
K=$WORK_DIR; P=$K/parser; C=$P/code
mkdir -p $P/adapters $P/raw/$LST $P/exec/$LST $P/test/$LST $P/select $P/freeze $P/logs
SB=silver_${LST}; [ -f $P/silver/${SB}_train.jsonl ] || SB=silver_${LST}_from_${SUB}
PIP="(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /w/logs/pip_peft_$LST.log 2>&1"
AD_DIR=$P/adapters/P_${LST}_s0
FZ=/kpct/parser/freeze/freeze_P_${LST}.json
log() { echo "$(date -u +%FT%TZ) [$LST] $*" | tee -a $P/logs/arm_$LST.log; }
if [ "$PHASE" = dev ] || [ "$PHASE" = all ]; then
  if [ ! -f $AD_DIR/train_stats.json ]; then
    log "1 train start"
    bash $C/run_card_p.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} train_$LST $P "$PIP; python3 /kpct/parser/code/pk_train.py --list $LST --silver /kpct/parser/silver/${SB}_train.jsonl --silver_manifest /kpct/parser/silver/${SB}_manifest.json --out /kpct/parser/adapters" > $P/logs/train_$LST.log 2>&1
    log "1 train done"
  fi
  CKS=$(python3 -c "import json; s=json.load(open('$AD_DIR/train_stats.json')); print(','.join(sorted(s['adapters'], key=lambda x: float(x[2:]))))")
  AD=$(echo $CKS | tr ',' '\n' | awk -v D=/kpct/parser/adapters/P_${LST}_s0 '{printf "%s%s=%s/%s", (NR>1?",":""), $1, D, $1}')
  RAW=$(echo $CKS | tr ',' '\n' | awk -v L=$LST '{printf "%s%s=/kpct/parser/raw/%s/raw_P_dev_sel2000_%s.jsonl.gz", (NR>1?",":""), $1, L, $1}')
  [ -f $P/dev_sel2000_quids.json ] || python3 -c "import json; m=json.load(open('$C/multitq_kpct_subsets.json')); json.dump(m['dev_selection']['quids'], open('$P/dev_sel2000_quids.json','w'))"
  if [ ! -f $P/raw/$LST/infer_stats_P_dev_sel2000.json ]; then
    log "2 dev decode start ($CKS)"
    bash $C/run_card_p.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} infer_dev_$LST $P "python3 /kpct/parser/code/pk_infer.py --split dev_sel2000 --bundles /kpct/bundles/$SUB/dev_sel2000.jsonl --quids /kpct/parser/dev_sel2000_quids.json --adapters $AD --out /kpct/parser/raw/$LST" > $P/logs/infer_dev_$LST.log 2>&1
    log "2 dev decode done"
  fi
  if [ ! -f $P/exec/$LST/exec_stats_P_${SUB}_dev_sel2000.json ]; then
    log "3 dev execute start"
    bash $C/run_cpu_p.sh exec_dev_$LST $P 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split dev_sel2000 --raw $RAW --preds /kpct/preds --out /kpct/parser/exec/$LST --workers 8" > $P/logs/exec_dev_$LST.log 2>&1
    log "3 dev execute done"
  fi
  if [ ! -f $P/select/select_P_$LST.json ]; then
    log "4 select start"
    bash $C/run_cpu_p.sh select_$LST $P 2 pylib "python3 /kpct/parser/code/pk_select.py --k $SUB --list $LST --exec_dir /kpct/parser/exec/$LST --exec_stats /kpct/parser/exec/$LST/exec_stats_P_${SUB}_dev_sel2000.json --ckpts $CKS --log /kpct/parser/selection_log_P.json --out /kpct/parser/select" > $P/logs/select_$LST.log 2>&1
    log "4 select done: $(tail -1 $P/logs/select_$LST.log | cut -c1-300)"
  fi
  if python3 -c "import json,sys; s=json.load(open('$P/select/select_P_$LST.json')); sys.exit(0 if s['STOP_DEV_FAILURE'] else 1)"; then
    log "5 STOP_DEV_FAILURE -> halted before test"; exit 3
  fi
  if [ ! -f $P/freeze/freeze_P_${LST}.json ]; then
    bash $C/run_cpu_p.sh freeze_$LST $P 2 none "python3 /kpct/parser/code/pk_freeze.py --mode pre_test --list $LST --subset $SUB --silver_manifest /kpct/parser/silver/${SB}_manifest.json --out $FZ" > $P/logs/freeze_pre_$LST.log 2>&1
    log "6 pre-test freeze: $(tail -1 $P/logs/freeze_pre_$LST.log)"
  fi
fi
if [ "$PHASE" = test ] || [ "$PHASE" = all ]; then
  test -f $P/freeze/freeze_P_${LST}.json
  CK=$(python3 -c "import json; print(json.load(open('$P/freeze/freeze_P_${LST}.json'))['chosen']['checkpoint'])")
  TN=$(python3 -c "import json; print(json.load(open('$P/freeze/freeze_P_${LST}.json'))['chosen']['tau'])")
  T=$(python3 -c "print({'off':'off','m030':'-0.30','m010':'-0.10','m003':'-0.03'}['$TN'])")
  if [ ! -f $P/raw/$LST/infer_stats_P_test.json ]; then
    log "7 test decode start ($CK)"
    bash $C/run_card_p.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} infer_test_$LST $P "python3 /kpct/parser/code/pk_infer.py --split test --bundles /kpct/bundles/$SUB/test.jsonl --adapters $CK=/kpct/parser/adapters/P_${LST}_s0/$CK --out /kpct/parser/raw/$LST" > $P/logs/infer_test_$LST.log 2>&1
    log "7 test decode done"
  fi
  if [ ! -f $P/test/$LST/exec_stats_P_${SUB}_test.json ]; then
    log "8 test execute start (tau $TN, fallback preds $FBP)"
    bash $C/run_cpu_p.sh exec_test_$LST $P 8 pylib "python3 /kpct/parser/code/pk_execute.py --k $SUB --split test --raw $CK=/kpct/parser/raw/$LST/raw_P_test_$CK.jsonl.gz --taus=$T --preds $FBP --out /kpct/parser/test/$LST --workers 8" > $P/logs/exec_test_$LST.log 2>&1
    log "8 test execute done"
  fi
  bash $C/run_cpu_p.sh freezet_$LST $P 2 none "python3 /kpct/parser/code/pk_freeze.py --mode test --list $LST --subset $SUB --silver_manifest /kpct/parser/silver/${SB}_manifest.json --out $FZ" > $P/logs/freeze_test_$LST.log 2>&1
  log "9 test freeze: $(tail -1 $P/logs/freeze_test_$LST.log)"
fi
