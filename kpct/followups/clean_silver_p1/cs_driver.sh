#!/bin/bash
# cs_driver.sh <phase: counts | gpu | post>   -- clean-silver P_1 follow-up (multitq_p1_clean_silver_rule.json); DEV ONLY.
#  counts: clean-silver filter for k01_s0 (+ clean training file) and counts for k05_s0, k10_s0, k100      [CPU]
#  gpu   : train P_1clean, P_1rerun, P_1order (one card each, sequential), then ONE vLLM session decoding
#          dev_sel2000 with all 9 checkpoints                                                             [GPU, usage_log]
#  post  : execution + fallback (3 S_k seeds), rule selection per variant, freeze, comparison            [CPU]
# GPU coordination: before every GPU container a card is used only if memory.used < 2000 MiB and util < 5% in two
# reads 30 s apart and no compute process sits on it; otherwise wait 600 s and re-check (<= 48 checks). Other
# processes are never touched. GPU-hour cap: track spend + item projection <= 2.0 GPU-h, usage_log total + item <= 52.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -eo pipefail
PHASE=$1
K=$WORK_DIR; P=$K/parser; Q=$K/parser_clean; C=$Q/code
mkdir -p $Q/silver $Q/adapters $Q/raw $Q/exec $Q/select $Q/logs $Q/freeze
L=$K/gpu_usage_kpct.tsv
log() { echo "$(date -u +%FT%TZ) $*" | tee -a $Q/logs/driver.log; }
spend() { awk -F'\t' -v T=$1 'NR>1 && (T=="" || $1==T){s+=$7} END{printf "%.4f", s+0}' $L; }
gpu_cap_ok() {   # $1 = projected GPU-h of the next item
  local tr=$(spend clean_silver_p1) tot=$(spend "")
  awk -v a=$tr -v b=$tot -v p=$1 'BEGIN{exit !((a+p)<=2.0 && (b+p)<=52.0)}'
}
pick_card() {
  local tries=0
  while [ $tries -lt 48 ]; do
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader >> $Q/logs/gpu_checks.log
    docker ps --format '{{.Names}} {{.Image}} {{.Status}}' >> $Q/logs/gpu_checks.log
    local busy_uuids=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | sort -u)
    for card in 1 0; do
      local uuid=$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i $card | tr -d ' ')
      local m1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $card | tr -d ' ')
      local u1=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $card | tr -d ' ')
      if [ "$m1" -lt 2000 ] && [ "$u1" -lt 5 ] && ! echo "$busy_uuids" | grep -q "$uuid"; then
        sleep 30
        local m2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $card | tr -d ' ')
        local u2=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $card | tr -d ' ')
        if [ "$m2" -lt 2000 ] && [ "$u2" -lt 5 ]; then echo $card; return 0; fi
      fi
    done
    log "no free card (check $tries); waiting 600 s" >&2; tries=$((tries+1)); sleep 600
  done
  return 1
}
PIP="(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /w/logs/pip_peft.log 2>&1"
if [ "$PHASE" = counts ]; then
  bash $C/run_cpu_p.sh cs_counts $Q 8 pylib "python3 /kpct/parser_clean/code/pk_clean_silver.py --subset k01_s0 --out /kpct/parser_clean/silver --workers 8 --write_train" > $Q/logs/counts_k01_s0.log 2>&1
  for S in k05_s0 k10_s0 k100; do
    bash $C/run_cpu_p.sh cs_counts $Q 8 pylib "python3 /kpct/parser_clean/code/pk_clean_silver.py --subset $S --out /kpct/parser_clean/silver --workers 8" > $Q/logs/counts_$S.log 2>&1
  done
  tail -n1 $Q/logs/counts_*.log
fi
if [ "$PHASE" = gpu ]; then
  FROZEN_ORDER=$(python3 -c "import json; print(json.load(open('$P/adapters/P_k01_s0_s0/train_stats.json'))['data_order_sha256'])")
  for V in clean rerun order; do
    [ -f $Q/adapters/P_k01_s0${V}_s0/train_stats.json ] && continue
    case $V in
      clean) SIL=/kpct/parser_clean/silver/silver_k01_s0_clean_train.jsonl; MAN=/kpct/parser_clean/silver/silver_k01_s0_clean_manifest.json; EX="" ;;
      rerun) SIL=/kpct/parser/silver/silver_k01_s0_train.jsonl; MAN=/kpct/parser/silver/silver_k01_s0_manifest.json; EX="" ;;
      order) SIL=/kpct/parser/silver/silver_k01_s0_train.jsonl; MAN=/kpct/parser/silver/silver_k01_s0_manifest.json; EX="--order_seed 1" ;;
    esac
    gpu_cap_ok 0.15 || { log "STOP_GPU_CAP before train $V"; exit 5; }
    CARD=$(pick_card) || { log "no free card after 48 checks: $V NOT_RUN"; exit 6; }
    log "train $V on card $CARD"
    bash $C/run_card_cs.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} train_$V $Q "$PIP; python3 /kpct/parser_clean/code/pk_train.py --list k01_s0 --silver $SIL --silver_manifest $MAN --manifest /kpct/parser/code/multitq_kpct_subsets.json --out /kpct/parser_clean/adapters --name P_k01_s0${V}_s0 --guided_common /kpct/guided/code/gk_common.py $EX" > $Q/logs/train_$V.log 2>&1 || { log "train $V failed"; exit 7; }
    if [ $V = rerun ]; then
      R=$(python3 -c "import json; print(json.load(open('$Q/adapters/P_k01_s0rerun_s0/train_stats.json'))['data_order_sha256'])")
      [ "$R" = "$FROZEN_ORDER" ] || { log "STOP_RERUN_DRIFT: data order $R != frozen $FROZEN_ORDER"; exit 8; }
      log "rerun data order identical to frozen ($R)"
    fi
  done
  if [ ! -f $Q/raw/infer_stats_P_dev_sel2000.json ]; then
    AD=$(for V in clean rerun order; do for E in ep1 ep2 ep3; do printf "%s_%s=/kpct/parser_clean/adapters/P_k01_s0%s_s0/%s," $V $E $V $E; done; done | sed 's/,$//')
    gpu_cap_ok 0.10 || { log "STOP_GPU_CAP before dev decoding"; exit 5; }
    CARD=$(pick_card) || { log "no free card after 48 checks: decoding NOT_RUN"; exit 6; }
    log "dev decode on card $CARD"
    bash $C/run_card_cs.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} infer_dev $Q "python3 /kpct/parser_clean/code/pk_infer.py --split dev_sel2000 --bundles /kpct/bundles/k01_s0/dev_sel2000.jsonl --quids /kpct/parser/dev_sel2000_quids.json --adapters $AD --out /kpct/parser_clean/raw" > $Q/logs/infer_dev.log 2>&1 || { log "dev decode failed"; exit 7; }
  fi
  log "gpu phase done; track spend $(spend clean_silver_p1) GPU-h"
fi
if [ "$PHASE" = post ]; then
  RAW=$(for V in clean rerun order; do for E in ep1 ep2 ep3; do printf "%s_%s=/kpct/parser_clean/raw/raw_P_dev_sel2000_%s_%s.jsonl.gz," $V $E $V $E; done; done | sed 's/,$//')
  [ -f $Q/exec/exec_stats_P_k01_s0_dev_sel2000.json ] || bash $C/run_cpu_p.sh cs_exec $Q 8 pylib "python3 /kpct/parser_clean/code/pk_execute.py --k k01_s0 --split dev_sel2000 --raw $RAW --preds /kpct/preds --out /kpct/parser_clean/exec --workers 8" > $Q/logs/exec_dev.log 2>&1
  for V in clean rerun order; do
    mkdir -p $Q/select/$V
    bash $C/run_cpu_p.sh cs_sel $Q 2 pylib "python3 /kpct/parser_clean/code/pk_select.py --k k01_s0 --list k01_s0 --exec_dir /kpct/parser_clean/exec --exec_stats /kpct/parser_clean/exec/exec_stats_P_k01_s0_dev_sel2000.json --ckpts ${V}_ep1,${V}_ep2,${V}_ep3 --log /kpct/parser_clean/select/selection_log_${V}.json --out /kpct/parser_clean/select/$V" > $Q/logs/select_$V.log 2>&1
  done
  bash $C/run_cpu_p.sh cs_freeze $Q 2 none "python3 /kpct/parser_clean/code/cs_freeze.py --out /kpct/parser_clean/freeze/multitq_p1_clean_silver_freeze.json" > $Q/logs/freeze.log 2>&1
  bash $C/run_cpu_p.sh cs_cmp $Q 2 pylib "python3 /kpct/parser_clean/code/cs_compare.py --freeze /kpct/parser_clean/freeze/multitq_p1_clean_silver_freeze.json --out /kpct/parser_clean/multitq_p1_clean_silver_results.json" > $Q/logs/compare.log 2>&1
  tail -3 $Q/logs/compare.log
fi
