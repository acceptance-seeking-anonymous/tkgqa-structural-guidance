#!/bin/bash
# nm_run_card.sh <card> <image> <tag> <max_seconds> <command string>
# Track "normalizer": ONE container pinned to ONE physical card (renumbered to 0 inside: --gpus "device=N" +
# CUDA_VISIBLE_DEVICES=0), no device_map. The frozen kpct tree is mounted READ-ONLY; only /kpct/normalizer is writable.
# Waits (up to 90 min, polling 60 s) while the card has any compute process; never stops anything it did not start.
# The free card memory at launch is passed to the container as CARD_FREE_MIB. A watchdog stops this container after
# <max_seconds>. Appends one row (track "normalizer") to the shared GPU usage log $WORK_DIR/gpu_usage_kpct.tsv
# (container wall-clock x 1 card; flock-serialised).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
CARD=$1; IMG=$2; TAG=$3; MAXS=$4; shift 4; CMD="$*"
K=$WORK_DIR; L=$K/gpu_usage_kpct.tsv; R=$K/normalizer; M=$MTQ_DIR
mkdir -p $R/logs
for i in $(seq 1 90); do
  n=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" = 0 ] && break
  echo "$(date -u +%FT%TZ) card $CARD busy ($n procs), waiting" >&2; sleep 60
done
[ "$n" = 0 ] || { echo "card $CARD still busy after 90 min; not launching $TAG" >&2; exit 75; }
FREE=$(nvidia-smi -i $CARD --query-gpu=memory.free --format=csv,noheader,nounits)
export CARD_FREE_MIB=$FREE
CN=nm_${TAG}_$(date +%s)
t0=$(date +%s)
( sleep $MAXS; docker stop $CN > /dev/null 2>&1 && echo "WATCHDOG stopped $CN after $MAXS s" >&2 ) & WD=$!
docker run --rm --name $CN --gpus "\"device=$CARD\"" -e CUDA_VISIBLE_DEVICES=0 --ipc=host --shm-size=16g \
  -v $K:/kpct:ro -v $R:/kpct/normalizer -v $M:/mtq:ro -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro -e CARD_FREE_MIB=$FREE \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_P2P_DISABLE=1 -e OMP_NUM_THREADS=8 -e GK_CODE=/kpct/guided/code \
  -w /kpct/normalizer $IMG bash -c "$CMD; rc=\$?; chmod -R a+rwX /kpct/normalizer 2>/dev/null; exit \$rc"
rc=$?; t1=$(date +%s)
kill $WD 2>/dev/null
(
  flock 9
  [ -s $L ] || printf "track\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" >> $L
  GH=$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')
  printf "normalizer\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CARD" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" \
    "$((t1-t0))" "$GH" "$rc" "$TAG" "$CN" >> $L
) 9>$L.lock
exit $rc
