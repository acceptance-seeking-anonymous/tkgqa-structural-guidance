#!/bin/bash
# run_card_r.sh <card: 0 | 1> <image> <tag> <command string>
# Robustness R2 copy of guided/code/run_card.sh: one container pinned to ONE physical card (renumbered to 0 inside:
# --gpus "device=N" + CUDA_VISIBLE_DEVICES=0), no device_map; appends one row (track "robustness") to the shared GPU
# usage_log $WORK_DIR/gpu_usage_kpct.tsv (container wall-clock x cards held; flock-serialised).
# Waits (up to 90 min, polling every 60 s) while the card has a foreign compute process; never stops anything.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
CARD=$1; IMG=$2; TAG=$3; shift 3; CMD="$*"
K=$WORK_DIR; L=$K/gpu_usage_kpct.tsv; M=$MTQ_DIR
for i in $(seq 1 90); do
  n=$(nvidia-smi -i $CARD --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .)
  [ "$n" = 0 ] && break
  echo "$(date -u +%FT%TZ) card $CARD busy ($n procs), waiting" >&2; sleep 60
done
[ "$n" = 0 ] || { echo "card $CARD still busy after 90 min; not launching $TAG" >&2; exit 75; }
CN=r2_${TAG}_$(date +%s)
t0=$(date +%s)
docker run --rm --name $CN --gpus "\"device=$CARD\"" -e CUDA_VISIBLE_DEVICES=0 --ipc=host --shm-size=16g \
  -v $K:/kpct -v $M:/mtq:ro -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $M/MultiTQ-94ad61c/data/MultiTQ:/data:ro \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_P2P_DISABLE=1 -e OMP_NUM_THREADS=8 \
  -w /kpct/robustness_r2 $IMG bash -c "$CMD; rc=\$?; chmod -R a+rwX /kpct/robustness_r2 2>/dev/null; exit \$rc"
rc=$?; t1=$(date +%s)
(
  flock 9
  [ -s $L ] || printf "track\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" >> $L
  GH=$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')
  printf "robustness_r2\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CARD" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" \
    "$((t1-t0))" "$GH" "$rc" "$TAG" "$CN" >> $L
) 9>$L.lock
exit $rc
