#!/bin/bash
# run_card_p.sh <cards: 0 | 1 | 0,1> <image> <tag> <workdir> <command string>
# P-track copy of guided/code/run_card.sh: one container pinned to the given physical card(s) (renumbered from 0
# inside: --gpus "device=N" + CUDA_VISIBLE_DEVICES=0), no device_map; appends one row (track "parser") to the shared
# GPU usage log $WORK_DIR/gpu_usage_kpct.tsv (container wall-clock x cards held; flock-serialised).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
CARDS=$1; IMG=$2; TAG=$3; W=$4; shift 4; CMD="$*"
K=$WORK_DIR; L=$K/gpu_usage_kpct.tsv
mkdir -p $K/parser
NC=$(echo "$CARDS" | tr ',' '\n' | wc -l)
if [ "$NC" = 1 ]; then CVD=0; else CVD=$(seq -s, 0 $((NC-1))); fi
CN=pk_${TAG}_$(date +%s)
t0=$(date +%s)
docker run --rm --name $CN --gpus "\"device=$CARDS\"" -e CUDA_VISIBLE_DEVICES=$CVD --ipc=host --shm-size=16g \
  -v $W:/w -v $K:/kpct -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_P2P_DISABLE=1 -e OMP_NUM_THREADS=8 \
  -w /w $IMG bash -c "$CMD; rc=\$?; chmod -R a+rX /w /kpct/parser 2>/dev/null; exit \$rc"
rc=$?; t1=$(date +%s)
(
  flock 9
  [ -s $L ] || printf "track\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" >> $L
  GH=$(awk -v s=$((t1-t0)) -v n=$NC 'BEGIN{printf "%.4f", s*n/3600}')
  printf "parser\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CARDS" "$NC" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" \
    "$((t1-t0))" "$GH" "$rc" "$TAG" "$CN" >> $L
) 9>$L.lock
exit $rc
