#!/bin/bash
# run_card_cs.sh <card> <image> <tag> <workdir> <command string>
# Clean-silver follow-up copy of parser/code/run_card_p.sh: one container pinned to ONE physical card
# (--gpus "device=N", renumbered to 0 inside -> CUDA_VISIBLE_DEVICES=0, model.to('cuda:0'), no device_map); appends one
# row with track "clean_silver_p1" to the shared usage log $WORK_DIR/gpu_usage_kpct.tsv (flock-serialised).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
CARD=$1; IMG=$2; TAG=$3; W=$4; shift 4; CMD="$*"
K=$WORK_DIR; L=$K/gpu_usage_kpct.tsv
CN=cs_${TAG}_$(date +%s)
t0=$(date +%s)
docker run --rm --name $CN --gpus "\"device=$CARD\"" -e CUDA_VISIBLE_DEVICES=0 --ipc=host --shm-size=16g \
  -v $W:/w -v $K:/kpct -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e NCCL_P2P_DISABLE=1 -e OMP_NUM_THREADS=8 \
  -w /w $IMG bash -c "$CMD; rc=\$?; chmod -R a+rX /w /kpct/parser_clean 2>/dev/null; exit \$rc"
rc=$?; t1=$(date +%s)
(
  flock 9
  [ -s $L ] || printf "track\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" >> $L
  GH=$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')
  printf "clean_silver_p1\t%s\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$CARD" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" \
    "$((t1-t0))" "$GH" "$rc" "$TAG" "$CN" >> $L
) 9>$L.lock
exit $rc
