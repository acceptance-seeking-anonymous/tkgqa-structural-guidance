#!/bin/bash
# run_card_pseed.sh <tag> <image> <command string>
# One container on the single RTX 4090 of the second GPU host (--gpus all = 1 card); mounts $SCRATCH_B as /w and the HF cache;
# appends one usage_log row (track parser_seeds; container wall-clock x 1 card) to $WORK_DIR_B/gpu_usage_pseeds.tsv
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${SCRATCH_B:?set SCRATCH_B (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
TAG=$1; IMG=$2; shift 2; CMD="$*"
W=${SCRATCH_B:-/tmp/pseeds}; LD=$WORK_DIR_B; L=$LD/gpu_usage_pseeds.tsv; mkdir -p $LD $W/logs
U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$U" -gt 1500 ]; then echo "GPU busy ($U MiB used) - refusing $TAG"; return 3 2>/dev/null || exit 3; fi
CN=ps_${TAG}_$(date +%s); t0=$(date +%s)
docker run --rm --name $CN --gpus all --ipc=host --shm-size=16g --entrypoint bash \
  -v $W:/w -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -e TOKENIZERS_PARALLELISM=false -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn -e OMP_NUM_THREADS=8 -w /w $IMG -c "$CMD; rc=\$?; chmod -R a+rwX /w 2>/dev/null; exit \$rc"
rc=$?; t1=$(date +%s)
[ -s $L ] || printf "track\thost\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" >> $L
GH=$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')
printf "parser_seeds\thost_b\t0\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" "$((t1-t0))" "$GH" "$rc" "$TAG" "$CN" >> $L
echo "USAGE_LOG $TAG seconds=$((t1-t0)) gpu_h=$GH rc=$rc"
return $rc 2>/dev/null || exit $rc
