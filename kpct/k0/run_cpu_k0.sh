#!/bin/bash
# run_cpu_k0.sh <tag> <workdir> <cpus> <image> <command string>  -- CPU-only container (no --gpus), not in the GPU usage log.
# OpenMP/BLAS pinned to 1 thread (process-parallel executor; forked torch workers deadlock in libgomp otherwise).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
TAG=$1; W=$2; NCPU=$3; IMG=$4; shift 4; CMD="$*"
K=$WORK_DIR
PP=""; [ "$IMG" = ${GPU_IMAGE:-tkgqa-gpu:latest} ] && PP="-e PYTHONPATH=/kpct/pylib"     # executor image only (as the parser / structural tracks)
docker run --rm --name k0cpu_${TAG}_$(date +%s) --cpus=$NCPU \
  -v $W:/w -v $K:/kpct -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
  $PP -e KPCT_PARSER_CODE=/kpct/k0/code/parser \
  -e KPCT_K0=${KPCT_K0:-/kpct/k0} -e KPCT_EXECUTOR=${KPCT_EXECUTOR:-/kpct/executor} \
  -w /w $IMG bash -c "$CMD; rc=\$?; chmod -R a+rwX /w /kpct/k0 2>/dev/null; exit \$rc"
