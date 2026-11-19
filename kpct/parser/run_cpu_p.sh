#!/bin/bash
# run_cpu_p.sh <tag> <workdir> <cpus> <pythonpath: pylib|none> <command string>   -- CPU-only container (no --gpus), not in the GPU usage log
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
TAG=$1; W=$2; NCPU=$3; PP=$4; shift 4; CMD="$*"
K=$WORK_DIR
# executor jobs (pylib) are process-parallel: OpenMP/BLAS pinned to 1 thread, else forked LightGBM/torch workers
# deadlock in libgomp (observed with OMP_NUM_THREADS=4)
NT=$NCPU; EXTRA=""; [ "$PP" = pylib ] && { EXTRA="-e PYTHONPATH=/kpct/pylib"; NT=1; }
docker run --rm --name pkcpu_${TAG}_$(date +%s) --cpus=$NCPU \
  -v $W:/w -v $K:/kpct -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=$NT -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=$NT $EXTRA \
  -w /w ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "$CMD; rc=\$?; chmod -R a+rwX /w 2>/dev/null; exit \$rc"
