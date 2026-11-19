#!/bin/bash
# run_cpu_gr.sh <tag> <cpus> <command string>  -- CPU-only container (no --gpus), not in the GPU usage log (= robustness_r3
# run_cpu_r3.sh with the writable root /kpct/grounded_reader; the frozen kpct tree is mounted READ-ONLY).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
TAG=$1; NCPU=$2; shift 2; CMD="$*"
K=$WORK_DIR; M=$MTQ_DIR; GR=$K/grounded_reader
docker run --rm --name grcpu_${TAG}_$(date +%s) --cpus=$NCPU \
  -v $K:/kpct:ro -v $GR:/kpct/grounded_reader -v $M:/mtq:ro -v $M/MultiTQ-94ad61c/data/MultiTQ:/data:ro \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONPATH=/kpct/pylib \
  -e PK_CODE=/kpct/parser/code -e KPCT_EXECUTOR=/kpct/executor -e GK_CODE=/kpct/guided/code \
  -w /kpct/grounded_reader ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "$CMD; rc=\$?; chmod -R a+rwX /kpct/grounded_reader 2>/dev/null; exit \$rc"
