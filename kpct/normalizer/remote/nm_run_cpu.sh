#!/bin/bash
# nm_run_cpu.sh <tag> <cpus> <command string>  -- CPU-only container (no --gpus), not in the GPU usage log (= run_cpu_r2.sh,
# but the frozen kpct tree is mounted READ-ONLY and only /kpct/normalizer is writable).
# Executor jobs are process-parallel: OpenMP/BLAS pinned to 1 thread (fork safety, as run_cpu_p.sh).
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
TAG=$1; NCPU=$2; shift 2; CMD="$*"
K=$WORK_DIR; M=$MTQ_DIR; R3=$K/normalizer
docker run --rm --name nmcpu_${TAG}_$(date +%s) --cpus=$NCPU \
  -v $K:/kpct:ro -v $R3:/kpct/normalizer -v $M:/mtq:ro -v $M/MultiTQ-94ad61c/data/MultiTQ:/data:ro \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONPATH=/kpct/pylib \
  -w /kpct/normalizer ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "$CMD; rc=\$?; chmod -R a+rwX /kpct/normalizer 2>/dev/null; exit \$rc"
