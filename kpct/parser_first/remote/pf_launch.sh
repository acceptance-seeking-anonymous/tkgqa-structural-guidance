#!/bin/bash
# pf_launch.sh -- run pf_replay_all.sh in a CPU-only container (no GPUs) from a workspace that holds
# pf_replay_all.sh and sa_parser_replay.py (analysis/structural_ablation/); the kpct tree is mounted read-only.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -e
K=$WORK_DIR
W=$(pwd)
docker run --rm --name pfirst_replay_$(date +%s) --cpus=30 -v $K:/kpct:ro -v $W:/workspace -w /workspace \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e PYTHONPATH=/kpct/pylib \
  -e PK_CODE=/kpct/parser/code -e KPCT_EXECUTOR=/kpct/executor ${GPU_IMAGE:-tkgqa-gpu:latest} bash /workspace/pf_replay_all.sh
cat logs/rc.txt
