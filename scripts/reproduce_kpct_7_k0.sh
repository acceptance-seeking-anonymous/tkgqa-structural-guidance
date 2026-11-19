#!/bin/bash
# reproduce_kpct_7_k0.sh -- k = 0 arms with Llama-3.1-70B-Instruct (bf16, vLLM tensor-parallel 2 on both cards):
# P_0 (few-shot JSON parser, 3 prompt designs on dev_sel2000, selected design on test, k0 executor) and G_0 (zero-shot
# Reader guided by P_0's bundles, designs g1-g3). `k0_run.sh dry` is a CPU plumbing run on 5 dev questions.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
JD=${JOBDIR:-$PWD/k0_job}; mkdir -p "$JD"
bash "$WORK_DIR/k0/code/k0_run.sh" p0 "$JD"
bash "$WORK_DIR/k0/code/k0_run.sh" g0 "$JD"
