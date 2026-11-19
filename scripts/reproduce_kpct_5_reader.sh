#!/bin/bash
# reproduce_kpct_5_reader.sh -- Reader G_k and Dense-RAG GU_k: LoRA training (Llama-3.1-8B-Instruct), dev_sel2000
# checkpoint selection, freeze, test generation (+ SHUF_G1 / QONLY_G1 controls in the G_1 session). Two GPUs, one
# process per card; training image + peft 0.21.2 (installed in-container), inference image with vLLM 0.10.2.
# LoRA training and vLLM greedy decoding are not bitwise deterministic: compare Hits@1 / top-1 agreement, not sha256.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
K=$WORK_DIR; JD=${JOBDIR:-$PWD/guided_job}; mkdir -p "$JD" "$K/guided/"{logs,checks,preds,adapters}
cp -r "$K/guided/code" "$JD/code"; cp "$K/guided/code/gk_run_arm.py" "$JD/"
# k = 1 core: card 0 = bundle checks, G_1 (two candidate orders), G_1 session with SHUF_G1 / QONLY_G1; card 1 = GU_1
(cd "$JD" && bash code/queue_core_k1.sh > core_k1.log 2>&1) & (cd "$JD" && bash code/queue_gu1_card1.sh > gu1_card1.log 2>&1); wait
# remaining arms from a shared queue on both cards (candidate order fixed by order_decision_k01_s0.json)
printf '%s\n' G_100 GU_100 G_10 GU_10 G_5 G_1s1 G_1s2 > "$K/guided/queue_main.txt"
(cd "$K/guided/code" && bash queue_runner.sh 0 > "$JD/runner0.log" 2>&1) & (cd "$K/guided/code" && bash queue_runner.sh 1 > "$JD/runner1.log" 2>&1); wait
