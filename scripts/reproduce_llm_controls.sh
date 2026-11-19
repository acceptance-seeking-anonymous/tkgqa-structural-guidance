#!/bin/bash
# reproduce_llm_controls.sh [item ...] -- LLM controls (amendment LC1 to protocol/multitq_kpct_decision_rule.json, with
# LC1-A1; protocol/multitq_kpct_amendment_LC1_*.json) on the GPU host, one card: GPLAIN_1 = Reader_1 with the same
# evidence rendered as a plain alphabetical list (no parse, ranks, scores, operator flags or links;
# kpct/llm_controls/gk_plain.py), trained and selected with the frozen Reader recipe; GU_5 = Dense-RAG at 5%.
# GPU items in the registered run order (cut point: GPLAIN_1 first, then GU_5):
#   setup gplain_train gplain_session gu5_train gu5_session rob_gplain rob_gu5 tokens
# Registered GPU-hour cap of the track: 5.5 GPU-h (checked before every GPU item; see lc_job.sh).
# Memory-only options (amendment LC1-A1; defaults = the reported runs): LC_GRAD_CKPT=1 (gradient checkpointing in LoRA
# training), LC_GPU_MEM=0.40 (vLLM gpu_memory_utilization; the frozen Reader recipe uses 0.85).
#
# reproduce_llm_controls.sh score -- CPU, after the freeze records exist: single scoring pass (kpct/llm_controls/lc_score.py
# -> lc_analyze.py). Inputs (not shipped): LC_OUT = copy of $WORK_DIR/llm_controls (preds/, robust/, tokens/);
# HITS = per-question hits csv.gz of the k% test pass; R1_SET / R2_SET = R1 / R2 perturbed sets; R1_PRED_DIR /
# R2_PRED_DIR = frozen probe prediction trees (preds/, G/, P/, GU/, checks/freeze_sha256.txt).
set -euo pipefail
if [ "${1:-}" = score ]; then
  source "$(dirname "$0")/env_check.sh" DATA_DIR OUT_DIR HITS R1_SET R2_SET R1_PRED_DIR R2_PRED_DIR
  python3 "$REPO/kpct/llm_controls/lc_score.py" --scorer "$REPO/src/multitq_eval.py" --test "$DATA_DIR/questions/test.json" \
    --hits "$HITS" --lc "${LC_OUT:-${WORK_DIR:?set LC_OUT or WORK_DIR}/llm_controls}" \
    --r1-set "$R1_SET" --r1-pred "$R1_PRED_DIR" --r2-set "$R2_SET" --r2-pred "$R2_PRED_DIR" \
    --outdir "$OUT_DIR/llm_controls"
  exit 0
fi
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
JD=${JOBDIR:-$PWD/llm_controls_job}; mkdir -p "$JD"
cp "$REPO"/kpct/llm_controls/* "$JD/"
(cd "$JD" && bash lc_job.sh ${@:-setup gplain_train gplain_session gu5_train gu5_session rob_gplain rob_gu5 tokens})
