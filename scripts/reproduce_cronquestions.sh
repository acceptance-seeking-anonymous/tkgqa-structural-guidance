#!/bin/bash
# reproduce_cronquestions.sh -- CronQuestions structural comparison (Rules vs Ranker; no LLM arm). run_cron_structural.py
# is a transcription of the executed steps (parser, interval operators, LightGBM lambdarank S, test prediction files);
# score with cronquestions_eval.py (official CronKGQA semantics). Needs $CRON_DATA (CronQuestions release).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" CRON_DATA OUT_DIR
O=$OUT_DIR/cronquestions; mkdir -p "$O"; cp "$REPO"/cronquestions/*.py "$O/"; mkdir -p "$O/models"
cp "$REPO/cronquestions/models/pipeline_config.json" "$O/models/"
cd "$O" && PYTHONHASHSEED=${PYTHONHASHSEED:-0} python3 run_cron_structural.py
