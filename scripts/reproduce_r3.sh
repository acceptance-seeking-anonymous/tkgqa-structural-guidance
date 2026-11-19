#!/bin/bash
# reproduce_r3.sh <step> [args] -- rewording probe R3 and the dev / training paraphrase sets of the method tracks
# (protocol/multitq_robustness_rule_R3.json, amendments A1-A3). Frozen arms only (no retraining).
#  sets        local, API: r3_common.draw_sample_r3 (2,000 test questions disjoint from R1 / R2; pool 600) ->
#              r3_pipeline.gen_gemini (gemini-2.5-pro, amendment A1; GEMINI_API_KEY) -> mech -> judge (JUDGE_MODEL_ID via
#              robustness/llm_client.py) -> attempt2_prompts -> gen_gemini -> judge -> build_r3_set; after step gen:
#              judge_calls on the first mechanically valid dev attempt -> build_dev_set; build_train_set. These are library
#              calls (robustness/r3/local/r3_pipeline.py; the reported sets were built step by step); this step only
#              prints the sequence.
#  integrity   hash the frozen files of remote/expected_sha_{r3,dev}.template.json into $WORK_DIR/integrity/
#  gen [card]  GPU host, one card: Qwen2.5-72B-Instruct-AWQ dev + training paraphrases (remote/gen_job_r3.sh);
#  completion  [card] the training completion session of amendment A3 (remote/gen_job_r3_completion.sh)
#  stageA      GPU host, CPU containers: frozen-file check, structural rebuilds (R1 rob_build.py), LLM-arm root
#              (remote/jobA3.sh), then the judge-rejected dev root of amendment A2 (remote/jobA3b.sh)
#  arms [card] GPU host, one card: frozen G / P / GU / GPLAIN arms on the rewrite rows (remote/jobBC3.sh)
#  freeze      sha256 freeze of every probe output + predictions archive (remote/jobD3b.sh)
#  dev_score   local: r3_dev_score.score_dev(frozen_rows, llm_dir, gold, acc_quids, jrej_quids, multitq_eval) on dev
#              ORIG vs DEV_ACCEPTED (+ DEV_MECH_OK, descriptive); PYTHONPATH as printed.
# R3 hits are computed only by the method tracks' scoring passes (Parser-first, grounded Reader, normalizer).
# Inputs of the GPU-host steps (not shipped): R3_INPUTS holds variants_r3.jsonl, variants_dev.jsonl,
# variants_devjrej.jsonl, dev_qtext.json and train_order.json written by step sets. JOBDIR (default ./r3_job) is the job
# directory the steps stage their code into. vLLM memory fraction of the arms: GPU_MEM (default 0.40, as run).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR
R=$REPO/robustness; JD=${JOBDIR:-$PWD/r3_job}; mkdir -p "$JD"
PYP="$R/r3:$R/r3/local:$R/r1/local:$R/r2/local:$R:$REPO/src/struct:$REPO/src"
stage() { for f in "$@"; do cp "$f" "$JD/"; done; }
inp() { : "${R3_INPUTS:?set R3_INPUTS (directory with the files written by step sets)}"; for f in "$@"; do cp "$R3_INPUTS/$f" "$JD/"; done; }
case "${1:?step}" in
  sets)
    echo "PYTHONPATH=$PYP; python3 -c 'import r3_common, r3_pipeline' and call, in order:"
    echo "  r3_common.draw_sample_r3 -> r3_pipeline.gen_gemini -> mech / judge -> attempt2_prompts -> gen_gemini -> judge -> build_r3_set"
    echo "  (after step gen) judge_calls on the first mechanically valid dev attempts -> build_dev_set; build_train_set" ;;
  integrity)
    source "$(dirname "$0")/env_check.sh" MTQ_DIR
    I=$WORK_DIR/integrity; mkdir -p "$I"
    for S in r3 dev; do
      [ -f "$I/expected_sha_$S.json" ] && { echo "kept existing $I/expected_sha_$S.json"; continue; }
      python3 "$R/fill_expected_sha.py" "$R/r3/remote/expected_sha_$S.template.json" "$I/expected_sha_$S.json" \
        --kpct "$WORK_DIR" --mtq "$MTQ_DIR" --strict
      chmod a-w "$I/expected_sha_$S.json"
    done ;;
  gen|completion)
    stage "$R/r1/local/rob_common.py" "$R/r2/local/r2_common.py" "$R/r3/r3_common.py" "$R/r3/remote/r3_qwen_gen.py" \
          "$R/r3/remote/run_card_r3.sh" "$R/r3/remote/run_cpu_r3.sh" "$R/r3/remote/gen_job_r3.sh" "$R/r3/remote/gen_job_r3_completion.sh"
    inp dev_qtext.json train_order.json
    J=gen_job_r3.sh; [ "$1" = completion ] && J=gen_job_r3_completion.sh
    (cd "$JD" && bash $J "${2:-0}") ;;
  stageA)
    stage "$R/r1/remote/rob_build.py" "$R/r1/remote/rob_templates.py" "$R/r1/remote/rob_collect_frozen.py" \
          "$R/r3/remote/r3_combine.py" "$R/r3/remote/run_cpu_r3.sh" "$R/r3/remote/run_card_r3.sh" "$R/r3/remote/jobA3.sh" \
          "$R/r3/remote/jobA3b.sh" "$R/r3/remote/jobBC3.sh" "$R/r3/remote/jobD3.sh" "$R/r3/remote/jobD3b.sh"
    inp variants_r3.jsonl variants_dev.jsonl variants_devjrej.jsonl
    cp "$WORK_DIR/integrity/expected_sha_r3.json" "$WORK_DIR/integrity/expected_sha_dev.json" "$JD/"
    (cd "$JD" && bash jobA3.sh && bash jobA3b.sh) ;;
  arms)   (cd "$JD" && bash jobBC3.sh "${2:-0}") ;;
  freeze) (cd "$JD" && bash jobD3b.sh) ;;
  dev_score) echo "PYTHONPATH=$PYP; r3_dev_score.score_dev(...) (see robustness/r3/local/r3_dev_score.py)" ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
