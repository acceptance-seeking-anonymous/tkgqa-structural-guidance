#!/bin/bash
# reproduce_normalizer.sh <step> [args] -- question normalizer N (protocol/multitq_normalizer_rule.json, amendment A1):
# a LoRA-tuned Llama-3.1-8B that rewrites a question into the benchmark's template wording before the UNCHANGED frozen
# pipeline (RULE, S, P_k, G_k at k = 5 and 100); guard + identity reuse in kpct/normalizer/nm_common.py.
#  items     local: nm_common.training_pairs(train rows, training paraphrases of the R3 track) -> train_pairs_k05.jsonl;
#            text-only dev / test item files (library calls; the reported files were built step by step)
#  train     second GPU host (one 24 GB card): dry run, registered epoch rule, LoRA training (host_b/nm_train_job.sh
#            <spent GPU-h>)
#  infer     second GPU host: normalizer decoding of an item file with the listed checkpoints (host_b/nm_infer_job.sh
#            <run> <items file> <ep0.5,ep1,...> <spent GPU-h> <projected GPU-h>)
#  root      local: nm_build_root.build(...) -> variants_<dev|test>.jsonl (texts needing a run + 200 identity controls)
#  arms      GPU host: integrity, structural rebuild + identity controls, frozen G / P arms on the normalized texts
#            (remote/nm_job.sh <dev|test> <spent host-b GPU-h> <G_5,P_5,P_100,G_100>); needs the R3 stage-A code staged
#  select    local: nm_select.score_dev(...) + nm_common.objective / select on dev (dev gold only)
#  score     local: the single scoring pass nm_score_driver.py <freeze.json> <out> (full test, R3 secondary, R1 / R2)
# JOBDIR (default ./nm_job) is the job directory; items / roots written by the local steps are read from NM_INPUTS
# (not shipped). vLLM memory fraction of the arms: GPU_MEM (default 0.40, as run); GPU index: NM_CARD (default 0).
set -euo pipefail
source "$(dirname "$0")/env_check.sh"
N=$REPO/kpct/normalizer; JD=${JOBDIR:-$PWD/nm_job}; mkdir -p "$JD"
PYP="$N:$REPO/robustness/r3:$REPO/robustness/r2/local:$REPO/robustness/r1/local:$REPO/src/struct:$REPO/src"
inp() { : "${NM_INPUTS:?set NM_INPUTS}"; for f in "$@"; do cp "$NM_INPUTS/$f" "$JD/"; done; }
case "${1:?step}" in
  items|root|select)
    echo "PYTHONPATH=$PYP; library calls: items -> nm_common.training_pairs; root -> nm_build_root.build;"
    echo "select -> nm_select.score_dev, nm_common.objective, nm_common.select (kpct/normalizer/)" ;;
  train)
    source "$(dirname "$0")/env_check.sh" WORK_DIR_B MODEL_DIR
    cp "$N/nm_common.py" "$N/nm_train.py" "$N/nm_infer.py" "$N/host_b/nm_train_job.sh" "$N/host_b/nm_infer_job.sh" "$JD/"
    inp train_pairs_k05.jsonl
    (cd "$JD" && bash nm_train_job.sh "${2:-0}") ;;
  infer)
    source "$(dirname "$0")/env_check.sh" WORK_DIR_B MODEL_DIR
    inp "${3:?items file}"
    (cd "$JD" && bash nm_infer_job.sh "${2:?run name}" "$3" "${4:?checkpoints}" "${5:-0}" "${6:-0.1}") ;;
  arms)
    source "$(dirname "$0")/env_check.sh" WORK_DIR
    S=${2:?dev or test}
    cp "$N/remote/nm_run_card.sh" "$N/remote/nm_run_cpu.sh" "$N/remote/nm_job.sh" "$N/remote/nm_verify.py" "$JD/"
    inp "variants_$S.jsonl"
    (cd "$JD" && bash nm_job.sh "$S" "${3:-0}" "${4:-G_5,P_5,P_100,G_100}") ;;
  score)
    PYTHONPATH=$PYP python3 "$N/nm_score_driver.py" "${2:?freeze json}" "${3:?out dir}" ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
