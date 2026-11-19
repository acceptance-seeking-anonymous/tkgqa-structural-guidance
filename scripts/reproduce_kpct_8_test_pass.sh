#!/bin/bash
# reproduce_kpct_8_test_pass.sh -- the single test pass.
#   1. build_test_manifest.py: every registered test file and its sha256, from the five freeze records
#   2. remote stage (GPU host, CPU container, frozen tree read-only): sha256 check of every file, then official scoring
#      (multitq_eval.hit_at_k, k = 1, 10) -> kpct_test_hits.npz
#   3. analyze / summarize (local, numpy): paired Poisson(1) bootstrap (B = 2000, seed 0), KP1-KP4, Holm over KP2-KP4,
#      strata -> multitq_kpct_test_results.json + per-question CSV
# FREEZES must hold the five freeze JSONs; CONFIG = filled copy of configs/test_pass_analyze_config.template.json.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" MTQ_DIR DATA_DIR FREEZES CONFIG
JD=${JOBDIR:-$PWD/kpct_testpass}; mkdir -p "$JD"; T=$REPO/kpct/test_pass
python3 "$T/build_test_manifest.py" --structural "$FREEZES/multitq_kpct_structural_freeze.json" --phase1 "$FREEZES/multitq_structural_freeze.json" \
  --guided "$FREEZES/multitq_kpct_guided_freeze.json" --parser "$FREEZES/multitq_kpct_parser_freeze.json" \
  --k0 "$FREEZES/multitq_kpct_k0_freeze.json" --out "$JD/kpct_test_manifest.json"
cp "$T/multitq_kpct_test_pass.py" "$T/multitq_kpct_test_pass_driver.sh" "$REPO/src/multitq_eval.py" "$DATA_DIR/questions/test.json" "$JD/"
(cd "$JD" && bash multitq_kpct_test_pass_driver.sh)
cd "$JD"
python3 multitq_kpct_test_pass.py analyze --hits kpct_test_hits.npz --meta kpct_test_scoring_meta.json \
  --hash-check hash_check.json --test-json test.json --scorer multitq_eval.py --config "$CONFIG"
python3 multitq_kpct_test_pass.py summarize --results multitq_kpct_test_results.json --config "$CONFIG"
# record the hashes of the frozen outputs now (checked later by the robustness probes and the parser seed scoring)
bash "$REPO/scripts/freeze_integrity_refs.sh"
