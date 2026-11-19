#!/bin/bash
# reproduce_analysis.sh -- post-hoc analyses from frozen outputs (CPU):
#   per-type breakdown (analysis/type_breakdown.py), Reader_1 copy / fix / loss mechanism (analysis/reader_mechanism.py),
#   R1 parser-vs-reader rewording contrasts (analysis/rewording_contrasts.py), R1 tipping point (analysis/tipping_point.py).
# Inputs (not shipped): HITS = per-question test hits CSV of the test pass; GUIDED_TGZ / BUNDLES_TAR = frozen G_1
# predictions and k01_s0 bundles / S lists; R1_GOLD = paraphrase_review_gold.json; R1_PRED_DIR = extracted R1 predictions.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR OUT_DIR HITS
A=$REPO/analysis; O=$OUT_DIR/analysis; mkdir -p "$O"
python3 "$A/type_breakdown.py" --test-json "$DATA_DIR/questions/test.json" --hits "$HITS" --out "$O/multitq_pertype_ci.csv"
if [ -n "${GUIDED_TGZ:-}" ] && [ -n "${BUNDLES_TAR:-}" ]; then
  python3 "$A/reader_mechanism.py" "$HITS" "$GUIDED_TGZ" "$BUNDLES_TAR" "$O/reader_mechanism.json"
fi
if [ -n "${R1_GOLD:-}" ] && [ -n "${R1_PRED_DIR:-}" ]; then
  python3 "$A/rewording_contrasts.py" --gold "$R1_GOLD" --pred-dir "$R1_PRED_DIR" --out "$O/robustness_posthoc_parser_vs_reader.csv"
  python3 "$A/tipping_point.py" --gold "$R1_GOLD" --pred-dir "$R1_PRED_DIR" --out "$O/tipping_point_R1_pre_review.csv"
fi
