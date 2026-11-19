#!/bin/bash
# reproduce_output_analyses.sh <A1|A2|A2_followup> -- descriptive output analyses of already frozen, already scored outputs
# (protocol/multitq_output_analyses_plan.json; CPU, no new predictions).
#   A1           absolute rewrite accuracy on R1 / R2 per arm and population (ALL, HUMAN_FILTERED, RANDOM100,
#                RANDOM100_ACCEPTED), drops and paired arm contrasts; reproduction check against the probes' human-audit
#                results (analysis/rewrite_tables/a1_rewrite.py)
#   A2           coverage-conditioned Hits@1 on the test split: gold in displayed facts (F) x gold in displayed candidates (C)
#                (analysis/coverage/a2_coverage_run.py)
#   A2_followup  Dense-RAG retrieved-fact coverage and the Parser's own top-20 coverage (analysis/coverage/a2_gu_parser_followup.py)
# Inputs (not shipped): see the variables below.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR OUT_DIR
O=$OUT_DIR/output_analyses; mkdir -p "$O"
case "${1:-}" in
  A1)
    : "${A1_WORK:?set A1_WORK to a directory holding R1/ and R2/ (frozen probe prediction trees)}"
    for v in R1_GOLD R1_KEY R1_SHEET R1_SEL R1_EXISTING R2_GOLD R2_SHEET R2_SEL R2_FREEZE R2_EXISTING; do : "${!v:?set $v}"; done
    (cd "$A1_WORK" && python3 "$REPO/analysis/rewrite_tables/a1_rewrite.py" --scorer "$REPO/src/multitq_eval.py" \
      --r1-gold "$R1_GOLD" --r1-key "$R1_KEY" --r1-sheet "$R1_SHEET" --r1-sel "$R1_SEL" --r1-existing "$R1_EXISTING" \
      --r2-gold "$R2_GOLD" --r2-sheet "$R2_SHEET" --r2-sel "$R2_SEL" --r2-freeze "$R2_FREEZE" --r2-existing "$R2_EXISTING" --outdir "$O/A1") ;;
  A2)
    : "${HITS:?set HITS}"; : "${BUNDLES:?set BUNDLES to the bundles/ folder (k01_s0, k05_s0, k100)}"
    : "${G_PRED_1:?}"; : "${G_PRED_5:?}"; : "${G_PRED_100:?set G_PRED_<k> to the selected Reader test prediction files}"
    python3 "$REPO/analysis/coverage/a2_coverage_run.py" --test-json "$DATA_DIR/questions/test.json" --hits "$HITS" \
      --bundle 1="$BUNDLES/k01_s0/test.jsonl" --bundle 5="$BUNDLES/k05_s0/test.jsonl" --bundle 100="$BUNDLES/k100/test.jsonl" \
      --g-pred 1="$G_PRED_1" --g-pred 5="$G_PRED_5" --g-pred 100="$G_PRED_100" --out "$O/A2" ;;
  A2_followup)
    : "${FETCH_DIR:?set FETCH_DIR (gu_test.jsonl.gz, the 9 Parser test prediction files .gz, istats__GU_* files)}"
    python3 "$REPO/analysis/coverage/a2_gu_parser_followup.py" --fetch-dir "$FETCH_DIR" --per-question "$O/A2/A2_coverage_per_question.csv" \
      --test-json "$DATA_DIR/questions/test.json" --scorer "$REPO/src/multitq_eval.py" --out "$O/A2_followup" ;;
  *) echo "usage: $0 A1|A2|A2_followup" >&2; exit 2 ;;
esac
