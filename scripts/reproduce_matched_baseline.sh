#!/bin/bash
# reproduce_matched_baseline.sh <step> -- matched MultiQA baseline on the registered nested k% train subsets
# (protocol/multitq_matched_baseline_rule.json; descriptive comparator, no pass/fail margin).
#   prepare  job directory: upstream MultiQA code, KG data and shipped TComplEx (baselines/multiqa_matched/fetch_multiqa.sh),
#            question inputs and subset lists (mbq_make_bundle.py), driver code and run plans
#   ner_tcx  GPU host: NER preprocessing + full-KG TComplEx retrain (jobA.sh)
#   smoke    GPU host, optional: dev-only feasibility run (jobB.sh)
#   train    GPU host: plan_C.json (1%, 5%) then plan_DE.json (10%, 100%) (jobQA.sh); runs kept in runs_C/, runs_DE/
#   freeze   local: dev_sel2000 embedding selection per level + sha256 freeze of every test prediction file (no test label)
#   score    local, once: official Hits@1 / Hits@10, Poisson bootstrap CIs (B = 2000, seed 0), paired contrasts with
#            RULE_k, S_k, G_k, P_k from the per-question hits of the k% test pass (HITS)
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR
B=$REPO/baselines/multiqa_matched; JD=${JOBDIR:-$PWD/mbq_job}
case "${1:-}" in
  prepare)
    : "${SWORK:?set SWORK (the k% structural tree holding subsets/multitq_kpct_subsets.json)}"
    bash "$B/fetch_multiqa.sh" "$JD"
    python3 "$B/mbq_make_bundle.py" --questions "$DATA_DIR/questions" --manifest "$SWORK/subsets/multitq_kpct_subsets.json" --out "$JD"
    cp "$B"/mbq_ner.py "$B"/mbq_qa.py "$B"/mbq_sched.py "$B"/mbq_tcx_full.py "$B"/plans/*.json "$B"/job*.sh "$JD/" ;;
  ner_tcx) (cd "$JD" && bash jobA.sh) ;;
  smoke)   (cd "$JD" && bash jobB.sh) ;;
  train)   (cd "$JD" && PLAN=plan_C.json bash jobQA.sh && mv runs runs_C && PLAN=plan_DE.json bash jobQA.sh && mv runs runs_DE) ;;
  freeze)
    : "${OUT_DIR:?set OUT_DIR}"; O=$OUT_DIR/matched_baseline; mkdir -p "$O"
    RID=${RULE_ID:-$(sha256sum "$REPO/protocol/multitq_matched_baseline_rule.json" | cut -c1-64)}
    python3 "$B/mbq_select_freeze.py" "$JD/runs_C,$JD/runs_DE" "$O" "$RID" ;;
  score)
    : "${OUT_DIR:?set OUT_DIR}"; : "${HITS:?set HITS to the per-question hits csv.gz of the k% test pass}"; O=$OUT_DIR/matched_baseline
    python3 "$B/mbq_score.py" "$O/multitq_matched_baseline_freeze.json" "$DATA_DIR/questions/test.json" "$REPO/src/multitq_eval.py" \
      "$HITS" "$O/multitq_matched_baseline_selection.json" "$O/multitq_matched_baseline" ;;
  *) echo "usage: $0 prepare|ner_tcx|smoke|train|freeze|score" >&2; exit 2 ;;
esac
