#!/bin/bash
# reproduce_parser_first.sh <step> -- Parser-first track (protocol/multitq_parser_first_rule.json): gated variants of the
# Parser arm P_k that re-use the CACHED Parser generations (no new LLM inference, CPU only). Variants (kpct/parser_first/
# pf_lib.py): V0 frozen P_k; V1 PALONE (Parser hypothesis only); V2 PALONE, else PK when the Parser-only parse fails;
# V3(t) PK when the mean token log-probability of the parse < t, else V2 (t in {-0.30, -0.10, -0.03, -0.01, -0.003}).
#  replay   GPU host, CPU container: analysis/structural_ablation/sa_parser_replay.py (unchanged) on dev_sel2000
#           (k = 1, 5, 10, 100) and on the R3 rewrite root robustness_r3/llm (k = 1, 5, 100) -> $PF_WS/replay
#  select   dev-only selection: pf_select.py -> multitq_parser_first_selection.json (reads dev.json gold only)
#  freeze   gold-free routing table + sha256 freeze: pf_freeze.py (PF_R3_SET = R3 perturbed set json)
#  score    the single scoring pass: pf_score.py (full test, R3 secondary analysis, R1 / R2 descriptive)
# Workspace PF_WS (not shipped): replay/ (step replay), saA/ (structural-ablation Part A predictions archive), r3pred/,
# r1raw/, r2raw/ (frozen R3 / R1 / R2 probe archives), multitq_eval.py (= src/multitq_eval.py); PF_FREEZE_A = the
# structural-ablation freeze_A json; PF_KPCT_PRED = local copy of the k% predictions (parser/dev_sel2000/<k>/, parser/test/<k>/).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR OUT_DIR PF_WS
P=$REPO/kpct/parser_first; O=$OUT_DIR/parser_first; mkdir -p "$O"
export PF_WS MTQ_DEV_JSON=$DATA_DIR/questions/dev.json MTQ_TEST_JSON=$DATA_DIR/questions/test.json
cp -n "$REPO/src/multitq_eval.py" "$PF_WS/" 2>/dev/null || true
case "${1:?step}" in
  replay)
    source "$(dirname "$0")/env_check.sh" WORK_DIR
    mkdir -p "$PF_WS/replay"; cp "$P/remote/pf_replay_all.sh" "$REPO/analysis/structural_ablation/sa_parser_replay.py" "$PF_WS/replay/"
    (cd "$PF_WS/replay" && bash "$P/remote/pf_launch.sh") ;;
  select)
    : "${PF_FREEZE_A:?set PF_FREEZE_A}"; export PF_FREEZE_A
    python3 "$P/pf_select.py" "$O" ;;
  freeze)
    : "${PF_R3_SET:?set PF_R3_SET}"; export PF_R3_SET
    python3 "$P/pf_freeze.py" "$O/multitq_parser_first_selection.json" "$O" ;;
  score)
    C=$O/r3_code; mkdir -p "$C"
    cp "$REPO/robustness/r1/local/rob_common.py" "$REPO/robustness/r2/local/r2_common.py" "$REPO/robustness/r3/r3_common.py" "$C/"
    R3_CODE=$C python3 "$P/pf_score.py" "$O/multitq_parser_first_freeze.json" "$O" ;;
  *) echo "unknown step $1" >&2; exit 2 ;;
esac
