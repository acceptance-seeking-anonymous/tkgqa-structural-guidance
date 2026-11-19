#!/bin/bash
# reproduce_structural_ablation.sh <part> -- structural ablation (protocol/multitq_structural_ablation_rule.json), CPU only.
#   A  Parser-alone vs Parser-plus-structural-alternatives: CPU replay of the cached P_k generations under four parse
#      modes (PK = frozen P_k, PALONE = Parser hypothesis only, PREL / PENT = relation / entity alternatives only) on the
#      test split (k = 1, 5, 10, 100) and on the R1 / R2 rewrites (k = 1, 5, 100); freeze; single scoring pass.
#   B  Ranker feature-group ablation: retrain the frozen S recipe with each of six feature groups (and two composites)
#      removed, k = 1 and 5, PYTHONHASHSEED 0 (all arms) and 7 (full only, noise floor); freeze; single scoring pass.
# Inputs (not shipped): KPCT_TREE = local copy of the k% outputs ($WORK_DIR layout: parser/test/<k>/raw_P_test_*.jsonl.gz
# and pred_P_*, bundles/<k>/test.jsonl, preds/<k>/test_S_seed*.jsonl); R1_PRED_DIR / R2_PRED_DIR = frozen probe prediction
# trees (preds/, P/); ROB_R1 / ROB_R2 = probe roots with inputs/variants.jsonl; SWORK = structural tree (stage/kpct/executor).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" DATA_DIR SWORK OUT_DIR
A=$REPO/analysis/structural_ablation; O=$OUT_DIR/structural_ablation; mkdir -p "$O"
RID=${RULE_ID:-$(sha256sum "$REPO/protocol/multitq_structural_ablation_rule.json" | cut -c1-64)}
export PK_CODE=$REPO/kpct/parser KPCT_EXECUTOR=$SWORK/stage/kpct/executor MTQ_ROOT=$DATA_DIR OMP_NUM_THREADS=1
declare -A CK=([k01_s0]=ep3 [k05_s0]=ep1 [k10_s0]=ep0.5 [k100]=ep0.5)
case "${1:-}" in
  A)
    : "${KPCT_TREE:?set KPCT_TREE}"; KT=$KPCT_TREE
    for k in k01_s0 k05_s0 k10_s0 k100; do
      python3 "$A/sa_parser_replay.py" --k $k --split test --raw "$KT/parser/test/$k/raw_P_test_${CK[$k]}.jsonl.gz" \
        --bundles "$KT/bundles/$k/test.jsonl" --fallback_dir "$KT/preds/$k" \
        --frozen_pred "$KT/parser/test/$k/pred_P_${k}_test_${CK[$k]}_tauoff_seed{b}.jsonl.gz" --out "$O/out/A_test" --workers ${WORKERS:-9}
    done
    for R in R1 R2; do
      PD=${R}_PRED_DIR; RB=ROB_$R; : "${!PD:?set $PD}"; : "${!RB:?set $RB}"
      for k in k01_s0 k05_s0 k100; do
        KPCT_EXECUTOR=$SWORK/stage/kpct/executor python3 "$REPO/robustness/r1/remote/rob_build.py" --k $k --root "${!RB}" --workers ${WORKERS:-8} \
          --frozen_preds "$KT/preds/$k" --frozen_bundles "$KT/bundles/$k/test.jsonl"
        python3 "$A/sa_parser_replay.py" --k $k --split $R --raw "${!PD}/P/raw/raw_P_variants_P_$k.jsonl.gz" \
          --bundles "${!RB}/bundles/$k/variants.jsonl" --fallback_dir "${!PD}/preds/$k" --fallback_prefix variants \
          --frozen_pred "${!PD}/P/$k/pred_P_${k}_variants_P_${k}_tauoff_seed{b}.jsonl" --out "$O/out/A_$R"
      done
    done
    python3 "$A/sa_freeze.py" A "$O/multitq_structural_ablation_freeze_A.json" "$RID" "$O/out/A_test" "$O/out/A_R1" "$O/out/A_R2"
    : "${HITS:?set HITS to the per-question hits csv.gz of the k% test pass}"
    SA_WORK=$O MTQ_TEST_JSON=$DATA_DIR/questions/test.json KPCT_HITS_CSV=$HITS \
      python3 "$A/sa_score_A.py" "$O/multitq_structural_ablation_freeze_A.json" "$O/results_A" ;;
  B)
    cp "$A/sb_ablate.py" "$A/sa_stats.py" "$SWORK/"; cd "$SWORK"
    export KPCT_NPROC=${KPCT_NPROC:-8} OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
    for HS in 0 7; do
      for k in k01_s0 k05_s0; do
        PYTHONHASHSEED=$HS KPCT_OUT=$SWORK/out_hs$HS KPCT_STAGE=fitA python3 kpct_retrain_fixed_vocab.py fitA $k --preserved "$SWORK/out/$k"
        if [ $HS = 0 ]; then PYTHONHASHSEED=0 KPCT_OUT=$SWORK/out_hs0 python3 sb_ablate.py train $k
        else PYTHONHASHSEED=7 KPCT_OUT=$SWORK/out_hs7 python3 sb_ablate.py train $k full; fi
      done
    done
    for k in k01_s0 k05_s0; do PYTHONHASHSEED=0 KPCT_OUT=$SWORK/out_hs0 python3 sb_ablate.py infer $k "$O/pred_B" full_hs7=$SWORK/out_hs7; done
    python3 "$A/sa_freeze.py" B "$O/multitq_structural_ablation_freeze_B.json" "$RID" "$O/pred_B"
    : "${HITS:?set HITS to the per-question hits csv.gz of the k% test pass}"
    SB_PRED=$O/pred_B SB_ROOT=$SWORK MTQ_QDIR=$DATA_DIR/questions KPCT_HITS_CSV=$HITS \
      python3 "$A/sa_score_B.py" "$O/multitq_structural_ablation_freeze_B.json" "$O/results_B" ;;
  *) echo "usage: $0 A|B" >&2; exit 2 ;;
esac
