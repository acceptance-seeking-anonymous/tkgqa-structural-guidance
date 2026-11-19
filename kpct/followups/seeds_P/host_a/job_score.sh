#!/bin/bash
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
set +e
K=$WORK_DIR; PS=$K/parser_seeds; mkdir -p $PS/score
# ref_seed0.json = sha256 of the frozen seed-0 test prediction files (paths relative to $WORK_DIR). It is written by
# scripts/freeze_integrity_refs.sh right after the test pass ($WORK_DIR/integrity/ref_seed0.json), so the check
# detects any change of those files made afterwards.
if [ ! -f ref_seed0.json ]; then
  if [ -f "$K/integrity/ref_seed0.json" ]; then cp "$K/integrity/ref_seed0.json" ref_seed0.json
  elif [ "${ALLOW_SELF_REFERENCE:-0}" = 1 ]; then
    echo "WARNING: ref_seed0.json built from the current files (ALLOW_SELF_REFERENCE=1): the seed-0 check is then vacuous" >&2
    python3 "${REPO:?set REPO to the repository root}/robustness/fill_expected_sha.py" ref_seed0.template.json ref_seed0.json --root "$K" --strict
  else
    echo "ERROR: no seed-0 reference. Run scripts/freeze_integrity_refs.sh after the test pass (or set ALLOW_SELF_REFERENCE=1)." >&2; exit 1
  fi
fi
cp ref_seed0.json $PS/score/
python3 $PS/code/ps_score.py --test_json $DATA_DIR/questions/test.json --scorer $K/executor/struct/multitq_eval.py --ref $PS/score/ref_seed0.json --out $PS/score/multitq_kpct_seed_per_question_scored_P.csv.gz
echo "score rc=$?"
mkdir -p out; cp $PS/score/* out/; ls -la out; tar czf out.tgz out
