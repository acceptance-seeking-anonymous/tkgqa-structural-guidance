#!/bin/bash
# freeze_integrity_refs.sh -- record the sha256 of the frozen k% outputs right after the test pass
# (scripts/reproduce_kpct_8_test_pass.sh calls it at the end). Later stages compare against these files:
#   $WORK_DIR/integrity/expected_sha_R1.json  -> robustness/r1/remote/jobA.sh  (STOP_TEST_INTEGRITY of the R1 probe)
#   $WORK_DIR/integrity/expected_sha_R2.json  -> robustness/r2/remote/jobA2.sh (same check for R2)
#   $WORK_DIR/integrity/ref_seed0.json        -> kpct/followups/seeds_P/host_a/job_score.sh (seed-0 test predictions)
# Existing freeze files are never overwritten (delete them by hand to re-freeze). Files are made read-only.
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR MTQ_DIR
I=$WORK_DIR/integrity; mkdir -p "$I"; F=$REPO/robustness/fill_expected_sha.py
freeze() {  # out template [extra args]
  local out=$1 tpl=$2; shift 2
  if [ -f "$I/$out" ]; then echo "kept existing $I/$out"; return; fi
  python3 "$F" "$tpl" "$I/$out" --kpct "$WORK_DIR" --mtq "$MTQ_DIR" --strict "$@"
  chmod a-w "$I/$out"; echo "frozen $I/$out sha256 $(python3 -c 'import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],"rb").read()).hexdigest())' "$I/$out")"
}
freeze expected_sha_R1.json "$REPO/robustness/r1/remote/expected_sha.template.json"
freeze expected_sha_R2.json "$REPO/robustness/r2/remote/expected_sha.template.json"
freeze ref_seed0.json "$REPO/kpct/followups/seeds_P/host_a/ref_seed0.template.json" --root "$WORK_DIR"
