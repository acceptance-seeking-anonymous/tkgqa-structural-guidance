#!/bin/bash
# smoke_test.sh -- CPU checks that need no data: byte-compile every module, run the evaluator unit tests, run the R2
# statistics / annotation-kit tests on synthetic data, and a toy end-to-end on 20 synthetic questions (Hits@1 with the
# official scorer + paired Poisson bootstrap + Holm, as in the test pass) in a temporary directory.
set -euo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd); export PYTHONDONTWRITEBYTECODE=1
PYTHONPYCACHEPREFIX="${TMPDIR:-/tmp}/tkgqa_pycache" python3 -m compileall -q "$REPO" && echo "compile ok"   # byte code kept outside the repo
python3 -m pytest -q "$REPO/tests" 2>/dev/null || { python3 "$REPO/tests/test_multitq_eval.py" && python3 "$REPO/tests/test_research_track.py" && python3 "$REPO/tests/test_robustness_methods.py"; }
python3 "$REPO/tests/toy_end_to_end.py"
