#!/bin/bash
: "${MTQ_DIR:?set MTQ_DIR (see configs/paths.env.example)}"
W=$(pwd)
rc=0
docker run --rm --name kpct_testpass_$(date +%s) --cpus 12 -e OMP_NUM_THREADS=1 -v "$W":"$W" -w "$W" -v "$MTQ_DIR":/mtq:ro ${GPU_IMAGE:-tkgqa-gpu:latest} \
  python3 multitq_kpct_test_pass.py remote --root /mtq --manifest kpct_test_manifest.json --test-json test.json --scorer multitq_eval.py --out out --workers 10 || rc=$?
docker run --rm -v "$W":"$W" ${GPU_IMAGE:-tkgqa-gpu:latest} chmod -R a+rX "$W/out" || true
cp out/* . 2>/dev/null || true
echo "rc=$rc"
exit $rc
