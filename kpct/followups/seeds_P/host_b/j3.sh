#!/bin/bash
# usage: bash j3.sh <LLM list> <adapters name=path,...  (paths under /w/adapters)>
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${SCRATCH_B:?set SCRATCH_B (see configs/paths.env.example)}"
set +e
L=$1; AD=$2
W=${SCRATCH_B:-/tmp/pseeds}; cp in/* $W/in/ 2>/dev/null; sha256sum $W/in/q_test.jsonl
O=$W/raw/${L}_test; mkdir -p $O
if [ ! -f $O/infer_stats_P_test.json ]; then
  bash run_card_pseed.sh test_${L} vllm/vllm-openai:v0.10.2 "cd /w/code && python3 pk_infer.py --split test --bundles /w/in/q_test.jsonl --adapters $AD --out /w/raw/${L}_test --gpu_mem 0.90 > /w/logs/test_$L.log 2>&1"
fi
mkdir -p out; cp $O/* out/; cp $W/logs/test_$L.log out/; cp $WORK_DIR_B/gpu_usage_pseeds.tsv out/
grep -v "INFO\|WARNING" $W/logs/test_$L.log | tail -5; tail -2 out/gpu_usage_pseeds.tsv; ls -la out
tar czf out.tgz out
