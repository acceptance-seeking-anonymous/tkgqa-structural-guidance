#!/bin/bash
# usage: bash job_test.sh <LLM list> <executor subset> "<seed:ckpt:tau ...>"
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
LST=$1; SUB=$2; SPEC=$3
K=$WORK_DIR; PS=$K/parser_seeds
tar xzf test_in.tgz
for x in $SPEC; do
  s=$(echo $x | cut -d: -f1); ck=$(echo $x | cut -d: -f2); tn=$(echo $x | cut -d: -f3); ARM=P_${LST}_lora$s
  cp test_in/infer_stats_P_test.json test_in/raw_P_test_s$s$ck.jsonl.gz $PS/raw/$ARM/
  bash $PS/code/ps_test.sh $ARM $LST $SUB $s $ck $tn
  mkdir -p out/$ARM; cp $PS/freeze/freeze_$ARM.json $PS/test/$ARM/exec_stats_P_${SUB}_test.json out/$ARM/
done
cp $PS/multitq_kpct_seed_freeze_P.json out/; sha256sum $PS/test/*/pred_P_*_test_*.jsonl
tar czf out.tgz out
