#!/bin/bash
# usage: bash job_dev.sh <LLM list> <executor subset> <seeds e.g. "1 2"> <ckpts e.g. ep1,ep2>
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set +e
LST=$1; SUB=$2; SEEDS=$3; CKS=$4
K=$WORK_DIR; PS=$K/parser_seeds; mkdir -p $PS/code $PS/train $PS/raw $PS/freeze out
cp ps_dev.sh ps_test.sh ps_freeze.py ps_score.py $PS/code/ 2>/dev/null
tar xzf dev_in.tgz
for s in $SEEDS; do
  ARM=P_${LST}_lora$s; mkdir -p $PS/train/$ARM $PS/raw/$ARM
  cp dev_in/train/P_${LST}_s$s/train_stats.json dev_in/train/P_${LST}_s$s/train_log.txt $PS/train/$ARM/
  cp dev_in/recipe_check_P_${LST}_s$s.json $PS/train/$ARM/recipe_check.json
  cp dev_in/raw/infer_stats_P_dev_sel2000.json $PS/raw/$ARM/
  MAP=""; for c in $(echo $CKS | tr ',' ' '); do cp dev_in/raw/raw_P_dev_sel2000_s$s$c.jsonl.gz $PS/raw/$ARM/; MAP="$MAP${MAP:+,}$c=raw_P_dev_sel2000_s$s$c.jsonl.gz"; done
  bash $PS/code/ps_dev.sh $ARM $LST $SUB $MAP $CKS
  FM=$(echo $CKS | tr ',' '\n' | awk -v s=$s '{printf "%s%s=s%s%s", (NR>1?",":""), $1, s, $1}')
  python3 $PS/code/ps_freeze.py --mode pre_test --arm $ARM --list $LST --sub $SUB --seed $s --map $FM
  echo "pre-test freeze $ARM rc=$?"
  mkdir -p out/$ARM; cp $PS/select/$ARM/* $PS/exec/$ARM/exec_stats_* $PS/freeze/freeze_$ARM.json out/$ARM/ 2>/dev/null
done
cp $PS/multitq_kpct_seed_freeze_P.json out/ 2>/dev/null
for s in $SEEDS; do python3 -c "
import json; s=json.load(open('out/P_${LST}_lora$s/select_P_${LST}.json')); print('P_${LST}_lora$s', s['chosen']['checkpoint'], s['chosen']['tau'], s['dev_sel2000_hits1'], s['dev_sel2000_hits1_ci95'], s['unseen_hits1'], s['STOP_DEV_FAILURE'])
for g in s['grid']: print('  ', g['checkpoint'], g['tau'], round(g['hits1'],4), g['hits1_per_seed'], round(g['fallback_rate'],4))"; done
tar czf out.tgz out
