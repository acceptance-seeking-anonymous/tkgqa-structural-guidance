#!/bin/bash
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${SCRATCH_B:?set SCRATCH_B (see configs/paths.env.example)}"
set +e
W=${SCRATCH_B:-/tmp/pseeds}; mkdir -p $W/adapters $W/raw $W/logs
cp in/* $W/in/; chmod -R a+rwX $W
sha256sum $W/in/multitq_kpct_seed_amendment_P.json
PIP="pip install --break-system-packages -q --no-deps peft==0.21.2 > /w/logs/pip_peft.log 2>&1"
L=k05_s0
TR="python3 /w/code/pk_train.py --list $L --silver /w/in/silver_${L}_train.jsonl --silver_manifest /w/in/silver_${L}_manifest.json --manifest /w/in/multitq_kpct_subsets.json --guided_common /w/in/gk_common.py --out /w/adapters"
for s in 1 2; do
  N=P_${L}_s$s
  if [ ! -f $W/adapters/$N/train_stats.json ]; then
    bash run_card_pseed.sh train_${N} ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --train_tok 768 > /w/logs/train_$N.out 2>&1"
    rc=$?
    if [ $rc != 0 ] && [ -f $W/adapters/$N/ckpt.pt ]; then
      echo "train $N rc=$rc -> resume once"; tail -5 $W/logs/train_$N.out
      bash run_card_pseed.sh train_${N}_resume ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --train_tok 768 > /w/logs/train_${N}_resume.out 2>&1"; rc=$?
    fi
    if [ $rc != 0 ]; then
      echo "train $N rc=$rc -> OOM fallback: grad_ckpt, train_tok 8192, from scratch"; tail -5 $W/logs/train_$N*.out
      mv $W/adapters/$N $W/adapters/${N}_failed_$(date +%s)
      bash run_card_pseed.sh train_${N}_gc ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --grad_ckpt --train_tok 8192 > /w/logs/train_${N}_gc.out 2>&1"
    fi
  fi
  python3 recipe_check_P.py $W/adapters/$N/train_stats.json $W/in/frozen_train_stats_${L}.json > $W/logs/recipe_check_$N.json; echo "recipe $N rc=$?"; cat $W/logs/recipe_check_$N.json
done
AD="s1ep1=/w/adapters/P_${L}_s1/ep1,s1ep2=/w/adapters/P_${L}_s1/ep2,s2ep1=/w/adapters/P_${L}_s2/ep1,s2ep2=/w/adapters/P_${L}_s2/ep2"
if [ ! -f $W/raw/$L/infer_stats_P_dev_sel2000.json ]; then
  mkdir -p $W/raw/$L
  bash run_card_pseed.sh dev_${L} vllm/vllm-openai:v0.10.2 "cd /w/code && python3 pk_infer.py --split dev_sel2000 --bundles /w/in/q_dev_sel2000.jsonl --adapters $AD --out /w/raw/$L --gpu_mem 0.90 > /w/logs/dev_$L.log 2>&1"
fi
mkdir -p out/adapters out/raw
for s in 1 2; do N=P_${L}_s$s; mkdir -p out/adapters/$N; cp $W/adapters/$N/train_stats.json $W/adapters/$N/train_log.txt out/adapters/$N/ 2>/dev/null; for c in ep1 ep2; do cp $W/adapters/$N/$c/ckpt_info.json out/adapters/$N/ckpt_info_$c.json 2>/dev/null; done; done
cp $W/raw/$L/* out/raw/ 2>/dev/null; cp $W/logs/recipe_check_* $W/logs/dev_$L.log out/ 2>/dev/null
cp $WORK_DIR_B/gpu_usage_pseeds.tsv out/
sha256sum $W/adapters/P_${L}_s*/ep*/adapter_model.safetensors
tail -2 $W/adapters/P_${L}_s*/train_log.txt; tail -3 $W/logs/dev_$L.log; cat out/gpu_usage_pseeds.tsv; df -h "$SCRATCH_B" | tail -1
tar czf out.tgz out
