#!/bin/bash
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${SCRATCH_B:?set SCRATCH_B (see configs/paths.env.example)}"
set +e
W=${SCRATCH_B:-/tmp/pseeds}; LG=$WORK_DIR_B/gpu_usage_pseeds.tsv
cp in/* $W/in/; sha256sum $W/in/multitq_kpct_seed_amendment_P_A1.json
for i in $(seq 1 180); do U=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1); [ "$U" -lt 1500 ] && break; sleep 30; done
echo "GPU free check: $U MiB after $i polls"; [ "$U" -lt 1500 ] || { echo "GPU still busy - abort"; exit 3; }
PIP="pip install --break-system-packages -q --no-deps peft==0.21.2 > /w/logs/pip_peft.log 2>&1"
L=k01_s0
TR="python3 /w/code/pk_train.py --list $L --silver /w/in/silver_${L}_train.jsonl --silver_manifest /w/in/silver_${L}_manifest.json --manifest /w/in/multitq_kpct_subsets.json --guided_common /w/in/gk_common.py --out /w/adapters"
spent() { awk -F'\t' 'NR>1{s+=$8} END{printf "%.4f", s}' $LG; }
train() {  # seed
  s=$1; N=P_${L}_s$s
  if [ ! -f $W/adapters/$N/train_stats.json ]; then
    bash run_card_pseed.sh train_${N} ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --train_tok 768 > /w/logs/train_$N.out 2>&1"; rc=$?
    if [ $rc != 0 ] && [ -f $W/adapters/$N/ckpt.pt ]; then echo "resume $N"; bash run_card_pseed.sh train_${N}_resume ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --train_tok 768 > /w/logs/train_${N}_resume.out 2>&1"; rc=$?; fi
    if [ $rc != 0 ]; then echo "OOM fallback $N"; mv $W/adapters/$N $W/adapters/${N}_failed_$(date +%s); bash run_card_pseed.sh train_${N}_gc ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --seed $s --name $N --grad_ckpt --train_tok 8192 > /w/logs/train_${N}_gc.out 2>&1"; fi
  fi
  python3 recipe_check_P.py $W/adapters/$N/train_stats.json $W/in/frozen_train_stats_${L}.json > $W/logs/recipe_check_$N.json; echo "recipe $N rc=$?"; cat $W/logs/recipe_check_$N.json
}
train 1
T1=$(awk -F'\t' 'index($10,"train_P_k01_s0_s1")==1{t+=$7} END{print t+0}' $LG)
TT=$(python3 -c "import json; d=json.load(open('$W/raw/k05_s0_test/infer_stats_P_test.json')); print(max(v['seconds'] for v in d['adapters'].values()))")
SP=$(spent)
PROJ=$(python3 -c "print(round($SP + ($T1 + 54 + 6*41.3 + 2*$TT + 50)/3600, 4))")
echo "hard_stop_1: spent $SP + lora2 train ${T1}s + dev 6 ckpts + test 2 x ${TT}s -> $PROJ GPU-h (limit 3.4)"
SEEDS="1"
if python3 -c "import sys; sys.exit(0 if $PROJ <= 3.4 else 1)"; then train 2; SEEDS="1 2"; else echo "HARD_STOP_1: P_1_lora2 not trained"; fi
AD=""; for s in $SEEDS; do for c in ep1 ep2 ep3; do AD="$AD${AD:+,}s$s$c=/w/adapters/P_${L}_s$s/$c"; done; done
if [ ! -f $W/raw/$L/infer_stats_P_dev_sel2000.json ]; then
  mkdir -p $W/raw/$L
  bash run_card_pseed.sh dev_${L} vllm/vllm-openai:v0.10.2 "cd /w/code && python3 pk_infer.py --split dev_sel2000 --bundles /w/in/q_dev_sel2000.jsonl --adapters $AD --out /w/raw/$L --gpu_mem 0.90 > /w/logs/dev_$L.log 2>&1"
fi
echo "spent after dev: $(spent)"
mkdir -p out/adapters out/raw
for s in $SEEDS; do N=P_${L}_s$s; mkdir -p out/adapters/$N; cp $W/adapters/$N/train_stats.json $W/adapters/$N/train_log.txt out/adapters/$N/; cp $W/logs/recipe_check_$N.json out/; done
cp $W/raw/$L/* out/raw/; cp $W/logs/dev_$L.log out/; cp $LG out/
mkdir -p out/k05_test; cp $W/raw/k05_s0_test/* out/k05_test/ 2>/dev/null
sha256sum $W/adapters/P_${L}_s*/ep*/adapter_model.safetensors; tail -1 $W/adapters/P_${L}_s*/train_log.txt; tail -4 $LG
tar czf out.tgz out
