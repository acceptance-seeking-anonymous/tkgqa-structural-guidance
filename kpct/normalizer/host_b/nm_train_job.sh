#!/bin/bash
# nm_train_job.sh <spent_total_gpu_h> -- normalizer track, training on the second GPU host RTX 4090 (rule section training).
# 0. waits until no other compute process holds the 4090 (polling 60 s, up to 60 min; never stops anything)
# 1. peft==0.21.2 installed ONCE into $W/pylib by a CPU-only container (no GPU time)
# 2. dry run (40 optimizer steps, adapter discarded) -> s/step; registered epoch rule: E = 2 if the projected two-epoch
#    training time is <= 1.5 h and spent + projection <= 4.5 - 1.0 (reserve for inference and arms), else E = 1
# 3. training (one container); every GPU container appends a usage_log row (container wall-clock x 1 card) to
#    $W/gpu_usage_normalizer_hostb.tsv (track normalizer; failed runs count)
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set +e
SPENT=${1:-0}
J=$PWD; W=$WORK_DIR_B/normalizer; L=$W/gpu_usage_normalizer_hostb.tsv; CAP=4.5
mkdir -p $W/code $W/inputs $W/adapters $W/logs $W/pylib
for f in nm_common.py nm_train.py nm_infer.py nm_train_job.sh nm_infer_job.sh; do
  if [ -f $W/code/$f ] && ! cmp -s $J/$f $W/code/$f; then echo "CODE_CHANGED $f -- refusing to overwrite staged code"; exit 2; fi
  cp -n $J/$f $W/code/$f
done
if [ -f $W/inputs/train_pairs_k05.jsonl ] && ! cmp -s $J/train_pairs_k05.jsonl $W/inputs/train_pairs_k05.jsonl; then echo "INPUT_CHANGED"; exit 2; fi
cp -n $J/train_pairs_k05.jsonl $W/inputs/
(cd $W && sha256sum code/* inputs/*) > $J/code_inputs_sha_train.txt
[ -s $L ] || printf "track\tcards\tn_cards\tstart_utc\tend_utc\tseconds\tgpu_h\trc\ttag\tcontainer\n" > $L
wait_free() {
  for i in $(seq 1 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [ "$n" = 0 ] && return 0
    echo "$(date -u +%FT%TZ) 4090 busy ($n compute procs), waiting" >&2; sleep 60
  done
  return 1
}
run_gpu() {   # run_gpu <tag> <max_s> <command>
  local TAG=$1 MAXS=$2; shift 2
  wait_free || { echo "4090 still busy after 60 min; not launching $TAG"; return 75; }
  local CN=nmb_${TAG}_$(date +%s); local t0=$(date +%s)
  ( sleep $MAXS; docker stop $CN > /dev/null 2>&1 && echo "WATCHDOG stopped $CN" >&2 ) & local WD=$!
  docker run --rm --name $CN --gpus all --ipc=host -v $W:$W -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
    -e PYTHONPATH=$W/pylib -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TOKENIZERS_PARALLELISM=false \
    -w $W ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "$*; rc=\$?; chmod -R a+rwX $W/adapters 2>/dev/null; exit \$rc"
  local rc=$?; local t1=$(date +%s); kill $WD 2>/dev/null
  printf "normalizer\thost_b\t1\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$(date -u -d @$t0 +%FT%TZ)" "$(date -u -d @$t1 +%FT%TZ)" "$((t1-t0))" \
    "$(awk -v s=$((t1-t0)) 'BEGIN{printf "%.4f", s/3600}')" "$rc" "$TAG" "$CN" >> $L
  return $rc
}
spent_b() { awk -F'\t' 'NR>1 && $1=="normalizer"{s+=$7} END{printf "%.4f", s+0}' $L; }
# 1. peft into pylib (CPU container)
[ -d $W/pylib/peft ] || docker run --rm -v $W:$W ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c "pip install --break-system-packages -q --no-deps --target $W/pylib peft==0.21.2; chmod -R a+rX $W/pylib" > $W/logs/pip.log 2>&1
[ -d $W/pylib/peft ] || { echo "peft install failed"; cat $W/logs/pip.log; exit 5; }
# 2. dry run
if [ ! -s $W/adapters/N_k05_s0_s0_dry/dry_stats.json ]; then
  run_gpu dry 1200 "python3 -u $W/code/nm_train.py --pairs $W/inputs/train_pairs_k05.jsonl --out $W/adapters --dry --max_steps 40" > $W/logs/train_dry.log 2>&1
  echo "dry rc=$?"
fi
cp $W/logs/train_dry.log $J/ 2>/dev/null; cp $W/adapters/N_k05_s0_s0_dry/dry_stats.json $J/ 2>/dev/null
[ -s $W/adapters/N_k05_s0_s0_dry/dry_stats.json ] || { echo "dry run failed"; tail -30 $W/logs/train_dry.log; exit 6; }
E=$(python3 - $W/adapters/N_k05_s0_s0_dry/dry_stats.json $SPENT $(spent_b) <<'EOF'
import json, sys
d = json.load(open(sys.argv[1])); spent = float(sys.argv[2]) + float(sys.argv[3])
proj2 = (d["s_per_step"] * d["steps_per_epoch"] * 2 + 120) / 3600
print(2 if (proj2 <= 1.5 and spent + proj2 <= 4.5 - 1.0) else 1)
EOF
)
echo "epoch rule -> E=$E (spent before $SPENT + host-b usage log $(spent_b))" | tee $J/epoch_rule.txt
# 3. training
if [ ! -s $W/adapters/N_k05_s0_s0/train_stats.json ]; then
  run_gpu train 9000 "python3 -u $W/code/nm_train.py --pairs $W/inputs/train_pairs_k05.jsonl --out $W/adapters --epochs $E" > $W/logs/train.log 2>&1
  echo "train rc=$?"
fi
cp $W/logs/train.log $J/ 2>/dev/null
cp $W/adapters/N_k05_s0_s0/train_stats.json $W/adapters/N_k05_s0_s0/probe_prompt_ids.json $J/ 2>/dev/null
for d in $W/adapters/N_k05_s0_s0/ep*; do [ -d $d ] && cp $d/ckpt_info.json $J/ckpt_info_$(basename $d).json; done
(cd $W/adapters/N_k05_s0_s0 && sha256sum ep*/adapter_model.safetensors ep*/adapter_config.json) > $J/adapter_sha256.txt 2>/dev/null
cp $L $J/usage_log_normalizer_hostb.tsv
echo "spent_b $(spent_b)"
