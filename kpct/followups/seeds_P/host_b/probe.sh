#!/bin/bash
: "${WORK_DIR_B:?set WORK_DIR_B (see configs/paths.env.example)}"
: "${SCRATCH_B:?set SCRATCH_B (see configs/paths.env.example)}"
set +e
W=${SCRATCH_B:-/tmp/pseeds}; mkdir -p $W/in $W/code $W/bench $W/logs
cp -r in/. $W/in/; cp code/*.py $W/code/; chmod -R a+rwX $W
sha256sum $W/in/frozen_P_k05_s0_ep1/adapter_model.safetensors $W/in/*.jsonl $W/in/*.json $W/code/*.py
df -h "$SCRATCH_B" | tail -1
PIP="pip install --break-system-packages -q --no-deps peft==0.21.2 > /w/logs/pip_peft.log 2>&1"
TR="python3 /w/code/pk_train.py --list k05_s0 --silver /w/in/silver_k05_s0_train.jsonl --silver_manifest /w/in/silver_k05_s0_manifest.json --manifest /w/in/multitq_kpct_subsets.json --guided_common /w/in/gk_common.py --seed 99 --out /w/bench"
# C: gradient checkpointing, frozen micro-batch budget 8192 (30 steps)
bash run_card_pseed.sh benchC ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --name benchC --grad_ckpt --train_tok 8192 --max_steps 30"
rcC=$?
if [ $rcC != 0 ]; then
  bash run_card_pseed.sh benchA ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --name benchA --grad_ckpt --train_tok 4096 --max_steps 30"
fi
# B: no checkpointing, small micro-batch (20 steps; speed reference only)
bash run_card_pseed.sh benchB ${GPU_IMAGE:-tkgqa-gpu:latest} "$PIP; cd /w/code && $TR --name benchB --train_tok 768 --max_steps 20"
# vLLM: frozen P_5 seed-0 ep1 adapter re-decoded on dev_sel2000 (throughput + cross-hardware agreement)
bash run_card_pseed.sh vllmdev vllm/vllm-openai:v0.10.2 "cd /w/code && python3 pk_infer.py --split dev_sel2000 --bundles /w/in/q_dev_sel2000.jsonl --adapters ep1=/w/in/frozen_P_k05_s0_ep1 --out /w/bench/dev_frozen_ep1 --gpu_mem 0.90 > /w/logs/vllmdev.log 2>&1"
mkdir -p out
for n in benchC benchA benchB; do [ -d $W/bench/$n ] && cp $W/bench/$n/train_log.txt out/train_log_$n.txt && cp $W/bench/$n/train_stats.json out/train_stats_$n.json 2>/dev/null; rm -rf $W/bench/$n/dry; done
cp -r $W/bench/dev_frozen_ep1 out/ ; cp $W/logs/*.log out/ 2>/dev/null
cp $WORK_DIR_B/gpu_usage_pseeds.tsv out/
tail -3 out/train_log_*.txt; tail -5 out/vllmdev.log; cat out/gpu_usage_pseeds.tsv
