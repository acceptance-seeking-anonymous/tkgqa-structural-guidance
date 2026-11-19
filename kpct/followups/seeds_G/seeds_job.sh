#!/bin/bash
# guided LoRA-seed replication (multitq_kpct_seed_amendment_G.json): G_1 lora1 then lora2 on the GPU host GPU 1 only
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
set -u
K=$WORK_DIR; W=$K/guided/code; S=$K/guided/seeds; C=/kpct/guided/code; M=$C/multitq_kpct_subsets.json; J=$PWD
mkdir -p $S/adapters $S/preds $S/logs $S/checks
cp -L run_card_seeds.sh recipe_check_G.py $S/
PIP="(pip install --break-system-packages -q --no-deps peft==0.21.2 || pip install --break-system-packages -q --no-deps peft) > /dev/null 2>&1"
card1_free() {
  u=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F', ' '$1==1{print $2" "$3}'); uuid=${u% *}; mem=${u#* }
  n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
  [ "$n" = "0" ] && [ "$mem" -lt 1000 ]
}
wait_card() { for i in $(seq 1 36); do card1_free && return 0; echo "$(date -u +%T) card 1 busy, re-check in 10 min"; nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader; sleep 600; done; return 1; }
for SEED in 1 2; do
  NAME=G_k01_s0_ordS_s$SEED; OUT=/kpct/guided/seeds/preds/G_k01_s0_lora$SEED
  if [ ! -s $S/adapters/$NAME/ep3/adapter_model.safetensors ] || [ ! -s $S/adapters/$NAME/train_stats.json ]; then
    wait_card || { echo "card 1 never free"; exit 3; }
    bash $S/run_card_seeds.sh 1 ${GPU_IMAGE:-tkgqa-gpu:latest} train_$NAME $W "$PIP; python3 -u $C/gk_train.py --arm G --list k01_s0 --bundles /kpct/bundles/k01_s0/train.jsonl --questions /data/questions/train.json --manifest $M --order S --seed $SEED --out /kpct/guided/seeds/adapters > /kpct/guided/seeds/logs/train_$NAME.log 2>&1" || { echo "train $NAME failed"; tail -20 $S/logs/train_$NAME.log; exit 4; }
  fi
  python3 $S/recipe_check_G.py $K/guided/adapters/G_k01_s0_ordS_s0/train_stats.json $S/adapters/$NAME/train_stats.json $S/checks/recipe_check_$NAME.json || { echo "STOP_RECIPE $NAME"; exit 5; }
  cp $S/checks/recipe_check_$NAME.json $J/
  wait_card || { echo "card 1 never free"; exit 3; }
  bash $S/run_card_seeds.sh 1 ${VLLM_IMAGE:-tkgqa-vllm:latest} sess_G_k01_s0_lora$SEED $W "python3 -u $C/gk_run_arm.py --arm G --list k01_s0 --runs main:S:/kpct/guided/seeds/adapters/$NAME --bundles_dir /kpct/bundles/k01_s0 --dev_q /data/questions/dev.json --test_q /data/questions/test.json --s_list_dev /kpct/preds/k01_s0/dev_sel2000_S_seed0.jsonl --s_list_test /kpct/preds/k01_s0/test_S_seed0.jsonl --controls= --manifest $M --eval $C/multitq_eval.py --kg /data/kg/full.txt --log /kpct/guided/seeds/multitq_kpct_seed_selection_log_G.jsonl --out $OUT > /kpct/guided/seeds/logs/sess_G_k01_s0_lora$SEED.log 2>&1" || { echo "session lora$SEED failed"; tail -30 $S/logs/sess_G_k01_s0_lora$SEED.log; exit 6; }
  cp $S/preds/G_k01_s0_lora$SEED/freeze_G_k01_s0.json $J/freeze_G_k01_s0_lora$SEED.json; cp $S/preds/G_k01_s0_lora$SEED/session_G_k01_s0.json $J/session_G_k01_s0_lora$SEED.json
  cp $S/preds/G_k01_s0_lora$SEED/selection_G_k01_s0_main.json $J/selection_G_k01_s0_lora$SEED.json 2>/dev/null
  cp $S/adapters/$NAME/train_stats.json $J/train_stats_$NAME.json
  grep guided_seeds $K/gpu_usage_kpct.tsv > $J/usage_log_guided_seeds.tsv
  awk -F'\t' '$1=="guided_seeds"{t+=$7} END{printf "guided_seeds spent %.4f\n", t}' $K/gpu_usage_kpct.tsv
done
cp $S/multitq_kpct_seed_selection_log_G.jsonl $J/ 2>/dev/null
cd $S/preds && sha256sum */pred_G_test_*.jsonl */raw_G_test_*.jsonl.gz */pred_G_dev_sel2000_*.jsonl > $J/pred_sha256.txt
tar czf $J/guided_seeds_preds.tgz --exclude='*/shards' G_k01_s0_lora1 G_k01_s0_lora2
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
