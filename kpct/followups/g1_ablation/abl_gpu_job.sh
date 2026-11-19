#!/bin/bash
# abl_gpu_job.sh -- one free card: GU retrieval (GPU image) -> CPU dry run (vllm image, no GPU) -> vLLM ablation session
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
: "${DATA_DIR:?set DATA_DIR (see configs/paths.env.example)}"
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set -u
K=$WORK_DIR; A=$K/ablation_g1; W=$PWD
cp -L abl_build_identity.json $A/abl_build_identity.json; cp -L abl_run.py gk_ablation.py run_card_abl.sh $A/code/
pick_card() {
  for i in 0 1; do
    u=$(nvidia-smi --query-gpu=index,uuid,memory.used --format=csv,noheader,nounits | awk -F', ' -v i=$i '$1==i{print $2" "$3}')
    uuid=${u% *}; mem=${u#* }
    napps=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader | grep -c "$uuid" || true)
    if [ "$napps" = "0" ] && [ "$mem" -lt 1000 ]; then echo $i; return; fi
  done
  echo none
}
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}'
CARD=$(pick_card); echo "card for retrieval: $CARD"
[ "$CARD" = none ] && { echo "NO FREE CARD"; exit 3; }
if [ ! -s $A/gu/gu_dev_abl.jsonl.gz ]; then
  bash $A/code/run_card_abl.sh $CARD ${GPU_IMAGE:-tkgqa-gpu:latest} abl_gu_retrieval $W "python3 -u /kpct/ablation_g1/code/gu_retrieve.py --mode queries --kg /data/kg/full.txt --emb /kpct/gu_retrieval/fact_emb_f16.npy --qfile /kpct/ablation_g1/code/dev_abl_questions.jsonl --out /kpct/ablation_g1/gu --name gu_dev_abl --bs 512" || { echo "retrieval failed"; exit 4; }
fi
ARGS="--adapter /kpct/guided/adapters/G_k01_s0_ordS_s0/ep2 --root /kpct/ablation_g1 --sel_bundles /kpct/bundles/k01_s0/dev_sel2000.jsonl --sel_gu /kpct/gu_retrieval/gu_dev_sel2000.jsonl.gz --sel_s0 /kpct/preds/k01_s0/dev_sel2000_S_seed0.jsonl --sel_frozen_pred /kpct/guided/preds/G_k01_s0/pred_G_dev_sel2000_ordS_ep2.jsonl --sel_quids /kpct/ablation_g1/code/dev_sel2000_quids.json --order /kpct/ablation_g1/code/D_main_order.json --main_bundles /kpct/ablation_g1/bundles/dev_abl_k01_s0.jsonl --main_s0 /kpct/ablation_g1/bundles/dev_abl_k01_s0_S_seed0.jsonl --main_gu /kpct/ablation_g1/gu/gu_dev_abl.jsonl.gz --build_identity /kpct/ablation_g1/abl_build_identity.json --dev_q /data/questions/dev.json --kg /data/kg/full.txt --eval /kpct/ablation_g1/code/multitq_eval.py --usage_log /kpct/gpu_usage_kpct.tsv"
mkdir -p $W/dry
docker run --rm --name abl_dry_$(date +%s) -v $K:/kpct -v $W/dry:/w -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_OFFLINE=1 \
  -v $DATA_DIR:/data:ro -e OMP_NUM_THREADS=8 -w /w ${VLLM_IMAGE:-tkgqa-vllm:latest} \
  bash -c "python3 -u /kpct/ablation_g1/code/abl_run.py $ARGS --out /w --dry; rc=\$?; chmod -R a+rwX /w /kpct/ablation_g1; exit \$rc" || { echo "DRY RUN FAILED"; exit 5; }
cp dry/abl_run_progress.json abl_dry_progress.json
CARD=$(pick_card); echo "card for vLLM: $CARD"
[ "$CARD" = none ] && { echo "NO FREE CARD after dry run"; exit 3; }
bash $A/code/run_card_abl.sh $CARD ${VLLM_IMAGE:-tkgqa-vllm:latest} abl_vllm $W "python3 -u /kpct/ablation_g1/code/abl_run.py $ARGS --out /w"
rc=$?
cp $A/gu/stats_queries_gu_dev_abl.json . 2>/dev/null; grep ablation_g1 $K/gpu_usage_kpct.tsv > usage_log_ablation_g1.tsv
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader
exit $rc
