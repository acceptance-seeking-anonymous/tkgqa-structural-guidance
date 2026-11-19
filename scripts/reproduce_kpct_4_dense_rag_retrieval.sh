#!/bin/bash
# reproduce_kpct_4_dense_rag_retrieval.sh -- Dense-RAG (GU) retrieval: BAAI/bge-large-en-v1.5 top-20 facts over all
# 461,329 verbalised MultiTQ facts, for dev_sel2000, test and the union of the train lists. GPU host, card 0.
# Needs $WORK_DIR/gu_retrieval/{dev_sel2000_quids.json,gu_train_union_quids.json} (quid lists from the subsets manifest).
set -euo pipefail
source "$(dirname "$0")/env_check.sh" WORK_DIR DATA_DIR MODEL_DIR
K=$WORK_DIR; cd "$K/guided/code"; mkdir -p "$K/gu_retrieval"
if [ ! -f "$K/gu_retrieval/fact_emb_f16.npy" ]; then
  bash run_card.sh 0 "${GPU_IMAGE:-tkgqa-gpu:latest}" gu_index "$PWD" "python3 -u gu_retrieve.py --mode facts --kg /data/kg/full.txt --out /kpct/gu_retrieval"
fi
cp "$K/gu_retrieval/dev_sel2000_quids.json" "$K/gu_retrieval/gu_train_union_quids.json" .
H=$(sha256sum "$K/gu_retrieval/fact_emb_f16.npy" | cut -d" " -f1)
bash a2_driver.sh "$H"
