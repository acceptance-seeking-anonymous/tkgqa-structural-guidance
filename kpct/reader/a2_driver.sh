#!/bin/bash
# A2: registered GU retriever -- top-20 bge-large facts for dev_sel2000, test and every train quid in the union of all
# k-subset lists, cap lists and reader_train60k (98,510). Reuses the 461,329-fact index embedded in the stage-0 run
# (same code, same model; sha256 checked). Card 0. Retrieval is independent of k.
: "${WORK_DIR:?set WORK_DIR (see configs/paths.env.example)}"
W=$PWD; K=$WORK_DIR
echo "== pre-dispatch check $(date -u +%FT%TZ)"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader; docker ps --format '{{.Names}} {{.Image}} {{.RunningFor}}'
EMB=/kpct/gu_retrieval/fact_emb_f16.npy
R="python3 -u gu_retrieve.py --mode queries --kg /data/kg/full.txt --emb $EMB --out /kpct/gu_retrieval"
bash run_card.sh 0 ${GPU_IMAGE:-tkgqa-gpu:latest} gu_retrieval $W "python3 -c \"import sys; sys.path.insert(0,'.'); import gk_common as GC; h=GC.sha256_file('$EMB'); print('emb', h); assert h=='$1'\" && \
  $R --qfile /data/questions/dev.json --quids dev_sel2000_quids.json --name gu_dev_sel2000 > a2_dev.log 2>&1 && \
  $R --qfile /data/questions/test.json --name gu_test > a2_test.log 2>&1 && \
  $R --qfile /data/questions/train.json --quids gu_train_union_quids.json --name gu_train_union > a2_train.log 2>&1 && \
  cp /kpct/gu_retrieval/stats_*.json . && cp dev_sel2000_quids.json gu_train_union_quids.json /kpct/gu_retrieval/"
cp $K/gpu_usage_kpct.tsv gpu_usage_kpct_snapshot.tsv
ls -la $K/gu_retrieval
