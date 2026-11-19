#!/bin/bash
# jobB.sh -- matched MultiQA baseline, optional feasibility smoke test (dev only): plans/plan_smoke.json (six concurrent
# MultiQA processes at 1% for 4 epochs, one 10% process for 1 epoch); timing only, no test score.
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set +e
W=$(pwd); CARD=${MBQ_CARD:-0}; IMG=${GPU_IMAGE:-tkgqa-gpu:latest}
cp -L tcomplex_fullkg.ckpt MultiQA/models/MultiTQ/kg_embeddings/
date -u +%FT%TZ > jobB_start.txt
docker run --rm --name mbq_smoke_$$ --gpus all -e CUDA_VISIBLE_DEVICES=$CARD -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --shm-size=16g --cpus 24 \
  -v $W:/workspace -w /workspace -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 $IMG \
  bash -c "pip install --break-system-packages -q transformers==4.57.6 > pip.log 2>&1; pip list 2>/dev/null | grep -i -E '^(transformers|torch) ' > pip_versions.txt; python3 mbq_sched.py plan_smoke.json" > smoke.out 2>&1
date -u +%FT%TZ > jobB_end.txt
exit 0
