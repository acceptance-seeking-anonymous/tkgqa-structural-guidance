#!/bin/bash
# jobQA.sh -- matched MultiQA baseline training stage (one GPU): PLAN=<plan json> (plan_C.json = 1% and 5%, three seeds,
# both embeddings; plan_DE.json = 10% (three seeds) and 100% (seed 0), both embeddings). Writes runs/<k>_<emb>_seed<s>/
# (run.json, driver.log, dev_sel2000 / answer-stripped test top-10 predictions) and sched_log.jsonl. No test score.
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set +e
: "${PLAN:?set PLAN to plan_C.json or plan_DE.json}"
W=$(pwd); CARD=${MBQ_CARD:-0}; IMG=${GPU_IMAGE:-tkgqa-gpu:latest}
cp -L tcomplex_fullkg.ckpt MultiQA/models/MultiTQ/kg_embeddings/
date -u +%FT%TZ > stage_start.txt
docker run --rm --name mbq_stage_$$ --gpus all -e CUDA_VISIBLE_DEVICES=$CARD -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True --shm-size=16g --cpus 24 \
  -v $W:/workspace -w /workspace -v $MODEL_DIR:/hf -e HF_HOME=/hf -e HF_HUB_DISABLE_XET=1 $IMG \
  bash -c "pip install --break-system-packages -q transformers==4.57.6 > pip.log 2>&1; pip list 2>/dev/null | grep -i -E '^(transformers|torch) ' > pip_versions.txt; python3 mbq_sched.py $PLAN" > stage.out 2>&1
date -u +%FT%TZ > stage_end.txt
exit 0
