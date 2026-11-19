#!/bin/bash
# jobA.sh -- matched MultiQA baseline, stage A on one GPU (run inside the job directory): NER preprocessing with the
# official ner_task.py logic (mbq_ner.py; flair 0.15.1 + transformers 4.57.6 are pip-installed inside the training
# container at run time) and the
# full-KG TComplEx retrain (mbq_tcx_full.py) side by side on the same card ($MBQ_CARD, default 0).
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
set +e
W=$(pwd); CARD=${MBQ_CARD:-0}; IMG=${GPU_IMAGE:-tkgqa-gpu:latest}
date -u +%FT%TZ > jobA_start.txt
docker run --rm --name mbq_tcx_$$ --gpus all -e CUDA_VISIBLE_DEVICES=$CARD --shm-size=8g --cpus 6 \
  -v $W:/workspace -w /workspace $IMG python3 mbq_tcx_full.py > tcx_full.out 2>&1 &
TCX=$!
docker run --rm --name mbq_ner_$$ --gpus all -e CUDA_VISIBLE_DEVICES=$CARD --shm-size=8g --cpus 16 \
  -v $W:/workspace -w /workspace -v $MODEL_DIR:/hf -e HF_HOME=/hf -e FLAIR_CACHE_ROOT=/hf/flair \
  -e HF_HUB_DISABLE_XET=1 -e NER_WORKERS=${NER_WORKERS:-12} ${HF_TOKEN:+-e HF_TOKEN} $IMG \
  bash -c "pip install --break-system-packages -q flair==0.15.1 transformers==4.57.6 > pip.log 2>&1; pip list 2>/dev/null | grep -i -E '^(flair|transformers|torch|huggingface.hub) ' > pip_versions.txt; python3 mbq_ner.py dev_sel2000,test,train_k10,train_rest" > ner.out 2>&1
echo "ner_rc=$?" > jobA_rc.txt
wait $TCX; echo "tcx_rc=$?" >> jobA_rc.txt
date -u +%FT%TZ > jobA_end.txt
exit 0
