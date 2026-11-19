#!/bin/bash
# dl_qwen.sh -- CPU-only download of Qwen/Qwen2.5-72B-Instruct-AWQ into the shared HF cache (no GPU, not in the GPU usage log).
# Plain HTTP (HF_HUB_DISABLE_XET=1), resumable, up to 12 passes. An HF token is optional (export HF_TOKEN).
: "${MODEL_DIR:?set MODEL_DIR (see configs/paths.env.example)}"
docker run --rm --name r2_dl_qwen_$(date +%s) --cpus=4 -v $MODEL_DIR:/hf -e HF_TOKEN -e HF_HOME=/hf \
  -e HF_HUB_DISABLE_XET=1 -e HF_HUB_ENABLE_HF_TRANSFER=0 -e HF_HUB_DOWNLOAD_TIMEOUT=60 ${GPU_IMAGE:-tkgqa-gpu:latest} bash -c '
export HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
python3 - <<PY
import time, os
from huggingface_hub import snapshot_download
for i in range(12):
    try:
        p = snapshot_download("Qwen/Qwen2.5-72B-Instruct-AWQ", max_workers=4)
        print("DONE", p, flush=True); break
    except Exception as e:
        print("pass", i, type(e).__name__, str(e)[:200], flush=True); time.sleep(30)
PY
chmod -R a+rX /hf/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ 2>/dev/null
du -sh /hf/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ; ls /hf/hub/models--Qwen--Qwen2.5-72B-Instruct-AWQ/snapshots/*/'
