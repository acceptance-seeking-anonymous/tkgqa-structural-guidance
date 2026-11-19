# Training / CPU-executor image (tag it tkgqa-gpu:latest or set GPU_IMAGE). Reconstructed from the recorded package
# list of the image used in the experiments (CUDA 12.8.1 base, system Python 3.12.3, torch 2.8.0+cu128).
#   docker build -f environment/Dockerfile -t tkgqa-gpu:latest environment/
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip python3-dev git ca-certificates \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/local/bin/python
RUN pip install --break-system-packages torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
COPY requirements-gpu.txt /tmp/requirements-gpu.txt
RUN pip install --break-system-packages -r /tmp/requirements-gpu.txt
CMD ["python3", "-c", "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"]
