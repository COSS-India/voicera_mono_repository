# miocodec requires Python >=3.12 and is a standard PyPI-torch project (hard-imports
# torchaudio). The official PyTorch images ship Python 3.11, so we use python:3.12-slim
# and install ABI-matched torch + torchaudio from the CUDA wheel index ourselves.
# GPU is provided by the nvidia container runtime.
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) torch + torchaudio (same version = ABI-matched) from the CUDA index FIRST, so
#    the miocodec install below sees them satisfied and won't pull a mismatched wheel.
#    cu128 wheels (torch >=2.7) ship sm_120 kernels required by Blackwell GPUs
#    (RTX PRO 6000). cu121 / torch 2.5.1 only go up to sm_90 and fail with
#    "no kernel image is available for execution on the device".
RUN pip install torch==2.7.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt constraints.txt ./
# constraints.txt re-pins torch/torchaudio so miocodec cannot replace them.
RUN pip install -r requirements.txt -c constraints.txt

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
