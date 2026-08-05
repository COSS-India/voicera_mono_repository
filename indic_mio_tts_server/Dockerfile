# miocodec is a standard PyPI-torch project: it hard-imports torchaudio at load
# (SSL feature extractor) and uses it in the mel path. The NGC PyTorch image ships
# a custom torch build with no ABI-compatible torchaudio, so we use plain PyPI
# torch + torchaudio (ABI-matched, CUDA wheels) instead — the environment miocodec
# is actually built for. GPU is provided by the nvidia container runtime at runtime.
FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1) Pin torch + torchaudio (same version = ABI-matched) from the CUDA index FIRST,
#    so the miocodec install below sees them already satisfied and does not pull a
#    mismatched CPU wheel. constraints.txt re-pins them for the second install too.
RUN pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

COPY requirements.txt constraints.txt ./
RUN pip install -r requirements.txt -c constraints.txt

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
