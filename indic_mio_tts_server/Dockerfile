# miocodec is a standard PyPI-torch project (hard-imports torchaudio). We base on
# the official PyTorch runtime image so torch==2.5.1 (+CUDA) is ALREADY present and
# is not re-downloaded — we only add the ABI-matched torchaudio and miocodec. This
# keeps the image small (the NGC base's custom torch has no compatible torchaudio).
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git libsndfile1 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt constraints.txt ./
# torch is already in the base image; add only torchaudio (same version = ABI-matched)
# from the CUDA wheel index, then miocodec + our deps. constraints.txt pins torch/
# torchaudio so the miocodec install cannot swap them for a mismatched wheel.
RUN pip install torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 \
    && pip install -r requirements.txt -c constraints.txt

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
