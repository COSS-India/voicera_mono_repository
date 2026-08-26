# AI4Bharat Indic Parler. Pins torch 2.10 + flashinfer, which is exactly why it
# cannot share an environment with the NeMo or vLLM images.
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-dev python3-pip ca-certificates libsndfile1 \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /tmp/requirements.txt

WORKDIR /app
COPY . /app

EXPOSE 8002
# Checkpoints live in ./checkpoints, where setup.sh already downloads them, and
# arrive via the bind mount.
CMD ["sh", "-c", "python3 server.py --host 0.0.0.0 --port 8002 --checkpoint ${CHECKPOINT_PATH_DEFAULT:-/app/checkpoints}"]
