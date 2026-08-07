FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev build-essential ffmpeg libsndfile1 sox \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir --break-system-packages torch torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    pip3 install --no-cache-dir --break-system-packages -r requirements.txt

# AI4Bharat NeMo fork - server.py imports nemo.collections.asr
COPY --from=nemo . /opt/nemo
RUN pip3 install --no-cache-dir --break-system-packages "/opt/nemo[asr]"

COPY . .
EXPOSE 8001

# /health is 503 until the checkpoint is loaded and the batch worker is beating.
HEALTHCHECK --interval=20s --timeout=10s --retries=3 --start-period=300s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8001/health', timeout=5)"

CMD ["python3", "server.py"]
