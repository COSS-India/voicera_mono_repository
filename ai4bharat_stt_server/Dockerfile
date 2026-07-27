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
CMD ["python3", "server.py"]
