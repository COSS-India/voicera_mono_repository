# AI4Bharat Indic Conformer (NeMo).
#
# Mirrors the image production already runs: a CUDA *runtime* base (not devel),
# torch from the cu130 index, and the NeMo fork supplied as a build context
# rather than cloned. Matching it means the layers are already cached on the
# GPU box, and more importantly that we build on an environment these models
# are known to work in.
FROM nvidia/cuda:13.0.1-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 PORT=8001

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev build-essential ffmpeg libsndfile1 sox \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# torch comes from the CUDA 13 index; strip it from requirements so pip does not
# pull a second, CPU-flavoured copy over the top.
RUN grep -vE '^(torch|torchaudio)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir --break-system-packages \
      torch torchaudio --index-url https://download.pytorch.org/whl/cu130 && \
    pip3 install --no-cache-dir --break-system-packages -r /tmp/req.txt

# The AI4Bharat NeMo fork, passed in as a named build context (see compose).
COPY --from=nemo . /opt/nemo
RUN pip3 install --no-cache-dir --break-system-packages "/opt/nemo[asr]"
# NeMo breaks against Lightning 2.5+; prod pins this too.
RUN pip3 install --no-cache-dir --break-system-packages "pytorch-lightning<2.5"

COPY . .

EXPOSE 8001
CMD ["python3", "server.py"]
