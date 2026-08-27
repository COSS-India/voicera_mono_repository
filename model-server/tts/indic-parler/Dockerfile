# AI4Bharat Indic Parler TTS.
#
# Same base production runs on, so the layers are already cached on the GPU box.
# The NVIDIA image ships torch plus a lot of extras we do not need; removing
# them keeps the image down and avoids version fights with our pinned torch.
FROM nvcr.io/nvidia/pytorch:25.06-py3

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY requirements.txt .
# PIP_CONSTRAINT is cleared because the NVIDIA base pins torch, which would
# otherwise block the exact version the Parler engine needs.
RUN grep -vE '^(torch|flashinfer)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt && \
    PIP_CONSTRAINT= pip3 install --no-cache-dir torch==2.10.0 \
      --index-url https://download.pytorch.org/whl/cu129 && \
    pip3 uninstall -y torchvision torchaudio flash_attn apex torch_tensorrt torchao && \
    PIP_CONSTRAINT= pip3 install --no-cache-dir \
      flashinfer-python==0.6.7 flashinfer-cubin==0.6.7

COPY . .

EXPOSE 8002
CMD ["sh", "-c", "python3 server.py --host 0.0.0.0 --port 8002 --checkpoint ${CHECKPOINT_PATH_DEFAULT:-/app/checkpoints}"]
