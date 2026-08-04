# Same CUDA/torch base as the AI4Bharat TTS server so MioCodec's torch ops match
# the driver stack already validated in this cluster.
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app
COPY requirements.txt .
# torch ships in the base image.  miocodec pulls torchaudio transitively —
# the PyPI torchaudio wheel links against a CUDA runtime that doesn't match
# this NGC image's CUDA (OSError: libcudart.so.13 not found).
# Fix: filter torch* and the miocodec git line, install torchaudio from the
# PyTorch wheel index that matches the container's CUDA, then install
# miocodec --no-deps so it doesn't drag in the wrong torchaudio again.
RUN grep -vE '^\s*(torch|git\+)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt && \
    CUDA_VER=$(python3 -c "import torch; print(torch.version.cuda.replace('.',''))") && \
    pip3 install --no-cache-dir torchaudio --index-url https://download.pytorch.org/whl/cu${CUDA_VER} && \
    pip3 install --no-cache-dir --no-deps git+https://github.com/Aratako/MioCodec

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
