# AI4Bharat Orpheus Indic TTS — single-container serving image.
#
# The base is a plain CPU Python image: the entire CUDA userspace (toolkit, cuDNN,
# NCCL) arrives as pip wheels pulled in by vllm/torch. The host supplies only the
# NVIDIA driver, exposed at run time by nvidia-container-toolkit. That keeps this
# image portable across GPU generations — nothing here is compiled for one arch.
#
# gcc/g++ are needed at RUN TIME, not just build time: vLLM's Triton kernels
# JIT-compile on first use against the pip CUDA toolkit in site-packages.
# libgomp1 and libsndfile1 are runtime dependencies of torch and soundfile
# (libsndfile is what encodes mp3/flac/opus responses).
FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/hf-cache \
    ORPHEUS_CONFIG=/app/config.yaml \
    # flashinfer cannot JIT against CUDA 13 headers, so use vLLM's own sampler.
    # Harmless on older stacks; required on CUDA 13.
    VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_LOGGING_LEVEL=WARNING

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 libsndfile1 curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so app edits do not invalidate the (large) wheel layer.
# requirements.txt holds the top-level packages; constraints.txt pins the full
# transitive set so builds are reproducible.
COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip && \
    pip install -r requirements.txt -c constraints.txt

# vLLM's Triton kernels JIT against nvcc from the pip CUDA toolkit, so CUDA_HOME
# has to point at it. This is the one genuinely environment-specific setting in
# the project; pinning it here at build time keeps it out of the application code.
# The build fails loudly if the expected layout is not there, rather than leaving
# a broken image to surface as a cryptic Triton error at first request.
ENV CUDA_HOME=/usr/local/lib/python3.12/site-packages/nvidia/cu13
ENV PATH=$CUDA_HOME/bin:$PATH
RUN test -x "$CUDA_HOME/bin/nvcc" || { \
        echo "ERROR: nvcc not found under CUDA_HOME=$CUDA_HOME"; \
        echo "The pinned torch/vllm wheels moved their CUDA toolkit. Available:"; \
        ls -d /usr/local/lib/python3.12/site-packages/nvidia/*/bin 2>/dev/null || true; \
        exit 1; \
    }

COPY src ./src
COPY config.yaml voices.json ./

# Non-root runtime user. It needs a writable HOME for Triton's JIT cache.
RUN useradd -m -u 1000 orpheus && \
    mkdir -p /home/orpheus/.triton /home/orpheus/.cache && \
    chown -R orpheus:orpheus /home/orpheus /app
USER orpheus

ENV PORT=8002
EXPOSE 8002
CMD ["sh", "-c", "exec python -m uvicorn orpheus_server.app:app --host 0.0.0.0 --port ${PORT:-8002}"]
