# Same CUDA/torch base as the AI4Bharat TTS server so MioCodec's torch ops match
# the driver stack already validated in this cluster.
FROM nvcr.io/nvidia/pytorch:25.06-py3

WORKDIR /app
COPY requirements.txt .
# torch ships in the base image (custom NVIDIA build 2.8.0a0).  No official
# torchaudio wheel is ABI-compatible with it (undefined symbol errors).
#
# torchaudio is only imported by miocodec's SSLFeatureExtractor (the *encoder*
# path).  We only use MioCodec.decode() (tokens → audio), so torchaudio is
# never called at runtime.  Strategy:
#   1. Install miocodec's other runtime deps (not torch/torchaudio).
#   2. Install miocodec itself with --no-deps.
#   3. Patch ssl_extractor.py to make the torchaudio import a soft try/except
#      so the module loads without it.
RUN grep -vE '^\s*(torch|git\+)' requirements.txt > /tmp/req.txt && \
    pip3 install --no-cache-dir -r /tmp/req.txt && \
    pip3 install --no-cache-dir jsonargparse safetensors transformers huggingface_hub && \
    pip3 install --no-cache-dir --no-deps git+https://github.com/Aratako/MioCodec && \
    SITE=$(python3 -c "import sysconfig; print(sysconfig.get_path('purelib'))") && \
    sed -i 's/^import torchaudio$/try:\n    import torchaudio\nexcept (ImportError, OSError):\n    torchaudio = None  # decode-only; encoder not used/' \
        "$SITE/miocodec/module/ssl_extractor.py"

COPY . .
EXPOSE 8003
CMD ["python3", "server.py", "--port", "8003"]
