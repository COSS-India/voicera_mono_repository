# indic-transcribe-core gateway — Canary 1.2B (25 Indic languages), AlignAtt streaming.
#
# Base is the CUDA 13 runtime matching this host's driver (CUDA 13.2, sm_120 Blackwell).
# Only the *base* image is needed: every CUDA library torch loads arrives as a pip wheel
# (nvidia-cublas-13.4, nvidia-cudnn-cu13-9.20, ...), so a -devel base would add ~5 GB for
# nothing. No nvcc is required — nothing is compiled here.
FROM nvidia/cuda:13.0.1-base-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_OFFLINE=1

# python + curl only. Notably NOT installed:
#   * libsndfile -- the soundfile wheel bundles libsndfile_x86_64.so
#   * ffmpeg     -- deliberate: 8 vCPUs here, so we never transcode per request. Uploads are
#                   decoded in-process and the WebSocket path takes raw PCM16.
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-venv python3-pip curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN python3.12 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# Heavy dependencies first, in their own layer, so the ~6.5 GB of torch/NeMo/CUDA wheels are
# cached across code edits AND across changes to the bench clients below.
# constraints.txt is the FULL 171-package transitive freeze of the known-good environment,
# passed with -c so pip resolves in one pass. Without it the resolver backtracks for a long time
# over lightning/nemo version combinations and can land on a different set than the one this
# stack was validated against.
COPY requirements.txt constraints.txt /tmp/
RUN pip install --upgrade pip wheel setuptools \
    && pip install -r /tmp/requirements.txt -c /tmp/constraints.txt \
        --extra-index-url https://download.pytorch.org/whl/cu132

# torchaudio, the one genuinely new dependency versus the Bhili image.
#
# models/core's feature extractor imports torchaudio at module top level
# (feature_extraction_indic_canary.py:26) for Resample(sinc_interp_hann), so unlike Bhili we
# cannot assert it absent -- the checkpoint's own code will not import without it.
#
# There is NO cu132 build of torchaudio. Verified against the index: torch publishes
# 2.12.0+cu132, while torchaudio's newest release is 2.11.0 and ships only
# cpu/cu126/cu128/cu129/cu130. Installing a CUDA-flavoured torchaudio next to our CUDA-13.2
# torch raises "PyTorch and TorchAudio were compiled with different CUDA versions", which
# breaks NeMo itself (NeMo imports torchaudio opportunistically), not merely the front-end.
#
# The +cpu build sidesteps that entirely: with no CUDA extension there is no version check to
# fail. Nothing is lost -- Resample is pure ATen and runs on CUDA tensors regardless, and all
# our audio is already 16 kHz so the resampler never actually runs. --no-deps so it cannot
# touch the pinned torch.
RUN pip install --no-deps torchaudio==2.11.0+cpu \
        --extra-index-url https://download.pytorch.org/whl/cpu

# silero-vad WITHOUT its dependency graph, for the same torchaudio reason: we want only the
# bundled .onnx, which onnxruntime drives on the CPU (see vad.py).
RUN pip install --no-deps silero-vad==6.2.1

# THE BUILD GATE. Everything above is worthless if these four cannot coexist, and the failure
# mode is a runtime import error inside a 5 GB model load, so it is asserted here instead.
RUN python -c "\
import torch, torchaudio; \
assert torch.version.cuda == '13.2', f'expected CUDA 13.2, got {torch.version.cuda}'; \
assert torch.__version__.startswith('2.12.0+cu132'), torch.__version__; \
import torchaudio.transforms as T; \
_ = T.Resample(24000, 16000, resampling_method='sinc_interp_hann')(torch.randn(1, 24000)); \
import nemo.collections.asr; \
from nemo.collections.asr.models import EncDecMultiTaskModel; \
print('BUILD GATE OK:', torch.__version__, '|', torchaudio.__version__)"

# Light bench clients in a separate layer -- see requirements-test.txt for why.
COPY requirements-test.txt /tmp/
RUN pip install -r /tmp/requirements-test.txt

WORKDIR /app
COPY nemo_patch/ /app/nemo_patch/
COPY vad.py core_engine.py batcher.py app.py /app/
COPY tools/ /app/tools/
COPY static/ /app/static/
COPY bench/ /app/bench/

# Install the vendored tokenizer into the site-packages copy of NeMo at BUILD time, so a cold
# start does not depend on a writable site-packages. Without this module the checkpoint cannot
# be restored at all -- see nemo_patch/README.md.
RUN python -c "import sys; sys.path.insert(0, '/app'); \
               from nemo_patch import ensure_installed; \
               print('tokenizer patch installed:', ensure_installed())" \
    && python -c "import sys; sys.path.insert(0,'/app'); \
               from nemo.collections.common.tokenizers.canary_multilingual_tokenizer \
               import CanaryMultilingualTokenizer as T; print('patch importable:', T.__name__)"

ENV CORE_CKPT=/artifacts/indic_transcribe_core.nemo \
    CORE_HF_DIR=/models/core \
    CORE_LANGUAGE=hi \
    CORE_PORT=8001

EXPOSE 8001

# One worker, always. The engine owns the GPU from a single thread; a second uvicorn worker
# would load a second copy of the model and contend for the same device.
CMD ["sh", "-c", "exec uvicorn app:app --host 0.0.0.0 --port ${PORT:-${CORE_PORT}} --workers 1"]
