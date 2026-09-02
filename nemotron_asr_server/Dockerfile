FROM python:3.12-slim

WORKDIR /app

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system dependencies for audio processing and C++ extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libsndfile1 \
    ffmpeg \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install PyTorch with CUDA 13.0 for native Blackwell SM 120 support
RUN pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu130

# Install NeMo ASR and ASGI production server dependencies
RUN pip install \
    "nemo_toolkit[asr]" \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    scipy \
    uvloop \
    websockets \
    pydantic \
    python-dotenv \
    numpy \
    soundfile \
    openai \
    requests

COPY . /app

EXPOSE 8000

CMD ["python3", "server.py"]
