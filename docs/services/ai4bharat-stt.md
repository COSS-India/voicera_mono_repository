# AI4Bharat STT Service

Optional on-premises **Indic speech-to-text** (NeMo). The core VoicERA stack can run with cloud STT only.

## HTTP API (port 8001)

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/` | GET | — | Status |
| `/health` | GET | — | Health |
| `/transcribe` | POST | `{ "audio_b64": string, "language_id": string }` (default `hi`) | `{ "text": string }` |
| `/transcribe/bhili` | POST | Same | Bhili model (`language_id` / agent `bhb`) |

Audio: base64 **16 kHz int16 PCM**. Batching: `MAX_BATCH_SIZE=16`, `BATCH_TIMEOUT=0.1s`. **Code:** `ai4bharat_stt_server/server.py`

### Model paths (environment)

| Variable | Purpose |
|----------|---------|
| `INDIC_NEMO_PATH` | Main Indic NeMo checkpoint file |
| `BHILI_NEMO_PATH` | Bhili checkpoint |
| `BHILI_ENABLE` | `"yes"` / `"no"` |
| `HF_TOKEN` | Optional HuggingFace token |
| `PORT` | Default `8001` |

### GPU / VRAM

| | |
|--|--|
| **Production** | NVIDIA GPU **strongly recommended** |
| **Development** | CPU fallback is supported but slow |
| **Pinned VRAM (GB)** | **Deferred** — not benchmarked in this documentation pass |

Exact VRAM depends on your **NeMo checkpoint file** (`INDIC_NEMO_PATH`, optional `BHILI_NEMO_PATH`), batch size (`MAX_BATCH_SIZE`), and whether both Indic and Bhili models are loaded. VoicERA does not ship a single reference checkpoint size for all deployments.

**Until measured values are published:** size GPUs with your hosting partner using a staging load test (`nvidia-smi` while serving `/transcribe` at expected concurrency), or use cloud STT providers and omit this server. Engineering may add reference GB figures after benchmarking on agreed hardware; that work is **out of scope for the initial main merge** and tracked in [source brief A5](../source-briefs/A5-ai4bharat-servers.md).

On **`dev`**, agents with language **`bhb`** and provider `indic-conformer-stt` use `/transcribe/bhili`.

## Overview

This optional service provides **high-accuracy speech-to-text for Indic languages** using AI4Bharat NeMo models.

**Supported Languages:**
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Bengali (bn)
- Punjabi (pa)
- Marathi (mr)
- Gujarat (gu)
- And more...

**Advantages:**
- Free and open-source
- Optimized for Indic languages
- Self-hosted (no API calls needed)
- Low latency
- Can run on GPU for better performance

## Quick Start

### Installation

```bash
cd ai4bharat_stt_server

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models
python download_models.py
```

### Running the Service

```bash
# Development
python server.py

# Via Docker
docker build -t ai4bharat-stt .
docker run -p 8001:8001 ai4bharat-stt

# With GPU support
docker run --gpus all -p 8001:8001 ai4bharat-stt
```

## Configuration

### Environment Variables

```env
# Server
HOST=0.0.0.0
PORT=8001
WORKERS=4

# Model
MODEL_NAME=indic-conformer-hi
MODEL_PATH=/models
DEVICE=cuda                    # cuda or cpu
BATCH_SIZE=32

# Audio
SAMPLE_RATE=16000
AUDIO_FORMAT=wav

# Logging
LOG_LEVEL=INFO
```

## API Endpoints

### Transcribe (Streaming)

**Endpoint:** `POST /transcribe`

**Request:**
```json
{
  "audio": "base64-encoded-audio",
  "language": "hi",
  "format": "wav"
}
```

**Response:**
```json
{
  "transcript": "नमस्ते, यह एक परीक्षण संदेश है।",
  "confidence": 0.98,
  "language": "hi"
}
```

### Health Check

```
GET /health
```

## Integration with VoiceERA

### Configuration

In `voice_2_voice_server/.env`:

```env
STT_PROVIDER=ai4bharat
STT_SERVICE_URL=http://ai4bharat_stt_server:8001
STT_LANGUAGE=hi
```

### Usage

```python
class AI4BharatSTT:
    def __init__(self, service_url, language="hi"):
        self.service_url = service_url
        self.language = language
    
    async def transcribe(self, audio_data):
        import base64
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.service_url}/transcribe",
                json={
                    "audio": base64.b64encode(audio_data).decode(),
                    "language": self.language
                }
            )
            result = await response.json()
            return result["transcript"]
```

## Performance

### Benchmarks

| Language | Accuracy | Speed | GPU Required |
|----------|----------|-------|--------------|
| Hindi | 95%+ | Real-time | Yes (recommended) |
| Tamil | 92%+ | Real-time | Yes (recommended) |
| Telugu | 93%+ | Real-time | Yes (recommended) |
| Kannada | 91%+ | Real-time | Yes (recommended) |

### Optimization Tips

- Use GPU for production deployments
- Batch audio chunks for better throughput
- Pre-download models to avoid startup delays
- Use appropriate sample rate (16kHz recommended)

## Troubleshooting

### Service won't start

```bash
# Check Python version
python --version  # Should be 3.10+

# Check dependencies
pip list | grep torch

# Download models
python download_models.py
```

### Low accuracy

- Verify audio quality (16kHz, mono)
- Check language configuration matches audio
- Ensure model is properly downloaded

### Out of memory

- Reduce BATCH_SIZE in config
- Use CPU instead of GPU for development
- Enable model quantization for production

---

## Next Steps

- **[TTS Service](ai4bharat-tts.md)** - Text-to-Speech documentation
- **[Configuration](../getting-started/configuration.md)** - Full configuration guide
- **[Quick Start](../getting-started/quickstart.md)** - Get VoiceERA running
