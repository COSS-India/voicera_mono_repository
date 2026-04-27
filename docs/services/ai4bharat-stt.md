# AI4Bharat STT Service

Documentation for the AI4Bharat Speech-to-Text (STT) service integration.

## Overview

This optional service provides **high-accuracy speech-to-text for Indic languages** using AI4Bharat's IndicConformer model.

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
