# AI4Bharat TTS Service

Documentation for the AI4Bharat Text-to-Speech (TTS) service integration.

## Overview

This optional service provides **natural speech synthesis for Indic languages** using AI4Bharat's IndicParler model.

**Supported Languages:**
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Kannada (kn)
- Malayalam (ml)
- Bengali (bn)
- Punjabi (pa)
- Marathi (mr)
- Gujarati (gu)
- And more...

**Advantages:**
- Free and open-source
- Optimized for Indic languages
- Self-hosted (no API calls needed)
- Natural-sounding voices
- Multiple speaker options
- Can run on GPU for better performance

## Quick Start

### Installation

```bash
cd ai4bharat_tts_server

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
docker build -t ai4bharat-tts .
docker run -p 8002:8002 ai4bharat-tts

# With GPU support
docker run --gpus all -p 8002:8002 ai4bharat-tts
```

## Configuration

### Environment Variables

```env
# Server
HOST=0.0.0.0
PORT=8002
WORKERS=2

# Model
MODEL_NAME=indic-parler-hi
MODEL_PATH=/models
DEVICE=cuda                    # cuda or cpu
ENABLE_CACHING=true

# Audio
SAMPLE_RATE=16000
AUDIO_FORMAT=wav
SPEED=1.0                      # 0.5 - 2.0

# Logging
LOG_LEVEL=INFO
```

## API Endpoints

### Synthesize

**Endpoint:** `POST /synthesize`

**Request:**
```json
{
  "text": "नमस्ते, आपका स्वागत है।",
  "language": "hi",
  "speaker": "female",
  "speed": 1.0
}
```

**Response:**
```
Content-Type: audio/wav
Body: Binary audio data (16-bit PCM, 16kHz)
```

### Health Check

```
GET /health
```

## Speakers & Voices

### Available Speakers

Each language supports multiple speakers:

```
Hindi (hi):
  - female (default)
  - male
  - child

Tamil (ta):
  - female (default)
  - male

Telugu (te):
  - female (default)
  - male

And so on for other languages...
```

## Integration with VoiceERA

### Configuration

In `voice_2_voice_server/.env`:

```env
TTS_PROVIDER=ai4bharat
TTS_SERVICE_URL=http://ai4bharat_tts_server:8002
TTS_LANGUAGE=hi
TTS_SPEAKER=female
TTS_SPEED=1.0
```

### Usage

```python
class AI4BharatTTS:
    def __init__(self, service_url, language="hi", speaker="female"):
        self.service_url = service_url
        self.language = language
        self.speaker = speaker
    
    async def synthesize(self, text, speed=1.0):
        import aiohttp
        
        async with aiohttp.ClientSession() as session:
            response = await session.post(
                f"{self.service_url}/synthesize",
                json={
                    "text": text,
                    "language": self.language,
                    "speaker": self.speaker,
                    "speed": speed
                }
            )
            audio_data = await response.read()
            return audio_data
```

## Voice Customization

### Adjust Speed

```python
# Slow down speech
await tts.synthesize("Hello", speed=0.8)

# Speed up speech
await tts.synthesize("Hello", speed=1.2)

# Range: 0.5 (very slow) to 2.0 (very fast)
```

### Choose Speaker

```python
# Female voice
response = await session.post(
    url,
    json={
        "text": "नमस्ते",
        "speaker": "female"
    }
)

# Male voice
response = await session.post(
    url,
    json={
        "text": "नमस्ते",
        "speaker": "male"
    }
)
```

## Performance

### Benchmarks

| Language | Quality | Speed | GPU Required |
|----------|---------|-------|--------------|
| Hindi | High | Real-time | Yes (recommended) |
| Tamil | High | Real-time | Yes (recommended) |
| Telugu | High | Real-time | Yes (recommended) |
| Kannada | High | Real-time | Yes (recommended) |

### Optimization Tips

- Enable audio caching for repeated text
- Use GPU for production deployments
- Pre-warm models on service startup
- Batch synthesis requests when possible
- Use appropriate sample rate (16kHz recommended)

## Caching Strategy

### Enable Caching

```env
ENABLE_CACHING=true
CACHE_SIZE_MB=500
CACHE_TTL_HOURS=24
```

### How Caching Works

```
Request: "नमस्ते" in Hindi
  │
  ├─► Check cache
  │      ├─► Cache HIT: Return cached audio
  │      └─► Cache MISS: Generate and cache
  │
  └─► Return audio to client
```

### Cache Key

```python
cache_key = f"{text}:{language}:{speaker}:{speed}"
# Example: "नमस्ते:hi:female:1.0"
```

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

### No audio output

- Verify text is in the correct language
- Check speaker name is valid for the language
- Ensure model is properly downloaded
- Check service logs for errors

### Audio quality issues

- Verify language matches text language
- Try different speaker
- Adjust speed parameter
- Check input text for special characters

### Out of memory

- Reduce cache size
- Disable caching if not needed
- Use CPU instead of GPU
- Enable model quantization

### Slow synthesis

- Enable GPU if available
- Check system resources (CPU, memory)
- Enable caching for repeated text
- Reduce batch size

## Advanced Configuration

### Docker Compose Integration

```yaml
services:
  ai4bharat_tts_server:
    build: ./ai4bharat_tts_server
    container_name: voicera_tts
    restart: unless-stopped
    ports:
      - "8002:8002"
    environment:
      HOST: 0.0.0.0
      PORT: 8002
      MODEL_PATH: /models
      DEVICE: cuda
    volumes:
      - ./models:/models
    devices:
      - /dev/nvidia.com/gpu=all  # GPU support
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 5s
      timeout: 5s
      retries: 5
```

---

## Next Steps

- **[STT Service](ai4bharat-stt.md)** - Speech-to-Text documentation
- **[Configuration](../getting-started/configuration.md)** - Full configuration guide
- **[Quick Start](../getting-started/quickstart.md)** - Get VoiceERA running
