---
description: Optional on-premises Indic text-to-speech server (Parler TTS over WebSocket).
---

# AI4Bharat TTS

Optional on-premises Indic text-to-speech using Parler TTS over **WebSocket**. The core VoicEra stack can run with cloud TTS only; this server is required only when an agent uses the `indic-parler-tts` provider.

Default port: **8002**. Code: `ai4bharat_tts_server/server.py`, `ai4bharat_tts_server/inference/runner.py`.

## Responsibilities

- Accept WebSocket utterance requests with prompt + voice description + language
- Stream synthesised audio back as binary float32 mono PCM chunks
- Continuous batching in an inference worker for throughput

## When you need it

| Scenario | Need this server? |
|----------|-------------------|
| Cloud TTS only | No |
| Local Indic TTS or Bhili | Yes (+ GPU recommended) |

## Architecture

PyTorch Parler TTS model behind a WebSocket frontend. An inference worker pulls from a per-connection queue and batches concurrent utterances.

- Sample rate: **44100 Hz** (`AUDIO_SAMPLE_RATE`)
- Output: float32 mono PCM, streamed as binary frames
- Framework: PyTorch + Parler TTS

## Configuration

The server reads model paths and runtime knobs from environment variables. Refer to `ai4bharat_tts_server/.env.example` for the active set in the repo.

## Endpoints / API surface

WebSocket per-connection protocol (one utterance per JSON message):

**Client -> server:**

```json
{
  "prompt": "text to speak",
  "description": "voice/style description",
  "language": "hi"
}
```

Use `"bhb"` or `"bhili"` for Bhili.

**Server -> client:**

1. Small JSON metadata frame
2. Binary frames: float32 mono PCM chunks
3. Final JSON: `{ "type": "done" }`

## Supported languages

Indic Parler covers the major Indian languages. Agent `language` codes used by the Voice Server:

| Language | Code |
|----------|------|
| Hindi | `hi` |
| Tamil | `ta` |
| Telugu | `te` |
| Kannada | `kn` |
| Malayalam | `ml` |
| Bengali | `bn` |
| Punjabi | `pa` |
| Marathi | `mr` |
| Gujarati | `gu` |
| Bhili | `bhb` (or `bhili`) |

Exact support depends on the loaded Parler weights. Per-provider language code mappings are defined in the voice server _(see source: `voice_2_voice_server/config/tts_mappings.py`)_.

## How it talks to other services

The Voice Server connects only when an agent's `tts_model.name = "indic-parler-tts"`. The base URL comes from `INDIC_TTS_SERVER_URL` (or `AI4BHARAT_TTS_URL`); the WebSocket client lives in `voice_2_voice_server/services/ai4bharat/tts.py`.

```bash
Voice Server (indic-parler-tts)
    | WS {INDIC_TTS_SERVER_URL}
    v
ai4bharat_tts_server (Parler, 44.1 kHz float32)
```

Telephony pipelines downsample / convert the 44.1 kHz float32 stream to the carrier sample rate as needed.

## GPU / VRAM

| | |
|--|--|
| Production | NVIDIA GPU **expected** for Parler TTS inference |
| Development | CPU possible but not recommended for latency |
| Recommended GPUs | **RTX 4000 series and newer** (see list below) |
| Pinned VRAM (GB) | **Deferred** — depends on checkpoint and concurrency |

**Hardware guidance.** GPUs at RTX 4000 series and above (NVIDIA Ada, Blackwell, and datacenter classes) generally run Parler TTS well for production. Examples teams have used or sized for:

RTX 5090, RTX 5080, RTX 5070 Ti, RTX 5070, RTX 5060 Ti, RTX 5060, RTX 4090, RTX 4080 Super, RTX 4080, RTX 4070 Ti Super, RTX 4070 Ti, RTX 4070 Super, RTX 4070, RTX 4060 Ti 16GB, RTX 4060 Ti, RTX 4060, RTX 6000 Ada, RTX 5000 Ada, RTX 4500 Ada, RTX 4000 Ada, L40S, L40, H100, H200.

Older or low-VRAM cards may work for light dev loads but are not recommended for production latency and concurrency.

{% hint style="warning" %}
Until pinned GB figures are published, VRAM still depends on your Parler weights, batching, and concurrent utterances. Consult your hosting partner or observe `nvidia-smi` on staging.
{% endhint %}

## Running

```bash
cd ai4bharat_tts_server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py
```

Monorepo: `make start-voice-only-services` (includes the optional AI4Bharat servers and the Voice Server).

Docker with GPU:

```bash
docker run --gpus all -p 8002:8002 \
  -v /path/to/models:/models \
  ai4bharat-tts
```

Docker Compose snippet for GPU passthrough:

```yaml
services:
  ai4bharat_tts_server:
    build: ./ai4bharat_tts_server
    restart: unless-stopped
    ports:
      - "8002:8002"
    volumes:
      - ./models:/models
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8002/health"]
      interval: 5s
      timeout: 5s
      retries: 5
```

## Troubleshooting

- [troubleshooting/voice-and-audio.md](../troubleshooting/voice-and-audio.md)
- [troubleshooting/common-issues.md](../troubleshooting/common-issues.md)
- Choppy or silent audio in the call -> verify the Voice Server is resampling the 44.1 kHz float32 stream to the carrier rate (`SAMPLE_RATE`, default `8000`).
- High latency -> ensure CUDA is being used (`nvidia-smi`) and that batching is not starved by a single long utterance.

## Next steps

- [services/ai4bharat-stt.md](ai4bharat-stt.md)
- [services/voice-server.md](voice-server.md)
- [concepts/voice-pipeline.md](../concepts/voice-pipeline.md)
