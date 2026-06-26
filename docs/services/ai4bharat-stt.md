---
description: Optional on-premises Indic speech-to-text server (NeMo Conformer).
---

# AI4Bharat STT

Optional on-premises Indic speech-to-text using NeMo ASR models. The core VoicEra stack can run with cloud STT only; this server is required only when an agent uses the `indic-conformer-stt` provider.

Default port: **8001**. Code: `ai4bharat_stt_server/server.py`.

## Responsibilities

- Serve HTTP transcription for Indic languages (NeMo Conformer)
- Provide a separate `/transcribe/bhili` endpoint for the Bhili (`bhb`) checkpoint
- Internal batching to amortise GPU cost across concurrent requests

## When you need it

| Scenario | Need this server? |
|----------|-------------------|
| Cloud STT only (Deepgram, Bhashini, etc.) | No |
| Local Indic STT or Bhili (`bhb`) | Yes (+ GPU recommended) |

## Architecture

FastAPI process with an internal request queue feeding a batched NeMo inference worker. CUDA is used if available; otherwise CPU.

- Device: `cuda:0` if CUDA available, else CPU
- Audio decoder: base64 -> **16 kHz int16 PCM** -> float32
- Batching: `MAX_BATCH_SIZE=16`, `BATCH_TIMEOUT=0.1s`

## Configuration

| Variable | Purpose |
|----------|---------|
| `INDIC_NEMO_PATH` | Main Indic NeMo checkpoint **file** path |
| `BHILI_NEMO_PATH` | Bhili NeMo checkpoint (when Bhili enabled) |
| `BHILI_ENABLE` | `"yes"` / `"no"` |
| `HF_TOKEN` | Optional; if the checkpoint is gated on HuggingFace |
| `PORT` | Server port (default `8001`) |

Checkpoint files are deployment-specific and are referenced by **on-disk path**, not HuggingFace model IDs.

## Endpoints / API surface

| Endpoint | Method | Body | Response |
|----------|--------|------|----------|
| `/` | GET | - | Status |
| `/health` | GET | - | Health |
| `/transcribe` | POST | `{ "audio_b64": string, "language_id": string }` (default `hi`) | `{ "text": string }` |
| `/transcribe/bhili` | POST | Same body | `{ "text": string }` (Bhili checkpoint) |

Example:

```bash
curl -X POST http://localhost:8001/transcribe \
  -H "Content-Type: application/json" \
  -d '{ "audio_b64": "<base64 16kHz int16 PCM>", "language_id": "hi" }'
```

## Supported languages

The Indic Conformer checkpoint covers the major Indian languages. Agent `language` codes used by the Voice Server:

| Language | Code | Endpoint |
|----------|------|----------|
| Hindi | `hi` | `/transcribe` |
| Tamil | `ta` | `/transcribe` |
| Telugu | `te` | `/transcribe` |
| Kannada | `kn` | `/transcribe` |
| Malayalam | `ml` | `/transcribe` |
| Bengali | `bn` | `/transcribe` |
| Punjabi | `pa` | `/transcribe` |
| Marathi | `mr` | `/transcribe` |
| Gujarati | `gu` | `/transcribe` |
| Bhili | `bhb` | `/transcribe/bhili` |

Exact code support depends on the checkpoint loaded via `INDIC_NEMO_PATH`. Per-provider language code mappings are defined in the voice server _(see source: `voice_2_voice_server/config/stt_mappings.py`)_.

## How it talks to other services

The Voice Server calls this server only when an agent's `stt_model.name = "indic-conformer-stt"`. The base URL comes from `INDIC_STT_SERVER_URL` (or `AI4BHARAT_STT_URL`) on the Voice Server; `/transcribe` or `/transcribe/bhili` is appended based on the agent's `language`.

```bash
Voice Server (indic-conformer-stt)
    | POST {INDIC_STT_SERVER_URL}/transcribe        (language != bhb)
    | POST {INDIC_STT_SERVER_URL}/transcribe/bhili  (language == bhb)
    v
ai4bharat_stt_server  (NeMo)
```

## GPU / VRAM

| | |
|--|--|
| Production | NVIDIA GPU **strongly recommended** |
| Development | CPU fallback is supported but slow |
| Pinned VRAM (GB) | Checkpoint-dependent; size with your hosting partner using a staging load test |

Exact VRAM depends on your NeMo checkpoint (`INDIC_NEMO_PATH`, optional `BHILI_NEMO_PATH`), `MAX_BATCH_SIZE`, and whether both Indic and Bhili models are loaded. VoicEra does not ship a single reference checkpoint size.

{% hint style="info" %}
VRAM requirements depend on the NeMo checkpoint loaded, `MAX_BATCH_SIZE`, and concurrent load. Size GPUs with your hosting partner using a staging load test (`nvidia-smi` while serving `/transcribe` at expected concurrency). As an alternative, cloud STT providers require no on-premises GPU.
{% endhint %}

## Running

```bash
cd ai4bharat_stt_server
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python server.py --port 8001
```

Monorepo: `make start-voice-only-services` (includes the optional AI4Bharat servers and the Voice Server).

Docker with GPU:

```bash
docker run --gpus all -p 8001:8001 \
  -e INDIC_NEMO_PATH=/models/indic_conformer.nemo \
  -e BHILI_ENABLE=yes \
  -e BHILI_NEMO_PATH=/models/bhili.nemo \
  -v /path/to/models:/models \
  ai4bharat-stt
```

## Troubleshooting

- [troubleshooting/voice-and-audio.md](../troubleshooting/voice-and-audio.md)
- [troubleshooting/common-issues.md](../troubleshooting/common-issues.md)
- "Model file not found" -> verify `INDIC_NEMO_PATH` (and `BHILI_NEMO_PATH` if `BHILI_ENABLE=yes`) point at real files inside the container.
- Slow transcription on CPU is expected; move to GPU for production loads.

## Next steps

- [services/ai4bharat-tts.md](ai4bharat-tts.md)
- [services/voice-server.md](voice-server.md)
- [concepts/voice-pipeline.md](../concepts/voice-pipeline.md)
