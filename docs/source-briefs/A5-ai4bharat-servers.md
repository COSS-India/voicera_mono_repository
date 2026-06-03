# Brief: AI4Bharat STT/TTS servers (A5)

**Review gap (original):** No model IDs, GPU VRAM requirements, or API spec for the STT/TTS servers in repository documentation.

**Merge status (VRAM):** API specs and env paths are documented in MkDocs and submodule READMEs. **Exact GPU VRAM (GB) is formally deferred** — checkpoints are customer-specific; engineering has not published reference benchmarks in this sprint. Operator-facing text in `docs/services/ai4bharat-*.md` states deferred status and hosting-partner sizing guidance.

These servers are **optional**. The core VoicERA stack can run using cloud providers only.

---

## ai4bharat_stt_server (default port 8001)

### Purpose

On-premises Indic speech-to-text using NeMo ASR models.

### HTTP API

| Endpoint | Method | Request | Response |
|----------|--------|---------|----------|
| `/` | GET | — | Status |
| `/health` | GET | — | Health |
| `/transcribe` | POST | `{ "audio_b64": string, "language_id": string }` (default `hi`) | `{ "text": string }` |
| `/transcribe/bhili` | POST | Same body | `{ "text": string }` (Bhili model) |

**Code:** `ai4bharat_stt_server/server.py`

### Models (on-disk paths, not HuggingFace IDs in .env.example)

| Env variable | Purpose |
|--------------|---------|
| `INDIC_NEMO_PATH` | Main Indic NeMo checkpoint **file** path |
| `BHILI_NEMO_PATH` | Bhili NeMo checkpoint (when Bhili enabled) |
| `BHILI_ENABLE` | `"yes"` / `"no"` |
| `HF_TOKEN` | Optional; if models are gated on HuggingFace |
| `PORT` | Server port (default 8001) |

### Runtime

- Device: `cuda:0` if CUDA available, else CPU
- Audio: base64-encoded **16 kHz int16 PCM** decoded to float32
- Batching: internal queue with `MAX_BATCH_SIZE=16`, `BATCH_TIMEOUT=0.1s`

### GPU / VRAM

**Status: deferred for main merge.** NVIDIA GPU strongly recommended for production; CPU for development only. Pinning `__ GB` is a follow-up after benchmarking agreed checkpoint files on reference hardware.

**Published interim text (MkDocs):** VRAM depends on checkpoint size and batch settings; consult your hosting partner or run staging load tests with `nvidia-smi`.

---

## ai4bharat_tts_server (default port 8002)

### Purpose

On-premises Indic text-to-speech (Parler TTS) over **WebSocket**.

### Protocol (from server docstring)

**Client → server** (one JSON object per utterance):

```json
{
  "prompt": "text to speak",
  "description": "voice/style description",
  "language": "hi"
}
```

Use `"bhb"` or `"bhili"` for Bhili.

**Server → client:**

1. Small JSON metadata frame
2. Binary frames: float32 mono PCM chunks
3. Final JSON: `{ "type": "done" }`

### Audio

- Sample rate: **44100 Hz** (`AUDIO_SAMPLE_RATE` in server)
- Uses PyTorch; continuous batching in inference worker

**Code:** `ai4bharat_tts_server/server.py`, `inference/runner.py`

### GPU / VRAM

**Production:** NVIDIA GPU expected. **RTX 4000 series and newer** run well (RTX 50xx/40xx consumer, RTX Ada workstation, L40S/L40, H100/H200) — full list in [ai4bharat-tts.md](../services/ai4bharat-tts.md#gpu-vram).

**Pinned VRAM (GB):** Still deferred; use hosting-partner / staging sizing in addition to the GPU class guidance above.

---

## How voice server connects

When an agent uses:

- STT: `indic-conformer-stt` → voice server calls `AI4BHARAT_STT_URL` or `INDIC_STT_SERVER_URL` + `/transcribe` or `/transcribe/bhili`
- TTS: `indic-parler-tts` → WebSocket/REST client in `voice_2_voice_server/services/ai4bharat/`

See `voice_2_voice_server/README.md` environment table for exact variable names.

### Start commands (development)

```bash
# STT
cd ai4bharat_stt_server && python server.py --port 8001

# TTS
cd ai4bharat_tts_server && python server.py
```

Monorepo: `make start-voice-only-services` (includes optional local AI4Bharat + voice server).
