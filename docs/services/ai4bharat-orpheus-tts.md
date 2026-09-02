---
description: Optional on-premises Indic text-to-speech server (Orpheus + vLLM + SNAC over WebSocket).
---

# AI4Bharat Orpheus TTS

Optional on-premises Indic text-to-speech using the AI4Bharat Orpheus checkpoint served by **vLLM**, decoded to audio by the **SNAC** codec, and streamed over **WebSocket**. The core VoicEra stack can run with cloud TTS only; this server is required only when an agent uses the `Orpheus` provider.

Default host port: **8004** (the container itself always listens on `9000`). Code: `ai4bharat_orpheustts_server/src/orpheus_server/app.py`, `ai4bharat_orpheustts_server/src/orpheus_server/api/native.py`, `ai4bharat_orpheustts_server/src/orpheus_server/codec.py`. Roster: `ai4bharat_orpheustts_server/voices.json`.

## Responsibilities

- Serve 22 scheduled Indian languages, 40 named speakers and 12 speaking styles from one checkpoint
- Stream synthesised audio back as binary int16 mono PCM frames over WebSocket
- Expose an OpenAI-compatible `/v1/audio/speech` surface plus a native REST surface for non-streaming callers
- Continuous batching across concurrent streams via vLLM, with batched SNAC decode

## When you need it

| Scenario | Need this server? |
|----------|-------------------|
| Cloud TTS only | No |
| English agent | No — Orpheus serves no English voice |
| Local Indic TTS with named speakers and style control | Yes (+ NVIDIA GPU) |
| Bhili (`bhb`) | No — use AI4Bharat Parler instead |

### Orpheus vs AI4Bharat Parler

Both are on-prem and need **no API key**. They are separate providers with separate agent configs; there is no fallback between them.

| | AI4Bharat Parler (`indic-parler-tts`) | AI4Bharat Orpheus (`Orpheus`) |
|--|--|--|
| Output | 44100 Hz float32 mono PCM | 24000 Hz int16 (s16le) mono PCM |
| Voice control | Free-text voice/style **description** | Named **speaker** + one of 12 discrete **styles** |
| Bhili (`bhb`) | Available | Not served |
| English | Available in the mapping | Not served |
| Serving | PyTorch inference worker with its own batching | vLLM continuous batching + batched SNAC decode |
| API key | None (on-prem) | None (on-prem) |

## Architecture

vLLM generates SNAC audio codes as ordinary LLM tokens; the SNAC codec turns them into 24 kHz PCM. FastAPI fronts both an OpenAI-compatible and a native API, plus the WebSocket endpoint the Voice Server uses.

- Sample rate: **24000 Hz**
- Output: **s16le (int16) mono PCM**, streamed as binary frames
- Frame size: 2048 samples = **4096 bytes = 85.33 ms** per WebSocket frame (`src/orpheus_server/codec.py:26-28`)
- Framework: FastAPI + vLLM + SNAC (`hubertsiuzdak/snac_24khz`, ~76 MB, fetched on first boot)
- Single-stream latency: **TTFA ~124 ms, RTF ~0.33** (FP8 on an RTX PRO 6000 Blackwell)

A measured example: one Hindi sentence with voice `Amit` produced 176128 bytes — 3.67 s of audio — in 1.26 s wall.

{% hint style="info" %}
`GET /health` returns **503** until the model is loaded **and** warmup is done. "Healthy" therefore means genuinely ready to serve; cold start is dominated by weight load, CUDA-graph capture and the warmup ladder.
{% endhint %}

## Configuration

Every setting lives in `config.yaml` and can be overridden by the matching `ORPHEUS_*` environment variable, which always wins. `docker-compose.yml` passes the whole `.env` into the container via `env_file`, and the boot log prints an `env overrides:` line listing exactly which variables were applied.

**Source of truth:** `ai4bharat_orpheustts_server/.env.example` and `ai4bharat_orpheustts_server/config.yaml`.

| Variable | `config.yaml` key | Default | Description |
|----------|-------------------|---------|-------------|
| `ORPHEUS_MODEL_PATH` | `model.path` | `models/orpheus-indic-5679` | Checkpoint directory (container path) or HuggingFace repo id |
| `ORPHEUS_DTYPE` | `model.dtype` | `auto` | `auto` lets vLLM pick bfloat16 / float16 |
| `ORPHEUS_QUANTIZATION` | `model.quantization` | `fp8` | Needs compute capability >= 8.9 (Ada / Hopper / Blackwell); use `none` below that |
| `ORPHEUS_MAX_MODEL_LEN` | `model.max_model_len` | `8192` | Context window: prompt + generated audio tokens |
| `ORPHEUS_TRUST_REMOTE_CODE` | `model.trust_remote_code` | `false` | Only for checkpoints shipping custom code |
| `ORPHEUS_GPU_MEMORY_UTILIZATION` | `engine.gpu_memory_utilization` | `0.90` | Fraction of VRAM for weights plus KV cache; up to ~0.95 on a dedicated card |
| `ORPHEUS_MAX_NUM_SEQS` | `engine.max_num_seqs` | `256` | Concurrent-stream **admission** limit — set to `64` for live voice |
| `ORPHEUS_ENFORCE_EAGER` | `engine.enforce_eager` | `false` | Skips CUDA-graph capture: faster boot, less VRAM, slower decode |
| `ORPHEUS_TENSOR_PARALLEL_SIZE` | `engine.tensor_parallel_size` | `1` | GPUs to shard the model across |
| `ORPHEUS_MAX_TOKENS_DEFAULT` | `engine.max_tokens_default` | `8192` | Generation cap when a request omits `max_tokens` |
| `ORPHEUS_MAX_TOKENS_LIMIT` | `engine.max_tokens_limit` | `8192` | Ceiling a client cannot exceed |
| `ORPHEUS_DECODER_DEVICE` | `decoder.device` | `cuda` | Torch device for the SNAC codec |
| `ORPHEUS_DECODER_MAX_BATCH` | `decoder.max_batch` | `256` | Keep **at or above** `ORPHEUS_MAX_NUM_SEQS` |
| `ORPHEUS_SNAC_MODEL_ID` | `decoder.model_id` | `hubertsiuzdak/snac_24khz` | Codec weights |
| `ORPHEUS_WARMUP_ENABLED` | `warmup.enabled` | `true` | Compiles sampling kernels per batch width at boot |
| `ORPHEUS_WARMUP_WIDTHS` | `warmup.concurrency_widths` | `1,2,4,8,16,32,64,128,256` | Should reach `ORPHEUS_MAX_NUM_SEQS` |
| `ORPHEUS_WARMUP_MAX_TOKENS` | `warmup.max_tokens` | `64` | Kernels compile per width, not per length |
| `ORPHEUS_PORT` | `server.port` | `9000` | **Host** port Compose publishes; the container port stays `9000` |
| `ORPHEUS_MODEL_NAME` | `server.model_name` | `orpheus-indic` | Id reported by `GET /v1/models` |
| `ORPHEUS_CORS_ORIGINS` | `server.cors_origins` | `*` | Comma-separated when set via environment |
| `ORPHEUS_DRAIN_TIMEOUT` | `server.drain_timeout` | `30.0` | Seconds in-flight streams get on shutdown; keep Compose `stop_grace_period` above it |
| `ORPHEUS_VOICES_FILE` | `voices_file` | `voices.json` | Speaker roster |
| `ORPHEUS_LOG_LEVEL` | – | `INFO` | Application log level |
| `VLLM_LOGGING_LEVEL` | – | `INFO` | `INFO` shows vLLM's KV-cache accounting at boot |

The sampling knobs (`ORPHEUS_TEMPERATURE=0.6`, `ORPHEUS_TOP_P=0.8`, `ORPHEUS_REPETITION_PENALTY=1.3`, `ORPHEUS_MIN_TOKENS=28`) are stack-verified; changing them degrades audio quality.

{% hint style="warning" %}
`ORPHEUS_PORT` in `.env` must be **`8004`** for a monorepo deployment. The standalone `docker-compose.yml` defaults to publishing `9000`, which is already MinIO's port in the VoicEra stack.
{% endhint %}

## Endpoints / API surface

Interactive docs: Swagger at `/docs`, ReDoc at `/redoc`. The WebSocket endpoint is not in the OpenAPI schema — OpenAPI cannot describe WebSockets.

| Method | Path | Notes |
|--------|------|-------|
| GET | `/health` | Liveness + readiness (503 until loaded and warmed up) |
| GET | `/metrics` | Aggregate counters |
| GET | `/v1/models` | OpenAI-compatible |
| POST | `/v1/audio/speech` | OpenAI-compatible; `response_format` `wav`/`pcm`/`mp3`/`flac`/`opus`, `stream_format` `audio`/`sse` |
| GET | `/v1/languages` | Catalog of languages, speakers and native-script samples |
| GET | `/v1/voices?language=hi` | Catalog; returns e.g. `{"hi":{"name":"Hindi","voices":["Amit"]}}` |
| GET | `/v1/styles` | 12 styles, default `CONV` |
| POST | `/v1/tts` | Complete 24 kHz WAV; timing headers `X-TTFA-Ms` / `X-RTF` |
| GET | `/v1/tts/stream` | Chunked WAV — `<audio src>`- and `curl`-friendly |
| **WS** | **`/v1/tts/ws`** | **Lowest latency; this is what the Voice Server uses** |

### WebSocket protocol

One utterance per connection: the socket closes once the stream ends, on success or failure alike. Clients open a connection per request.

**Client -> server** (one JSON message):

```json
{
  "text": "text to speak",
  "voice": "Amit",
  "language": "hi",
  "style": "CONV",
  "max_tokens": 8192
}
```

`language`, `style` and `max_tokens` are optional.

**Server -> client:**

1. JSON start frame:

```json
{
  "type": "start",
  "sample_rate": 24000,
  "format": "s16le",
  "channels": 1,
  "language": "hi",
  "voice": "Amit",
  "style": "CONV"
}
```

2. Binary frames: int16 mono PCM, 4096 bytes each
3. Final JSON: `{ "type": "done", "metrics": { ... } }`

Errors arrive as `{ "type": "error", "message": "..." }` — a bad request is answered with a single error frame instead of a start frame followed by silence.

{% hint style="info" %}
**The speaker name selects the language.** All 40 names are globally unique, so `Amit` is unambiguously Hindi and `Anitha` unambiguously Tamil. `language` is optional; when both are sent, the server validates the pair.
{% endhint %}

{% hint style="warning" %}
`speed` is accepted on `/v1/audio/speech` but **ignored** — Orpheus has no rate control, and resampling would shift pitch. A non-default value sets an `X-Speed-Ignored` response header.
{% endhint %}

## Supported languages

All 22 scheduled Indian languages, 40 speakers. **No English and no Bhili (`bhb`)** — an English agent cannot use Orpheus.

| Language | Code | Speakers |
|----------|------|----------|
| Hindi | `hi` | Amit |
| Marathi | `mr` | Anagha, Chinmay |
| Tamil | `ta` | Anitha, Arun |
| Telugu | `te` | Sravani, Vamsi |
| Kannada | `kn` | Adarsh, Deepika |
| Malayalam | `ml` | Kiran, Lakshmi |
| Gujarati | `gu` | Parth, Dhara |
| Punjabi | `pa` | Kaur, Manpreet |
| Bengali | `bn` | Ishita, Sourav |
| Assamese | `as` | Ankur, Prastuti |
| Odia | `or` | Akash, Itishree |
| Urdu | `ur` | Saba, Zaid |
| Kashmiri | `ks` | Ishfaq, Zoon |
| Sindhi | `sd` | Moomal |
| Maithili | `mai` | Madhukar, Vaidehi |
| Dogri | `doi` | Preeti, Sham |
| Konkani | `kok` | Anjali, Sandeep |
| Bodo | `brx` | Gwrbw, Sansuma |
| Manipuri | `mni` | Chaoba |
| Santali | `sat` | Sibu |
| Nepali | `ne` | Srijana, Sagar |
| Sanskrit | `sa` | Aryaman, Bharati |

Display-name to code mapping for agents lives in `TTS_LANGUAGE_MAP["Orpheus"]` in `voice_2_voice_server/config/tts_mappings.py`. `GET /v1/languages` is the live roster and also returns a native-script `sample` per language.

{% hint style="warning" %}
**Kashmiri (`ks`)** is served by this server but has no `ks` language key in the frontend's `voicera_frontend/tts.json`, so it is not yet selectable in the dashboard.
{% endhint %}

## Speaking styles

12 styles, sent as `style` on the request. Default is `CONV`.

`CONV` (default), `WIKI`, `NEWS`, `BOOK`, `NAMES`, `INDIC`, `ALEXA`, `BB`, `UMANG`, `DIGI`, `SANGRAH`, `INDICTTS`

`GET /v1/styles` returns the list served by the loaded roster.

## How it talks to other services

The Voice Server connects only when an agent's `tts_model.name = "Orpheus"`. `create_tts_service` in `voice_2_voice_server/api/services.py` dispatches on that official provider name; the lowercase ids `orpheus` and `orpheus-indic` both normalise to it. The WebSocket client is `OrpheusTTSService` in `voice_2_voice_server/services/orpheus/tts.py`, a subclass of Pipecat's `TTSService`.

The base URL comes from `ORPHEUS_TTS_SERVER_URL` — **base only, no path**: the service class appends `/v1/tts/ws` itself. Optional `ORPHEUS_TTS_GAIN` (float, default `1.0`) scales the output.

```
Voice Server (Orpheus)
    | WS {ORPHEUS_TTS_SERVER_URL}/v1/tts/ws
    v
ai4bharat_orpheustts_server (Orpheus + vLLM + SNAC, 24 kHz int16)
```

Agent config that selects it:

```json
{
  "tts_model": {
    "name": "Orpheus",
    "model": "orpheus-indic",
    "speaker": "Amit",
    "style": "CONV"
  }
}
```

The int16 PCM is passed straight through to `TTSAudioRawFrame` with **no dtype conversion** — unlike the Parler provider, which converts float32 first. The Voice Server then resamples 24 kHz down to the carrier rate (`SAMPLE_RATE`, default `8000` for telephony, `16000` for the browser test client).

Mid-call language switching is supported and gated in `voice_2_voice_server/api/bot.py` alongside AI4Bharat Parler. See [concepts/mid-call-language-switching.md](../concepts/mid-call-language-switching.md).

## Concurrency and tuning

`ORPHEUS_MAX_NUM_SEQS` is an **admission policy, not a throughput control**. Raising it does not make the GPU faster; it changes what happens to traffic the GPU cannot keep up with. A low value makes extra requests **wait** in a queue; a high value **admits** them all and makes every stream slower. Total throughput is fixed by the hardware either way.

Measured on one RTX PRO 6000 Blackwell Server Edition (96 GB, FP8, `gpu_memory_utilization` 0.95, 85.3 GiB KV cache). All N streams start at the same instant; a frame is **late** if it misses its real-time playback slot after a 100 ms prebuffer, and *clean* is the share of streams with none late. Table reproduced from `ai4bharat_orpheustts_server/config.yaml` lines 77-100.

| Concurrent | audio-s/s | TTFA p50 | TTFA p95 | RTF | Clean |
|------------|-----------|----------|----------|-----|-------|
| 1 | 3.0 | 124 ms | 124 ms | 0.33 | 100% |
| 16 | 30.8 | 169 ms | 175 ms | 0.47 | 100% |
| 32 | 50.5 | 171 ms | 180 ms | 0.59 | 100% |
| 64 | 76.9 | 192 ms | 229 ms | 0.78 | 100% |
| 128 | 103.4 | 265 ms | 355 ms | 1.16 | 0% |
| 256 | 111.6 | 468 ms | 694 ms | 2.19 | 0% |

Throughput keeps rising past 64, but the GPU stops making streams *wait* and starts making them *stutter*. So pick by what the client does with the audio:

| Value | Use for |
|-------|---------|
| `64` | **Live voice** — voice agents, phone lines. Highest value where every stream stays real-time; costs about a third of peak throughput. |
| `32` | Live delivery with spikier traffic than the test, or a shared GPU. |
| `256` | Clients that **buffer before playing** (batch synthesis, file generation), where a late frame costs nothing. This is the `config.yaml` default. |

{% hint style="danger" %}
`config.yaml` alone defaults to `256`, which is wrong for VoicEra calls: at that width **0%** of streams are clean. `ai4bharat_orpheustts_server/.env.example` therefore ships `ORPHEUS_MAX_NUM_SEQS=64`, and the root `docker-compose.yml` falls back to `64` as well. Only raise it if this server is repurposed for buffered clients.
{% endhint %}

Whatever you pick, keep `ORPHEUS_DECODER_MAX_BATCH` at or above it (the server warns at boot otherwise) and make sure `ORPHEUS_WARMUP_WIDTHS` reaches it.

## GPU / VRAM

| | |
|--|--|
| Production | NVIDIA GPU **required** — vLLM has no CPU serving path here |
| Minimum GPU memory | **16 GB free** |
| Model weights | **7.6 GB** (`model.safetensors`) |
| SNAC codec | **~0.5 GB** on top of vLLM's reservation, plus the CUDA context |
| Disk | **12 GB** — 4.4 GB image, 7.6 GB weights |
| FP8 quantization | Compute capability **>= 8.9** (Ada / Hopper / Blackwell); set `ORPHEUS_QUANTIZATION=none` below that |
| Host prerequisites | NVIDIA driver + `nvidia-container-toolkit`. No CUDA toolkit, Python or vLLM on the host. |
| Pinned VRAM per concurrency level | **Deferred** — depends on `gpu_memory_utilization`, quantization and `max_num_seqs` |

`gpu_memory_utilization` is the real lever: it decides how much KV cache exists, and KV cache is what concurrency is made of. Boot with `VLLM_LOGGING_LEVEL=INFO` and read the `Available KV cache memory: X GiB` line to see the effect of a change.

## Running

Standalone, from the submodule directory:

```bash
cd ai4bharat_orpheustts_server
cp .env.example .env          # already ships ORPHEUS_PORT=8004 and ORPHEUS_MAX_NUM_SEQS=64
mkdir -p hf-cache             # must exist before the first up, or Docker creates it root-owned
docker compose up -d --build
docker compose logs -f tts    # watch model load and warmup
curl -sf http://localhost:8004/health
```

Verify the deployment with the bundled test suite:

```bash
python3 tests/orpheus_test.py --suite api          # fast functional check
python3 tests/orpheus_test.py --suite concurrency  # reproduces the table above
```

From the monorepo root, as part of the full stack:

```bash
# Everything, including Orpheus
docker compose up -d --build

# Or just Orpheus plus what a call needs
docker compose up -d mongodb backend minio voice_server orpheus_tts

docker compose logs -f orpheus_tts
curl -sf http://localhost:8004/health
```

The root `docker-compose.yml` defines the service as `orpheus_tts` on
`voicera_network`, publishing `${ORPHEUS_PORT:-8004}:9000`, and already sets
`ORPHEUS_TTS_SERVER_URL=ws://orpheus_tts:9000` on the `voice_server` service — so
no extra wiring is needed for a Compose deployment.

{% hint style="warning" %}
Create `ai4bharat_orpheustts_server/hf-cache` before the first `up`
(`mkdir -p ai4bharat_orpheustts_server/hf-cache`). If Docker creates it, it is
owned by root and the container — which runs as uid 1000 — cannot write the SNAC
download.
{% endhint %}

In-network the Voice Server then uses `ORPHEUS_TTS_SERVER_URL=ws://orpheus_tts:9000`; from the host it is `ws://localhost:8004`. See [reference/ports-and-defaults.md](../reference/ports-and-defaults.md).

## Troubleshooting

- [troubleshooting/voice-and-audio.md](../troubleshooting/voice-and-audio.md)
- [troubleshooting/common-issues.md](../troubleshooting/common-issues.md)
- `/health` returns 503 for several minutes -> expected during startup. If it persists, check the logs for out-of-memory and lower `ORPHEUS_GPU_MEMORY_UTILIZATION`.
- Stuttering playback or requests hanging under load -> `ORPHEUS_MAX_NUM_SEQS` is too high. Reduce to `64`, or `32` under sustained load. Queued requests produce no signal other than latency, so set client-side timeouts.
- Port `9000` already allocated -> MinIO owns `9000` in the VoicEra stack. Set `ORPHEUS_PORT=8004`.
- A configuration change has no effect -> check the `env overrides` line at boot. Five variables are pinned in `docker-compose.yml` and win over `.env`.
- Permission denied on `/models` or `/hf-cache` -> `sudo chown -R 1000:1000 models hf-cache && docker compose restart tts`.
- Empty or very short audio -> the input is not in the language's native script. Compare with the `sample` field from `GET /v1/languages`.
- Audio truncated mid-sentence -> `max_tokens` was reached. Raise `ORPHEUS_MAX_TOKENS_DEFAULT` and `ORPHEUS_MAX_MODEL_LEN` together.
- Choppy or silent audio in the call -> verify the Voice Server is resampling the 24 kHz int16 stream to the carrier rate (`SAMPLE_RATE`, default `8000`).
- `ORPHEUS_TTS_SERVER_URL environment variable not set` -> the Voice Server raises this at service construction; set the base URL with no path.

## Next steps

- [services/ai4bharat-tts.md](ai4bharat-tts.md)
- [services/ai4bharat-stt.md](ai4bharat-stt.md)
- [services/voice-server.md](voice-server.md)
- [concepts/voice-pipeline.md](../concepts/voice-pipeline.md)
- [reference/environment-variables.md](../reference/environment-variables.md)
- [reference/ports-and-defaults.md](../reference/ports-and-defaults.md)
