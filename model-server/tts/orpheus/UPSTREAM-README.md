<!-- Upstream documentation, kept verbatim from the dev-Orpheustts branch.
     Renamed from Readme.md because Windows checkouts treat that as the same
     file as README.md, which silently clobbered it once already.
     README.md in this folder covers how the model fits the TTS slot. -->

# AI4Bharat Orpheus Indic TTS

Streaming text-to-speech server for 22 Indian languages, exposing an
OpenAI-compatible API. Runs as a single Docker container on one NVIDIA GPU.

## Overview

| | |
|---|---|
| Languages | 22 — Hindi, Marathi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Bengali, Assamese, Odia, Urdu, Kashmiri, Sindhi, Maithili, Dogri, Konkani, Bodo, Manipuri, Santali, Nepali, Sanskrit |
| Speakers | 40, each assigned to exactly one language |
| Styles | 12 — `CONV` (default), `NEWS`, `WIKI`, `BOOK`, `NAMES`, `INDIC`, `ALEXA`, `BB`, `UMANG`, `DIGI`, `SANGRAH`, `INDICTTS` |
| Output | 24 kHz mono — `pcm`, `wav`, `mp3`, `flac`, `opus` |
| Interfaces | OpenAI `/v1/audio/speech`, native REST, WebSocket |
| Model | AI4Bharat Orpheus checkpoint, SNAC codec, served by vLLM |
| Interactive docs | Swagger at `/docs`, ReDoc at `/redoc` |


## Requirements

| Component | Requirement |
|---|---|
| GPU | NVIDIA, 16 GB free memory minimum |
| Driver | NVIDIA driver only. CUDA toolkit, Python and vLLM are not required on the host. |
| Runtime | Docker with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| Disk | 12 GB — 4.4 GB image, 7.6 GB weights |
| Checkpoint | Obtained separately, see Installation step 1 |

FP8 quantization is enabled by default and requires compute capability 8.9 or
higher (Ada, Hopper, Blackwell). On earlier GPUs set `ORPHEUS_QUANTIZATION=none` in
`.env`.

## Installation

### 1. Obtain the checkpoint

Download the checkpoint from the Google Drive link 

```bash
pip install gdown
gdown --folder '<drive-folder-url>' -O /tmp/checkpoint-download
```

The Drive folder contains the raw training output. Copy the five required files into
a single directory:

```bash
DRIVE=/tmp/checkpoint-download
mkdir -p models/orpheus-indic-5679
cp "$DRIVE"/checkpoint-5679/{config.json,generation_config.json,model.safetensors} \
   models/orpheus-indic-5679/
cp "$DRIVE"/llama-3-audio-tok_trimmed/{tokenizer.json,tokenizer_config.json} \
   models/orpheus-indic-5679/
```

Resulting layout:

```
models/orpheus-indic-5679/
├── config.json              862 B
├── generation_config.json   179 B
├── model.safetensors        7.6 GB
├── tokenizer.json           22 MB
└── tokenizer_config.json    326 B
```

`tokenizer.json` and `tokenizer_config.json` come from the tokenizer folder, not the
checkpoint folder. Do not copy `optimizer.bin` or `pytorch_model_fsdp.bin`; they are
training states.



### 2. Create bind-mount directories

```bash
mkdir -p models hf-cache
sudo chown 1000:1000 models hf-cache
```

Both directories must exist and be owned by uid 1000 before the first start.

### 3. Configure

```bash
cp .env.example .env
```

Default settings target batch synthesis. For live audio delivery, set the following
in `.env`:

```bash
ORPHEUS_MAX_NUM_SEQS=64                   # 32 for sustained load
ORPHEUS_DECODER_MAX_BATCH=64
ORPHEUS_GPU_MEMORY_UTILIZATION=0.95
ORPHEUS_WARMUP_WIDTHS=1,2,4,8,16,32,64
```

### 4. Build and start

```bash
docker compose up -d --build
docker compose logs -f tts
```

### 5. Verify

```bash
curl -s localhost:9000/health | jq
```

```json
{"status":"ok","ready":true,"model":"orpheus-indic","quantization":"fp8"}
```

`/health` returns 503 until warmup completes.

```bash
curl -s localhost:9000/v1/audio/speech -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","voice":"Amit","input":"नमस्ते, आज मौसम बहुत अच्छा है।","response_format":"wav"}' \
  -o hello.wav
```

## Usage

The speaker name determines the language. Every speaker name is unique across the
roster, so OpenAI clients require only the standard `voice` field. Input text must
be in the language's native script.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")
client.audio.speech.create(
    model="orpheus", voice="Amit",
    input="नमस्ते, आज मौसम बहुत अच्छा है।",
).write_to_file("hello.mp3")
```

Query the roster:

```bash
curl -s localhost:9000/v1/languages | jq '.[] | {code, name, voices}'
curl -s localhost:9000/v1/voices?language=ta | jq
curl -s localhost:9000/v1/styles | jq
```

## API reference

### POST /v1/audio/speech

| Field | Default | Description |
|---|---|---|
| `input` | required | Text in the language's native script |
| `voice` | required | Speaker name; determines the language |
| `model` | `orpheus` | Accepted for compatibility |
| `response_format` | `mp3` | `pcm`, `wav`, `mp3`, `flac`, `opus` |
| `stream_format` | `audio` | `audio` or `sse` |
| `speed` | `1.0` | Accepted, no effect. Sets `X-Speed-Ignored`. |
| `instructions` | — | Applied as style when it names one |
| `language` | — | Extension. Required only for non-unique speaker names. |
| `style` | — | Extension. Takes precedence over `instructions`. |
| `max_tokens` | `8192` | Extension. 12.2 ms of audio per token; 8192 ≈ 100 s. |

Response headers: `X-TTFA-Ms`, `X-RTF`, `X-Audio-Duration-Sec`, `X-Generation-Ms`,
`X-Language`, `X-Voice`.

Invalid input returns 400 before generation begins, in the OpenAI error envelope:

```json
{"error": {"message": "unknown voice 'Bob'. See GET /v1/voices for the 40 available speakers.",
           "type": "invalid_request_error", "param": null, "code": null}}
```

#### Deviations from the OpenAI specification

| Area | OpenAI | This server |
|---|---|---|
| `response_format` | `mp3`, `opus`, `aac`, `flac`, `wav`, `pcm` | `aac` is not supported; libsndfile cannot encode it. The other five are present. |
| `input` length | Capped at 4096 characters | Capped by prompt tokens against `max_model_len` instead, which is the limit that actually applies to the model. For Indic scripts this permits longer input than 4096 characters. |
| `voice` | Fixed set (`alloy`, `echo`, …) | The 40 speakers from `GET /v1/voices`. The speaker name also selects the language. |
| `model` | Selects a model | Accepted and ignored; one model is served. `GET /v1/models` reports its id. |
| `speed` | `0.25`–`4.0`, applied | Accepted and validated but ignored; sets `X-Speed-Ignored`. |
| Errors | `{"error": {...}}` | Same envelope. Body-validation failures return 400 rather than FastAPI's default 422. |
| SSE | `speech.audio.delta`, `speech.audio.done` | Same event types. `flac` and `opus` are rejected with 400 because their encoders emit nothing until close. |
| Extensions | — | `language`, `style`, `max_tokens` are additions. Standard clients never need them. |

### Response format support

| Format | `stream_format: audio` | `stream_format: sse` |
|---|---|---|
| `pcm` | Streamed | Streamed |
| `mp3` | Streamed | Streamed |
| `wav` | Buffered | Streamed |
| `flac` | Buffered | 400 |
| `opus` | Buffered | 400 |

Buffered formats return complete, valid audio in a single response body. Use `pcm`
for live playback.

#### Buffered request

```bash
curl -s localhost:9000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","voice":"Amit","input":"नमस्ते, आज मौसम बहुत अच्छा है।",
       "response_format":"wav"}' \
  -D headers.txt -o hello.wav
```

Response is the audio body with timing headers:

```
HTTP/1.1 200 OK
content-type: audio/wav
x-language: hi
x-voice: Amit
x-audio-duration-sec: 3.33
x-generation-ms: 1148.7
x-ttfa-ms: 124.2
x-rtf: 0.345
```

#### Chunked streaming request

Identical body with `"response_format":"pcm"`. The response is
`transfer-encoding: chunked`; bytes arrive as they are produced. Timing headers are
absent, because they are not known when the headers are sent.

```python
with client.audio.speech.with_streaming_response.create(
    model="orpheus", voice="Amit", input="…", response_format="pcm",
) as response:
    for chunk in response.iter_bytes():
        play(chunk)
```

PCM chunks are 4096 bytes (one 85.33 ms frame at 24 kHz), except the first and last,
which are 8192 bytes.

#### SSE streaming request

```bash
curl -N -s localhost:9000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","voice":"Amit","input":"नमस्ते।",
       "response_format":"pcm","stream_format":"sse"}'
```

```
data: {"type":"speech.audio.delta","audio":"<base64 pcm>"}
data: {"type":"speech.audio.delta","audio":"<base64 pcm>"}
data: {"type":"speech.audio.done","usage":{"input_tokens":32,"output_tokens":288,
       "total_tokens":320},"audio":{"duration_ms":3242.7,"format":"pcm",
       "sample_rate":24000},"timings":{"ttfa_ms":121.4,"audio_ms":3242.7,
       "gen_ms":1116.2,"rtf":0.344,"tokens":288,"tokens_per_s":258.0,"frames":38}}
```

`speech.audio.delta` and `speech.audio.done` are OpenAI's event types. The `audio`
and `timings` objects on the done event are additions; OpenAI clients ignore unknown
fields. The stream is not terminated by a `[DONE]` sentinel.

### Endpoints

| Endpoint | Description |
|---|---|
| `POST /v1/audio/speech` | OpenAI-compatible synthesis |
| `GET /v1/models` | OpenAI model discovery |
| `GET /v1/languages` | Languages, speakers and sample text |
| `GET /v1/voices?language=<code>` | Speakers, optionally filtered |
| `GET /v1/styles` | Available styles and the default |
| `POST /v1/tts` | Complete WAV with timing headers |
| `GET /v1/tts/stream` | Streaming WAV via URL |
| `GET /health` | Readiness; 503 until warm |
| `GET /metrics` | Aggregate counters |
| `WS /v1/tts/ws` | Lowest-latency streaming |

#### Catalog responses

```jsonc
// GET /v1/languages  -> array, one entry per language
{"code":"hi","name":"Hindi","n_voices":1,"voices":["Amit"],
 "sample":"नमस्ते, आज मौसम बहुत अच्छा है। मैं ठीक हूँ।"}

// GET /v1/voices?language=ta
{"ta":{"name":"Tamil","voices":["Anitha","Arun"]}}

// GET /v1/styles
{"styles":["CONV","WIKI","NEWS","BOOK","NAMES","INDIC","ALEXA","BB","UMANG",
           "DIGI","SANGRAH","INDICTTS"],"default":"CONV"}

// GET /v1/models
{"object":"list","data":[{"id":"orpheus-indic","object":"model",
                          "created":1786540845,"owned_by":"local"}]}

// GET /health   (503 with the same body until ready is true)
{"status":"ok","ready":true,"model":"orpheus-indic",
 "model_path":"/models/orpheus-indic-5679","quantization":"fp8",
 "max_num_seqs":256,"streams_active":1}

// GET /metrics
{"requests_total":80,"streams_active":1,"errors_total":0,
 "audio_seconds_total":984.32,"ready":true,"uptime_seconds":30497.9}
```

#### Native synthesis

```bash
# POST /v1/tts -> complete WAV, same timing headers as /v1/audio/speech
curl -s localhost:9000/v1/tts -H 'Content-Type: application/json' \
  -d '{"text":"नमस्ते।","voice":"Amit","style":"NEWS"}' -o out.wav

# GET /v1/tts/stream -> streaming WAV, usable directly as a URL
curl -N -s 'localhost:9000/v1/tts/stream?voice=Amit&text=नमस्ते' -o out.wav
```

Body fields for `POST /v1/tts`: `text`, `voice`, optional `language`, `style`,
`max_tokens`. Query parameters for `GET /v1/tts/stream` are the same names.

### WebSocket

```json
--> {"text":"नमस्ते, आज मौसम बहुत अच्छा है।","voice":"Amit"}
<-- {"type":"start","sample_rate":24000,"format":"s16le","channels":1,
     "language":"hi","voice":"Amit","style":"CONV"}
<-- <binary pcm frame>  ...
<-- {"type":"done","metrics":{"ttfa_ms":120.6,"audio_ms":3242.7,"gen_ms":1116.2,
     "rtf":0.344,"tokens":288,"tokens_per_s":238.9,"frames":38,"jitter_p99_ms":33.0}}
```

Errors arrive as `{"type":"error","message":"..."}` before the socket closes.
Binary frames are 8192 bytes for the first and last, 4096 bytes in between.
One utterance per connection.

## Configuration

[`config.yaml`](config.yaml) contains all settings with inline documentation. It and
`voices.json` are bind-mounted, so changes require a restart
(`docker compose restart tts`), not a rebuild.

Precedence, lowest first: built-in defaults → `config.yaml` → environment variable.
Every `ORPHEUS_*` variable in `.env` is passed into the container via the
`env_file:` entry in `docker-compose.yml`, so `.env` overrides `config.yaml` for
every key below. The five variables named in the compose `environment:` block
(`MAX_NUM_SEQS`, `QUANTIZATION`, `GPU_MEMORY_UTILIZATION`, `MODEL_PATH`,
`LOG_LEVEL`) have defaults there and take effect even with no `.env` present.

The `env overrides:` line printed at boot lists exactly which variables were
applied. Check it whenever a setting appears not to take effect.

**Model**

| Variable | `config.yaml` key | Default | Description |
|---|---|---|---|
| `ORPHEUS_MODEL_PATH` | `model.path` | `models/orpheus-indic-5679` | Directory, or an HF repo id. Under Docker this is a **container** path. |
| `ORPHEUS_QUANTIZATION` | `model.quantization` | `fp8` | `none` below compute capability 8.9 |
| `ORPHEUS_DTYPE` | `model.dtype` | `auto` | `auto` selects bf16 on Ampere and newer, fp16 below |
| `ORPHEUS_MAX_MODEL_LEN` | `model.max_model_len` | `8192` | Context window in tokens |
| `ORPHEUS_TRUST_REMOTE_CODE` | `model.trust_remote_code` | `false` | Required only for checkpoints shipping custom code |

**Engine capacity**

| Variable | `config.yaml` key | Default | Description |
|---|---|---|---|
| `ORPHEUS_MAX_NUM_SEQS` | `engine.max_num_seqs` | `256` | Concurrent-stream admission limit. An admission policy, not a throughput control. |
| `ORPHEUS_GPU_MEMORY_UTILIZATION` | `engine.gpu_memory_utilization` | `0.90` | Fraction of VRAM for weights plus KV cache. Up to ~0.95 on a dedicated GPU. |
| `ORPHEUS_TENSOR_PARALLEL_SIZE` | `engine.tensor_parallel_size` | `1` | GPUs to shard the model across |
| `ORPHEUS_ENFORCE_EAGER` | `engine.enforce_eager` | `false` | Skips CUDA graphs: lower VRAM and faster boot, slower decode |
| `ORPHEUS_MAX_TOKENS_DEFAULT` | `engine.max_tokens_default` | `8192` | Generation cap when a request omits `max_tokens`. A value below `MAX_MODEL_LEN` truncates long input without error. |
| `ORPHEUS_MAX_TOKENS_LIMIT` | `engine.max_tokens_limit` | `8192` | Server-side ceiling a client cannot exceed |

**Audio decoder**

| Variable | `config.yaml` key | Default | Description |
|---|---|---|---|
| `ORPHEUS_DECODER_DEVICE` | `decoder.device` | `cuda` | `cuda:1` moves the codec to a second GPU |
| `ORPHEUS_DECODER_MAX_BATCH` | `decoder.max_batch` | `256` | Must be ≥ `MAX_NUM_SEQS`, or decode competes with generation |
| `ORPHEUS_SNAC_MODEL_ID` | `decoder.model_id` | `hubertsiuzdak/snac_24khz` | SNAC codec weights, fetched to `HF_HOME` on first boot |

**Sampling** — stack-verified for Orpheus; changing these degrades audio quality.

| Variable | `config.yaml` key | Default |
|---|---|---|
| `ORPHEUS_TEMPERATURE` | `sampling.temperature` | `0.6` |
| `ORPHEUS_TOP_P` | `sampling.top_p` | `0.8` |
| `ORPHEUS_REPETITION_PENALTY` | `sampling.repetition_penalty` | `1.3` |
| `ORPHEUS_MIN_TOKENS` | `sampling.min_tokens` | `28` — one decode window; prevents an empty clip |

**Warmup**

| Variable | `config.yaml` key | Default | Description |
|---|---|---|---|
| `ORPHEUS_WARMUP_ENABLED` | `warmup.enabled` | `true` | Disabling shortens startup and moves JIT cost to first traffic |
| `ORPHEUS_WARMUP_WIDTHS` | `warmup.concurrency_widths` | `1,…,256` | Comma-separated batch widths to pre-compile. Should reach `MAX_NUM_SEQS`. |
| `ORPHEUS_WARMUP_MAX_TOKENS` | `warmup.max_tokens` | `64` | Tokens per warmup request |

**Server**

| Variable | `config.yaml` key | Default | Description |
|---|---|---|---|
| `ORPHEUS_PORT` | `server.port` | `9000` | Host port Compose publishes. The container always listens on 9000. |
| `ORPHEUS_HOST` | `server.host` | `0.0.0.0` | Bind address. Used only by `python -m orpheus_server`; the image's CMD fixes it. |
| `ORPHEUS_MODEL_NAME` | `server.model_name` | `orpheus-indic` | Id reported by `GET /v1/models` |
| `ORPHEUS_CORS_ORIGINS` | `server.cors_origins` | `*` | Comma-separated allowed origins |
| `ORPHEUS_DRAIN_TIMEOUT` | `server.drain_timeout` | `30` | Seconds in-flight streams get to finish on shutdown |
| `ORPHEUS_VOICES_FILE` | `voices_file` | `voices.json` | Path to the speaker roster |

**Read directly, not from `config.yaml`**

| Variable | Default | Description |
|---|---|---|
| `ORPHEUS_CONFIG` | `<repo>/config.yaml` | Path to the config file itself. Set to `/app/config.yaml` in the image. |
| `ORPHEUS_LOG_LEVEL` | `INFO` | Application log level |
| `VLLM_LOGGING_LEVEL` | `INFO` | vLLM's own log level; `INFO` shows KV-cache accounting at boot |
| `HF_HOME` | `/hf-cache` | HuggingFace cache location, set in the image |


## Troubleshooting

| Symptom | Resolution |
|---|---|
| `/health` returns 503 for several minutes | Expected during startup. If persistent, check the logs for out-of-memory and reduce `ORPHEUS_GPU_MEMORY_UTILIZATION`. |
| Permission denied on `/models` or `/hf-cache` | `sudo chown -R 1000:1000 models hf-cache && docker compose restart tts` |
| A configuration change has no effect | Check the `env overrides` line at boot. Five variables are set by Compose and must be changed in `.env`. |
| `no CUDA device visible` | Verify `nvidia-container-toolkit` is installed and `nvidia-smi` works on the host. |
| Output is silence or noise | `tokenizer.json` does not match the checkpoint. Re-copy all five files from one source. |
| Audio truncated mid-sentence | `max_tokens` reached. Increase `ORPHEUS_MAX_TOKENS_DEFAULT` and `ORPHEUS_MAX_MODEL_LEN` together. |
| Empty or very short audio | Input is not in the language's native script. Compare with the `sample` field in `/v1/languages`. |
| Stuttering playback, or requests hanging under load | `ORPHEUS_MAX_NUM_SEQS` is too high. Reduce to 64, or 32 under sustained load. Queued requests produce no signal other than latency; set client-side timeouts. |
| Model files not found | `ORPHEUS_MODEL_PATH` is a container path. Compose mounts `./models` at `/models`. A path that is not an existing directory is treated as a HuggingFace repo id. |

## Testing

`tests/orpheus_test.py` verifies every endpoint and measures latency and
throughput against a running server. It requires only `requests` and exits
non-zero if any functional check fails, so it can gate a deployment.

```bash
pip install requests
python3 tests/orpheus_test.py --suite api                    # fast functional check
python3 tests/orpheus_test.py --suite all --json results.json
```

Suites: `api` (endpoints, schemas, validation), `batch` (non-live buffered),
`live` (streaming latency), `concurrency` (continuous batching), `latency`
(sustained load and fixed arrival rate). See [tests/README.md](tests/README.md)
for what each measures and how to read the output.

## Repository layout

```
config.yaml          All settings, documented inline
voices.json          Language, speaker and style roster
.env.example         Environment overrides
requirements.txt     Top-level dependency pins
constraints.txt      Full transitive dependency pins
src/orpheus_server/  Application source
tests/               Functional and performance test suite
```

`models/` and `hf-cache/` are excluded from version control.




## Measured performance

One RTX PRO 6000 Blackwell Server Edition (96 GB), FP8, `max_num_seqs=256`,
`gpu_memory_utilization=0.95`, 85.3 GiB KV cache. Reproduce with
`python3 tests/orpheus_test.py --suite all`.

**Single stream** — time to first audio is independent of output length:

| Text | Audio | TTFA | RTF | Jitter p99 |
|---|---|---|---|---|
| 3.2 s | 37 frames | 122.1 ms | 0.34 | 28.5 ms |
| 11.9 s | 139 frames | 122.3 ms | 0.33 | 28.7 ms |
| 37.3 s | 437 frames | 123.5 ms | 0.34 | 29.9 ms |

**Concurrency** — all streams started simultaneously. `clean` is the percentage of
streams in which no frame arrived after its real-time playback deadline:

| Concurrent | Aggregate audio-s/s | TTFA p50 | TTFA p95 | RTF | Clean |
|---|---|---|---|---|---|
| 1 | 3.0 | 124 ms | 124 ms | 0.33 | 100% |
| 16 | 30.8 | 169 ms | 175 ms | 0.47 | 100% |
| 32 | 50.5 | 171 ms | 180 ms | 0.59 | 100% |
| **64** | **76.9** | 192 ms | 229 ms | 0.78 | **100%** |
| 128 | 103.4 | 265 ms | 355 ms | 1.16 | 0% |
| 256 | 111.6 | 468 ms | 694 ms | 2.19 | 0% |

**64 concurrent streams is the most this GPU delivers with every stream smooth.**
Throughput keeps climbing to 111.6 audio-s/s at 256 — 37× the single-stream rate —
but past 64 individual streams stop meeting playback deadlines. Choose by workload:
`ORPHEUS_MAX_NUM_SEQS=64` for live delivery, `256` for batch throughput.

**Sustained load** (30 s, N kept in flight) reproduces the same knee: 64 → 76.1
audio-s/s with 100% clean and 100% started within 500 ms; 128 → 105.2 audio-s/s
with 0% clean.

**Open-loop arrival rate** — requests issued on a clock regardless of completion:

| Rate | Aggregate audio-s/s | TTFA p50 | TTFA p95 | Clean |
|---|---|---|---|---|
| 2 rps | 15.6 | 174 ms | 187 ms | 100% |
| 5 rps | 34.8 | 194 ms | 227 ms | 100% |
| 10 rps | 59.7 | 238 ms | 321 ms | 100% |
| 20 rps | 89.4 | 305 ms | 488 ms | 20% |

**Continuous batching**: eight requests took 33.3 s sequentially and 5.5 s
concurrently — a **6.1× speedup** for a 26% increase in per-request latency.

Full run: 1,786 requests, 22,753 audio-seconds, **0 errors**, GPU at 99.6% mean
utilisation under sustained load.


## License


Components: Orpheus Indic checkpoint ([AI4Bharat](https://ai4bharat.iitm.ac.in/),
IIT Madras) · [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) (Canopy Labs) ·
[SNAC](https://github.com/hubertsiuzdak/snac) ·
[vLLM](https://github.com/vllm-project/vllm)
