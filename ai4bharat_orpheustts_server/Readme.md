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

Reference performance on one RTX PRO 6000 Blackwell (96 GB), FP8: 122 ms to first
audio, 0.36 real-time factor, 64 concurrent real-time streams. Measurement tables
are in [`config.yaml`](config.yaml).

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

`aac` is not supported. All other OpenAI response formats are available.

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

```python
with client.audio.speech.with_streaming_response.create(
    model="orpheus", voice="Amit", input="…", response_format="pcm",
) as response:
    for chunk in response.iter_bytes():
        play(chunk)
```

PCM chunks are 4096 bytes, except the first and last, which are 8192 bytes.

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

### WebSocket

Send one JSON request with `text`, `voice`, and optional `language`, `style`,
`max_tokens`. The server returns a `start` frame, binary 24 kHz s16le PCM frames,
and a `done` frame containing stream metrics. One utterance per connection; the
socket closes when the stream ends.

## Configuration

[`config.yaml`](config.yaml) contains all settings with inline documentation. It and
`voices.json` are bind-mounted, so changes require a restart
(`docker compose restart tts`), not a rebuild.

Environment variables take precedence over `config.yaml`. `docker-compose.yml` sets
`ORPHEUS_MAX_NUM_SEQS`, `ORPHEUS_QUANTIZATION`, `ORPHEUS_GPU_MEMORY_UTILIZATION`,
`ORPHEUS_MODEL_PATH` and `ORPHEUS_LOG_LEVEL`. **These five must be changed in `.env`;
editing them in `config.yaml` has no effect under Docker.**

| Variable | Default | Description |
|---|---|---|
| `ORPHEUS_QUANTIZATION` | `fp8` | `none` below compute capability 8.9 |
| `ORPHEUS_GPU_MEMORY_UTILIZATION` | `0.90` | Up to 0.95 on a dedicated GPU |
| `ORPHEUS_MAX_MODEL_LEN` | `8192` | Context window in tokens |
| `ORPHEUS_MAX_TOKENS_DEFAULT` | `8192` | Default generation cap. Values below `MAX_MODEL_LEN` truncate long input without error. |
| `ORPHEUS_DECODER_MAX_BATCH` | `256` | Must be greater than or equal to `MAX_NUM_SEQS` |
| `ORPHEUS_DECODER_DEVICE` | `cuda` | Set to `cuda:1` to move the codec to a second GPU |
| `ORPHEUS_WARMUP_WIDTHS` | `1,…,256` | Batch widths to pre-compile. Must reach `MAX_NUM_SEQS`. |
| `ORPHEUS_WARMUP_ENABLED` | `true` | Disabling shortens startup and moves JIT cost to first traffic |
| `ORPHEUS_ENFORCE_EAGER` | `false` | Disables CUDA graphs; lower VRAM, slower decode |
| `ORPHEUS_TENSOR_PARALLEL_SIZE` | `1` | GPUs to shard the model across |
| `ORPHEUS_DRAIN_TIMEOUT` | `30` | Seconds allowed for in-flight streams to complete on shutdown |
| `ORPHEUS_MODEL_PATH` | `/models/orpheus-indic-5679` | Container path or HuggingFace repo id |
| `ORPHEUS_PORT` | `9000` | Published host port |



No source code is specific to a GPU model. Compute capability is read at boot and
reported as a warning only.

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

## Deployment without Compose

```bash
docker run -d --name orpheus-indic-tts \
  --gpus all --shm-size 8g --init --stop-timeout 60 \
  -v "$PWD/models:/models:ro" -v "$PWD/hf-cache:/hf-cache" \
  -v "$PWD/config.yaml:/app/config.yaml:ro" -v "$PWD/voices.json:/app/voices.json:ro" \
  -e ORPHEUS_MODEL_PATH=/models/orpheus-indic-5679 \
  -p 9000:9000 ai4bharat-orpheus-indic-tts:latest
```

`--shm-size 8g` is required by vLLM's workers. `--init` releases GPU memory held by
the engine subprocess on stop. `--stop-timeout` must exceed `ORPHEUS_DRAIN_TIMEOUT`.

Direct execution is supported but requires a host CUDA toolkit matching the pinned
torch wheels:

```bash
pip install -r requirements.txt -c constraints.txt
PYTHONPATH=src ORPHEUS_MODEL_PATH=./models/orpheus-indic-5679 python -m orpheus_server
```

## Repository layout

```
config.yaml          All settings, documented inline
voices.json          Language, speaker and style roster
.env.example         Environment overrides
requirements.txt     Top-level dependency pins
constraints.txt      Full transitive dependency pins
src/orpheus_server/  Application source
```

`models/` and `hf-cache/` are excluded from version control.

## License

Serving code is provided as-is. Model weights are not included and are licensed
separately by AI4Bharat; review their terms before use, particularly for commercial
deployment.

Components: Orpheus Indic checkpoint ([AI4Bharat](https://ai4bharat.iitm.ac.in/),
IIT Madras) · [Orpheus TTS](https://github.com/canopyai/Orpheus-TTS) (Canopy Labs) ·
[SNAC](https://github.com/hubertsiuzdak/snac) ·
[vLLM](https://github.com/vllm-project/vllm)
