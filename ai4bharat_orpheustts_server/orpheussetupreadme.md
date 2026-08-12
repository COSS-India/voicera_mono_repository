# AI4Bharat Orpheus Indic TTS

Streaming text-to-speech for **22 Indian languages** and **40 speakers**, served
behind an **OpenAI-compatible API**. One Docker container, one GPU, one config file.

Built on the [AI4Bharat](https://ai4bharat.iitm.ac.in/) Orpheus checkpoint with the
[SNAC](https://github.com/hubertsiuzdak/snac) neural codec, served by
[vLLM](https://github.com/vllm-project/vllm) with continuous batching.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:9000/v1", api_key="not-needed")

client.audio.speech.create(
    model="orpheus",
    voice="Amit",                                  # a Hindi speaker
    input="नमस्ते, आज मौसम बहुत अच्छा है।",
).write_to_file("hello.mp3")
```

That is the whole integration. Any existing OpenAI TTS code works by changing
`base_url`.

---

## Contents

- [What you get](#what-you-get)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Choosing a voice and language](#choosing-a-voice-and-language)
- [API reference](#api-reference)
  - [OpenAI-compatible: `POST /v1/audio/speech`](#openai-compatible-post-v1audiospeech)
  - [Streaming](#streaming)
  - [Native endpoints](#native-endpoints)
  - [WebSocket](#websocket-v1ttsws)
- [Configuration](#configuration)
- [Sizing for your GPU](#sizing-for-your-gpu)
- [Concurrency: what `max_num_seqs` really does](#concurrency-what-max_num_seqs-really-does)
- [Improving performance](#improving-performance)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)
- [Credits and licensing](#credits-and-licensing)

---

## What you get

| | |
|---|---|
| **Languages** | 22 — Hindi, Marathi, Tamil, Telugu, Kannada, Malayalam, Gujarati, Punjabi, Bengali, Assamese, Odia, Urdu, Kashmiri, Sindhi, Maithili, Dogri, Konkani, Bodo, Manipuri, Santali, Nepali, Sanskrit |
| **Speakers** | 40, each belonging to exactly one language |
| **Styles** | 12 — `CONV`, `NEWS`, `WIKI`, `BOOK`, `NAMES`, and more |
| **Audio** | 24 kHz mono; `wav`, `mp3`, `flac`, `opus`, raw `pcm` |
| **APIs** | OpenAI `/v1/audio/speech` (buffered, chunked, and SSE), native REST, WebSocket |
| **Docs** | Swagger UI at `/docs`, ReDoc at `/redoc` |

Measured on one NVIDIA RTX PRO 6000 Blackwell (96 GB) with the shipped FP8 defaults:
**122 ms** to first audio, **0.36** real-time factor, and **64 concurrent streams**
all staying real-time (aggregate 82 audio-seconds per wall-clock second, zero
errors). All 22 languages verified end to end. Your hardware will differ — see
[Sizing for your GPU](#sizing-for-your-gpu) and
[Improving performance](#improving-performance).

## Requirements

- **NVIDIA GPU** with roughly 16 GB of free memory or more. The checkpoint is
  ~7.6 GB; whatever is left becomes KV cache, which is what concurrency is made of.
- **NVIDIA driver** on the host. You do **not** need a CUDA toolkit installed —
  the entire CUDA userspace ships as pip wheels inside the image.
- **Docker** with [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
- **The model checkpoint** — not included. See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).
- About **12 GB of disk**: a 4.4 GB image plus ~7.2 GB of weights and codec cache.

> FP8 is enabled by default and needs **compute capability ≥ 8.9** (Ada, Hopper,
> Blackwell). On an older GPU, set `ORPHEUS_QUANTIZATION=none`. The server prints a
> clear warning at boot if your GPU cannot do what the config asks for.

## Quick start

**1. Get the checkpoint.** Follow [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md) and put it at
`models/orpheus-indic-5679/`. When you are done:

```
models/orpheus-indic-5679/
├── config.json
├── generation_config.json
├── model.safetensors        # 7.6 GB
├── tokenizer.json
└── tokenizer_config.json
```

If what you were given is the **raw training checkpoint** — a `checkpoint-5679/`
folder plus a separate tokenizer folder, ~29 GB in total — those five files have to be
assembled out of it first. [MODEL_SETUP.md](docs/MODEL_SETUP.md#assembling-from-the-raw-training-checkpoint)
has the exact copy commands and the checksums to verify against.

**2. Start the service.**

```bash
mkdir -p hf-cache            # do this before the first `up` — see the note below
cp .env.example .env         # optional; defaults are fine
docker compose up -d --build
docker compose logs -f tts   # watch it load
```

> `hf-cache/` holds the SNAC codec (~76 MB, fetched on first boot) and is a bind
> mount, so it has to exist and be writable by the container, which runs as **uid
> 1000**. If the directory is missing, Docker creates it owned by `root` and the first
> boot fails with a permission error. If your host account is not uid 1000, also run
> `sudo chown 1000:1000 hf-cache`.

First boot takes a while — several minutes. It loads 7.6 GB of weights, captures
CUDA graphs, and pre-compiles GPU kernels for every batch width it expects to
serve ([why](#why-startup-takes-minutes)). `/health` returns **503 until it is
genuinely ready**, so nothing is sent to a half-loaded engine:

```bash
curl -s localhost:9000/health | jq
# {"status":"ok","ready":true,"model":"orpheus-indic","quantization":"fp8", ...}
```

**3. Synthesize.**

```bash
curl -s localhost:9000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"orpheus","voice":"Amit","input":"नमस्ते, आज मौसम बहुत अच्छा है।","response_format":"wav"}' \
  -o hello.wav
```

Then open **<http://localhost:9000/docs>** to try every endpoint from the browser.

## Choosing a voice and language

**The speaker name selects the language.** The model has no language code in its
prompt — it infers the language from the speaker, and every speaker name in
`voices.json` is unique. So `voice="Amit"` is Hindi and `voice="Anitha"` is Tamil,
and OpenAI clients need nothing beyond the standard `voice` field.

```bash
curl -s localhost:9000/v1/languages | jq '.[] | {code, name, voices}'
curl -s 'localhost:9000/v1/voices?language=ta' | jq
curl -s localhost:9000/v1/styles | jq
```

A few to start with:

| Language | Speakers | Language | Speakers |
|---|---|---|---|
| Hindi (`hi`) | Amit | Bengali (`bn`) | see `/v1/voices` |
| Tamil (`ta`) | Anitha, Arun | Marathi (`mr`) | Anagha, Chinmay |

**Text must be in the language's native script.** Romanized input will not produce
correct speech. Each language entry in `/v1/languages` carries a `sample` field with
valid example text.

**Styles** change delivery — `CONV` (default) is conversational, `NEWS` is a news
read, `WIKI` is expository. Pass one as `style`, or as `instructions` (OpenAI's
standard field) when it names a style:

```json
{"model":"orpheus","voice":"Amit","input":"...","instructions":"NEWS"}
```

## API reference

### OpenAI-compatible: `POST /v1/audio/speech`

| Field | Type | Default | Notes |
|---|---|---|---|
| `model` | string | `orpheus` | Accepted for compatibility; one model is served. |
| `input` | string | — | Text, in the native script. |
| `voice` | string | — | Speaker name. Selects the language. |
| `response_format` | enum | `mp3` | `mp3`, `wav`, `flac`, `opus`, `pcm`. Only `pcm` and `mp3` stream — [see below](#streaming). |
| `stream_format` | enum | `audio` | `audio` = audio body, `sse` = event stream. |
| `speed` | float | `1.0` | **Accepted but ignored** — see below. |
| `instructions` | string | — | Used as the style when it names one. |
| `language` | string | — | *Extension.* Only needed if a speaker name is not unique. |
| `style` | string | — | *Extension.* Takes precedence over `instructions`. |
| `max_tokens` | int | `8192` | *Extension.* ~85 ms of audio per token. |

`speed` is accepted so clients that always send it keep working, but it has no
effect: Orpheus has no native rate control, and resampling the output would shift
pitch. A non-default value sets `X-Speed-Ignored` on the response rather than
failing silently.

`GET /v1/models` lists the served model, so SDK model discovery works.

Every buffered response carries its own measurements — `X-TTFA-Ms`, `X-RTF`,
`X-Audio-Duration-Sec`, `X-Generation-Ms`, `X-Language`, `X-Voice`. These are
per-request, not process-global, so they stay correct under concurrency.

### Streaming

Two independent things are called "streaming", and both work:

**1. Chunked audio (`stream_format: "audio"`, the default).** Audio bytes are sent
as they are generated. A client calling `create()` just gets the whole body; a
client that wants bytes early reads the response incrementally:

```python
with client.audio.speech.with_streaming_response.create(
    model="orpheus", voice="Amit", input="…", response_format="pcm",
) as response:
    for chunk in response.iter_bytes():
        play(chunk)          # first chunk arrives in ~120 ms
```

**2. OpenAI SSE (`stream_format: "sse"`).** Server-sent events carrying base64
audio, terminated by a `done` event with usage and timings:

```
data: {"type":"speech.audio.delta","audio":"<base64>"}
data: {"type":"speech.audio.delta","audio":"<base64>"}
data: {"type":"speech.audio.done","usage":{...},"audio":{...},"timings":{...}}
```

**Which formats actually stream.** Only two, and this was measured rather than
assumed — see the note below, because the intuitive answer is wrong:

| Format | Delivery | Streams? |
|---|---|---|
| `pcm` | chunked, 4096 B per 85 ms frame | ✅ **best for real-time** |
| `mp3` | chunked, 546–1066 B per frame | ✅ (adds ~80 ms of encoder padding) |
| `wav` | one body, exact RIFF length | ❌ buffered by design |
| `flac` | one body | ❌ cannot be chunked |
| `opus` | one body | ❌ cannot be chunked |

All five formats return correct, fully valid audio. The difference is only *when*
the bytes arrive. **For low latency, ask for `pcm` or `mp3`.**

<details>
<summary>Why <code>flac</code> and <code>opus</code> cannot stream (worth knowing before you try to "fix" it)</summary>

libsndfile seeks back and **patches the file header when the file is closed** —
FLAC's `STREAMINFO` total-sample count, Ogg's page structure. That patch is
invisible if you inspect the finished buffer, but in a stream those first bytes
were sent long ago and the correction can never reach the client.

Measured by concatenating the emitted chunks and decoding *that* (rather than the
final buffer, which silently includes the patch):

| Format | Chunk-concatenation decodes? |
|---|---|
| `pcm` | yes |
| `mp3` | yes — the patched Xing header only carries trim hints |
| `flac` | **no** — the unpatched sample count makes decoders reject it |
| `opus` | yes, but libsndfile emits ~nothing until close, so there is nothing to stream |

So they are encoded in one pass and returned as a complete file. `wav` is buffered
for the same family of reasons: a RIFF header declares its length up front, and
buffering makes that length truthful so tools like Python's `wave` report the real
duration. If you specifically want a streaming WAV, use
[`GET /v1/tts/stream`](#native-endpoints) — it sends a max-length header and streams
the body.

</details>

### Native endpoints

Beyond the OpenAI surface, because OpenAI's schema has nowhere to put 22 languages:

| Endpoint | Purpose |
|---|---|
| `GET /v1/languages` | Languages, speakers, and sample text |
| `GET /v1/voices?language=ta` | Speakers, optionally filtered |
| `GET /v1/styles` | The 12 styles and the default |
| `POST /v1/tts` | Complete WAV with timing headers |
| `GET /v1/tts/stream` | Streaming WAV as a plain URL |
| `GET /health` | Readiness — 503 until loaded and warm |
| `GET /metrics` | Aggregate counters |

`GET /v1/tts/stream` is a GET so it can be used as a URL directly:

```html
<audio controls src="http://localhost:9000/v1/tts/stream?voice=Amit&text=नमस्ते"></audio>
```

### WebSocket: `/v1/tts/ws`

The lowest-latency path, for voice agents. Not in Swagger — OpenAPI cannot describe
WebSockets.

Send one JSON request, receive a `start` frame, then raw PCM frames, then a `done`
frame with that stream's metrics:

```python
import asyncio, json, websockets

async def speak(text, voice="Amit"):
    async with websockets.connect("ws://localhost:9000/v1/tts/ws", max_size=None) as ws:
        await ws.send(json.dumps({"text": text, "voice": voice}))
        async for msg in ws:
            if isinstance(msg, bytes):
                play(msg)                      # 24 kHz mono s16le, one 85 ms frame
            else:
                event = json.loads(msg)
                if event["type"] == "done":
                    print(event["metrics"])    # ttfa_ms, rtf, jitter_p99_ms, ...
                elif event["type"] == "error":
                    raise RuntimeError(event["message"])

asyncio.run(speak("नमस्ते, आज मौसम बहुत अच्छा है।"))
```

Request fields: `text`, `voice`, optional `language`, `style`, `max_tokens`. A bad
request returns an `error` frame instead of dropping the connection.

## Configuration

[`config.yaml`](config.yaml) is the source of truth and documents every key inline,
including what changes if you tune it. Every key also has an `ORPHEUS_*` environment
variable, which **wins over the file** — so a deployment can be retuned without
editing or rebuilding anything.

`config.yaml` and `voices.json` are bind-mounted by Compose, which makes tuning a
**restart**, not a rebuild:

```bash
docker compose restart tts
```

The knobs you are most likely to touch:

| Variable | Default | Effect |
|---|---|---|
| `ORPHEUS_MAX_NUM_SEQS` | `256` | Concurrent-stream admission. [Read this before changing it.](#concurrency-what-max_num_seqs-really-does) |
| `ORPHEUS_QUANTIZATION` | `fp8` | `none` on GPUs below compute capability 8.9. |
| `ORPHEUS_GPU_MEMORY_UTILIZATION` | `0.90` | Lower when sharing the GPU; up to ~0.95 when dedicated. |
| `ORPHEUS_MAX_MODEL_LEN` | `8192` | Context window. Lower it to fit more streams. |
| `ORPHEUS_MAX_TOKENS_DEFAULT` | `8192` | Default length cap. **Too low silently truncates.** |
| `ORPHEUS_ENFORCE_EAGER` | `false` | `true` saves VRAM and boot time, decodes slower. |
| `ORPHEUS_TENSOR_PARALLEL_SIZE` | `1` | Shard across GPUs. |
| `ORPHEUS_DECODER_DEVICE` | `cuda` | Move the codec to another GPU, e.g. `cuda:1`. |
| `ORPHEUS_DECODER_MAX_BATCH` | `256` | Keep ≥ `MAX_NUM_SEQS`. |
| `ORPHEUS_WARMUP_ENABLED` | `true` | `false` boots fast, moves the cost to real traffic. |
| `ORPHEUS_WARMUP_WIDTHS` | `1,…,256` | Batch widths to pre-compile. |
| `ORPHEUS_MODEL_PATH` | `/models/orpheus-indic-5679` | Local dir or HF repo id. |
| `ORPHEUS_PORT` | `9000` | Published port. |

[`.env.example`](.env.example) carries these, ready to uncomment.
[`config.yaml`](config.yaml) documents **every** key, each with its `ORPHEUS_*` name —
including the ones you rarely touch (sampling, warmup width, SNAC model id, CORS).

**Nothing in the source code is specific to a GPU model.** vLLM dispatches kernels
for whatever card it finds. The only hardware-aware code in this repository reads
your compute capability at boot and *warns* if the config asks for something
impossible — it never changes behaviour behind your back.

## Sizing for your GPU

The defaults target a large datacentre GPU, because this is meant to be shared
across teams. If yours is smaller, change these two things first:

**GPUs with compute capability ≥ 8.9** (RTX 40-series, L4/L40S, H100, Blackwell) —
keep FP8. Measured against BF16 on the identical checkpoint:

| Metric | FP8 | BF16 | FP8 advantage |
|---|---|---|---|
| Decode rate, single stream | **239.2 tok/s** | 164.7 tok/s | +45% |
| Time to first audio | **118 ms** | 171 ms | 31% lower |
| Real-time factor | **0.343** | 0.498 | 31% lower |
| KV cache available | **75.8 GiB** | 73.4 GiB | +2.4 GiB |

Word error rate was unchanged within measurement noise. FP8 is simply better where
it is supported.

**GPUs below 8.9** (A100, A10, T4, V100) — turn FP8 off:

```bash
ORPHEUS_QUANTIZATION=none
```

**Smaller cards (16–24 GB)** — the weights need ~7.6 GB in BF16, so there is less
left for KV cache. Reduce demand rather than fighting it:

```bash
ORPHEUS_QUANTIZATION=none          # or fp8 if capability >= 8.9, which halves weights
ORPHEUS_MAX_NUM_SEQS=16            # fewer concurrent streams
ORPHEUS_MAX_MODEL_LEN=4096         # ~50 s of audio; less KV cache per stream
ORPHEUS_MAX_TOKENS_DEFAULT=4096    # keep in step with max_model_len
ORPHEUS_DECODER_MAX_BATCH=16
ORPHEUS_WARMUP_WIDTHS=1,2,4,8,16
ORPHEUS_ENFORCE_EAGER=true         # last resort: skips CUDA graphs, decodes slower
```

> Only the Blackwell configuration above has been measured. The other paths are
> correct by construction — they are ordinary vLLM settings, not special cases —
> but the numbers on your card will be your own.

## Concurrency: what `max_num_seqs` really does

This is the setting people reach for first and misunderstand most, so it is worth
being precise:

> **`max_num_seqs` is an admission policy, not a throughput control.**

Raising it does not make the GPU faster. It changes what happens to traffic the GPU
cannot keep up with. A low value makes surplus requests **wait in a queue**. A high
value **admits them all** and makes every stream slower. Total throughput is set by
the hardware either way.

Measured on one RTX PRO 6000 Blackwell (96 GB, FP8) under identical over-capacity load:

| `max_num_seqs` | Delivered audio-s/s | Started ≤ 500 ms | Streams with **no** late frame |
|---|---|---|---|
| 8 | 15.6 | 5.1% | **100%** |
| 16 | 24.3 | 9.7% | **100%** |
| 32 | 34.1 | 21.0% | **100%** ← last stutter-free value |
| 48 | 38.5 | 42.6% | 33.5% |
| 64 | 41.5 | 67.0% | 24.4% |
| 96 | 42.3 | 95.5% | 28.4% |
| 128 | 42.4 | 98.3% | 26.7% |

Two things to read off that table. Throughput **plateaus near 42 audio-seconds per
second** from 96 upward — that is the GPU's ceiling under this load, and no setting
raises it. And past 32, the GPU stops making streams *wait* and starts making them
*stutter*.

**A burst behaves better than sustained overload.** The table above is the harsh
case: arrivals keep coming faster than the GPU can serve them, forever. A burst of
N streams that then stops is easier, and the shipped configuration was measured
separately on it (all N started simultaneously, `max_num_seqs=256`, one long Hindi
sentence each, a frame counted late if it misses its playback slot after a 100 ms
prebuffer):

| Concurrent burst | Errors | Aggregate audio-s/s | TTFA p50 | TTFA p95 | Worst RTF | Stutter-free streams |
|---|---|---|---|---|---|---|
| 1 | 0 | 2.8 | 143 ms | 143 ms | 0.35 | 100% |
| 8 | 0 | 17.9 | 146 ms | 146 ms | 0.42 | 100% |
| 32 | 0 | 52.9 | 192 ms | 194 ms | 0.55 | 100% |
| 64 | 0 | 81.8 | 236 ms | 258 ms | 0.73 | 100% |
| 128 | 0 | 107.8 | 313 ms | 365 ms | **1.09** | **0%** |

Zero failures at every level, and the knee sits between 64 and 128 rather than at
32. The aggregate rate is higher than the 42 above because a finite burst gets the
full benefit of batching without ever falling behind — **do not read these two
tables as competing measurements of the same thing.** The first tells you what
happens when demand outruns the GPU indefinitely; the second tells you what a
finite burst costs. Plan capacity with the first and expect the second.

Practical reading: **64 concurrent is the most this GPU serves with every stream
smooth.** Past that, throughput keeps climbing while individual streams degrade.

**So choose by what your clients do with the audio:**

| Setting | Behaviour | Right when |
|---|---|---|
| **`256`** (default) | Nothing queues; everything starts at once and shares the GPU | Clients **buffer before playing** — batch jobs, file generation, an app that downloads then plays. A late frame costs nothing if nobody is listening yet. |
| **`64`** | Every stream in a finite burst stays smooth (measured above) | **Bursty live playback** — traffic arrives in waves and then quiets down. |
| **`32`** | Every admitted stream stays real-time even under sustained overload; surplus queues | **Continuous live playback** — voice agents, phone lines, where demand never lets up and a gap is audible. |

The default of 256 is above the highest value in that table. It admits more; it does
not deliver more than ~42 audio-s/s. That is the intended trade for a shared,
buffer-then-play service.

**One caveat if you raise it:** surplus requests queue **silently**. A client can
wait a long time with no signal other than latency. Set client-side timeouts.

## Improving performance

Work through these in order — the first two are usually where the wins are.

**1. Decide what you are optimizing.** Throughput (total audio per second) and
latency (time to first audio for one stream) pull in opposite directions.
`max_num_seqs` is the lever between them, and the table above is the map.

**2. Turn on FP8 if your GPU supports it.** +45% decode rate, 31% lower latency,
and more KV cache. Biggest single win available, and it is one config line.

**3. Give vLLM more memory.** `ORPHEUS_GPU_MEMORY_UTILIZATION` up towards `0.95` on
a dedicated card. Everything above the weights becomes KV cache, and KV cache is
what lets streams run concurrently. Leave headroom for the SNAC codec (~0.5 GB) and
the CUDA context.

**4. Shorten the context if your text is short.** `ORPHEUS_MAX_MODEL_LEN=4096`
halves the KV cache each stream reserves, so more fit. Keep
`ORPHEUS_MAX_TOKENS_DEFAULT` in step or you will truncate long inputs.

**5. Keep `DECODER_MAX_BATCH` ≥ `MAX_NUM_SEQS`.** Audio decoding is batched across
all live streams into one GPU call. If this ceiling is below your admission limit,
decode starts competing with generation and streams lose real-time well before the
engine is actually full. The server warns at boot if you get this wrong.

**6. Move the codec to a second GPU.** With more than one card,
`ORPHEUS_DECODER_DEVICE=cuda:1` takes audio decoding off the generation GPU
entirely.

**7. Keep warmup on, and covering your real width.** `ORPHEUS_WARMUP_WIDTHS` should
reach `MAX_NUM_SEQS`. GPU kernels compile the first time each batch width appears;
if warmup never exercised your busiest width, the first burst that does pays a
multi-second stall mid-stream.

**8. Split text into sentences client-side for long documents.** Latency is
dominated by output length. Several short requests pipeline through continuous
batching better than one long one, and the user hears audio sooner.

**9. Scale out, not up.** One GPU tops out somewhere around 42 audio-s/s under
sustained overload (107 in a finite burst); no setting changes that ceiling. Beyond
it, run more replicas behind a load balancer. `tensor_parallel_size` shards one
model across GPUs — it buys KV-cache headroom, not raw speed, and for a 7.6 GB model
that fits on one card it is rarely the right tool.

**What to measure.** `X-TTFA-Ms` and `X-RTF` on each response, and the WebSocket
`done` frame's `jitter_p99_ms`. RTF below 1.0 means faster than real time. Watch
`streams_active` in `/metrics` against your `max_num_seqs` to see whether you are
actually saturated or just queueing.

### Why startup takes minutes

Roughly 150 s goes to loading 7.6 GB of weights and capturing CUDA graphs. Warmup
then runs a small synthesis at each width in `ORPHEUS_WARMUP_WIDTHS`, because vLLM's
sampling kernels JIT-compile the first time each batch width is seen. Paying that at
boot — where nobody is waiting — beats paying it as a multi-second stall in a real
user's first sentence.

To boot faster while iterating on config: `ORPHEUS_WARMUP_WIDTHS=1,4` or
`ORPHEUS_WARMUP_ENABLED=false`.

## Troubleshooting

**`/health` returns 503 for a long time.** Normal for the first few minutes.
`docker compose logs -f tts` shows load and warmup progress. If it never becomes
ready, look for an out-of-memory error and lower
`ORPHEUS_GPU_MEMORY_UTILIZATION`.

**"no CUDA device visible".** The container cannot see the GPU. Check
`nvidia-container-toolkit` is installed and `nvidia-smi` works on the host, then
confirm the `deploy.resources.reservations.devices` block in `docker-compose.yml`.

**A warning about `quantization=fp8` and compute capability.** Your GPU cannot do
native FP8. Set `ORPHEUS_QUANTIZATION=none`.

**Out of memory at startup.** Lower `ORPHEUS_GPU_MEMORY_UTILIZATION`, then
`ORPHEUS_MAX_MODEL_LEN`, then set `ORPHEUS_ENFORCE_EAGER=true`. Also check nothing
else holds GPU memory (`nvidia-smi`).

**Audio cut off mid-sentence.** `max_tokens` was hit. Each token is ~85 ms, so 8192
is ~100 s. Raise `ORPHEUS_MAX_TOKENS_DEFAULT` (and `ORPHEUS_MAX_MODEL_LEN` with it),
or pass a larger `max_tokens` per request.

**Empty or very short audio.** Check the text is in the language's native script —
romanized input does not work. Compare against the `sample` text in
`/v1/languages`.

**`unknown voice 'X'`.** `GET /v1/voices` lists all 40. Names are case-sensitive.

**Requests hang for a long time under load.** You are over capacity and requests are
queueing silently. See [the concurrency section](#concurrency-what-max_num_seqs-really-does).

**Model files not found.** `ORPHEUS_MODEL_PATH` is a path *inside the container*;
Compose mounts `./models` at `/models`. See [docs/MODEL_SETUP.md](docs/MODEL_SETUP.md).

**Permission denied writing `/hf-cache`, or a failed SNAC download on first boot.**
The directory was created by Docker as `root`. Fix the ownership and restart:

```bash
sudo chown -R 1000:1000 hf-cache && docker compose restart tts
```

**Stuttering audio during live playback.** `max_num_seqs` is too high for
real-time delivery. Set `ORPHEUS_MAX_NUM_SEQS=32`.

## Project layout

```
.
├── config.yaml              # every tunable, documented inline
├── voices.json              # 22 languages, 40 speakers, 12 styles
├── docker-compose.yml
├── Dockerfile
├── requirements.txt         # top-level pins
├── constraints.txt          # full transitive pin set (reproducible builds)
├── docs/MODEL_SETUP.md      # how to obtain and place the checkpoint
└── src/orpheus_server/
    ├── config.py            # YAML + env loading, hardware advisory
    ├── voices.py            # roster, voice -> language resolution
    ├── prompt.py            # prompt construction as token ids
    ├── codec.py             # SNAC: token arithmetic, batched decode
    ├── engine.py            # vLLM lifecycle, warmup, PCM generation
    ├── audio.py             # wav/pcm/mp3/flac/opus encoding
    ├── app.py               # FastAPI assembly
    └── api/
        ├── openai_speech.py # /v1/audio/speech, /v1/models
        ├── native.py        # catalog, /v1/tts, /v1/tts/stream, WebSocket
        ├── meta.py          # /health, /metrics
        └── deps.py          # shared application state
```

### Running the image without Compose

Compose supplies four things the container needs; if you run `docker run` by hand,
supply them yourself. In particular `ORPHEUS_MODEL_PATH` must be the **container**
path — the default in `config.yaml` is repo-relative and will not find a bind mount,
and anything that is not an existing directory is treated as a HuggingFace repo id:

```bash
docker build -t ai4bharat-orpheus-indic-tts:latest .
mkdir -p hf-cache

docker run -d --name orpheus-indic-tts \
  --gpus all --shm-size 8g --init \
  -v "$PWD/models:/models:ro" \
  -v "$PWD/hf-cache:/hf-cache" \
  -v "$PWD/config.yaml:/app/config.yaml:ro" \
  -v "$PWD/voices.json:/app/voices.json:ro" \
  -e ORPHEUS_MODEL_PATH=/models/orpheus-indic-5679 \
  -p 9000:9000 \
  ai4bharat-orpheus-indic-tts:latest
```

`--shm-size 8g` matters: vLLM's workers talk over shared memory and Docker's 64 MB
default is not enough. `--init` reaps the engine subprocess so nothing keeps holding
GPU memory after the container stops.

### Running without Docker

```bash
pip install -r requirements.txt -c constraints.txt
export PYTHONPATH=src
export ORPHEUS_MODEL_PATH=./models/orpheus-indic-5679
python -m orpheus_server
```

You will need a CUDA toolkit whose `nvcc` matches the installed torch wheels, since
vLLM's Triton kernels JIT-compile against it. The Docker image pins `CUDA_HOME`
for exactly this reason — Docker is the supported path.

## Credits and licensing

- **Model** — Orpheus Indic checkpoint by [AI4Bharat](https://ai4bharat.iitm.ac.in/), IIT Madras
- **Orpheus TTS** — [Canopy Labs](https://github.com/canopyai/Orpheus-TTS)
- **SNAC codec** — [hubertsiuzdak/snac](https://github.com/hubertsiuzdak/snac)
- **Serving** — [vLLM](https://github.com/vllm-project/vllm)

The serving code in this repository is provided as-is. **The model weights are not
included and carry their own license** — check AI4Bharat's terms before using them,
especially commercially. Nothing here grants rights to the checkpoint.
