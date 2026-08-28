# model-server

Every self-hosted model in one place, behind one gateway.

```
model-server/
├── gateway/               the only published port    :8100
├── stt/                   speech-to-text slot        :8001 (internal)
│   └── indic-conformer/
├── tts/                   text-to-speech slot        :8002 (internal)
│   ├── indic-parler/
│   └── orpheus/
├── llm/                   language-model slot        :8003 (internal)
│   └── qwen3.5-4b/
├── models.yaml            catalogue of every model, served at /models
└── tests/                 run without a GPU
```

Three **slots**, each holding as many **models** as you have folders for. A slot
is one container on a fixed port; which model fills it is a folder name in
`.env`. Adding a model is adding a folder — no service, no port, no gateway
change, which is what makes swapping one a one-line edit rather than a project.

## Running it

```bash
cp .env.example .env          # pick which model fills each slot
docker compose -f compose.model-server.yml up -d --build
```

`setup.sh` at the repo root does this for you as part of a full deployment, and
asks which model should fill each slot:

```
  Speech to text
    1) indic-conformer
    0) none
  Choose [1]:
```

The list is the folders in `stt/`, `tts/` and `llm/` — so a model you add shows
up in the installer without anyone editing the installer. Set `STT_MODEL`,
`TTS_MODEL` or `LLM_MODEL` in the environment to skip a menu and run unattended.

`.env` answers two separate questions:

```bash
COMPOSE_PROFILES=stt,tts      # which slots run at all
STT_MODEL=indic-conformer     # which folder under stt/ fills the slot
TTS_MODEL=indic-parler
LLM_MODEL=qwen3.5-4b
```

The profile is the slot name, never a model name. Keeping those jobs apart is
what lets you change model without touching anything that starts containers.

Switching model, in full:

```bash
sed -i 's/^LLM_MODEL=.*/LLM_MODEL=gemma-3-4b/' .env
docker compose -f compose.model-server.yml up -d --build llm
```

The service is still called `llm`, still on 8003, so the gateway never learns
that anything changed. `STT_MODEL` and friends are also what tell the gateway a
slot is deployed — one variable, so Compose and the gateway cannot disagree
about what is running.

## Endpoints

| | |
|---|---|
| `POST /v1/audio/transcriptions` | speech to text, one segment per request |
| `WS /v1/asr/ws` | speech to text, incremental — when the deployed model serves it |
| `POST /v1/audio/speech` | text to speech |
| `POST /v1/chat/completions` | LLM, when a model is deployed |
| `GET /models` | every model, and which are running |
| `GET /v1/models` | OpenAI-compatible: only what can be called right now |
| `GET /health` | gateway and upstreams |

`/models` and `/v1/models` differ on purpose. OpenAI clients read `/v1/models`
as "what can I call", so it must never list something that would answer 503.

`/v1/asr/ws` is the one route that is not OpenAI-shaped, because OpenAI has no
equivalent. It exists because live transcription is genuinely two-directional:
audio flows in for as long as someone is talking while partial transcripts flow
back, and neither side knows when the other will speak next. TTS is not like
that -- it is one-directional, so it moved *off* WebSockets to plain HTTP, which
gave cancellation for free. Direction of travel decides the transport, not
fashion.

**This route is not what makes transcription live.** Every STT model here
returns partial transcripts while the caller is still speaking, and always has —
that is a telephony requirement, not a feature. What differs is where the
partials come from:

| | how partials are produced | cost |
|---|---|---|
| `indic-conformer` | the client re-transcribes the open segment every 600 ms (`AI4BHARAT_INTERIM_MS`) over the POST route | grows with utterance length |
| `indic-transcribe` | the model decodes incrementally over the WebSocket | one word costs one word |

So a model without the WebSocket route is not a model that waits for you to
finish. It is a model whose partials the client produces on its behalf, by
brute force. `models.yaml` records both facts separately — `partial_transcripts`
for what the caller gets, `streaming_endpoint` for what the model serves —
because collapsing them into one `streaming:` flag misled a reader once already.

`tests/test_partial_transcripts.py` pins the client-side path, which is what
production runs today.

All three follow the OpenAI shape, TTS included. Barge-in still works: when the
caller interrupts, Pipecat stops reading the response, the connection drops, and
that drop travels through the gateway to the TTS server, which frees the GPU
slot. `POST /v1/audio/speech` streams raw float32 PCM as it is generated, with
the sample rate in an `X-Sample-Rate` header rather than assumed by the client.

## Adding a model

Two steps, and neither of them is Compose:

1. A folder, `<slot>/<id>/`, containing a `Dockerfile`
2. An entry in `models.yaml` with the same `id`

Then `<SLOT>_MODEL=<id>` in `.env`. Nothing in `compose.model-server.yml` or
`gateway/` changes, ever. Tests enforce both directions — a `ready` catalogue
entry without a folder fails, and a folder nobody catalogued fails too.

### What a model folder has to provide

The container is the contract. Whatever is inside the folder, the image it
builds must:

| | |
|---|---|
| listen on | its slot's port — STT 8001, TTS 8002, LLM 8003 |
| answer `GET /health` | 2xx once the model is loaded and can serve |
| answer its slot's OpenAI route | `/v1/audio/transcriptions`, `/v1/audio/speech`, or `/v1/chat/completions` |
| stop work when the client hangs up | for TTS this is what makes barge-in free the GPU |
| say what it is sending | TTS only: `X-Audio-Format` and `X-Sample-Rate` on every response |
| accept a WAV upload | STT only — see below |

That last row is what keeps the slot model-agnostic. Two TTS models here
disagree on the wire — Indic Parler streams 44.1 kHz float32 under the name
`pcm_f32le`, Orpheus streams 24 kHz signed 16-bit under OpenAI's own name `pcm` —
and the client decodes whichever arrives by reading the headers. Nothing is
mandated about *what* a model sends, only that it says so. A format the client
cannot decode produces a clear error naming it, never silence or noise.

The STT row is the same principle pointed the other way. Uploads are a real
audio file, not a bare PCM stream: `soundfile`-based models answer 415 to
headerless bytes, and headerless bytes cannot state their own sample rate
anyway. We were the off-spec side here -- OpenAI's transcriptions endpoint takes
files -- so the client now wraps its buffer in a 44-byte WAV header, which costs
nothing and every model reads.

Optionally, a folder may also contain either of two files, both found by
existence so that adding one never edits `setup.sh`:

**`fetch.sh`** — a model that needs weights brings its own download step. It
must resolve paths from its own location and be safe to re-run. It runs *before*
the build, which is the right time for a download and the wrong time for
anything that needs the built image.

**`compose.extra.yml`** — an overlay merged on top of the base Compose file. Two
quite different needs turned out to have the same answer: `tts/indic-mio/` uses
it to bring a vLLM sidecar it delegates token generation to, and
`stt/indic-transcribe/` uses it to mount the weights its image is forbidden to
fetch. Either way the slot contract is unchanged -- one service, one port, one
route. Paths in an overlay resolve against the **project directory**
(`model-server/`), not against the overlay's own folder.

That is the whole interface. Some folders are a full server — `tts/indic-parler/`
carries a paged-KV-cache engine. Some are a Dockerfile and nothing else:
`llm/qwen3.5-4b/` is 30 lines, because vLLM already serves the spec, so there is
no adapter to write. Copy that folder, change the model name and flags, and the
new model is done — it appears in `setup.sh`'s menu on its own.

One thing setup.sh still knows about a specific model: the AI4Bharat NeMo fork
that `indic-conformer` needs. That is a *build context*, so Compose needs its
path before the image exists, which is too early for `fetch.sh`. Any model that
does not reference the `nemo` context never triggers it.

## Tests

```bash
pip install -r tests/requirements-dev.txt
pytest tests/ -v
```

No GPU needed — the model layer is stubbed, everything else is real code. They
cover the things this setup could break quietly: audio surviving the trip
unchanged, `voice` + `instructions` recomposing into exactly the prompt the model
used to get, a chunk boundary splitting a float without corrupting the audio, the
gateway streaming rather than buffering, a disconnect actually stopping
generation, the KV page allocator never handing one page to two calls, naming a
different model actually building a different folder, the LLM's model id meaning
the same string in all four files that have to agree on it, an undeployed slot
refusing calls clearly instead of hanging, two TTS models with different sample
widths both decoding correctly, a live transcription socket relaying both frame
types both ways and carrying a hang-up through to the model, and every model
marked `ready` actually being nameable by an agent config.

That last one is worth a sentence, because it is the failure with no symptom
until a call drops: a model can be catalogued, built, healthy and listed at
`/models` while the voice server has never heard of its name. Deploying it looks
like success right up to the first agent that asks for it.

## Current state

Verified on ace-h200 (26 Aug), running beside the prod and translate stacks:

| | |
|---|---|
| Both models on GPU 1 | via the shared MPS daemon, ~12.3 GB |
| TTS time-to-first-audio | 1.5 s cold, **~250 ms warm** |
| Realtime factor | 0.69x |
| Round trip | TTS speaks a sentence, STT transcribes it back word for word |
| Effect on prod | none — `voicera-prod` stayed `running(11)` throughout |

Not yet tested: a real call through `voice_2_voice_server`. That needs a second
voice server pointed at the gateway via `MODEL_SERVER_URL`, plus an agent
configured for `indic-conformer-stt` and `indic-parler-tts`.

**Not yet run on hardware at all: the LLM slot.** `llm/qwen3.5-4b/` is written
but has never been built or started, so the vLLM flags in it are unverified
against a live model. Treat the numbers above as covering STT and TTS only.

### Gotchas that cost us time

- **`ai4bharat/indic-parler-tts` is a gated HuggingFace repo.** The Parler
  checkpoint comes from elsewhere, but the tokenizer and T5 encoder do not.
  Either supply a token with access, or read a cache that already has them
  (what `HF_CACHE_VOLUME` + `HF_HUB_OFFLINE` do here).
- **Weights are not in the repo.** `stt/indic-conformer/models/IndicConformer.nemo`
  and `tts/indic-parler/checkpoints/` are gitignored. Fetch or copy them before
  building. They live inside the model's own folder, which is what the slot
  bind-mounts.
- **`additional_contexts` paths resolve against the compose file, not the build
  context.** The NeMo fork path stays `../../ai4bharat_nemo` even though the
  model folders moved a level deeper. A test pins this.
- **Build one image at a time on a tight disk.** Building both in parallel
  doubles peak usage at the export stage, which is where it fails.

The model containers publish nothing on the host, so this stack can run beside
others without competing for ports. To reach a model directly for debugging, use
`docker compose exec` or add a temporary `ports:` mapping.

Pick the GPU with `GPU_DEVICE_IDS` in `.env`. Check `nvidia-smi` first: avoid any
GPU in `Exclusive Process` mode, and avoid splitting a tensor-parallel pair.
