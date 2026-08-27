---
description: Self-hosted STT, TTS and LLM behind one OpenAI-compatible gateway on port 8100.
---

# Model server

The model server is an **optional** component that runs speech and language
models on your own hardware instead of calling hosted APIs. It replaces the
separate AI4Bharat STT and TTS servers that earlier versions ran on ports 8001
and 8002, and the standalone vLLM launcher on 8003.

Everything now sits behind **one published port, 8100**, speaking the OpenAI
API. Existing OpenAI clients work against it unchanged, and the voice server
needs a single setting — `MODEL_SERVER_URL` — rather than one per modality.

## Slots and models

There are three **slots**: `stt`, `tts` and `llm`. A slot is one container on a
fixed internal port. Which **model** fills it is a folder name:

```
model-server/
├── gateway/                the only published port    :8100
├── stt/
│   └── indic-conformer/    AI4Bharat Indic Conformer 600M
├── tts/
│   └── indic-parler/       AI4Bharat Indic Parler TTS
├── llm/
│   └── qwen3.5-4b/         Qwen3.5-4B on vLLM
└── models.yaml             the catalogue, served at /models
```

Two questions, answered separately in `model-server/.env`:

```bash
COMPOSE_PROFILES=stt,tts       # which slots run at all
STT_MODEL=indic-conformer      # which folder fills each one
TTS_MODEL=indic-parler
LLM_MODEL=                     # empty = that slot is not deployed
```

The profile is always a slot name, never a model name. Keeping those apart is
what lets you change model without touching anything that starts containers.

A slot with no model deployed answers **503** naming the slot, rather than 404 —
so a misconfigured deployment is distinguishable from a wrong URL. It does not
mark the gateway unhealthy: choosing not to run an LLM is a valid configuration,
not a fault.

## Endpoints

All on `:8100`. The model containers bind nothing on the host, so they are
reachable only through the gateway or `docker compose exec`.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/audio/transcriptions` | Speech to text. Multipart `file` plus a `language` field. |
| `POST /v1/audio/speech` | Text to speech. Streams raw float32 PCM as it generates. |
| `POST /v1/chat/completions` | LLM. Streams SSE. |
| `GET /models` | The whole catalogue, and which models are running. |
| `GET /v1/models` | OpenAI-compatible: only what can be called right now. |
| `GET /health` | Gateway plus each slot. |

`/models` and `/v1/models` differ deliberately. OpenAI clients read `/v1/models`
as "what can I call", so it must never list something that would answer 503.
`/models` is the fuller picture, including models that are catalogued but not
deployed.

### Audio format

`POST /v1/audio/speech` returns raw PCM, not a container format. The response
carries `X-Sample-Rate` (44100 for Indic Parler) and `X-Audio-Format`. **Read
the rate off the header rather than assuming it** — a different TTS model will
report a different one.

Chunked HTTP can split a 4-byte float across two reads, so a client that
concatenates chunks and decodes blindly gets noise partway through. Carry the
remainder forward to the next chunk.

### Interrupting the bot

Barge-in works by hanging up: the client stops reading the response, the
connection drops, that drop reaches the TTS server, and it evicts the request
from its batch. The GPU slot is freed immediately rather than finishing a
sentence nobody is listening to. No special protocol is involved — this is
ordinary HTTP cancellation.

## Running it

```bash
cd model-server
cp .env.example .env          # pick a model for each slot
docker compose -f compose.model-server.yml --project-directory . up -d --build
curl localhost:8100/health
```

`setup.sh` at the repository root does this as part of a full deployment and
asks which model should fill each slot, listing whatever folders exist.

First build takes 20–40 minutes. Weights are several GB and are downloaded on
first start, so a slot can look hung when it is working — watch
`docker compose logs -f` rather than waiting on `/health`.

### Switching model

```bash
sed -i 's/^LLM_MODEL=.*/LLM_MODEL=gemma-3-4b/' .env
docker compose -f compose.model-server.yml --project-directory . up -d --build llm
```

The service is still called `llm` and still on 8003 internally, so the gateway
never learns anything changed. Nothing else in the stack is touched.

### Adding a model

Two steps, and neither is editing Compose:

1. A folder, `<slot>/<id>/`, containing a `Dockerfile`
2. An entry in `models.yaml` with the same `id`

Then `<SLOT>_MODEL=<id>` in `.env`. Tests enforce both directions — a catalogue
entry with no folder fails, and a folder nobody catalogued fails too.

The container is the whole contract. Whatever is inside the folder, the image
must listen on its slot's port (STT 8001, TTS 8002, LLM 8003 — internal only),
answer `GET /health` once loaded, answer its slot's OpenAI route, and stop work
when the client hangs up. An optional `fetch.sh` in the folder downloads weights;
`setup.sh` runs it if present.

Some folders hold a full server — `tts/indic-parler/` carries a paged-KV-cache
inference engine. Some hold nothing but a Dockerfile: `llm/qwen3.5-4b/` is about
30 lines, because vLLM already serves the OpenAI spec and there is no adapter to
write.

## Models currently shipped

| Slot | Model | Notes |
|------|-------|-------|
| `stt` | `indic-conformer` | AI4Bharat Indic Conformer 600M via NeMo. 23 Indic languages; Bhili (`bhb`) uses a second checkpoint, enabled with `BHILI_ENABLE=yes`. |
| `tts` | `indic-parler` | AI4Bharat Indic Parler. Voice chosen by free-text description rather than a preset list. Needs a HuggingFace token — the tokenizer and T5 encoder are in a gated repo. |
| `llm` | `qwen3.5-4b` | Qwen3.5-4B on vLLM. Off by default. |

Each folder has its own README with the model-specific detail — flags, GPU
notes, and known upstream bugs. That is deliberately not repeated here, so there
is one place to correct when it changes.

## GPU

One GPU serves all deployed slots. Pick it with `GPU_DEVICE_IDS` in `.env`.
Check `nvidia-smi` first: avoid a GPU in `Exclusive Process` mode unless you are
attaching as an MPS client, and avoid splitting a tensor-parallel pair.

Sizing is best measured rather than predicted. On an H200, Indic Conformer and
Indic Parler together draw roughly 12 GB. For vLLM, `VLLM_GPU_MEMORY_UTILIZATION`
is a fraction of the card's **total** memory, not of what is free — it is a hard
reservation taken at startup, so on a shared GPU an oversized value takes memory
from whatever else is running rather than failing cleanly.

## Related

- [Voice server](voice-server.md) — the client of these endpoints
- [Ports and defaults](../reference/ports-and-defaults.md)
- [Environment variables](../reference/environment-variables.md)
- `model-server/README.md` in the repository — the maintainer-facing detail
