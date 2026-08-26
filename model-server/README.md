# model-server

Every self-hosted model in one place, behind one gateway.

```
model-server/
├── gateway/     the only published port    :8100
├── stt/         AI4Bharat Indic Conformer  (internal)
├── tts/         AI4Bharat Indic Parler     (internal)
├── llm/         vLLM (off by default)      (internal)
├── models.yaml  catalogue of every model, served at /models
└── tests/       run without a GPU
```

## Running it

```bash
cp .env.example .env          # pick which model fills each slot
docker compose -f compose.model-server.yml up -d --build
```

`setup.sh` at the repo root does this for you as part of a full deployment.

Which models start is decided by `.env`:

```bash
STT_MODEL=indic-conformer
TTS_MODEL=indic-parler
LLM_MODEL=                    # empty = that slot is not deployed
```

That one file drives both which containers start and what the gateway believes
is available. Keeping it as the only source of truth matters: if the two ever
disagree, the gateway reports a model as deployed that was never started.

## Endpoints

| | |
|---|---|
| `POST /v1/audio/transcriptions` | speech to text |
| `WS /v1/audio/speech` | text to speech |
| `POST /v1/chat/completions` | LLM, when a model is deployed |
| `GET /models` | every model, and which are running |
| `GET /v1/models` | OpenAI-compatible: only what can be called right now |
| `GET /health` | gateway and upstreams |

`/models` and `/v1/models` differ on purpose. OpenAI clients read `/v1/models`
as "what can I call", so it must never list something that would answer 503.

**TTS is a WebSocket, not the OpenAI REST shape.** Barge-in has to kill a
generation mid-stream, and closing a socket does that for free. The OpenAI spec
has no equivalent, so this is a deliberate exception; STT and the LLM follow the
spec normally.

## Adding a model

1. An entry in `models.yaml`
2. A folder with a `Dockerfile`
3. A service in `compose.model-server.yml` with `profiles: [<id>]`

Nothing in `gateway/` changes. A test enforces that anything marked `ready` in
the catalogue actually has a profile to start it.

## Tests

```bash
pip install -r tests/requirements-dev.txt
pytest tests/ -v
```

No GPU needed — the model layer is stubbed, everything else is real code. They
cover the things this setup could break quietly: audio surviving the trip
unchanged, both TTS request formats producing identical speech, the gateway
streaming rather than buffering, a disconnect actually stopping generation, and
the KV page allocator never handing one page to two calls.

## Current state

Working, but **never run on a GPU**. The images have not been built. Until that
happens this is a tested design, not a proven deployment.

The model containers publish nothing on the host, so this stack can run beside
others without competing for ports. To reach a model directly for debugging, use
`docker compose exec` or add a temporary `ports:` mapping.

Pick the GPU with `GPU_DEVICE_IDS` in `.env`. Check `nvidia-smi` first: avoid any
GPU in `Exclusive Process` mode, and avoid splitting a tensor-parallel pair.
