---
description: Add a model folder and a catalogue entry — nothing else changes.
---

# Adding a model

Adding a model to the model server is two steps, and neither of them is Compose. This page is the contract a model folder has to satisfy, the two optional files it may bring, and what the test suite checks.

## The two steps

1. A folder, `<slot>/<id>/`, containing a `Dockerfile`.
2. An entry in `models.yaml` with the same `id`.

Then `<SLOT>_MODEL=<id>` in `.env`. Nothing in `compose.model-server.yml` or `gateway/` changes, ever. The new folder appears in `setup.sh`'s menu on its own, because that menu is built by listing folders.

A catalogue entry is minimal:

```yaml
llm:
  - id: gemma
    name: Gemma
    status: planned
    runtime: vllm
```

`status` is `ready` — the folder exists with a Dockerfile in it, so `<KIND>_MODEL=<id>` will deploy it — or `planned`, meaning chosen but not built yet.

## The container contract

The container is the contract. Whatever is inside the folder, the image it builds must:

| Requirement | Detail |
| --- | --- |
| listen on its slot's port | STT 8001, TTS 8002, LLM 8003 |
| answer `GET /health` | 2xx once the model is loaded and can serve |
| answer its slot's OpenAI route | `/v1/audio/transcriptions`, `/v1/audio/speech`, or `/v1/chat/completions` |
| stop work when the client hangs up | for TTS this is what makes barge-in free the GPU |
| say what it is sending | TTS only: `X-Audio-Format` and `X-Sample-Rate` on every response |
| accept a WAV upload | STT only |

The image must also honour `PORT`, so the folder is not welded to the slot's numbering. That is the one change every vendored model folder gets.

Nothing is mandated about *what* a TTS model sends, only that it says so. Two TTS models here disagree on the wire — Indic Parler streams 44.1 kHz float32 under the name `pcm_f32le`, Orpheus streams 24 kHz signed 16-bit under OpenAI's own name `pcm` — and the client decodes whichever arrives by reading the headers. A format the client cannot decode produces a clear error naming it, never silence or noise. See [TTS models](tts-models.md).

The STT row is the same principle pointed the other way. Uploads are a real audio file, not a bare PCM stream: `soundfile`-based models answer `415` to headerless bytes, and headerless bytes cannot state their own sample rate anyway. Voicera was the off-spec side here — OpenAI's transcriptions endpoint takes files — so the client wraps its buffer in a 44-byte WAV header, which costs nothing and every model reads.

`/health` returning 503 while a model loads is the useful behaviour, not a defect: it means "healthy" is "will answer fast", not "the process is alive", which is exactly what the gateway's probe wants.

## fetch.sh

A model that needs weights brings its own download step. If `<slot>/<id>/fetch.sh` exists, `setup.sh` runs it — found by existence, so adding one never edits `setup.sh`.

It must resolve paths from its own location and be safe to re-run. It runs **before** the build, which is the right time for a download and the wrong time for anything that needs the built image.

`stt/indic-conformer/fetch.sh` is the model of it: it computes its own directory, checks whether the target file already exists, and returns early if so.

```bash
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$HERE/models/IndicConformer.nemo"
if [ -f "$TARGET" ]; then
  echo "  IndicConformer.nemo already present"
  exit 0
fi
```

Not every model has one, and the absence is sometimes deliberate:

* The vLLM-backed models (`llm/qwen3.5-4b`, `tts/orpheus`, `tts/indic-mio`) have none — vLLM downloads its own weights from HuggingFace into the `hf_cache` volume on first start.
* `stt/indic-transcribe` has none because preparing it is a download, a conversion and two verification gates, and the conversion has to run *inside the built image*. `setup.sh` runs `fetch.sh` before it builds, so the ordering the contract offers does not fit. Doing it by hand is honest; a `fetch.sh` that silently could not work is not.

## compose.extra.yml overlays

A folder may also contain `compose.extra.yml`, an overlay merged on top of the base Compose file. `compose-files.sh` and `setup.sh` both find it by existence.

Two quite different needs turned out to have the same answer:

| Folder | What the overlay does |
| --- | --- |
| `tts/indic-mio/` | brings a vLLM sidecar it delegates token generation to |
| `stt/indic-transcribe/` | mounts the weights its image is forbidden to fetch, and declares engine tuning |

Either way the slot contract is unchanged — one service, one port, one route.

{% hint style="warning" %}
Paths in an overlay resolve against the **project directory** (`model-server/`), not against the overlay's own folder. `additional_contexts` in the base file follows a *different* rule — it resolves against the compose file's directory — so the two cannot be reasoned about interchangeably.
{% endhint %}

That is why `stt/indic-transcribe/compose.extra.yml` spells its mounts from the project root even though the file lives inside the model folder:

```yaml
volumes:
  - ${CORE_MODELS_DIR:-./stt/indic-transcribe/models/core}:/models/core:ro
  - ${CORE_ARTIFACTS_DIR:-./stt/indic-transcribe/artifacts}:/artifacts
```

A folder may also carry its own `compose.mps.yml`, added alongside the shared one only when a daemon is really there. See [Running on GPUs](gpu-operations.md).

## What the tests enforce

The suite runs without a GPU: the model layer is stubbed, everything else is real code.

```bash
pip install -r tests/requirements-dev.txt
pytest tests/ -v
```

For a new model specifically:

| Test | What it enforces |
| --- | --- |
| `test_catalogue.py` | catalogue and folders agree in both directions — a `ready` entry without a folder fails, and a folder nobody catalogued fails too; ids are unique within a kind; profiles stay slot names |
| `test_model_switching.py` | naming a different model really builds a different folder, and the slot's service name and port do not move |
| `test_model_extras.py` | a `compose.extra.yml` merges cleanly and does not disturb its slot's service name, port or route, and the sidecar publishes nothing on the host |
| `test_setup_selection.py` | `setup.sh` offers a menu per slot instead of assuming a model, and runs the chosen model's `fetch.sh` |
| `test_client_selection.py` | every model marked `ready` can actually be named by an agent config, and every name the client accepts has a model behind it |
| `test_tts_format_negotiation.py` | TTS only: the client decodes by declared format, at either sample width |

{% hint style="info" %}
`test_client_selection.py` is the one worth knowing about, because it is the failure with no symptom until a call drops. A model can be catalogued, built, healthy and listed at `/models` while the runtime has never heard of its name. The convention it enforces is `<catalogue id>-<slot>` — folder `orpheus` in `tts/` is `orpheus-tts` to an agent.
{% endhint %}

`test_model_switching`, `test_model_extras` and `test_mps` shell out to `docker compose config`, which interpolates without needing a running daemon; they skip if the `docker` CLI is absent.

Lint applies to the whole tree, with vendored model code excluded path by path in `ruff.toml` rather than by a blanket `stt/**` glob — the files written to fit a model into the slot sit *inside* those folders, so a blanket rule would quietly stop checking Voicera's own code the day someone adds a file.

## A worked example

`llm/qwen3.5-4b/` is the smallest complete model folder in the repo: a Dockerfile and a README, no code at all. vLLM already serves `/v1/chat/completions`, `/v1/models` and `/health` in the shape the gateway forwards, so there is no adapter to write.

Adding a second vLLM model is:

1. Copy `llm/qwen3.5-4b/` to `llm/gemma-3-4b/`.
2. Change `MODEL_ID`, `SERVED_NAME` and `EXTRA_ARGS` in the Dockerfile. `SERVED_NAME` must equal the folder name and the catalogue id — vLLM rejects any request whose `model` field is not the name it was started with.
3. Add the id to `models.yaml` under `llm:`.
4. Set `LLM_MODEL=gemma-3-4b` in `.env` and `docker compose ... up -d --build llm`.

It appears in `setup.sh`'s menu on its own.

Other folders are much larger — `tts/indic-parler/` carries a paged-KV-cache engine — but the interface is the same either way. The size of the folder is a property of the model, not of the contract.

One thing `setup.sh` still knows about a specific model: the AI4Bharat NeMo fork that `indic-conformer` needs. That is a *build context*, so Compose needs its path before the image exists, which is too early for `fetch.sh`. Any model that does not reference the `nemo` context never triggers it.

## Related

* [Slots and models](slots-and-models.md)
* [Gateway API](gateway-api.md)
* [Provider registry](../../guides/concepts/provider-registry.md)
