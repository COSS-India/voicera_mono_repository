# Indic-Transcribe (Canary 1.2B)

The second model for the STT slot. Set `STT_MODEL=indic-transcribe` in
`model-server/.env`.

25 Indian languages, and — the reason it is here — **incremental decoding**,
using NeMo's AlignAtt streaming decoder with a Silero VAD in front.

Be precise about what that buys, because it is easy to overstate. Both STT
models return partial transcripts while the caller is still talking; the
pipeline has done that since before the model-server existed. Indic-Conformer
gets there by re-transcribing the whole open segment every 600 ms, so the work
per utterance grows as the utterance does. This model decodes forward from where
it left off, so each new word costs one word.

The gain is latency and GPU cost, not the existence of partial transcripts.

> **This source is not public yet.**
> Read the next section before pushing anything.

## Before you push this repository

`Indic-transcribe-core` is a **private, unreleased** repository. VoicEra's
monorepo is public. Vendoring it here means the first push publishes it.

That was a deliberate decision, made once, with the alternatives on the table
(a submodule, a build-time clone, waiting for the release around 5 September).
Vendoring won because every other model folder is vendored and the whole point
of the folder contract is that a model is a folder you can copy in. But it is
worth knowing three things:

1. **A push cannot be taken back.** Deleting the files later leaves them in git
   history; undoing it properly means rewriting history on a public repo.
2. **The checkpoint is not here and cannot be** — it lives in a private
   HuggingFace repo and needs a token with read access. Publishing the code does
   not publish the weights.
3. **Check the release has happened**, or that its authors are content, before
   the push that includes this folder. If neither is true yet, hold the folder
   back — nothing else in `model-server/` depends on it, and `STT_MODEL` simply
   defaults to `indic-conformer`.

`LICENSE` is upstream's and travels with the code.

## Vendored, not written here

Everything except `README.md`, `compose.extra.yml` and the port change in the
Dockerfile is upstream's, lifted from `COSS-India/Indic-transcribe-core`. Their
own documentation is preserved:

| file | what it covers |
|---|---|
| [UPSTREAM-README.md](UPSTREAM-README.md) | the project as its authors describe it |
| [SETUP.md](SETUP.md) | checkpoint conversion, step by step, with the verification gates |
| [REPORT.md](REPORT.md) | accuracy and latency measurements |
| [LOADTEST.md](LOADTEST.md) | concurrency behaviour under load |

It was called `README.md`, and so is this. On a Windows-backed checkout those
are one file — renaming theirs to `UPSTREAM-README.md` is what stops this page
silently destroying theirs. The same rename was applied to Orpheus and Indic-Mio
after it happened there twice.

The folder is excluded from our ruff config, like every other vendored model:
restyling it would turn each future sync into a merge conflict for no
behavioural gain.

**One change was made**, and it is the same one every model folder gets: the
Dockerfile honours `PORT` so the folder is not welded to our numbering. It
listens on 8001, which is what the STT slot is addressed as.

## Two blockers before this can run

Neither has been tested on hardware. Both are real.

**1. The image is built for a GPU we do not have.** The Dockerfile pins
`torch 2.12.0+cu132` and its base image targets CUDA 13 / sm_120 — Blackwell.
Our H200 is Hopper, sm_90. There is a build gate that asserts
`torch.version.cuda == '13.2'`, so a mismatch fails at build rather than at
runtime, which is the good version of this problem. What is not yet known is
whether the driver on this box serves CUDA 13.2 at all:

```bash
nvidia-smi --query-gpu=driver_version,name --format=csv
```

If it does not, the pins have to come down to a cu12x build — which is a real
piece of work, because the torchaudio situation upstream documents at length
(there is no cu132 torchaudio; they install a CPU build `--no-deps` to dodge a
version check) changes shape at every CUDA version.

**2. The checkpoint needs a private HuggingFace token.** See below.

## Weights: a one-time manual step

There is no `fetch.sh` here, and that is deliberate rather than an omission.
Preparing this model is not a download — it is a download, a conversion, and two
verification gates, and the conversion has to run *inside the built image*
because it needs that image's torch and NeMo. `setup.sh` runs `fetch.sh` before
it builds, so the ordering the contract offers does not fit. Doing it by hand is
honest; a `fetch.sh` that silently could not work is not.

`SETUP.md` is the authority. In outline:

1. Pull the HF checkpoint into `models/core/` (needs the private-repo token).
2. Build the image: `docker compose --profile stt build`.
3. Run `tools/transcribe_hf.py --verify-only` to get a reference transcript and
   its token ids from the HuggingFace implementation.
4. Run `tools/hf_to_nemo.py` to convert — ~4.6 GB, 1926 tensors, vocab 7152.
5. Run `tools/verify_nemo.py --expect-ids` with the ids from step 3. The gate is
   **byte-identical token ids**, not similar text, which is the right gate: a
   conversion where "all the keys matched" can still land weights wrong.

`models/core/` and `artifacts/` are gitignored. `compose.extra.yml` mounts them
in — `/models/core` read-only, `/artifacts` writable — and pins
`HF_HUB_OFFLINE=1`, because an image that cannot fetch should fail fast rather
than hang on a network call it will never be allowed to make.

Override the locations with `CORE_MODELS_DIR` and `CORE_ARTIFACTS_DIR` in
`.env` if the weights live elsewhere on the box.

## What the slot gets

```
POST /v1/audio/transcriptions    OpenAI-shaped, whole utterance
WS   /v1/asr/ws                  live: PCM16 in, JSON partials and finals out
GET  /health                     503 while loading, 200 once warm
GET  /v1/languages               the languages this checkpoint really has
```

The gateway forwards both routes. `/metrics`, `/admin/*` and the demo page at
`/` are not part of the slot contract and are reachable only with
`docker compose exec`.

### Uploads must be a real audio file

This model reads uploads with `soundfile` and answers **415** to headerless
PCM. Indic-Conformer accepted raw int16, so the client used to send it — that
was us being off-spec, not this model being strict: OpenAI's transcriptions
endpoint takes audio files, and a bare PCM stream cannot say its own sample rate.

The client now wraps its buffer in a 44-byte WAV header before uploading, which
both models accept — Indic-Conformer already sniffed for `RIFF`. Pinned by
`tests/test_stt_audio_parity.py`.

### Languages: a superset, so switching is safe

Every language our `STT_LANGUAGE_MAP` can ask for is supported here, and this
model adds Bhojpuri (`bho`) and English (`en`). Switching `STT_MODEL` from
`indic-conformer` to this cannot lose a language an existing agent config uses.

Unlike Indic-Conformer, this model **validates** the language against its
checkpoint and returns 400 with the reason. The client now logs that reason once
per call rather than swallowing it — a rejected language would otherwise look
exactly like a caller who never spoke.

The authoritative list comes from the checkpoint's own `tokenizer_config.json`
at load time; `GET /v1/languages` reports it. The 25 named in the code are not
the 27 in the shared `indic_transcribe.LANGUAGES` tuple — upstream is explicit
that `bgc` and `hne` are not in this vocabulary.

## Streaming, and what still has to be built

The gateway relays `/v1/asr/ws` in both directions
(`tests/test_stt_streaming.py`), so the route is reachable end to end today.

What does **not** exist yet is the Pipecat side: the voice pipeline still uses
the REST path for both models, and produces its own partials from it, because
that works against either one. Moving it onto the socket for models that serve
one is a separate change to `ModelServerSTTService`.

Until that is done, deploying this model is safe but buys little — the client
keeps re-transcribing segments over POST and the incremental decoder goes
unused. The socket work is where this model's advantage actually lands.

## GPU

One uvicorn worker, always. The engine owns the GPU from a single thread; a
second worker would load a second copy of the model and contend for the device.
Budget ~30 GB for the image and weights during setup.
