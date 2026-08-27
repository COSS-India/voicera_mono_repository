# Orpheus Indic TTS

AI4Bharat's Orpheus — a Llama-3.2-3B backbone with a SNAC codec, served by vLLM
with continuous batching. Fills the TTS slot; set `TTS_MODEL=orpheus` in
`model-server/.env`.

22 Indian languages. **The speaker name picks the language** — every speaker in
the roster belongs to exactly one, so `voice="Amit"` is Hindi and
`voice="Anitha"` is Tamil. `GET /v1/voices` lists them. That is different from
Indic Parler, where the voice is a free-text description and language is a
separate field.

## Vendored, not written here

The upstream project's own documentation is preserved here as
[UPSTREAM-README.md](UPSTREAM-README.md) — 522 lines covering the full API,
the speaker roster, styles and tuning. It was called `Readme.md`, which a
Windows checkout treats as the same file as `README.md`; renaming it is what
stops the two clobbering each other.


`src/orpheus_server/` is the upstream project as its authors wrote it, lifted
from the `dev-Orpheustts` branch. It is excluded from our ruff config on
purpose: restyling it would turn every future sync from upstream into a merge
conflict for no behavioural gain.

Two changes were made, both small and both about fitting the slot rather than
changing the model:

1. **Port.** The image listened on 9000; the TTS slot is addressed as `tts:8002`.
   `PORT` is honoured so the folder is not welded to our numbering.
2. **Self-description.** `POST /v1/audio/speech` now sets `X-Audio-Format`,
   `X-Sample-Rate` and `X-Channels`. It already sent `X-Language` and `X-Voice`.
   See below for why this matters.

## Audio format — the reason the headers were added

Two TTS models in this slot disagree on the wire:

| | Indic Parler | Orpheus |
|---|---|---|
| sample rate | 44,100 Hz | 24,000 Hz |
| sample width | float32 | signed 16-bit |
| format name | `pcm_f32le` (an extension) | `pcm` (OpenAI's own name) |

Neither is wrong. OpenAI's `response_format` vocabulary is `mp3`, `opus`, `aac`,
`flac`, `wav`, `pcm` — so Orpheus is the compliant one, and `pcm_f32le` is
something Indic Parler serves because float32 is what its engine produces.

The client therefore cannot assume a width or a rate. It reads `X-Audio-Format`
and `X-Sample-Rate` off the response and decodes accordingly, which is why a
model must declare them. Getting this wrong does not raise an error — it
produces plausible bytes that sound like noise on a phone line.

`tests/test_tts_format_negotiation.py` pins both the decoder table and the
chunk-boundary handling, which is width-dependent: a sample split across two
HTTP reads desynchronises everything after it, and a 2-byte model re-opens that
bug if the width is hardcoded to 4.

## No `fetch.sh`

vLLM downloads the weights from HuggingFace into the `hf_cache` volume on first
start, as with the LLM slot. First start therefore takes several minutes with
nothing on `/health` — watch `docker compose logs -f tts` rather than assuming it
has hung. `/health` returns **503 while loading** and 200 once warmup and CUDA
graph capture have finished, which is exactly what the gateway's probe wants.

## Beyond the OpenAI endpoint

The upstream server also exposes `/v1/tts` (one complete WAV),
`/v1/tts/stream` (a playable URL), a `/v1/tts/ws` WebSocket, `/v1/voices`,
`/v1/styles` and `/metrics`. The gateway forwards only the OpenAI routes, so
those are reachable with `docker compose exec` for debugging but are not part of
the slot contract.

Worth knowing if you touch the streaming path: the authors measured which
formats survive being sent in chunks, and `flac` does not — libsndfile seeks back
and patches the header at close, and that patch never reaches a client whose
first bytes already went out. Only `pcm` and `mp3` stream.

## GPU

vLLM-backed, so it takes a memory reservation at startup the same way the LLM
slot does — see `config.yaml`, where the KV-cache ratios are documented. Roughly
7 GB for a 3B backbone in bf16 plus cache. Not yet run on hardware.
