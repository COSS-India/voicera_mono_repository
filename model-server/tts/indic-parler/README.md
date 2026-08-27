# Indic Parler (TTS)

AI4Bharat's Indic Parler TTS with a custom inference engine — paged KV cache,
continuous batching, CUDA graphs, and incremental DAC decoding. Fills the TTS
slot; set `TTS_MODEL=indic-parler` in `model-server/.env`.

Covers the same 23 Indic languages as the STT model. The voice is chosen by a
free-text description rather than a fixed preset list, so `voice` and
`instructions` are recomposed into one prompt on the server.

## Run

Nothing to run by hand — the slot brings it up:

```bash
cd model-server
docker compose -f compose.model-server.yml --project-directory . up -d --build tts
```

`fetch.sh` in this folder downloads the checkpoints into `checkpoints/`.

## API

Reached through the gateway on `:8100`, never directly.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/audio/speech` | OpenAI-compatible; streams raw float32 PCM as it generates |
| `GET /health` | ready to serve |

The response carries `X-Sample-Rate` (44100) and `X-Audio-Format`. Clients must
read the rate off the header rather than assume it. Chunked HTTP can split a
4-byte float across reads, so a client that concatenates chunks blindly gets
noise — carry the remainder forward.

Hanging up mid-sentence is how barge-in works: the client stops reading, the
connection drops, and the server evicts the request from the batch, freeing the
GPU slot immediately rather than finishing a sentence nobody is listening to.

## The gated repo

The Parler checkpoint comes from Drive, but the tokenizer and T5 encoder are
pulled from `ai4bharat/indic-parler-tts`, which is **gated**. The container will
not start without either `HF_TOKEN` set to an account with access, or a
HuggingFace cache that already holds those files:

```bash
docker compose -f compose.model-server.yml -f compose.shared-hf-cache.yml ...
```

That overlay mounts an existing cache read-only and sets offline mode, which
stops HuggingFace making the gated check that would 401.

## GPU

NVIDIA, Ada generation or newer — the engine leans on CUDA graphs and flashinfer.
VRAM scales with concurrent utterances; measure on staging. Warm
time-to-first-audio on an H200 is around 250 ms, roughly 1.5 s cold.
