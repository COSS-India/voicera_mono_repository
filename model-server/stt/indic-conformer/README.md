# Indic Conformer (STT)

AI4Bharat's 600M hybrid RNNT/CTC Conformer, served through NeMo. Fills the STT
slot; set `STT_MODEL=indic-conformer` in `model-server/.env`.

Covers 23 Indic languages. Bhili (`bhb`) uses a separate checkpoint — enable it
with `BHILI_ENABLE=yes` and point `BHILI_NEMO_PATH` at the file. The server
routes on the request's `language` field, so callers use the same endpoint
either way.

## Run

Nothing to run by hand — the slot brings it up:

```bash
cd model-server
docker compose -f compose.model-server.yml --project-directory . up -d --build stt
```

`setup.sh` at the repo root does this as part of a full deployment, and
`fetch.sh` in this folder downloads the checkpoint (~2.4 GB) into `models/`.

## API

Reached through the gateway on `:8100`, never directly — the container binds
nothing on the host.

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/audio/transcriptions` | OpenAI-compatible; multipart `file` plus a `language` field |
| `GET /health` | ready to serve |

## Build context

The image installs the AI4Bharat NeMo fork from a local checkout rather than
cloning during the build, matching production. Compose passes it in as a named
build context; `NEMO_CONTEXT_PATH` in `.env` says where it lives, defaulting to
`~/ai4bharat_nemo`.

```bash
git clone --branch nemo-v2 --depth 1 https://github.com/AI4Bharat/NeMo.git ~/ai4bharat_nemo
```

## Configuration

`INDIC_NEMO_PATH`, `BHILI_ENABLE`, `BHILI_NEMO_PATH`, `HF_TOKEN`, `PORT` — see
`.env.example` here, and `model-server/.env.example` for what Compose passes in.

## GPU

An NVIDIA GPU is strongly recommended; CPU works but is far too slow for a live
call. VRAM depends on the checkpoint and batch settings — measure on staging
rather than trusting a number here. On the H200 this and Indic Parler together
draw roughly 12 GB.
