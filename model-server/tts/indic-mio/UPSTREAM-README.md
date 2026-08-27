<!-- Upstream documentation, kept verbatim from the feat/indic-mio-tts-v1
     branch. Renamed so it cannot collide with the slot README beside it;
     Windows checkouts treat README.md and Readme.md as one file.
     Note: it still describes the WebSocket protocol, which this folder no
     longer speaks. README.md has the current contract. -->

# Indic-Mio TTS server

On-prem TTS for [`SPRINGLab/Indic-Mio`](https://huggingface.co/SPRINGLab/Indic-Mio)
(22 scheduled Indian languages + English, code-mixed, emotion tags). Runs in
parallel to the AI4Bharat (Parler) TTS server and speaks the identical WebSocket
contract, so the voice pipeline treats it as just another provider.

## Two-stage pipeline

Indic-Mio is a Qwen3-0.6B fine-tune, not a classic acoustic model:

1. **Token generation — vLLM.** `vllm serve SPRINGLab/Indic-Mio` generates *speech
   tokens*. vLLM owns the concurrency (continuous batching + paged KV). This
   server does **not** hold the LLM.
2. **Codec decode — MioCodec.** This server turns the content-token indices into a
   44.1 kHz waveform and streams it as float32 PCM.

```
voice pipeline ──WS──> server.py (this) ──HTTP /v1/chat/completions──> vllm-mio
                            │                                            (GPU, 0.5 mem)
                            └── MioCodec.decode() ── float32 PCM ──> WS
```

## Run locally

```bash
# 1) token generator (leaves VRAM for the codec on the same GPU)
vllm serve SPRINGLab/Indic-Mio --gpu-memory-utilization 0.5 --port 8100

# 2) this server
INDIC_MIO_VLLM_URL=http://localhost:8100/v1 python server.py

# 3) smoke test -> writes a wav
python tests/ws_smoke.py
```

## Wire contract

Client sends one JSON per utterance; `prompt` is the only required field:

```json
{"prompt": "नमस्ते <happy>", "description": "...", "language": "hi"}
```

Server replies: a `meta` JSON frame, then binary float32 mono PCM frames, then a
`done` JSON frame (`error` on failure). `sample_rate` in `meta` is read from the
codec at load — trust it, do not assume 44100.

- **Emotion / stress**: put the tags in `prompt` (`<happy>` at the end, `*word*`
  for stress). The model reads them from the text; nothing special server-side.
- `description` / `language` are accepted for contract parity but currently
  informational — Indic-Mio picks voice/script from the text itself.

## Performance model

This server shares **nothing** with the AI4Bharat/Parler server except the WS
wire format (a transport interface, so the pipeline adapter is a drop-in). There
is no hand-written batching loop and no per-process model copy — the two things
that made the Parler server slow and prone to OOM-on-load.

- **Concurrency & scale** live entirely in vLLM (continuous batching + paged KV).
  Scale up = `--max-num-seqs` / more `mio-tts` + `vllm-mio` capacity. `vllm-mio`
  is launched with `--max-model-len 1024 --dtype bfloat16` to cap KV memory (fast
  load, more concurrent sequences, no OOM-on-load).
- **Low TTFB**: generation streams token-by-token and the codec decodes
  incrementally (`MIO_STREAM_DECODE=true`), pushing PCM as it is produced instead
  of waiting for the whole utterance. Each incremental decode uses the full token
  prefix and holds back a short look-ahead tail (`MIO_LOOKAHEAD_TOKENS`) so
  emitted audio always had enough right context — no flush-boundary artifacts.
  Kill-switch: `MIO_STREAM_DECODE=false` → single whole-utterance decode.

## Config

All env-driven — see `.env.example` and `config.py`. Key ones:
`INDIC_MIO_VLLM_URL`, `MIO_LLM_MODEL`, `MIO_CODEC_MODEL_ID`, `MIO_DECODE_CONCURRENCY`,
`MIO_STREAM_DECODE`, `MIO_FLUSH_TOKENS`, `MIO_LOOKAHEAD_TOKENS`.

## Speaker embedding (required by the codec)

`MioCodecModel.decode(global_embedding, content_token_indices)` requires a speaker
`global_embedding` — the LLM only produces speaker-independent content tokens. On
first boot the engine downloads one Indic-Mio reference sample
(`MIO_REFERENCE_REPO`/`MIO_REFERENCE_FILE`), encodes it once to a `global_embedding`,
and caches the vector at `MIO_SPEAKER_EMBED_PATH` (in the shared HF volume). Every
decode reuses it; the encoder/reference is never touched again. To use a different
default voice, delete the cached `.pt` (or point `MIO_SPEAKER_EMBED_PATH` at another
precomputed embedding) and restart.

## Environment

miocodec is a **PyPI-torch** project (hard-imports `torchaudio`). The image uses
plain `torch==2.5.1 + torchaudio==2.5.1` (CUDA wheels), **not** the NGC PyTorch
base — that base's custom torch has no ABI-compatible torchaudio. GPU comes from the
nvidia container runtime.

## Deploy-time seams to confirm on first boot

- **`skip_special_tokens: false`** is sent to vLLM — required, else the `<|s_N|>`
  speech tokens are stripped and no audio is produced.
- **Sample rate** is read from `codec.config.sample_rate` (the `25Hz-24kHz` codec
  is 24 kHz), reported in the `meta` frame; the pipeline resamples as needed.
- **First boot is slower**: it downloads the codec, the reference sample, and the
  SSL encoder bundle (for the one-time embedding). All cached in the HF volume.
