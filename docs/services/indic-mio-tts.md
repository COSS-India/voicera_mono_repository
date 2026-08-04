# Indic-Mio TTS (on-prem)

Second on-prem TTS provider, parallel to [AI4Bharat](./ai4bharat-tts.md). Serves
[`SPRINGLab/Indic-Mio`](https://huggingface.co/SPRINGLab/Indic-Mio) — 22 scheduled
Indian languages + English, code-mixed, emotion tags — via vLLM + MioCodec.

## Topology

```
voice_2_voice_server ──WS /ttslb2──> mio-tts (8003) ──HTTP /v1──> vllm-mio (8100)
   IndicMioRESTTTSService                │ MioCodec decode              (GPU, 0.5 mem)
                                         └─ float32 PCM ── WS ──> pipeline
```

- **vllm-mio** — `vllm serve SPRINGLab/Indic-Mio --gpu-memory-utilization 0.5`.
  The concurrency engine (continuous batching + paged KV). Scale in-flight with
  `--max-num-seqs`.
- **mio-tts** — thin async WS server (`indic_mio_tts_server/`). Generates via
  vLLM, extracts `<|s_N|>` speech tokens, decodes with MioCodec, streams PCM.
  Both share one GPU (vLLM 0.5, codec in the rest).

## Wire contract

Identical to AI4Bharat TTS. Client → `{"prompt","description","language"}`
(only `prompt` required). Server → `meta` JSON, float32 mono PCM frames, `done`
JSON. `meta.sample_rate` comes from the codec — trust it.

Emotion/stress live in the text: `... <happy>` (sentence end), `*word*` (stress).

## Selecting it

- Agent config `tts_model`: `{"name": "indic-mio", "model": "indic-mio"}`
  (optional `args.emotion`, `speaker`, `description`).
- Voice server factory: `create_tts_service` → `IndicMio` branch
  (`voice_2_voice_server/api/services.py`).
- Language codes: `TTS_LANGUAGE_MAP["IndicMio"]`
  (`voice_2_voice_server/config/tts_mappings.py`).
- Frontend: `voicera_frontend/tts.json` lists `indic-mio` per language.

## Env

| Var | Where | Purpose |
|---|---|---|
| `INDIC_MIO_SERVER_URL` | voice_2_voice_server | WS URL of mio-tts. Prod: `ws://nginx:8080/ttslb2/` |
| `INDIC_MIO_GAIN` | voice_2_voice_server | Optional output gain (default 1.0) |
| `INDIC_MIO_VLLM_URL` | mio-tts | vLLM `/v1` base (compose sets `http://vllm-mio:8100/v1`) |

See `indic_mio_tts_server/.env.example` for the rest.

## Deploy

Compose services `vllm-mio` + `mio-tts` in `prod/` and `deploy/compose/`; nginx
`/ttslb2` in `prod/nginx.prod.conf` and `deploy/nginx/nginx.conf`. Point voice
server `INDIC_MIO_SERVER_URL` at `ws://nginx:8080/ttslb2/` (not `mio-tts:8003`
directly — aiohttp's DNS cache would pin one replica).

## First-boot checks

See `indic_mio_tts_server/README.md` → the `miocodec` decode-API seam,
`skip_special_tokens: false`, and the sample-rate resolution. Confirm the smoke
test (`tests/ws_smoke.py`) yields a valid WAV before wiring the pipeline.
