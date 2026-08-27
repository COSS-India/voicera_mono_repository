---
description: Canonical reference for every environment variable used across VoicEra services, grouped by service with defaults and requirement flags.
---

# Environment Variables

Every VoicEra service is configured through environment variables, typically loaded from a per-service `.env` file. This page is the canonical list. For port-level defaults see [ports-and-defaults.md](ports-and-defaults.md); for credentials, see [../quickstart/default-credentials.md](../quickstart/default-credentials.md).

{% hint style="warning" %}
**Vobiz Auth ID / Token are not env vars in production.** They are stored per-organization in the database via **Dashboard -> Integrations** and consumed at call time by `fetch_integration_key(org_id, ...)`. The env entries below exist only for legacy single-tenant dev setups. See [../concepts/telephony-model.md](../concepts/telephony-model.md).
{% endhint %}

## Configuration files

| Service | File |
|---------|------|
| Backend | `voicera_backend/.env` (template: `voicera_backend/env.example`) |
| Voice server | `voice_2_voice_server/.env` |
| Frontend | `voicera_frontend/.env.local` |
| Model server (optional) | `model-server/.env` |

In Docker Compose deployments the same files are mounted via `env_file:` in `docker-compose.yml`. Service-name aliases (e.g. `mongodb`, `minio`, `backend`) resolve inside the `voicera_network` bridge.

---

## Backend (`voicera_backend/.env`)

### Database

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `MONGODB_HOST` | backend | `mongodb` (Docker) / `localhost` | yes | MongoDB hostname |
| `MONGODB_PORT` | backend | `27017` | yes | MongoDB port |
| `MONGODB_USER` | backend | `admin` | yes | MongoDB username — change in production |
| `MONGODB_PASSWORD` | backend | `admin123` | yes | MongoDB password — change in production |
| `MONGODB_DATABASE` | backend | `voicera` | yes | Database name |
| `MONGODB_AUTH_SOURCE` | backend | `admin` | no | Authentication database |

### Object storage (MinIO)

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `MINIO_ENDPOINT` | backend | `minio:9000` | yes | MinIO host:port |
| `MINIO_ACCESS_KEY` | backend | `minioadmin` | yes | Access key — change in production |
| `MINIO_SECRET_KEY` | backend | `minioadmin` | yes | Secret key — change in production |
| `MINIO_SECURE` | backend | `false` | no | Set `true` when MinIO is behind TLS |

### Security and auth

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `SECRET_KEY` | backend | `secret_key` | yes | JWT signing secret — must be changed in production |
| `INTERNAL_API_KEY` | backend | – | yes | Shared secret for voice-server-to-backend calls. Generate with `python -c "import secrets; print(secrets.token_urlsafe(32))"` |
| `DEBUG` | backend | `False` | no | Enable verbose error responses |

### Email (Mailtrap)

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `MAILTRAP_API_TOKEN` | backend | – | no | Mailtrap API token for transactional email |
| `MAILTRAP_FROM_EMAIL` | backend | `noreply@voicera.com` | no | From address |
| `MAILTRAP_FROM_NAME` | backend | `VoicEra` | no | From name |
| `FRONTEND_URL` | backend | `http://localhost:3000` | no | Used in password-reset links |

### Vobiz (legacy / dev only)

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `VOBIZ_API_BASE_URL` | backend | – | no | Vobiz API base URL for application CRUD |
| `VOBIZ_ACCOUNT_ID` | backend | – | no | Single-tenant dev fallback |
| `VOBIZ_AUTH_ID` | backend | – | no | Single-tenant dev fallback (prefer Integrations) |
| `VOBIZ_AUTH_TOKEN` | backend | – | no | Single-tenant dev fallback (prefer Integrations) |

### RAG / Knowledge base

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `CHROMA_BASE_DIR` | backend | `voicera_backend/rag_system/chroma_data` | no | Override Chroma persistence root |

### Application

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `FRONTEND_URL` | backend | `http://localhost:3000` | no | Used in password-reset email links |
| `VOICE_SERVER_URL` | backend | `http://localhost:7860` | no | Voice server URL for outbound call proxy; also accepts `JOHNAIC_SERVER_URL` as fallback |
| `BATCH_SCHEDULER_POLL_SECONDS` | backend | `5` | no | Polling interval for the outbound batch scheduler |

---

## Voice server (`voice_2_voice_server/.env`)

### Public URLs

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `JOHNAIC_SERVER_URL` | voice server | – | yes (prod) | Public HTTPS base for webhooks (e.g. `https://voice.example.com`) |
| `JOHNAIC_WEBSOCKET_URL` | voice server | – | yes (prod) | Public WSS base for the audio stream |

`JOHNAIC_*` is the **public voice server URL**, not a third-party product. For local development use an ngrok tunnel; see [../guides/developer/local-setup.md](../guides/developer/local-setup.md).

### Telephony — Vobiz

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `VOBIZ_API_BASE` | voice server | `https://api.vobiz.in/v1` | yes | Vobiz API base URL |
| `VOBIZ_CALLER_ID` | voice server | – | no | Default outbound caller ID |

### Backend integration

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `VOICERA_BACKEND_URL` | voice server | `http://backend:8000` | yes | Backend API URL (Compose alias) |
| `INTERNAL_API_KEY` | voice server | – | yes | Must match the backend's `INTERNAL_API_KEY` |
| `BACKEND_API_TIMEOUT` | voice server | `30` | no | Request timeout in seconds |

### Object storage

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `MINIO_ENDPOINT` | voice server | `minio:9000` | yes | MinIO host:port |
| `MINIO_ACCESS_KEY` | voice server | `minioadmin` | yes | Access key |
| `MINIO_SECRET_KEY` | voice server | `minioadmin` | yes | Secret key |
| `MINIO_SECURE` | voice server | `false` | no | Use HTTPS |

### Provider API keys (fallback)

Provider selection is per-agent in MongoDB. Per-org keys are stored in **Dashboard → Integrations** and take priority. The env vars below are fallbacks for local dev or single-tenant installs.

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `OPENAI_API_KEY` | voice server | – | conditional | Fallback when org has no OpenAI Integration |
| `DEEPGRAM_API_KEY` | voice server | – | conditional | Fallback Deepgram key |
| `SARVAM_API_KEY` | voice server | – | conditional | Fallback Sarvam key |
| `ELEVENLABS_API_KEY` | voice server | – | conditional | Fallback ElevenLabs key |
| `XAI_API_KEY` | voice server | – | conditional | Grok (xAI) key |
| `BHASHINI_API_KEY` | voice server | – | conditional | Bhashini STT/TTS key |
| `VLLM_API_KEY` | voice server | – | conditional | vLLM server API key |
| `VLLM_BASE_URL` | voice server | – | conditional | vLLM base URL |
| `GOOGLE_STT_CREDENTIALS_PATH` | voice server | `credentials/google_stt.json` | conditional | Google STT service-account JSON |
| `GOOGLE_TTS_CREDENTIALS_PATH` | voice server | `credentials/google_tts.json` | conditional | Google TTS service-account JSON |
| `MODEL_SERVER_URL` | voice server | – | conditional | Model server gateway, e.g. `http://localhost:8100`. One URL for STT, TTS and LLM — the gateway routes on the endpoint. From inside a container use `http://host.docker.internal:8100`. |
| `KENPATH_JWT_PRIVATE_KEY_PATH` | voice server | – | conditional | RS256 private key for Kenpath Vistaar |

### Server, audio, logging

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `HOST` | voice server | `0.0.0.0` | no | Bind address |
| `PORT` | voice server | `7860` | no | Listen port |
| `SAMPLE_RATE` | voice server | `8000` | no | Telephony wire sample rate: `8000` = µ-law, `16000` = L16 PCM |
| `MAX_CONCURRENT_CALLS` | voice server | `100` | no | Concurrency cap |
| `LOG_LEVEL` | voice server | `INFO` | no | Logging level |
| `DEBUG_MODE` | voice server | `false` | no | Verbose pipeline logs |
| `ENABLE_AUDIO_LOGGING` | voice server | `false` | no | Log raw audio (CPU intensive) |

---

## Frontend (`voicera_frontend/.env.local`)

Public frontend env vars must be prefixed `NEXT_PUBLIC_` to be exposed to the browser bundle.

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | frontend | – | yes | Voice server base — Vobiz/Plivo answer URLs and **Test on Browser** WebSocket (`http`→`ws`, `https`→`wss`). See `lib/johnaic-config.ts`. |
| `VOICE_SERVER_URL` | frontend | `http://localhost:7860` | no | Voice server URL for server-side Next.js routes (outbound call, telemetry) |
| `API_URL` | frontend (Docker) | `http://localhost:8000` | no | Backend URL for Next.js `/api/*` proxies. Hardcoded to `http://localhost:8000` in `lib/api-config.ts` when unset. Set to `http://backend:8000` in Docker Compose. |

---

## Model server (`model-server/.env`)

| Variable | Service | Default | Required | Description |
|----------|---------|---------|----------|-------------|
| `COMPOSE_PROFILES` | model-server | – | yes | Which slots run, by slot name: `stt,tts,llm` |
| `STT_MODEL` | model-server | – | no | Folder under `model-server/stt/`; empty means the slot is not deployed |
| `TTS_MODEL` | model-server | – | no | Folder under `model-server/tts/` |
| `LLM_MODEL` | model-server | – | no | Folder under `model-server/llm/` |
| `GATEWAY_PORT` | model-server | `8100` | no | The only published port |
| `GPU_DEVICE_IDS` | model-server | `1` | no | Which GPU the model containers claim |
| `HF_TOKEN` | model-server | – | conditional | Needed for models pulling from a gated HuggingFace repo |
| `NEMO_CONTEXT_PATH` | model-server | `../../ai4bharat_nemo` | conditional | Local checkout of the AI4Bharat NeMo fork, used as a build context by `indic-conformer` |
| `VLLM_MAX_MODEL_LEN` | model-server | `8192` | no | Context cap for the LLM slot |
| `VLLM_MAX_NUM_SEQS` | model-server | `20` | no | Concurrent sequences for the LLM slot |
| `VLLM_GPU_MEMORY_UTILIZATION` | model-server | `0.10` | no | Fraction of the card's **total** memory vLLM reserves at startup |
| `VLLM_QUANTIZATION` | model-server | – | no | Empty means bf16 |

`model-server/.env.example` is the annotated list. Per-model settings live in
`model-server/<slot>/<model>/`, not here.

### Slot variables passed through to containers

Compose sets these from `model-server/.env`; they are per-slot, not per-model.

| Name | Slot | Default | Required | Description |
|------|------|---------|----------|-------------|
| `PORT` | all | `8001` / `8002` / `8003` | no | The slot's port **inside** the model-server network. Nothing binds these on the host. |
| `INDIC_NEMO_PATH` | stt | `/app/models/IndicConformer.nemo` | conditional | Path inside the container to the main Indic checkpoint (`indic-conformer`) |
| `BHILI_ENABLE` | stt | `no` | no | `yes` to load the Bhili checkpoint as well |
| `BHILI_NEMO_PATH` | stt | – | conditional | Path inside the container to the Bhili checkpoint |
| `CHECKPOINT_PATH_DEFAULT` | tts | `/app/checkpoints` | no | Where the TTS model finds its weights |
| `HF_HUB_OFFLINE` | tts | `0` | no | `1` when reading a cache that already holds gated files, so HuggingFace skips the check that would 401 |
| `HUGGING_FACE_HUB_TOKEN` | all | – | conditional | Set from `HF_TOKEN`; needed for gated repos |

Batching constants are model code, not configuration: STT uses `MAX_BATCH_SIZE=16`
and `BATCH_TIMEOUT=0.1s` in `model-server/stt/<model>/server.py`, and the LLM slot
takes `VLLM_MAX_NUM_SEQS` from `.env`.

---

## Generating secrets

```bash
# JWT / INTERNAL_API_KEY
python -c "import secrets; print(secrets.token_urlsafe(32))"
openssl rand -hex 32

# MinIO credentials
openssl rand -base64 32
```

## Validation checklist

- All `Required: yes` variables are set.
- `INTERNAL_API_KEY` is identical between backend and voice server.
- `MONGODB_PASSWORD`, `MINIO_SECRET_KEY`, `SECRET_KEY` are changed from defaults before any exposure to a network.
- CORS `allow_origins` in `voicera_backend/app/main.py` is restricted to your real frontend origin (currently hardcoded as `["*"]` — tighten before production).
- `JOHNAIC_*` URLs are HTTPS / WSS in production.
- AI provider keys (OpenAI, Deepgram, Cartesia) belong to non-trial accounts in production.

For hardening guidance, see [../guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md).

## Next steps

- [ports-and-defaults.md](ports-and-defaults.md) — Port map and default URLs
- [../quickstart/default-credentials.md](../quickstart/default-credentials.md) — Out-of-the-box passwords
- [../guides/deployment/docker-compose.md](../guides/deployment/docker-compose.md) — How env vars flow into Compose
- [../guides/deployment/production.md](../guides/deployment/production.md) — Production checklist
