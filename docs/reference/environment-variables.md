---
description: Canonical reference for every environment variable used across VoicEra services, grouped by service with defaults and requirement flags.
---

# Environment Variables

Every VoicEra service is configured through environment variables, typically loaded from a per-service `.env` file. This page is the canonical list. For port-level defaults see [ports-and-defaults.md](ports-and-defaults.md); for credentials, see [../quickstart/default-credentials.md](../quickstart/default-credentials.md).

{% hint style="warning" %}
**Vobiz Auth ID / Token are not env vars in production.** They are stored per-organization in the database via **Dashboard -> Integrations** and consumed at call time by `fetch_integration_key(org_id, ...)`. The env entries below are development fallbacks for single-tenant setups; use Dashboard → Integrations for all production deployments. See [../concepts/telephony-model.md](../concepts/telephony-model.md).
{% endhint %}

## Configuration files

| Service | File |
|---------|------|
| Backend | `voicera_backend/.env` (template: `voicera_backend/env.example`) |
| Voice server | `voice_2_voice_server/.env` |
| Frontend | `voicera_frontend/.env.local` |
| AI4Bharat STT (optional) | `ai4bharat_stt_server/.env` |
| AI4Bharat TTS (optional) | `ai4bharat_tts_server/.env` |

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

### Vobiz (development fallback)

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
| `GOOGLE_STT_CREDENTIALS_PATH` | voice server | `voice_2_voice_server/credentials/google_stt.json` | conditional | Google STT service-account JSON |
| `GOOGLE_TTS_CREDENTIALS_PATH` | voice server | `voice_2_voice_server/credentials/google_tts.json` | conditional | Google TTS service-account JSON |
| `AI4BHARAT_STT_URL` | voice server | – | conditional | Local AI4Bharat STT base URL |
| `AI4BHARAT_TTS_URL` | voice server | – | conditional | Local AI4Bharat TTS base URL |
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

All public frontend env vars must be prefixed `NEXT_PUBLIC_` to be exposed to the browser bundle.

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | frontend | `http://localhost:8000` | yes | Backend REST base URL (browser-side) |
| `API_URL` | frontend | – | no | Backend URL for server-side routes; falls back to `NEXT_PUBLIC_API_URL` |
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | frontend | – | yes (prod) | Public HTTPS voice server URL — used for Vobiz answer URLs and browser test |
| `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL` | frontend | – | no | Explicit `wss://` base for **Test on Browser**; derived from `NEXT_PUBLIC_JOHNAIC_SERVER_URL` if absent |
| `VOICE_SERVER_URL` | frontend | `http://localhost:7860` | no | Voice server URL for server-side Next.js routes |

---

## AI4Bharat STT (`ai4bharat_stt_server/.env`)

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `PORT` | ai4bharat-stt | `8001` | no | Listen port |
| `INDIC_NEMO_PATH` | ai4bharat-stt | – | yes | On-disk path to the main Indic NeMo checkpoint |
| `BHILI_ENABLE` | ai4bharat-stt | `no` | no | `yes` to load the Bhili checkpoint |
| `BHILI_NEMO_PATH` | ai4bharat-stt | – | conditional | On-disk path to the Bhili checkpoint |
| `HF_TOKEN` | ai4bharat-stt | – | no | HuggingFace token for gated checkpoints |

For the batch size and timeout constants (`MAX_BATCH_SIZE=16`, `BATCH_TIMEOUT=0.1s`) see `ai4bharat_stt_server/server.py`.

## AI4Bharat TTS (`ai4bharat_tts_server/.env`)

| Name | Service | Default | Required | Description |
|------|---------|---------|----------|-------------|
| `PORT` | ai4bharat-tts | `8002` | no | Listen port |
| `AUDIO_SAMPLE_RATE` | ai4bharat-tts | `44100` | no | Output sample rate (Hz); Parler TTS default |

Refer to `ai4bharat_tts_server/.env.example` for the active variable set — checkpoint and batching knobs vary by deployment.

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
