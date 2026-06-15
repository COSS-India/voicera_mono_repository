<div align="center">

# VoicERA

**Open-source framework for building production-ready voice AI agents.**

Build, test, and deploy real-time voice agents — try them in your **browser first**, then
connect production telephony (Plivo / Vobiz) when you're ready. Provider-agnostic across
LLM, STT, and TTS, with built-in knowledge bases (RAG) and self-hostable Docker deployment.

<!-- TODO(branding): add VoicERA logo here once available (docs/assets/logo.png) -->
<!-- TODO(demo): add a hero demo GIF of the in-browser test flow (docs/assets/demo.gif) -->

![Browser Testing](https://img.shields.io/badge/Browser_Testing-✓-brightgreen)
![Plivo](https://img.shields.io/badge/Plivo-supported-blue)
![Vobiz](https://img.shields.io/badge/Vobiz-supported-blue)
![RAG](https://img.shields.io/badge/Knowledge_Base-RAG-orange)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED)
![Self Hostable](https://img.shields.io/badge/Self_Hostable-✓-brightgreen)
![License](https://img.shields.io/badge/License-MIT-green)

**LLM:** OpenAI · Anthropic · Grok (xAI) · Qwen (vLLM) · Kenpath  
**STT:** Deepgram · Google · OpenAI · ElevenLabs · Sarvam · AI4Bharat · Bhashini  
**TTS:** ElevenLabs · Cartesia · OpenAI · Google · Deepgram · Sarvam · AI4Bharat · Bhashini

</div>

---

> [!NOTE]
> VoicERA is **browser-first**: you can talk to a working voice agent from the dashboard
> without any telephony account, public URL, or webhooks. Telephony (Plivo / Vobiz) is the
> next step in the journey, not a prerequisite.

![Architecture](docs/assets/architecture.png)

<!-- TODO(screenshots): add dashboard + browser-test-dialog screenshots here -->

---

## Why VoicERA?

Most voice-AI projects require telephony setup, webhooks, public URLs, and multiple provider
configurations **before you can even test an agent**. VoicERA flips that: build and test
agents directly in the browser first, then connect production telephony when ready.

- **Browser-first** — talk to your agent in the dashboard, no phone number required.
- **Provider-agnostic** — mix and match LLM / STT / TTS providers per agent.
- **Production-ready** — Plivo and Vobiz telephony with inbound, outbound, and recording.
- **Knowledge-aware** — attach documents and let agents answer from them (RAG).
- **Self-hostable** — runs entirely on your own infrastructure via Docker Compose.
- **Multi-tenant** — organization-scoped agents, integrations, and documents.
- **Open source** — MIT licensed.

---

## Features

### Voice Agents
- Real-time, low-latency conversations
- Streaming responses with natural turn-taking
- Barge-in / interruption handling
- Configurable greeting, system prompt, and per-language voices

### Telephony
- **Plivo** support (applications, number linking, inbound & outbound)
- **Vobiz** support (inbound & outbound, with native call recording)
- WebSocket audio streaming via [Pipecat](https://github.com/pipecat-ai/pipecat)

### AI Providers
- **LLM:** OpenAI, Anthropic (Claude), Grok (xAI), Qwen (self-hosted vLLM), Kenpath/Vistaar
- **STT:** Deepgram, Google, OpenAI/Whisper, ElevenLabs, Sarvam, AI4Bharat, Bhashini
- **TTS:** ElevenLabs, Cartesia, OpenAI, Google, Deepgram, Sarvam, AI4Bharat, Bhashini
- Strong **Indic-language** coverage (22+ languages) via Sarvam, AI4Bharat, and Bhashini

### Knowledge Base (RAG)
- Upload documents (PDF) per organization
- Automatic chunking + embeddings into a vector store
- Retrieval-augmented answers at call time (configurable top-k)

### Developer Experience
- **Browser testing** — no telephony, public URL, or webhooks needed
- **Docker Compose** one-command stack
- Per-service `.env.example` files and `make` targets
- Self-hostable optional **AI4Bharat** local STT/TTS servers

---

## Quick Start

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.10+ (for local voice server development)
- CUDA-capable GPU (optional, only for local AI4Bharat servers)

### 1. Clone

```bash
git clone <repository-url>
cd voicera_mono_repository
```

### 2. Configure environment

```bash
cp voicera_backend/env.example       voicera_backend/.env
cp voicera_frontend/.env.example     voicera_frontend/.env.local
cp voice_2_voice_server/.env.example voice_2_voice_server/.env
# Optional local Indic STT/TTS servers:
cp ai4bharat_stt_server/.env.example ai4bharat_stt_server/.env
cp ai4bharat_tts_server/.env.example ai4bharat_tts_server/.env
```

See [Environment Configuration](#environment-configuration) for variable details.

### 3. Build and start

```bash
make build-all-services
make start-all-services
```

Then open the dashboard at **http://localhost:3000**.

---

## Try It in the Browser (no telephony required)

The fastest way to experience VoicERA — **no Plivo, Vobiz, SIP, webhooks, or public URL**:

1. Open **http://localhost:3000** and sign up / sign in.
2. Create an agent — set its **system prompt**, **LLM**, **STT**, and **TTS** providers.
3. Open the agent and click **Test in Browser**.
4. Allow microphone access and start talking — you'll see live transcripts and hear the
   agent respond.

Under the hood this uses the voice server's browser WebSocket endpoint
(`/browser/agent/{agent_id}` in [voice_2_voice_server/api/server.py](voice_2_voice_server/api/server.py))
and the dashboard's test dialog
([test-browser-dialog.tsx](voicera_frontend/components/assistants/test-browser-dialog.tsx)).

---

## Supported Providers

| Category | Supported Providers |
|----------|---------------------|
| **Telephony** | Plivo, Vobiz |
| **LLM** | OpenAI, Anthropic (Claude), Grok (xAI), Qwen (self-hosted vLLM), Kenpath/Vistaar |
| **STT** | Deepgram, Google, OpenAI/Whisper, ElevenLabs, Sarvam, AI4Bharat, Bhashini |
| **TTS** | ElevenLabs, Cartesia, OpenAI, Google, Deepgram, Sarvam, AI4Bharat, Bhashini |

> Provider wiring lives in
> [voice_2_voice_server/api/services.py](voice_2_voice_server/api/services.py) with default
> models in [config/llm_mappings.py](voice_2_voice_server/config/llm_mappings.py) and language
> maps in [config/stt_mappings.py](voice_2_voice_server/config/stt_mappings.py). See
> [AGENTS.md](AGENTS.md) for how to add a new provider.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                       VoicERA_mono_repository                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   Frontend   │    │   Backend    │    │ Voice Server │         │
│  │   (Next.js)  │◄──►│  (FastAPI)   │◄──►│  (Pipecat)   │         │
│  │   :3000      │    │   :8000      │    │   :7860      │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│                             │                   │                  │
│                             ▼                   ▼                  │
│                      ┌──────────────┐    ┌──────────────┐         │
│                      │   MongoDB    │    │    MinIO     │         │
│                      │   :27017     │    │  :9000/:9001 │         │
│                      └──────────────┘    └──────────────┘         │
│                                                                   │
│  ┌──────────────────────────────────────────────────────┐        │
│  │           Optional: Local AI4Bharat Servers           │        │
│  │  ┌──────────────┐              ┌──────────────┐        │        │
│  │  │  STT Server  │              │  TTS Server  │        │        │
│  │  │   :8001      │              │   :8002      │        │        │
│  │  └──────────────┘              └──────────────┘        │        │
│  └──────────────────────────────────────────────────────┘        │
└───────────────────────────────────────────────────────────────────┘
```

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js web dashboard for agent management |
| `backend` | 8000 | FastAPI REST API for data, auth, integrations & RAG |
| `voice_server` | 7860 | Real-time voice processing with Pipecat |
| `mongodb` | 27017 | Primary database |
| `minio` | 9000/9001 | Object storage for recordings & transcripts |
| `ai4bharat_stt_server` | 8001 | Local Indic STT (optional) |
| `ai4bharat_tts_server` | 8002 | Local Indic TTS (optional) |

**Browser mode** (testing):

```
Mic → Browser → Voice Server (/browser/agent) → STT → LLM (+RAG) → TTS → Browser
```

**Telephony mode** (production):

```
Caller → Plivo/Vobiz → Voice Server (/{provider}/agent) → STT → LLM (+RAG) → TTS → Caller
```

More diagrams: [docs/assets/](docs/assets/) and
[docs/architecture/](docs/architecture/overview.md).

---

## Production & Telephony Setup

Once your agent works in the browser, connect a telephony provider:

- **Plivo** & **Vobiz** integration — [docs/services/telephony.md](docs/services/telephony.md)
- **Public voice URLs** (exposing the voice server) — [docs/deployment/public-voice-urls.md](docs/deployment/public-voice-urls.md)
- **Docker deployment** — [docs/deployment/docker.md](docs/deployment/docker.md)
- **Production hardening** — [docs/deployment/production.md](docs/deployment/production.md) · [docs/deployment/security-hardening.md](docs/deployment/security-hardening.md)

Telephony credentials are configured per organization via **Integrations** in the dashboard
and/or environment variables — see [Environment Configuration](#environment-configuration).

---

## Makefile Commands

| Command | Description |
|---------|-------------|
| `make build-all-services` | Build Docker images for all core services (mongodb, backend, minio, frontend, voice_server) |
| `make start-all-services` | Start all core services (detached) |
| `make stop-all-services` | Stop all core services |
| `make build-backend-services` | Build only backend infrastructure (mongodb, backend, minio) |
| `make start-backend-services` | Start backend services without frontend/voice |
| `make stop-backend-services` | Stop backend services |
| `make start-frontend` | Start frontend dev server locally (frees :3000 first) |
| `make start-voice-only-services` | Start AI4Bharat STT/TTS + voice server locally (requires venvs) |
| `make start-dev` | Start everything for local development |
| `make stop-dev` | Stop all development services |
| `make stop-all-ports` | Force-kill all service ports (3000, 27017, 8000, 8001, 8002, 7860) |

---

## Environment Configuration

<details>
<summary><strong>Backend (<code>voicera_backend/.env</code>)</strong></summary>

```bash
# MongoDB
MONGODB_HOST=localhost          # Use 'mongodb' in Docker
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin

# Application
DEBUG=False
SECRET_KEY=your-secret-key      # python -c "import secrets; print(secrets.token_urlsafe(32))"

# Email (Mailtrap)
MAILTRAP_API_TOKEN=your-mailtrap-token
MAILTRAP_FROM_EMAIL=noreply@voicera.com
MAILTRAP_FROM_NAME=VoicERA
FRONTEND_URL=http://localhost:3000

# Internal API (service-to-service auth)
INTERNAL_API_KEY=your-internal-api-key

# MinIO Storage
MINIO_ENDPOINT=minio:9000       # 'localhost:9000' for local dev
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Vobiz Telephony API
VOBIZ_API_BASE_URL=https://api.vobiz.in/v1
VOBIZ_ACCOUNT_ID=your-account-id
VOBIZ_AUTH_ID=your-auth-id
VOBIZ_AUTH_TOKEN=your-auth-token
```
</details>

<details>
<summary><strong>Frontend (<code>voicera_frontend/.env.local</code>)</strong></summary>

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
API_URL=http://localhost:8000
VOICE_SERVER_URL=http://localhost:7860

# In Docker, use service names:
# NEXT_PUBLIC_API_URL=http://nginx:8080/api/v1
# API_URL=http://backend:8000
# VOICE_SERVER_URL=http://voice_server:7860
```
</details>

<details>
<summary><strong>Voice Server (<code>voice_2_voice_server/.env</code>)</strong></summary>

```bash
# Vobiz Telephony API
VOBIZ_AUTH_ID=your-vobiz-auth-id
VOBIZ_AUTH_TOKEN=your-vobiz-auth-token
VOBIZ_API_BASE=https://api.vobiz.in/v1
VOBIZ_CALLER_ID=+91XXXXXXXXXX

# Server URLs (your public domain)
JOHNAIC_SERVER_URL=https://your-server-domain.com
JOHNAIC_WEBSOCKET_URL=wss://your-server-domain.com

# Backend API
VOICERA_BACKEND_URL=http://localhost:8000   # 'http://backend:8000' in Docker
INTERNAL_API_KEY=your-internal-api-key      # Must match backend's INTERNAL_API_KEY

# MinIO Storage
MINIO_ENDPOINT=localhost:9000               # 'minio:9000' in Docker
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Bhashini STT (cloud)
BHASHINI_API_KEY=your-bhashini-api-key
BHASHINI_SOCKET_URL=wss://dhruva-api.bhashini.gov.in

# Local AI4Bharat Servers (optional)
AI4BHARAT_STT_URL=http://localhost:8001
AI4BHARAT_TTS_URL=http://localhost:8002
```

> Provider API keys (OpenAI, Anthropic, Deepgram, ElevenLabs, Cartesia, Sarvam, …) are
> typically configured **per organization** via the dashboard's Integrations, with
> environment variables as a fallback.
</details>

<details>
<summary><strong>AI4Bharat STT / TTS servers (optional)</strong></summary>

```bash
# ai4bharat_stt_server/.env  and  ai4bharat_tts_server/.env
HF_TOKEN=your-huggingface-token   # if the model is gated
PORT=8001                          # STT default; TTS default is 8002
```
</details>

---

## Local Development (without Docker)

```bash
# 1. Infrastructure via Docker
make start-backend-services

# 2. Frontend
cd voicera_frontend && npm install && npm run dev

# 3. Voice server
cd voice_2_voice_server && python -m venv venv && source venv/bin/activate \
  && pip install -r requirements.txt && python main.py
```

Optional AI4Bharat servers (require GPU): see
[docs/development/local-setup.md](docs/development/local-setup.md).

---

## API Endpoints

**Backend (`:8000`)** — Swagger at `http://localhost:8000/docs`
- `GET /api/v1/agents` · `POST /api/v1/agents` — manage agents
- `GET /api/v1/meetings` — call history · `GET /api/v1/call-recordings` — recordings

**Voice Server (`:7860`)** — Swagger at `http://localhost:7860/docs`
- `GET /health` — health check · `POST /outbound/call/` — outbound call
- `WS /agent/{agent_id}` · `WS /plivo/agent/{agent_id}` · `WS /browser/agent/{agent_id}`

Full reference: [docs/api/](docs/api/rest-api.md).

---

## Troubleshooting

Common quick fixes:

```bash
make stop-all-ports               # port already in use
docker compose logs -f backend    # view service logs (backend|voice_server|frontend)
docker compose down -v            # reset (removes volumes/data)
```

Full guide with audio, API-key, telephony, Docker, and provider-credential issues:
**[docs/troubleshooting.md](docs/troubleshooting.md)**.

---

## Documentation

- 📖 Full docs: [docs/](docs/index.md) (MkDocs)
- 🏗️ Architecture: [docs/architecture/overview.md](docs/architecture/overview.md)
- 🤖 For contributors & coding agents: [AGENTS.md](AGENTS.md)
- 🔌 Services: [docs/services/](docs/services/voice-server.md)
- 🚀 Deployment: [docs/deployment/](docs/deployment/docker.md)

---

## Contributing

Contributions are welcome! Please read **[CONTRIBUTING.md](CONTRIBUTING.md)** and our
**[Code of Conduct](CODE_OF_CONDUCT.md)** before opening an issue or PR. Security issues:
see **[SECURITY.md](SECURITY.md)**. Notable changes are tracked in
**[CHANGELOG.md](CHANGELOG.md)**.

## License

MIT License — Copyright (c) 2026 COSS India. See [LICENSE](LICENSE).
