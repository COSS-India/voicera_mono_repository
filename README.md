# VoicEra Mono Repository

A complete voice AI building block with telephony integration, featuring real-time speech-to-text, text-to-speech, and LLM-powered conversational agents.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         VoicEra_mono_repository                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Frontend   │    │   Backend    │    │ Voice Server │       │
│  │   (Next.js)  │◄──►│  (FastAPI)   │◄──►│  (Pipecat)   │       │
│  │   :3000      │    │   :8000      │    │   :7860      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                   │               │
│                             ▼                   ▼               │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   MongoDB    │    │    MinIO     │       │
│                      │   :27017     │    │  :9000/:9001 │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                 │
│  ┌──────────────────────────────────────────────────────┐       │
│  │   Optional: model-server — self-hosted models        │       │
│  │   One published port; models are reachable only      │       │
│  │   inside it, so nothing else binds to the host.      │       │
│  │  ┌────────────────────────────────────────────────┐  │       │
│  │  │  Gateway  :8100   (OpenAI-compatible)          │  │       │
│  │  ├────────────────────────────────────────────────┤  │       │
│  │  │   stt slot   │   tts slot   │   llm slot       │  │       │
│  │  │  (internal)  │  (internal)  │  (internal)      │  │       │
│  │  └────────────────────────────────────────────────┘  │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js web dashboard for agent management |
| `backend` | 8000 | FastAPI REST API for data management |
| `voice_server` | 7860 | Real-time voice processing with Pipecat |
| `ferretdb` | 27017 | Mongo-compatible DB (FerretDB over PostgreSQL) |
| `postgres` | (internal) | FerretDB backing store (DocumentDB extension) |
| `minio` | 9000/9001 | Object storage for recordings & transcripts |
| `model-server` | 8100 | Self-hosted STT / TTS / LLM behind one OpenAI-compatible gateway (optional) |

---

## Quick Start

### Prerequisites

- Docker & Docker Compose
- Node.js 18+ (for local frontend development)
- Python 3.10+ (for local voice server development)
- CUDA-capable GPU (optional, for local AI4Bharat servers)

### 1. Clone and Setup Environment

```bash
git clone <repository-url>
cd voicera_mono_repository
```

### 2. Configure Environment Variables

Copy the example environment files and configure them:

```bash
# Backend
cp voicera_backend/env.example voicera_backend/.env

# Frontend
cp voicera_frontend/.env.example voicera_frontend/.env.local

# Voice Server
cp voice_2_voice_server/.env.example voice_2_voice_server/.env

# Self-hosted models (optional)
cp model-server/.env.example model-server/.env
```

See [Environment Configuration](#environment-configuration) below for detailed variable descriptions.

### 3. Start All Services

```bash
# Build all Docker images
make build-all-services

# Start all services
make start-all-services

# Stop all services
make stop-all-services
```

---

## Makefile Commands

The Makefile provides convenient commands for managing the services:

### Primary Commands

| Command | Description |
|---------|-------------|
| `make build-all-services` | Build Docker images for all core services (backend, minio, frontend, voice_server) |
| `make start-all-services` | Start all core services in detached mode (postgres, ferretdb, …) |
| `make stop-all-services` | Stop all core services |
| `make migrate-to-ferretdb` | Dump MongoDB and cut over to FerretDB |

### Backend-Only Commands

| Command | Description |
|---------|-------------|
| `make build-backend-services` | Build only backend infrastructure (backend, minio) |
| `make start-backend-services` | Start postgres, ferretdb, backend, minio |
| `make stop-backend-services` | Stop backend services |

### Development Commands

| Command | Description |
|---------|-------------|
| `make start-frontend` | Start frontend dev server locally (kills existing :3000 process) |
| `make start-voice-only-services` | Bring up the model-server containers and run the voice server locally |
| `make start-dev` | Start everything for local development |
| `make stop-dev` | Stop all development services |
| `make stop-all-ports` | Force kill service ports (3000, 27017, 8000, 7860, 8100) and stop the model-server stack |

---

## Environment Configuration

### Backend (`voicera_backend/.env`)

```bash
# MongoDB Configuration
MONGODB_HOST=localhost          # Use 'mongodb' (FerretDB alias) when running in Docker
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin

# Application
DEBUG=False
SECRET_KEY=your-secret-key      # Generate: python -c "import secrets; print(secrets.token_urlsafe(32))"

# Email (Mailtrap)
MAILTRAP_API_TOKEN=your-mailtrap-token
MAILTRAP_FROM_EMAIL=noreply@voicera.com
MAILTRAP_FROM_NAME=VoicEra
FRONTEND_URL=http://localhost:3000

# Internal API (service-to-service auth)
INTERNAL_API_KEY=your-internal-api-key

# MinIO Storage
MINIO_ENDPOINT=minio:9000       # Use 'localhost:9000' for local dev
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

# Vobiz Telephony API
VOBIZ_API_BASE_URL=https://api.vobiz.in/v1
VOBIZ_ACCOUNT_ID=your-account-id
VOBIZ_AUTH_ID=your-auth-id
VOBIZ_AUTH_TOKEN=your-auth-token
```

### Frontend (`voicera_frontend/.env.local`)

```bash
NEXT_PUBLIC_JOHNAIC_SERVER_URL=https://your-public-voice-host
VOICE_SERVER_URL=http://localhost:7860

# Backend is proxied via Next.js /api/* routes to http://localhost:8000 (hardcoded).
# Docker Compose sets API_URL=http://backend:8000 on the frontend container.
```

### Voice Server (`voice_2_voice_server/.env`)

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
VOICERA_BACKEND_URL=http://localhost:8000   # Use 'http://backend:8000' in Docker
INTERNAL_API_KEY=your-internal-api-key      # Must match backend's INTERNAL_API_KEY

# MinIO Storage
MINIO_ENDPOINT=localhost:9000               # Use 'minio:9000' in Docker
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false

# Bhashini STT (cloud-based)
BHASHINI_API_KEY=your-bhashini-api-key
BHASHINI_SOCKET_URL=wss://dhruva-api.bhashini.gov.in

# Self-hosted models (optional). One URL for all three modalities -- the
# gateway routes on the endpoint, so this does not change when the model does.
# From inside a container, use http://host.docker.internal:8100
MODEL_SERVER_URL=http://localhost:8100
```

### Self-hosted models (`model-server/.env`)

```bash
# Which slots run, by slot name -- never a model name.
COMPOSE_PROFILES=stt,tts

# Which model fills each slot: a folder name under model-server/<slot>/.
# Switching model is editing one of these and rebuilding that one service.
STT_MODEL=indic-conformer
TTS_MODEL=indic-parler
LLM_MODEL=                     # empty = that slot is not deployed

# The only published port. The model containers bind nothing on the host.
GATEWAY_PORT=8100

# Which GPU, and how much of it vLLM may reserve.
GPU_DEVICE_IDS=1

# Needed by models that pull from a gated HuggingFace repo -- Indic Parler's
# tokenizer and T5 encoder are gated, so TTS will not start without either this
# or a cache that already holds them.
HF_TOKEN=
```

`model-server/.env.example` is the full list with comments. See
[model-server/README.md](model-server/README.md) for the slot layout and how to
add a model.

---

## Development Setup

### Local Development (without Docker)

1. **Start infrastructure with Docker:**
   ```bash
   make start-backend-services
   ```

2. **Start frontend locally:**
   ```bash
   cd voicera_frontend
   npm install
   npm run dev
   ```

3. **Start voice server locally:**
   ```bash
   cd voice_2_voice_server
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python main.py
   ```

4. **Start the self-hosted models (optional, requires GPU):**

   These run as containers, not venvs. Pick a model per slot in
   `model-server/.env`, then:

   ```bash
   cd model-server
   docker compose -f compose.model-server.yml --project-directory . up -d --build
   curl localhost:8100/health
   ```

   First build takes 20–40 minutes and the weights are several GB, so the first
   start of a slot is slow with little output — watch
   `docker compose logs -f` rather than assuming it has hung.

   `setup.sh` at the repo root does all of this as part of a full deployment,
   and asks which model should fill each slot.

### Using the Combined Dev Command

```bash
# Start everything for development
make start-dev

# Stop everything
make stop-dev
```

---

## API Endpoints

### Backend API (`:8000`)
- `GET /api/v1/agents` - List agents
- `POST /api/v1/agents` - Create agent
- `GET /api/v1/meetings` - List call meetings
- `GET /api/v1/call-recordings` - List recordings
- Swagger docs: `http://localhost:8000/docs`

### Voice Server (`:7860`)
- `GET /` - Health check
- `GET /health` - Detailed health
- `POST /outbound/call/` - Initiate outbound call
- `WS /agent/{agent_id}` - WebSocket for audio streaming
- Swagger docs: `http://localhost:7860/docs`

### Model Server (`:8100`, optional)

OpenAI-compatible, so existing OpenAI clients work against it unchanged.

- `POST /v1/audio/transcriptions` — speech to text
- `POST /v1/audio/speech` — text to speech, streaming PCM
- `POST /v1/chat/completions` — LLM
- `GET /models` — every model in the catalogue, and which are running
- `GET /v1/models` — only what can be called right now
- `GET /health` — gateway and each slot

A slot with no model deployed answers 503 naming the slot, rather than 404.

### MinIO Console (`:9001`)
- Web UI for managing object storage
- Default credentials: `minioadmin` / `minioadmin`

---

## Troubleshooting

### Port Already in Use
```bash
make stop-all-ports
```

### Docker Network Issues
```bash
docker compose down -v
docker network prune
make start-all-services
```

### View Service Logs
```bash
docker compose logs -f backend
docker compose logs -f voice_server
docker compose logs -f frontend
```

### Reset Database
```bash
docker compose down -v
# Optional: remove legacy Mongo volume after FerretDB cutover is verified
# docker volume rm voicera_mono_repository_mongodb_data
docker volume rm voicera_mono_repository_ferretdb_postgres_data
make start-all-services
```

---

## License

MIT License — Copyright (c) 2026 COSS India. See [LICENSE](LICENSE).