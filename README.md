# Voicera Mono Repository

A complete voice AI platform with telephony integration, featuring real-time speech-to-text, text-to-speech, and LLM-powered conversational agents.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                         voicera_mono_repository                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Frontend   │    │   Backend    │    │ Voice Server │       │
│  │   (Next.js)  │◄──►│  (FastAPI)   │◄──►│  (Pipecat)   │       │
│  │   :3000      │    │   :8000      │    │   :7860      │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                   │                │
│                             ▼                   ▼                │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   MongoDB    │    │    MinIO     │       │
│                      │   :27017     │    │  :9000/:9001 │       │
│                      └──────────────┘    └──────────────┘       │
│                                                                  │
│  ┌──────────────────────────────────────────────────────┐       │
│  │            Optional: Local AI4Bharat Servers          │       │
│  │  ┌──────────────┐              ┌──────────────┐      │       │
│  │  │  STT Server  │              │  TTS Server  │      │       │
│  │  │   :8001      │              │   :8002      │      │       │
│  │  └──────────────┘              └──────────────┘      │       │
│  └──────────────────────────────────────────────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

## Services

| Service | Port | Description |
|---------|------|-------------|
| `frontend` | 3000 | Next.js web dashboard for agent management |
| `backend` | 8000 | FastAPI REST API for data management |
| `voice_server` | 7860 | Real-time voice processing with Pipecat |
| `mongodb` | 27017 | Primary database |
| `minio` | 9000/9001 | Object storage for recordings & transcripts |
| `ai4bharat_stt_server` | 8001 | Local Indic STT (optional) |
| `ai4bharat_tts_server` | 8002 | Local Indic TTS (optional) |

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

# AI4Bharat servers (optional)
cp ai4bharat_stt_server/.env.example ai4bharat_stt_server/.env
cp ai4bharat_tts_server/.env.example ai4bharat_tts_server/.env
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
| `make build-all-services` | Build Docker images for all core services (mongodb, backend, minio, frontend, voice_server) |
| `make start-all-services` | Start all core services in detached mode |
| `make stop-all-services` | Stop all core services |

### Backend-Only Commands

| Command | Description |
|---------|-------------|
| `make build-backend-services` | Build only backend infrastructure (mongodb, backend, minio) |
| `make start-backend-services` | Start backend services without frontend/voice |
| `make stop-backend-services` | Stop backend services |

### Development Commands

| Command | Description |
|---------|-------------|
| `make start-frontend` | Start frontend dev server locally (kills existing :3000 process) |
| `make start-voice-only-services` | Start AI4Bharat STT/TTS and voice server locally (requires venv) |
| `make start-dev` | Start everything for local development |
| `make stop-dev` | Stop all development services |
| `make stop-all-ports` | Force kill all service ports (3000, 27017, 8000, 8001, 8002, 7860) |

---

## Environment Configuration

### Backend (`voicera_backend/.env`)

```bash
# MongoDB Configuration
MONGODB_HOST=localhost          # Use 'mongodb' when running in Docker
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
MAILTRAP_FROM_NAME=Voicera
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
# API URLs
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
API_URL=http://localhost:8000
VOICE_SERVER_URL=http://localhost:7860

# When running in Docker, use service names:
# NEXT_PUBLIC_API_URL=http://nginx:8080/api/v1
# API_URL=http://backend:8000
# VOICE_SERVER_URL=http://voice_server:7860
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

# Local AI4Bharat Servers (optional)
AI4BHARAT_STT_URL=http://localhost:8001
AI4BHARAT_TTS_URL=http://localhost:8002
```

### AI4Bharat STT Server (`ai4bharat_stt_server/.env`)

```bash
# HuggingFace Token (if model is gated)
HF_TOKEN=your-huggingface-token

# Server port (default: 8001)
PORT=8001
```

### AI4Bharat TTS Server (`ai4bharat_tts_server/.env`)

```bash
# HuggingFace Token (if model is gated)
HF_TOKEN=your-huggingface-token

# Server port (default: 8002)
PORT=8002
```

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

4. **Start AI4Bharat servers (optional, requires GPU):**
   ```bash
   # STT Server
   cd ai4bharat_stt_server
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python server.py --port 8001

   # TTS Server (in another terminal)
   cd ai4bharat_tts_server
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python server.py
   ```

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
docker volume rm voicera_mono_repository_mongodb_data
make start-all-services
```

---

## License

Proprietary - All rights reserved.
