---
description: Canonical port map, service URLs, and pointer to default credentials for a stock VoicEra deployment.
---

# Ports and Defaults

A stock `docker-compose up -d` from the repository root brings up six containers on the `voicera_network` bridge. This page lists the host ports they expose, the in-network DNS aliases, and where to find default credentials.

{% hint style="info" %}
**Source of truth:** `docker-compose.yml` at the repo root. If ports drift, that file wins.
{% endhint %}

## Host port map

| Host port | Container | Protocol | Purpose |
|-----------|-----------|----------|---------|
| `3000` | `voicera_frontend` | HTTP | Next.js dashboard |
| `8000` | `voicera_backend` | HTTP | Backend REST API + Swagger |
| `7860` | `voicera_voice_server` | HTTP + WS | Voice server REST + WebSocket audio |
| `27017` | `voicera_mongodb` | TCP | MongoDB |
| `9000` | `voicera_minio` | HTTP | MinIO S3 API |
| `9001` | `voicera_minio` | HTTP | MinIO web console |
| `8080` | `voicera_nginx` | HTTP | Optional reverse proxy |

## Service URLs

Use these when developing against a local stack.

| Service | Local URL | Notes |
|---------|-----------|-------|
| Frontend | `http://localhost:3000` | Dashboard, login at `/login` |
| Backend API | `http://localhost:8000` | REST routes under `/api/v1` |
| Backend Swagger | `http://localhost:8000/docs` | Interactive OpenAPI explorer |
| Backend ReDoc | `http://localhost:8000/redoc` | Static OpenAPI viewer |
| Voice server | `http://localhost:7860` | Health at `/health` |
| Voice server Swagger | `http://localhost:7860/docs` | REST surface only |
| Voice server WebSocket | `ws://localhost:7860/agent/{agent_id}` | See [websocket-api.md](websocket-api.md) |
| MinIO API | `http://localhost:9000` | S3-compatible endpoint |
| MinIO Console | `http://localhost:9001` | Web UI for buckets |
| MongoDB | `mongodb://localhost:27017` | Use a client like `mongosh` |
| Nginx | `http://localhost:8080` | Only when `nginx.conf` is configured |

## In-cluster DNS

Containers reach each other via service names on `voicera_network`. This is what you put in `.env` files when running under Compose.

| Service alias | Resolves to | Port(s) inside the network |
|---------------|-------------|----------------------------|
| `mongodb` | `voicera_mongodb` | `27017` |
| `backend` | `voicera_backend` | `8000` |
| `voice_server` | `voicera_voice_server` | `7860` |
| `frontend` | `voicera_frontend` | `3000` |
| `minio` | `voicera_minio` | `9000`, `9001` |
| `nginx` | `voicera_nginx` | `8080` |

So inside a container the backend lives at `http://backend:8000`, MongoDB at `mongodb://mongodb:27017`, and MinIO at `minio:9000` — not `localhost`.

## Default credentials

The repository ships with these out-of-the-box passwords. **Change every one of them before exposing the stack to a network.**

| Service | Username | Password |
|---------|----------|----------|
| MongoDB | `admin` | `admin123` |
| MinIO | `minioadmin` | `minioadmin` |

For the full table including dashboard seed users, the rotation procedure, and rotation order, see [../quickstart/default-credentials.md](../quickstart/default-credentials.md).

{% hint style="danger" %}
The defaults above are public knowledge — they're in `docker-compose.yml` on GitHub. Treat any deployment that still uses them as effectively unauthenticated.
{% endhint %}

## Reserved or commonly-conflicting ports

Before `docker-compose up`, free these ports on the host:

| Port | Often clashes with |
|------|---------------------|
| `3000` | React/Next dev servers, Grafana |
| `8000` | Other Python dev servers (Django, FastAPI) |
| `8080` | Tomcat, Jenkins, kubectl proxy |
| `7860` | Gradio apps |
| `9000` | Portainer, Prometheus pushgateway |
| `9001` | Portainer agent, MinIO peers |
| `27017` | A locally-installed MongoDB |

Check with `lsof -i :<port>` (Linux/macOS) or `netstat -ano | findstr :<port>` (Windows).

## Healthchecks and readiness

| Service | Endpoint | Expected response |
|---------|----------|-------------------|
| Backend | `GET /health` | `{ "status": "ok" }` |
| Voice server | `GET /health` | `{ "status": "ok" }` |
| MongoDB | `db.adminCommand('ping')` | `{ ok: 1 }` |
| MinIO | `mc ready local` | exit 0 |

The Compose file wires healthchecks into `depends_on: condition: service_healthy` so the backend will not start until MongoDB and MinIO are ready.

## Default LLM / STT / TTS

The voice server defaults align with the stock `.env.example`:

| Pipeline stage | Default provider |
|----------------|------------------|
| LLM | OpenAI (`gpt-4`) |
| STT | Deepgram (`nova-2`) |
| TTS | Cartesia (`english_male`) |

All three require an API key set via env vars (see [environment-variables.md](environment-variables.md)). Local alternatives via AI4Bharat are documented at [../services/ai4bharat-stt.md](../services/ai4bharat-stt.md) and [../services/ai4bharat-tts.md](../services/ai4bharat-tts.md).

## Public URLs (production)

In a production deployment the voice server must be reachable from the telephony provider over HTTPS. Configure:

- `JOHNAIC_SERVER_URL=https://voice.yourdomain.com`
- `JOHNAIC_WEBSOCKET_URL=wss://voice.yourdomain.com`
- `NEXT_PUBLIC_JOHNAIC_SERVER_URL=https://voice.yourdomain.com`

See [../guides/deployment/production.md](../guides/deployment/production.md) and [../concepts/telephony-model.md](../concepts/telephony-model.md).

## Next steps

- [environment-variables.md](environment-variables.md) — Every configuration variable, per service
- [../quickstart/default-credentials.md](../quickstart/default-credentials.md) — Full default credential table
- [../guides/deployment/docker-compose.md](../guides/deployment/docker-compose.md) — How Compose wires the services together
- [../troubleshooting/deployment.md](../troubleshooting/deployment.md) — Port conflicts and health-check failures
