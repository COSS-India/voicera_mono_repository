# VoicEra Backend

**FastAPI REST API** for the platform: auth, agents, telephony (Vobiz/Plivo), campaigns, meetings, integrations, knowledge base (RAG), and call recordings. Persists data in **MongoDB**; object storage via **MinIO**.

## When you need it

| Scenario | Need this service? |
|----------|-------------------|
| Dashboard, agent CRUD, org integrations | **Yes** — always |
| Voice server handling calls | **Yes** — voice server reads agent config and integration keys from here |
| Voice-only experiments without dashboard | **Yes** — at least MongoDB + backend for agent/integration data |

## Run

### Monorepo (Docker)

```bash
# From repository root
make build-backend-services   # mongodb, backend, minio
make start-backend-services
```

Docker Compose service name: **`backend`** (port **8000**). Swagger: http://localhost:8000/docs

```bash
make stop-backend-services
```

### This folder only

```bash
cd voicera_backend
cp env.example .env
docker compose up -d          # MongoDB + API
# or local API with MongoDB already up:
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run.py
```

## Dependencies

Must be running **before** the API is useful:

| Service | Port | Role |
|---------|------|------|
| **MongoDB** | 27017 | Users, agents, meetings, integrations |
| **MinIO** | 9000 / 9001 | Call recordings, uploads (when enabled) |

Typically also used by the full stack:

| Service | Notes |
|---------|--------|
| **voice_server** | `VOICERA_BACKEND_URL`, `INTERNAL_API_KEY` |
| **frontend** | `NEXT_PUBLIC_API_URL` → backend |

## Configuration

Copy `env.example` → `.env`. Key variables:

| Variable | Purpose |
|----------|---------|
| `MONGODB_*` | Database connection |
| `SECRET_KEY` | Auth / sessions |
| `INTERNAL_API_KEY` | Service-to-service (voice server → backend) |
| `MINIO_*` | Recording storage |
| `FRONTEND_URL` | Links in email flows |

Telephony credentials for production are usually stored per org via **Dashboard → Integrations**, not only in `.env`.

## Documentation

- [Backend service (MkDocs)](../docs/services/backend.md)
- [REST API](../docs/api/rest-api.md)
- [Integrations](../docs/services/integrations.md)
- [Knowledge base / RAG](../docs/services/knowledge-base.md)
- [Source brief A1](../docs/source-briefs/A1-submodule-readmes.md) · [A2 API](../docs/source-briefs/A2-api-reference.md)
