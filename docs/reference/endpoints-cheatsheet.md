---
description: Dense one-page reference card with every REST endpoint, common shell commands, and default URLs.
---

# Endpoints and Commands Cheatsheet

A consolidated quick-reference for the VoicEra platform. Use it as a lookup for endpoints, common curl calls, and routine operator commands. For full schemas, see Swagger at `http://<host>:8000/docs` and `http://<host>:7860/docs`.

{% hint style="info" %}
**Source of truth:** Swagger / OpenAPI. Router registration lives in `voicera_backend/app/main.py` and `voice_2_voice_server/api/server.py`.
{% endhint %}

## Service URLs

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | `http://localhost:3000` | Web dashboard |
| Backend API | `http://localhost:8000` | REST API |
| Backend Swagger | `http://localhost:8000/docs` | Interactive API docs |
| Voice server | `http://localhost:7860` | HTTP + WebSocket |
| Voice server Swagger | `http://localhost:7860/docs` | Voice server API docs |
| MinIO API | `http://localhost:9000` | S3-compatible storage |
| MinIO Console | `http://localhost:9001` | Object browser UI |
| MongoDB | `mongodb://localhost:27017` | Database |
| Nginx (optional) | `http://localhost:8080` | Reverse proxy |

Default credentials are listed at [../quickstart/default-credentials.md](../quickstart/default-credentials.md). For port details and exposure rules, see [ports-and-defaults.md](ports-and-defaults.md).

---

## Backend routers (prefix `/api/v1`)

| Area | Router | Main operations |
|------|--------|-----------------|
| Auth / users | `/users` | signup, login, me, forgot/reset password |
| Agents | `/agents` | CRUD, config by id/phone |
| Meetings (calls) | `/meetings` | create, patch, list, get by id |
| Recordings | `/call-recordings` | recording metadata, download, transcript |
| Phone numbers | `/phone-numbers` | list, attach, detach |
| Vobiz | `/vobiz` | applications, link numbers |
| Plivo | `/plivo` | applications, link numbers |
| Integrations | `/integrations` | per-org API keys |
| Campaigns | `/campaigns` | create, list, launch, pause |
| Audience | `/audience` | create, list, get |
| Batches | `/batches` | CSV upload, run/stop campaigns |
| Knowledge | `/knowledge` | upload PDFs, list, delete |
| RAG | `/rag` | retrieve chunks for voice agent |
| Analytics | `/analytics` | org-level analytics |
| Members | `/members` | org members |

## Auth and users

| Method | Endpoint | Auth | Description |
|--------|----------|------|-------------|
| POST | `/api/v1/users/signup` | – | Register new user |
| POST | `/api/v1/users/login` | – | Exchange credentials for JWT |
| GET | `/api/v1/users/me` | bearer | Current user |
| POST | `/api/v1/users/forgot-password` | – | Request reset email |
| POST | `/api/v1/users/reset-password` | – | Reset password with token |

## Agents

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/agents` | List agents |
| POST | `/api/v1/agents` | Create agent |
| GET | `/api/v1/agents/{agent_id}` | Get agent |
| PUT | `/api/v1/agents/{agent_id}` | Update agent |
| DELETE | `/api/v1/agents/{agent_id}` | Delete agent |
| GET | `/api/v1/agents/config/id/{agent_id}` | Runtime config (internal, `X-API-Key`) |
| GET | `/api/v1/agents/by-phone/{phone}` | Resolve by phone |

## Campaigns, audience, batches

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/campaigns` | List campaigns |
| POST | `/api/v1/campaigns` | Create campaign |
| POST | `/api/v1/campaigns/{id}/launch` | Launch |
| POST | `/api/v1/campaigns/{id}/pause` | Pause |
| GET | `/api/v1/audience` | List audiences |
| POST | `/api/v1/audience` | Create audience |
| POST | `/api/v1/batches/upload` | Upload CSV |
| POST | `/api/v1/batches/{id}/run` | Start dialing |
| POST | `/api/v1/batches/{id}/stop` | Halt batch |

## Calls (meetings) and recordings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/meetings` | List calls |
| GET | `/api/v1/meetings/{id}` | Get call detail |
| GET | `/api/v1/call-recordings/{call_id}` | Recording metadata |
| GET | `/api/v1/call-recordings/{call_id}/download` | WAV download |
| GET | `/api/v1/call-recordings/{call_id}/transcript` | Transcript text |
| POST | `/api/v1/call-recordings/{call_id}/transcribe` | Re-transcribe |

## Knowledge base and RAG

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/knowledge/upload` | Upload PDF |
| GET | `/api/v1/knowledge` | List documents |
| DELETE | `/api/v1/knowledge/{doc_id}` | Delete document |
| POST | `/api/v1/rag/retrieve` | Retrieve top-k chunks |

## Integrations and telephony

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/integrations` | List integrations |
| POST | `/api/v1/integrations` | Store credential |
| PUT | `/api/v1/integrations/{id}` | Update |
| DELETE | `/api/v1/integrations/{id}` | Delete |
| POST | `/api/v1/vobiz/applications` | Create Vobiz app |
| POST | `/api/v1/vobiz/numbers/link` | Link number |
| GET | `/api/v1/phone-numbers` | List numbers |

## Analytics and members

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/analytics/calls` | Call statistics |
| GET | `/api/v1/analytics/sentiment` | Sentiment breakdown |
| GET | `/api/v1/analytics/top-phrases` | Common phrases |
| GET | `/api/v1/analytics/agent-performance` | Agent metrics |
| GET | `/api/v1/analytics/export` | CSV export |
| GET | `/api/v1/members` | List org members |
| POST | `/api/v1/members` | Invite member |
| DELETE | `/api/v1/members/{id}` | Remove |

## Voice server HTTP

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Status |
| GET | `/health` | Health check |
| POST | `/outbound/call/` | Start outbound call |
| GET/POST | `/answer` | Vobiz answer webhook |
| GET/POST | `/plivo/answer` | Plivo answer webhook |
| GET/POST | `/plivo/hangup` | Plivo hangup webhook |
| WS | `/agent/{agent_id}` | Vobiz / browser audio stream |
| WS | `/plivo/agent/{agent_id}` | Plivo audio stream |

---

## Common API calls

### Get a token

```bash
curl -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!"}'
```

### Use the token

```bash
TOKEN="eyJhbGc..."
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/agents
```

### Create an agent

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support Bot",
    "llm_provider": "openai",
    "system_prompt": "You are a helpful support agent"
  }'
```

### Place an outbound call

```bash
curl -X POST http://localhost:7860/outbound/call/ \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "<agent-uuid>",
    "customer_number": "+1234567890"
  }'
```

### Download a recording

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/call-recordings/<call-id>/download \
  -o recording.wav
```

---

## Shell commands

### First-time install

```bash
git clone https://github.com/COSS-India/voicera_mono_repository.git
cd voicera_mono_repository

cp voicera_backend/env.example voicera_backend/.env
cp voice_2_voice_server/.env.example voice_2_voice_server/.env

docker-compose up -d
docker-compose ps
```

See [../quickstart/install-and-run.md](../quickstart/install-and-run.md) for the full walkthrough.

### Docker container management

```bash
docker-compose up -d                     # Start all services
docker-compose down                      # Stop all services
docker-compose ps                        # Status
docker-compose logs -f backend           # Tail backend logs
docker-compose restart voice_server      # Restart one service
docker-compose build backend             # Rebuild one image
docker-compose build --no-cache backend  # Clean rebuild
docker stats                             # Live resource usage
```

### Backend development

```bash
cd voicera_backend
pip install -r requirements.txt
uvicorn app.main:app --reload
pytest
pytest --cov=app
```

### Frontend development

```bash
cd voicera_frontend
npm install
npm run dev          # http://localhost:3000
npm run build
npm test
npm run lint
```

### Voice server development

```bash
cd voice_2_voice_server
pip install -r requirements.txt
python main.py
DEBUG_MODE=true python main.py
```

### MongoDB shell

```bash
docker exec -it voicera_mongodb mongosh -u admin -p admin123

# Inside the shell
show databases
use voicera
show collections
db.agents.find()
db.agents.countDocuments()
```

### Troubleshooting

```bash
docker-compose logs <service>     # Service logs
lsof -i :8000                     # Port conflict check
docker stats                      # Resource usage
docker-compose build --no-cache   # Force rebuild
```

See [../troubleshooting/common-issues.md](../troubleshooting/common-issues.md) and [../troubleshooting/deployment.md](../troubleshooting/deployment.md).

---

## Query parameters

### Pagination

```
?skip=0&limit=10
```

### Filtering

```
?status=active
?agent_id=<uuid>
?campaign_id=<uuid>
?start_date=2026-01-01
?end_date=2026-01-31
?phone_number=%2B1234567890
?search=keyword
```

### Sorting

```
?sort=created_at&order=desc
?sort=-created_at        # descending shorthand
?sort=name,created_at    # multi-field
```

---

## Common headers

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
Accept: application/json
```

---

## Common response shapes

### Success

Success responses return the resource object directly — no wrapper envelope.

### Error

FastAPI returns a standard `detail` field on errors:

```json
{ "detail": "Agent not found" }
```

Validation errors (422):

```json
{
  "detail": [
    { "loc": ["body", "name"], "msg": "field required", "type": "value_error.missing" }
  ]
}
```

## Next steps

- [rest-api.md](rest-api.md) — Detailed REST reference with examples
- [websocket-api.md](websocket-api.md) — Audio streaming protocol
- [environment-variables.md](environment-variables.md) — All configuration knobs
- [ports-and-defaults.md](ports-and-defaults.md) — Ports, URLs, default credentials
