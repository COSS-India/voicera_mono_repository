---
description: Complete REST API reference for the VoicEra backend, grouped by resource, with request/response examples.
---

# REST API Reference

VoicEra exposes its REST API from the backend service. This page is a grouped reference for the most common operations. For the complete, always-current schema, use Swagger UI at `http://<host>:8000/docs` or fetch the OpenAPI spec from `http://<host>:8000/openapi.json`.

{% hint style="info" %}
**Swagger is the source of truth.** Hand-written documentation can drift; the OpenAPI spec is generated from the FastAPI routes in `voicera_backend/app/main.py`. When in doubt, check `/docs`.
{% endhint %}

## Base URL

```
http://localhost:8000        # Local development
https://api.yourdomain.com   # Production
```

All resource routes are mounted under the `/api/v1` prefix.

## Authentication

Most endpoints require a JWT bearer token, obtained via `POST /api/v1/users/login`.

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
Accept: application/json
```

Service-to-service calls (voice server to backend) use an internal API key configured via the `INTERNAL_API_KEY` environment variable. See [environment-variables.md](environment-variables.md).

## Response format

Success responses return the resource object directly. Error responses follow FastAPI's standard `HTTPException` shape:

```json
{ "detail": "Invalid credentials" }
```

Validation errors (422) return a list of field-level errors:

```json
{
  "detail": [
    { "loc": ["body", "email"], "msg": "field required", "type": "value_error.missing" }
  ]
}
```

{% hint style="info" %}
The canonical response schema for each endpoint is in Swagger at `/docs`. The examples in this page show representative fields only.
{% endhint %}

## Status codes

| Code | Meaning |
|------|---------|
| 200 | OK — request succeeded |
| 201 | Created — resource was created |
| 204 | No Content — successful, no body |
| 400 | Bad Request — validation failure |
| 401 | Unauthorized — missing or invalid token |
| 403 | Forbidden — authenticated but not permitted |
| 404 | Not Found |
| 409 | Conflict — duplicate or invalid state transition |
| 422 | Unprocessable Entity — schema validation failed |
| 429 | Too Many Requests — rate limited |
| 500 | Internal Server Error |

---

## Auth and users (`/api/v1/users`)

User signup, login, profile, and password reset.

### POST /api/v1/users/signup

Register a new user account. The first user in an organization becomes its admin.

{% tabs %}
{% tab title="curl" %}
```bash
curl -X POST http://localhost:8000/api/v1/users/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!",
    "first_name": "Jane",
    "last_name": "Doe"
  }'
```
{% endtab %}

{% tab title="python" %}
```python
import requests

requests.post(
    "http://localhost:8000/api/v1/users/signup",
    json={
        "email": "user@example.com",
        "password": "SecurePass123!",
        "first_name": "Jane",
        "last_name": "Doe",
    },
)
```
{% endtab %}

{% tab title="javascript" %}
```javascript
await fetch("http://localhost:8000/api/v1/users/signup", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    email: "user@example.com",
    password: "SecurePass123!",
    first_name: "Jane",
    last_name: "Doe",
  }),
});
```
{% endtab %}
{% endtabs %}

**Response:** `201 Created`

```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "first_name": "Jane",
  "last_name": "Doe",
  "created_at": "2026-06-19T10:30:00Z"
}
```

### POST /api/v1/users/login

Exchange email and password for a JWT.

```bash
curl -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"SecurePass123!"}'
```

**Response:** `200 OK`

```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400,
  "user": { "id": "user-uuid", "email": "user@example.com", "role": "admin" }
}
```

### GET /api/v1/users/me

Return the authenticated user profile.

### POST /api/v1/users/forgot-password

Request a password reset email (sent via Mailtrap when configured).

### POST /api/v1/users/reset-password

Complete a password reset using the token from the email.

---

## Agents (`/api/v1/agents`)

CRUD over voice agents. Each agent encodes its LLM, STT, TTS, prompt, language, and telephony provider.

### GET /api/v1/agents

List agents in the caller's organization. Supports `?skip=`, `?limit=`, `?search=`, `?status=`.

### POST /api/v1/agents

Create an agent.

```json
{
  "name": "Support Agent",
  "description": "Handles customer support",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "stt_provider": "deepgram",
  "tts_provider": "cartesia",
  "system_prompt": "You are a helpful support agent.",
  "language": "en",
  "telephony_provider": "vobiz"
}
```

**Response:** `201 Created` with the new agent.

### GET /api/v1/agents/{agent_id}

Return a single agent including its full configuration.

### PUT /api/v1/agents/{agent_id}

Patch any subset of fields.

### DELETE /api/v1/agents/{agent_id}

Delete an agent. `204 No Content`.

### GET /api/v1/agents/config/id/{agent_id}

Internal endpoint used by the voice server to load runtime config when a call is answered. Requires `INTERNAL_API_KEY`.

### GET /api/v1/agents/by-phone/{phone_number}

Resolve an agent from a linked phone number.

---

## Campaigns (`/api/v1/campaigns`) and audience (`/api/v1/audience`)

Outbound calling at scale.

### GET /api/v1/campaigns

List campaigns. Filter by `?agent_id=`, `?status=`.

### POST /api/v1/campaigns

```json
{
  "name": "Q2 Marketing Campaign",
  "agent_id": "agent-uuid",
  "audience_id": "audience-uuid",
  "start_time": "2026-07-01T00:00:00Z",
  "end_time": "2026-09-30T23:59:59Z",
  "max_concurrent_calls": 50
}
```

### Batches (`/api/v1/batches`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/batches/upload` | Upload a CSV of phone numbers |
| POST | `/api/v1/batches/{batch_id}/run` | Start dialing |
| POST | `/api/v1/batches/{batch_id}/stop` | Halt the batch |

---

## Meetings and call logs (`/api/v1/meetings`)

`/meetings` is the canonical record of a placed or received call. Older clients may still see `/call-logs`-style paths in the dashboard; the storage backend is the same collection.

### GET /api/v1/meetings

List calls. Filters: `?campaign_id=`, `?agent_id=`, `?status=`, `?start_date=`, `?end_date=`, `?phone_number=`.

### GET /api/v1/meetings/{meeting_id}

```json
{
  "id": "meeting-uuid",
  "campaign_id": "campaign-uuid",
  "agent_id": "agent-uuid",
  "phone_number": "+1234567890",
  "status": "completed",
  "duration_seconds": 120,
  "transcript": "...",
  "summary": "Customer asked about billing...",
  "sentiment": "positive",
  "recording_url": "https://minio:9000/recordings/meeting-uuid.wav",
  "created_at": "2026-06-19T10:30:00Z"
}
```

---

## Recordings (`/api/v1/call-recordings`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/call-recordings/{call_id}` | Recording metadata |
| GET | `/api/v1/call-recordings/{call_id}/download` | WAV download |
| GET | `/api/v1/call-recordings/{call_id}/transcript` | Plain-text transcript |
| POST | `/api/v1/call-recordings/{call_id}/transcribe` | Request re-transcription |

Audio is stored in the MinIO `recordings` bucket; downloads return a short-lived presigned URL.

---

## Knowledge base (`/api/v1/knowledge`) and RAG (`/api/v1/rag`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/knowledge/upload` | Upload a PDF for indexing |
| GET | `/api/v1/knowledge` | List indexed documents |
| DELETE | `/api/v1/knowledge/{doc_id}` | Remove a document |
| POST | `/api/v1/rag/retrieve` | Retrieve top-k chunks for an agent at runtime |

Embeddings use OpenAI; vector storage is Chroma. See [../concepts/knowledge-base-rag.md](../concepts/knowledge-base-rag.md).

---

## Integrations (`/api/v1/integrations`)

Per-organization API keys for Vobiz, Plivo, Bhashini, OpenAI, etc. The voice server reads these at call time via `fetch_integration_key(org_id, key_name)`.

{% hint style="warning" %}
**Vobiz Auth ID and Auth Token live here, not in `.env`.** Putting them in `.env` only works for single-tenant dev setups. See [../concepts/telephony-model.md](../concepts/telephony-model.md).
{% endhint %}

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/integrations` | List integrations for the org |
| POST | `/api/v1/integrations` | Store a new integration credential |
| PUT | `/api/v1/integrations/{integration_id}` | Update |
| DELETE | `/api/v1/integrations/{integration_id}` | Remove |

---

## Telephony (`/api/v1/vobiz`, `/api/v1/plivo`, `/api/v1/phone-numbers`)

Provider-specific resource management.

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/vobiz/applications` | Create a Vobiz application bound to an agent |
| DELETE | `/api/v1/vobiz/applications/{app_id}` | Delete |
| GET | `/api/v1/vobiz/numbers` | List numbers on the Vobiz account |
| POST | `/api/v1/vobiz/numbers/link` | Link a number to an application |
| POST | `/api/v1/vobiz/numbers/unlink` | Unlink |
| POST | `/api/v1/plivo/applications` | Plivo equivalent |
| GET | `/api/v1/phone-numbers` | List numbers attached to the org |

---

## Analytics (`/api/v1/analytics`) and members (`/api/v1/members`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/analytics/calls` | Aggregate call counts and durations |
| GET | `/api/v1/analytics/sentiment` | Sentiment distribution |
| GET | `/api/v1/analytics/top-phrases` | Most-spoken phrases |
| GET | `/api/v1/analytics/agent-performance` | Per-agent metrics |
| GET | `/api/v1/analytics/export` | CSV export |
| GET | `/api/v1/members` | List org members |
| POST | `/api/v1/members` | Invite a member |
| DELETE | `/api/v1/members/{member_id}` | Remove |

---

## Voice server HTTP

The voice server is a separate FastAPI app on port `7860`. Most traffic is WebSocket (see [websocket-api.md](websocket-api.md)); the HTTP surface is small.

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/` | Status string |
| GET | `/health` | Health check |
| GET | `/docs` | Swagger UI |
| POST | `/outbound/call/` | Start an outbound call (server-to-server) |
| GET/POST | `/answer` | Vobiz answer webhook — returns XML pointing the call at `/agent/{agent_id}` |
| GET/POST | `/plivo/answer` | Plivo answer webhook |
| GET/POST | `/plivo/hangup` | Plivo hangup webhook |

---

## Health and observability

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Service health (backend and voice server) |
| GET | `/docs` | Swagger UI |
| GET | `/redoc` | ReDoc UI |
| GET | `/openapi.json` | OpenAPI spec |

---

## Query parameters

### Pagination

```
?skip=0&limit=10   # default limit is 10, max is 100
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
?sort=-created_at         # shorthand for descending
?sort=name,created_at     # multi-field
```

---

## Next steps

- [websocket-api.md](websocket-api.md) — Real-time audio protocol
- [endpoints-cheatsheet.md](endpoints-cheatsheet.md) — One-page index of every route
- [../concepts/architecture.md](../concepts/architecture.md) — How the REST API fits into the system
- [../guides/developer/local-setup.md](../guides/developer/local-setup.md) — Run the backend locally
