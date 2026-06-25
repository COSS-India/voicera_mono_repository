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

### GET /api/v1/users/{email}

Return a user profile by email address.

### POST /api/v1/users/forgot-password

Request a password reset email (sent via Mailtrap when configured).

### POST /api/v1/users/reset-password

Complete a password reset using the token from the email.

---

## Agents (`/api/v1/agents`)

CRUD over voice agents. Each agent encodes its LLM, STT, TTS, prompt, language, and telephony provider.

### GET /api/v1/agents/org/{org_id}

List all agents for a given organization. The caller's `org_id` must match the path parameter.

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

### GET /api/v1/agents/{agent_type}

Return a single agent by its `agent_type` slug.

### PUT /api/v1/agents/{agent_type}

Update any subset of fields. If `agent_type` is renamed and the agent has a linked Vobiz application, the application name is updated automatically.

### DELETE /api/v1/agents/{agent_type}

Delete an agent. Also accepts `DELETE /api/v1/agents?agent_type=<value>` as an alternative for agent types that contain `/`.

### GET /api/v1/agents/config/{agent_type}

Internal endpoint — requires `X-API-Key`. Used by the voice server to load runtime config when a call is answered.

### GET /api/v1/agents/config/id/{agent_id}

Internal endpoint — requires `X-API-Key`. Same as above but looks up by MongoDB `_id` instead of `agent_type`.

### GET /api/v1/agents/by-phone/{phone_number}

Internal endpoint — requires `X-API-Key`. Resolve an agent from a linked phone number.

---

## Campaigns (`/api/v1/campaigns`) and audience (`/api/v1/audience`)

Outbound calling at scale.

### GET /api/v1/campaigns/org/{org_id}

List all campaigns for a given organization.

### GET /api/v1/campaigns/{campaign_name}

Get a single campaign by name.

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

### Audience (`/api/v1/audience`)

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/audience` | Create a new audience entry |
| GET | `/api/v1/audience` | List all audiences (optional `?phone_number=` filter) |
| GET | `/api/v1/audience/{audience_name}` | Get audience by name |

### Batches (`/api/v1/batches`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| GET | `/api/v1/batches` | JWT | List batches for the org (optional `?agent_type=`) |
| POST | `/api/v1/batches/upload` | JWT | Upload a CSV of phone numbers |
| DELETE | `/api/v1/batches/{batch_id}` | JWT | Delete a batch |
| POST | `/api/v1/batches/{batch_id}/run` | JWT | Start dialing |
| POST | `/api/v1/batches/{batch_id}/stop` | JWT | Halt the batch |
| POST | `/api/v1/batches/{batch_id}/schedule` | JWT | Schedule a batch for a future time |
| POST | `/api/v1/batches/{batch_id}/schedule/cancel` | JWT | Cancel a scheduled batch |
| POST | `/api/v1/batches/{batch_id}/schedule/reschedule` | JWT | Change a scheduled batch's time |
| POST | `/api/v1/batches/worker/claim-next` | X-API-Key | Voice server: claim the next contact to dial |
| POST | `/api/v1/batches/worker/agent-config` | X-API-Key | Voice server: fetch agent call config |
| POST | `/api/v1/batches/worker/report` | X-API-Key | Voice server: report a contact result |
| POST | `/api/v1/batches/worker/finalize` | X-API-Key | Voice server: mark batch complete |

The `worker/*` endpoints are internal — called by the voice server only, authenticated with `X-API-Key`.

---

## Meetings and call logs (`/api/v1/meetings`)

`/meetings` is the canonical record of a placed or received call.

### GET /api/v1/meetings

List calls with pagination. Supported query params: `?page=`, `?limit=`, `?for_export=`, `?agent_type=`, `?from_number=`, `?to_number=`, `?inbound=`, `?call_status=`, `?date_from=`, `?date_to=`, `?date_sort_order=`, `?duration_sort_order=`.

### GET /api/v1/meetings/filter-options

Returns distinct `agent_type` and phone number values for use in History filter dropdowns.

### GET /api/v1/meetings/{meeting_id}

```json
{
  "id": "meeting-uuid",
  "agent_type": "support",
  "phone_number": "+1234567890",
  "status": "completed",
  "duration_seconds": 120,
  "transcript": "...",
  "summary": "Customer asked about billing...",
  "sentiment": "positive",
  "recording_url": "minio://recordings/meeting-uuid.wav",
  "created_at": "2026-06-19T10:30:00Z"
}
```

### GET /api/v1/meetings/{meeting_id}/recording

Stream the audio recording for a meeting directly. Returns a `StreamingResponse` (WAV, MP3, or M4A) proxied from MinIO. The meeting must belong to the caller's organization.

### POST /api/v1/meetings and PATCH /api/v1/meetings/{meeting_id}

These are **internal bot endpoints** called by the voice server (authenticated with `X-API-Key`). `POST` creates the meeting record when a call starts; `PATCH` updates the end time when the call ends. They are not intended for dashboard or third-party use.

---

## Call recordings (`/api/v1/call-recordings`)

This router has a single internal endpoint called by the voice server after a call completes:

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/v1/call-recordings` | None (internal) | Voice server saves recording URL, transcript, and call metadata to the meeting record |

To stream or download a recording from the frontend, use `GET /api/v1/meetings/{meeting_id}/recording` (see above).

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

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/v1/integrations/bot/get-api-key` | X-API-Key | Voice server fetches a decrypted key by `model` name |
| GET | `/api/v1/integrations` | JWT | List all integrations for the org |
| POST | `/api/v1/integrations` | JWT | Store or update a credential |
| GET | `/api/v1/integrations/{model}` | JWT | Get a single integration by model name |
| DELETE | `/api/v1/integrations/{model}` | JWT | Remove an integration |

There is no PUT — to update a credential, POST again with the same `model` value (upsert behaviour).

## Custom LLM integrations (`/api/v1/custom-llm-integrations`)

Per-organization custom LLM endpoint configurations (base URL, model name, API key for self-hosted or third-party OpenAI-compatible servers). An org can store multiple custom LLM configs.

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/v1/custom-llm-integrations/bot/get-config` | X-API-Key | Voice server fetches full config by `custom_llm_id` |
| GET | `/api/v1/custom-llm-integrations` | JWT | List all custom LLM configs for the org |
| POST | `/api/v1/custom-llm-integrations` | JWT | Create a new custom LLM config |
| PUT | `/api/v1/custom-llm-integrations/{custom_llm_id}` | JWT | Update a custom LLM config |
| DELETE | `/api/v1/custom-llm-integrations/{custom_llm_id}` | JWT | Delete a custom LLM config |

---

## Telephony (`/api/v1/vobiz`, `/api/v1/plivo`, `/api/v1/phone-numbers`)

Provider-specific resource management.

### Vobiz

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/vobiz/application` | Create a Vobiz application bound to an agent |
| DELETE | `/api/v1/vobiz/application/{application_id}` | Delete a Vobiz application |
| GET | `/api/v1/vobiz/numbers` | List numbers on the Vobiz account |
| POST | `/api/v1/vobiz/numbers/link` | Link a number to an application |
| DELETE | `/api/v1/vobiz/numbers/unlink` | Unlink a number |

### Plivo

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/v1/plivo/application` | Create a Plivo application bound to an agent |
| DELETE | `/api/v1/plivo/application/{application_id}` | Delete a Plivo application |
| GET | `/api/v1/plivo/numbers` | List numbers on the Plivo account |
| POST | `/api/v1/plivo/numbers/link` | Link a number to an application |
| DELETE | `/api/v1/plivo/numbers/unlink` | Unlink a number |

### Phone numbers

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/api/v1/phone-numbers/org/{org_id}` | List all numbers attached to an org |
| GET | `/api/v1/phone-numbers/agent/{agent_type}` | Get the number attached to a specific agent |
| POST | `/api/v1/phone-numbers/attach` | Attach a number to an agent |
| DELETE | `/api/v1/phone-numbers/detach` | Detach a number |

---

## Analytics (`/api/v1/analytics`)

### GET /api/v1/analytics

Returns aggregate metrics for the caller's organization: calls attempted, calls connected, average call duration, total minutes connected, and most-used agent. Metrics are calculated on-demand from the `CallLogs` collection.

Supported query params:

| Param | Description |
|-------|-------------|
| `agent_type` | Filter by a specific agent |
| `phone_number` | Filter by a specific phone number |
| `start_date` | ISO date/datetime (inclusive) |
| `end_date` | ISO date/datetime (inclusive) |

---

## Members (`/api/v1/members`)

| Method | Endpoint | Auth | Purpose |
|--------|----------|------|---------|
| POST | `/api/v1/members/add-member` | None (public) | Add a new member to an org (used for invite links) |
| GET | `/api/v1/members/{org_id}` | JWT | List all members of an org |
| POST | `/api/v1/members/delete-member` | JWT | Remove a member from an org |

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
| WS | `/agent/{agent_id}` | Vobiz inbound/outbound and browser test audio stream |
| WS | `/plivo/agent/{agent_id}` | Plivo audio stream |
| WS | `/browser/agent/{agent_id}` | In-browser test client audio stream |

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
