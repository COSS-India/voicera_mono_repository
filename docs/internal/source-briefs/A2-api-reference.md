# Brief: API reference (A2)

**Review gap:** Root README documents only a few endpoints. Auth, user management, telephony webhooks, and recordings are absent. Voice server WebSocket format is not in main docs.

**Writer task:**

- Backend: group by feature; point to **Swagger** `http://<host>:8000/docs` for full request/response schemas.
- Voice server: `http://<host>:7860/docs` plus WebSocket section below and [A8-johnaic-public-urls.md](./A8-johnaic-public-urls.md).

**Do not hand-copy every field** — OpenAPI/Swagger is the source of truth for schemas.

---

## Backend routers (all under `/api/v1`)

| Area | Prefix | Main operations |
|------|--------|-----------------|
| Auth / users | `/users` | signup, login, me, forgot/reset password |
| Agents | `/agents` | create, list by org, get/update/delete by agent_type, config by id/phone |
| Meetings (calls) | `/meetings` | create, patch, list with filters, get by id |
| Recordings | `/call-recordings` | post recording metadata |
| Phone numbers | `/phone-numbers` | list by org/agent, attach, detach |
| Vobiz | `/vobiz` | create/delete application, list numbers, link/unlink number |
| Plivo | `/plivo` | same pattern as Vobiz |
| Integrations | `/integrations` | store API keys per org (Vobiz, Plivo, Bhashini, etc.) |
| Campaigns | `/campaigns` | create, list by org |
| Audience | `/audience` | create, list, get |
| Batches | `/batches` | upload CSV, run/schedule/stop campaigns, worker endpoints |
| Knowledge | `/knowledge` | upload PDFs, list, delete |
| RAG | `/rag` | retrieve for agent |
| Analytics | `/analytics` | org analytics |
| Members | `/members` | add/list/delete org members |

**Also:** `GET /health` on backend (outside versioned prefix if configured — check Swagger).

**Code reference:** `voicera_backend/app/main.py` — router registration.

---

## Voice server HTTP

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Status |
| `/health` | GET | Health |
| `/outbound/call/` | POST | Start outbound call |
| `/answer` | GET/POST | Vobiz answer webhook → XML with WebSocket URL |
| `/plivo/answer` | GET/POST | Plivo answer webhook |
| `/plivo/hangup` | GET/POST | Plivo hangup logging |

**Code reference:** `voice_2_voice_server/api/server.py`

---

## Voice server WebSocket (critical)

| Path | Purpose |
|------|---------|
| `/agent/{agent_id}` | Vobiz / browser audio stream |
| `/plivo/agent/{agent_id}` | Plivo audio stream |

### Connection lifecycle

1. Client (telephony platform or browser) connects; server accepts.
2. Server loads agent config from backend (`fetch_agent_config_from_backend`).
3. **First message** must be JSON text:

```json
{"event":"start","start":{"callSid":"...","streamSid":"..."}}
```

(`callId` / `streamId` aliases also accepted.)

4. **Uplink audio** (client → server):

```json
{
  "event": "media",
  "media": {
    "contentType": "audio/x-l16",
    "sampleRate": 16000,
    "payload": "<base64 PCM>"
  }
}
```

5. **Downlink:** server sends play-audio frames with base64 payload (see browser test doc).
6. On error or hangup, WebSocket closes; meeting may be updated via webhooks.

### Audio specifications

- Format: **L16** (16-bit linear PCM)
- Sample rate: **16000 Hz** for Vobiz/Plivo streaming
- Rationale: Vobiz serializer supports 16 kHz L16; μ-law is 8 kHz only (`voice_2_voice_server/serializer/vobiz_serializer.py`)

### Browser test (same protocol)

Detailed client behavior: `voice_2_voice_server/docs/talk-on-browser-feature.md`

Path: `wss://<public-host>/agent/{agent_id}`

---

## Auth model (high level)

| Traffic | Auth |
|---------|------|
| Dashboard → backend | User login (JWT/session via frontend) |
| Voice server → backend | `INTERNAL_API_KEY` (service-to-service) |
| Telephony → voice server | Public URLs; provider callbacks |
| Integrations API keys | Per-org records in MongoDB |
