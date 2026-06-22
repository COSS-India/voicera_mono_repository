# Brief: Vobiz telephony (A3)

**Review gap:** Env vars are listed but there is no documentation on webhook configuration, call flow logic, or call state machine.

---

## Important correction

**Vobiz Auth ID and Auth Token are NOT read from `.env` for normal operation.**

They are stored **per organization** in:

- **Dashboard → Integrations**
- MongoDB models: `VobizAuthId`, `VobizAuthToken`
- Backend: `voicera_backend/app/services/vobiz.py` — `_get_vobiz_auth_for_org`
- Voice server outbound: `voice_2_voice_server/api/server.py` — `fetch_integration_key(org_id, "VobizAuthId")` etc.

**Do not tell operators to put Vobiz auth in a `.env` file** unless documenting a legacy/dev exception.

### What still comes from environment

| Variable | Service | Purpose |
|----------|---------|---------|
| `VOBIZ_API_BASE` | voice server | Vobiz API base URL |
| `VOBIZ_CALLER_ID` | voice server | Optional default outbound caller ID |
| `JOHNAIC_SERVER_URL` | voice server | Public HTTPS base for webhooks (see A8) |
| `JOHNAIC_WEBSOCKET_URL` | voice server | Public WSS base for audio |
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | frontend | Builds answer URLs when creating agents |

Backend may also use `VOBIZ_API_BASE_URL` from settings for application CRUD APIs.

---

## Inbound call flow (Vobiz)

1. Caller dials a number linked to a Vobiz **application**.
2. Vobiz sends HTTP request to: `{PUBLIC_HTTPS_URL}/answer?agent_id=<agent_uuid>`
3. Voice server handles `vobiz_answer_webhook`:
   - On `Event=StartApp`: logs meeting, returns **XML** with WebSocket URL `{PUBLIC_WSS_URL}/agent/{agent_id}`
4. Vobiz opens WebSocket; audio streams; Pipecat pipeline runs (`api/bot.py`).
5. On `Hangup` and other events: `log_meeting` updates call record in backend.

**Code:** `voice_2_voice_server/api/server.py` — `vobiz_answer_webhook`, `websocket_endpoint`

---

## Outbound call flow

1. Trigger: `POST /outbound/call/` on voice server (body includes customer number, agent id/type).
2. Server loads agent config from backend.
3. Reads `telephony_provider` from agent (default Vobiz).
4. **Vobiz:** loads auth from Integrations for agent's `org_id`; POST to `{VOBIZ_API_BASE}/Account/{auth_id}/Call/`
5. **Plivo:** parallel path via `make_outbound_call_plivo` and Plivo integration keys.

---

## Dashboard setup (operator-facing)

1. **Integrations:** enter Vobiz Auth ID + Auth Token (and Plivo keys if using Plivo).
2. **Create agent:** choose telephony provider (Vobiz or Plivo).
3. For Vobiz: system creates Vobiz application with answer URL like `{NEXT_PUBLIC_JOHNAIC_SERVER_URL}/answer?agent_id={id}`.
4. **Numbers page:** link purchased number to agent's Vobiz application (`linkVobizNumber`).

**Frontend references:**

- `voicera_frontend/app/(dashboard)/integrations/page.tsx`
- `voicera_frontend/app/(dashboard)/assistants/page.tsx`
- `voicera_frontend/app/(dashboard)/numbers/page.tsx`

---

## Plivo (parallel documentation)

| Item | Value |
|------|-------|
| Answer webhook | `{PUBLIC_URL}/plivo/answer?agent_id=...` |
| WebSocket | `{PUBLIC_WSS}/plivo/agent/{agent_id}` |
| Hangup webhook | `/plivo/hangup` |
| Credentials | Integrations: Plivo auth (see `voicera_backend/app/services/plivo.py`) |

---

## Call state machine (simplified)

| Stage | System behavior |
|-------|-----------------|
| Call offered / answered | Vobiz hits `/answer` |
| StartApp | XML returned; WebSocket URL issued |
| Streaming | STT → LLM → TTS pipeline active |
| Hangup | Webhook event; meeting end metadata saved |
| User busy | `HangupCause=USER_BUSY` logged |

**Writer:** Add sequence diagram and document the Vobiz portal application URL field (which URL to paste) from staging deployment.
