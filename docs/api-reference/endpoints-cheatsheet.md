---
description: Every Voicera endpoint on one page, grouped by service.
---

# Endpoints cheatsheet

Every HTTP and WebSocket route the Voicera stack exposes, extracted from the routers. Use this to find a path fast; use `/docs` on a running API for the always-current schema, request bodies, and response models.

The API serves interactive OpenAPI docs at `http://localhost:8000/docs` and ReDoc at `/redoc`. Those are generated from the same routers this page was extracted from, so they never drift.

## Auth column values

| Value | Dependency in the route signature | How you satisfy it |
|---|---|---|
| `public` | none | No header. |
| `Bearer` | `Depends(get_current_user)` | `Authorization: Bearer <jwt>` from `POST /users/login`. |
| `X-API-Key` | `Depends(verify_api_key)` | `X-API-Key: <INTERNAL_API_KEY>` — service-to-service only. |
| `Bearer (admin)` | `Depends(get_current_user)` plus an in-body role check against `{super_admin, admin}` | Bearer token whose `role` claim is `admin` or `super_admin`. |
| `Bearer (super_admin)` | `Depends(get_current_user)` plus an in-body check against `super_admin` | Bearer token whose `role` claim is `super_admin`. |

The dependencies are defined in `apps/api/app/auth.py`. Role checks are not dependencies — they are explicit `HTTPException` raises inside the handler, so a wrong role returns `403`, not `401`.

## API — `:8000`

Every router below is mounted under `settings.API_V1_PREFIX`, which defaults to `/api/v1` (`apps/api/app/config.py`). The paths in this table already include that prefix. `GET /` and `GET /health` are declared directly on the app in `apps/api/app/main.py` and are **not** prefixed.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/` | public | Welcome payload: project name, version, link to `/docs`. |
| GET | `/health` | public | Liveness and readiness; pings FerretDB and reports `ok` or `degraded`. |
| POST | `/api/v1/users/signup` | public | Create a user, a new organisation, and a `super_admin` membership. |
| POST | `/api/v1/users/login` | public | Exchange email and password for a JWT. |
| POST | `/api/v1/users/bot/token` | `X-API-Key` | Service bootstrap: exchange the internal key for an org-scoped JWT. |
| POST | `/api/v1/users/switch-organisation` | Bearer | Reissue the JWT against a different organisation you belong to. |
| GET | `/api/v1/users/organisations` | Bearer | List organisations the caller is a member of. |
| GET | `/api/v1/users/me` | Bearer | Current user profile plus memberships. |
| GET | `/api/v1/users/check/{email}` | public | Invite helper: whether the email exists and is already in the organisation. |
| GET | `/api/v1/users/{email}` | Bearer | User profile by email. Self only — any other email returns `403`. |
| POST | `/api/v1/users/forgot-password` | public | Request a password-reset email. |
| POST | `/api/v1/users/reset-password` | public | Reset a password with a reset token. |
| POST | `/api/v1/members/invite` | Bearer (admin) | Invite a user into the caller's active organisation. |
| GET | `/api/v1/members/{org_id}` | Bearer | List members. The caller must hold a membership in that organisation. |
| POST | `/api/v1/members/assign-admin` | Bearer (super_admin) | Promote a member to `admin`. |
| POST | `/api/v1/members/remove` | Bearer (super_admin) | Remove a member from the active organisation. |
| DELETE | `/api/v1/organisations/{org_id}` | Bearer (super_admin) | Delete an organisation and its memberships. Must be your active org. |
| GET | `/api/v1/languages` | Bearer | Supported language catalogue. |
| GET | `/api/v1/configuration/stt` | Bearer | List registered STT providers. |
| GET | `/api/v1/configuration/tts` | Bearer | List registered TTS providers. |
| GET | `/api/v1/configuration/llm` | Bearer | List registered LLM providers. |
| GET | `/api/v1/configuration/telephony` | Bearer | List registered telephony providers. |
| GET | `/api/v1/configuration/stt/setting/{provider}` | Bearer | Config schema for one STT provider. |
| GET | `/api/v1/configuration/tts/setting/{provider}` | Bearer | Config schema for one TTS provider. |
| GET | `/api/v1/configuration/llm/setting/{provider}` | Bearer | Config schema for one LLM provider. |
| GET | `/api/v1/configuration/telephony/setting/{provider}` | Bearer | Config schema for one telephony provider. |
| GET | `/api/v1/auth/catalog` | Bearer | Credential-field catalogue for every provider that takes credentials. |
| GET | `/api/v1/auth/catalog/{provider}` | Bearer | Credential-field catalogue for one provider. |
| GET | `/api/v1/auth/configured` | Bearer | Which providers the organisation has stored credentials for. |
| POST | `/api/v1/auth` | Bearer (admin) | Upsert encrypted credentials for a provider. |
| GET | `/api/v1/auth/{provider}` | Bearer | Read stored credentials. Masked for non-admin roles. |
| DELETE | `/api/v1/auth/{provider}` | Bearer (admin) | Delete stored credentials for a provider. |
| POST | `/api/v1/agents` | Bearer | Create an agent in the active organisation. |
| GET | `/api/v1/agents` | Bearer | List agents in the active organisation. |
| GET | `/api/v1/agents/by-phone/{phone_number}` | `X-API-Key` | Resolve an agent from an inbound phone number. Used by the runtime. |
| GET | `/api/v1/agents/{agent_id}` | Bearer | Fetch one agent. |
| PATCH | `/api/v1/agents/{agent_id}` | Bearer | Partial update of an agent. |
| DELETE | `/api/v1/agents/{agent_id}` | Bearer (admin) | Delete an agent. |
| GET | `/api/v1/phone-numbers` | Bearer | List numbers in the organisation inventory. |
| GET | `/api/v1/phone-numbers/agent/{agent_id}` | Bearer | The number attached to one agent. |
| POST | `/api/v1/phone-numbers/attach` | Bearer | Add a number to the inventory and optionally bind it to an agent. |
| DELETE | `/api/v1/phone-numbers/detach` | Bearer | Unbind a number from its agent; the inventory row stays. |
| GET | `/api/v1/phone-numbers/providers/{provider}/inventory` | Bearer | Numbers held in the telephony provider account. |
| POST | `/api/v1/calls/outbound` | Bearer | Place an outbound call. |
| POST | `/api/v1/calls/inbound` | Bearer | Register an inbound call from the runtime answer webhook. |
| POST | `/api/v1/calls/web` | Bearer | Register a browser websocket session so it gets a CallLog and artifacts. |
| PATCH | `/api/v1/calls/by-provider-sid/{provider_call_sid}` | Bearer | Patch a call log found by provider call SID. |
| PATCH | `/api/v1/calls/{call_id}` | Bearer | Patch artifact URLs, end time, status, or disposition. |
| GET | `/api/v1/calls/{call_id}/recording` | Bearer | Stream the call recording from MinIO. |
| GET | `/api/v1/calls/{call_id}/transcript` | Bearer | Stream the call transcript from MinIO. |
| GET | `/api/v1/calls/org/{org_id}` | Bearer | Paginated call logs for an organisation. |
| GET | `/api/v1/calls/{call_id}` | Bearer | Fetch one call log. |
| POST | `/api/v1/campaign/upload` | Bearer | Upload a contact CSV; returns the `source_id` to pass to create. |
| POST | `/api/v1/campaign/create` | Bearer | Create a campaign against an agent and an uploaded CSV. |
| GET | `/api/v1/campaign/` | Bearer | List campaigns in the active organisation. |
| GET | `/api/v1/campaign/{campaign_id}` | Bearer | Fetch one campaign. |
| POST | `/api/v1/campaign/{campaign_id}/start` | Bearer | Start a campaign. |
| POST | `/api/v1/campaign/{campaign_id}/pause` | Bearer | Pause a running campaign. |
| POST | `/api/v1/campaign/{campaign_id}/resume` | Bearer | Resume a paused campaign. |
| PATCH | `/api/v1/campaign/{campaign_id}` | Bearer | Update name, rate limit, concurrency, retry, schedule, or circuit breaker. |
| GET | `/api/v1/campaign/{campaign_id}/runs` | Bearer | List queued runs for a campaign. |
| GET | `/api/v1/campaign/{campaign_id}/progress` | Bearer | Row counts and progress percentage. |
| POST | `/api/v1/campaign/{campaign_id}/redial` | Bearer | Create a follow-up campaign over the failed calls. |
| GET | `/api/v1/campaign/{campaign_id}/source-download-url` | Bearer | Presigned URL for the uploaded source CSV. |
| GET | `/api/v1/campaign/{campaign_id}/report` | Bearer | Campaign outcome report. |
| POST | `/api/v1/campaign/internal/call-status` | `X-API-Key` | Call-status callback from the runtime into the orchestrator. |
| GET | `/api/v1/knowledge` | Bearer | List knowledge documents for the organisation. |
| POST | `/api/v1/knowledge/upload` | Bearer | Upload a PDF and schedule background ingest into Chroma. |
| DELETE | `/api/v1/knowledge/{document_id}` | Bearer | Delete a document and its vectors. |
| POST | `/api/v1/rag/retrieve` | `X-API-Key` | Service-to-service chunk retrieval for the runtime. |


`POST /api/v1/calls/inbound` and `PATCH /api/v1/calls/*` take a Bearer token, not `X-API-Key`, even though the runtime is the usual caller. The runtime obtains that token from `POST /api/v1/users/bot/token`, which is the one route that trades the internal key for a JWT.

## Runtime — `:7860`

Three routes, all declared in `apps/runtime/routes/` and mounted with no prefix (`apps/runtime/app.py`). Nothing here is authenticated: the runtime trusts the network and the identifiers in the URL.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | public | Returns `{"status": "ok", "service": "voicera-runtime"}`. |
| GET, POST | `/answer` | public | Telephony answer webhook. Returns Stream XML pointing at the media WebSocket. Requires `agent_id` and `org_id` query parameters. |
| WS | `/agent/{org_id}/{agent_id}` | public | Media WebSocket for the Pipecat pipeline. Accepts an optional `call_id` query parameter for CallLog correlation. |

{% hint style="warning" %}
The runtime authenticates nothing. `/answer` and `/agent/{org_id}/{agent_id}` are reachable by anyone who can reach the port, and `/answer` is the route your telephony provider calls, so it must be publicly resolvable. Put it behind a reverse proxy and restrict by source where you can — see [Security hardening](../guides/deployment/security-hardening.md).
{% endhint %}

## Model-server gateway — `:8100`

Declared in `model-server/gateway/app/main.py`. The gateway is the only published port in the model-server stack; the STT, TTS, and LLM containers are reachable only by service name on the internal network. No route requires authentication.

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | public | Probes each deployed slot. Returns `503` when a deployed slot is unreachable; an undeployed slot is not a fault. |
| GET | `/models` | public | Full catalogue from `models.yaml`, each entry flagged with whether it is deployed. |
| GET | `/v1/models` | public | OpenAI-compatible list of only the models callable right now. |
| GET | `/demo` | public | Live STT demo page, proxied from whichever model fills the STT slot. |
| GET | `/v1/languages` | public | Languages the STT checkpoint in the slot actually speaks. |
| POST | `/v1/audio/transcriptions` | public | OpenAI-compatible transcription, proxied to the STT slot. |
| WS | `/v1/asr/ws` | public | Streaming transcription relay to the STT slot. Not an OpenAI route. |
| WS | `/v1/realtime` | public | OpenAI Realtime transcription relay to the STT slot. |
| POST | `/v1/audio/speech` | public | OpenAI-compatible speech synthesis, proxied to the TTS slot. |
| POST | `/v1/chat/completions` | public | OpenAI-compatible chat, proxied to the LLM slot. |

A request to a slot with no model deployed returns `503` with `"type": "upstream_not_configured"`; a WebSocket route sends a JSON error frame and closes with code `1013`.

{% hint style="warning" %}
`model-server/README.md` states that the LLM slot has never been built or started, so the vLLM flags behind `/v1/chat/completions` are unverified. Treat that route as untested.
{% endhint %}

## Related

* [REST API](overview.md)
* [WebSocket API](websocket-api.md)
* [Ports and defaults](../developer/reference/ports-and-defaults.md)
* [Environment variables](../developer/reference/environment-variables.md)
* [Data model](../developer/reference/data-model.md)
