---
description: Base URL, versioning, request and response conventions.
---

# API overview

The Voicera API is a JSON REST API served by the `api` container on port `8000`. Everything the platform does — users, agents, credentials, numbers, calls, campaigns, documents — is available through it. There is no private admin surface.

{% hint style="success" %}
A running API serves an **interactive console** at `http://localhost:8000/docs` and ReDoc at `/redoc`, generated from the same routers as these pages. Use it to try requests against a real token; use these pages to understand what a route is for and how it behaves.
{% endhint %}

Every router is mounted under `settings.API_V1_PREFIX`, which defaults to `/api/v1` (`apps/api/app/config.py`). Paths on this page already include it.

Two routes are declared on the app itself in `apps/api/app/main.py` and are **not** prefixed:

| Method | Path | Auth | Response |
| --- | --- | --- | --- |
| GET | `/` | public | `{"message": "Welcome to …", "version": "…", "docs": "/docs"}` |
| GET | `/health` | public | `{"status": "ok" \| "degraded", "database": "up" \| "down"}` |

`/health` pings FerretDB. It always returns `200`; read the body, not the status code.

## Request format

Send `Content-Type: application/json` on every request with a body. Two routes take `multipart/form-data` instead, because they carry a file: `POST /api/v1/campaign/upload` and `POST /api/v1/knowledge/upload`.

## Browse by resource

| Page | Covers |
| --- | --- |
| [Agents](agents.md) | Create and manage voice agents |
| [Calls](calls.md) | Place calls, register them, fetch artifacts |
| [Campaigns](campaigns.md) | Outbound campaigns end to end |
| [Phone numbers](phone-numbers.md) | Number inventory and agent attachment |
| [Knowledge and RAG](knowledge-and-rag.md) | Documents and retrieval |
| [Configuration catalogs](configuration.md) | Which providers exist and what they accept |
| [Provider credentials](provider-auth.md) | Storing encrypted API keys |
| [Users and organisations](users-and-orgs.md) | Signup, login, membership, roles |

Prefer one flat list? See the [Endpoints cheatsheet](endpoints-cheatsheet.md).

## Related

* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
* [WebSocket API](websocket-api.md) — the media protocol
* [Operating via the API](../guides/operator/operating-via-api.md) — task-shaped recipes
