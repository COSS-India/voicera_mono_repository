---
description: Every HTTP and WebSocket surface Voicera exposes.
---

# API Reference

The complete Voicera API: 71 REST routes across twelve routers, plus the realtime media WebSocket. Every page here was extracted from the routers and verified against them.

{% hint style="success" %}
A running API serves an **interactive console** at `http://localhost:8000/docs`, generated from the same code. Use it to fire real requests; use these pages to learn what a route does and how it behaves.
{% endhint %}

## Start here

| Page | Covers |
| --- | --- |
| [Introduction](overview.md) | Base URL, versioning, request and response conventions |
| [Authentication](authentication.md) | JWTs, `X-API-Key`, roles, organisation scoping |
| [Errors](errors.md) | Status codes and error shapes |
| [Endpoints cheatsheet](endpoints-cheatsheet.md) | Every route across all three services, one page |

## Endpoints by resource

| Page | Routes | What it covers |
| --- | --- | --- |
| [Agents](agents.md) | 6 | Create, read, update, delete voice agents |
| [Calls](calls.md) | 9 | Outbound, inbound, web calls; recordings and transcripts |
| [Campaigns](campaigns.md) | 14 | CSV upload, scheduling, start/pause/resume, progress, reports |
| [Phone numbers](phone-numbers.md) | 5 | Number inventory, attaching to agents |
| [Knowledge and RAG](knowledge-and-rag.md) | 4 | Document upload and retrieval |
| [Configuration catalogs](configuration.md) | 9 | Which providers exist and what each accepts |
| [Provider credentials](provider-auth.md) | 6 | Storing encrypted API keys |
| [Users and organisations](users-and-orgs.md) | 15 | Signup, login, membership, roles, bot tokens |

## Realtime

* [**WebSocket API**](websocket-api.md) — `WS /agent/{org_id}/{agent_id}` in both modes: telephony frame serializers at 8 kHz, browser protobuf/RTVI at 16 kHz.

## Conventions on every page

Each endpoint lists its **method and path**, the **auth** it requires, its **request** fields, its **response** shape, and the **errors** it can return. Paths already include the `/api/v1` prefix.

## Related

* [Operating via the API](../guides/operator/operating-via-api.md) — task-shaped recipes built on these routes
* [Connecting a client](../developer/clients/README.md) — choosing a surface
* [Data model](../developer/reference/data-model.md) — the documents behind these routes
