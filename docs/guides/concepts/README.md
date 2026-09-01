---
description: How Voicera works — the ideas behind the code, in dependency order.
---

# Core concepts

The engineering model behind Voicera. These pages explain *why* the system is shaped the way it is; the [Reference](../../api-reference/overview.md) section gives the exact contracts.

{% hint style="info" %}
Reading in order works, but is not required. If you only have ten minutes, read [Architecture](architecture.md) and [Voice pipeline](voice-pipeline.md) — everything else builds on those two.
{% endhint %}

## Start here

| Page | What it answers |
| --- | --- |
| [Architecture](architecture.md) | What the containers are and how they fit together. |
| [Voice pipeline](voice-pipeline.md) | What happens between a caller speaking and the agent replying. |
| [Data flow](data-flow.md) | What moves where, for each call scenario. |

## The domain

| Page | What it answers |
| --- | --- |
| [Agents and agent categories](agents.md) | What an agent is, and why `telephony` and `websocket` behave differently. |
| [Calls and call artifacts](calls.md) | Call types, statuses, and where recordings and transcripts land. |
| [Campaigns](campaigns.md) | CSV-driven outbound at volume: batches, retries, circuit breakers. |
| [Call concurrency and rate limiting](call-concurrency.md) | Why calls get queued or refused, and how the limits are enforced. |
| [Knowledge base (RAG)](knowledge-base-rag.md) | Grounding answers in your own documents. |

## Providers and telephony

| Page | What it answers |
| --- | --- |
| [Provider registry](provider-registry.md) | How 23 vendors plug in without a central `if/elif`. |
| [Provider credentials (ProviderAuth)](provider-auth.md) | Where API keys live and how they are encrypted. |
| [Telephony model](telephony-model.md) | One `/answer` webhook serving multiple carriers. |

## Platform

| Page | What it answers |
| --- | --- |
| [Multi-tenancy and roles](multi-tenancy.md) | Organisations, roles, active-org switching, bot tokens. |
| [Data store (FerretDB)](data-store.md) | MongoDB wire protocol over PostgreSQL, and the 27018/27017 split. |
| [Glossary](glossary.md) | Every term this documentation assumes. |

## Related

* [Services](../../developer/services/README.md) — the same system, from the operator's side
* [Repository layout](../../developer/guides/repository-layout.md) — where each concept lives in code
