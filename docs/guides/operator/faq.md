---
description: Common questions about running Voicera.
---

# FAQ

## Where is the dashboard?

The core stack does not include one. Voicera is API-first: `http://localhost:8000/docs` gives you an interactive console for every endpoint.

A web dashboard exists but is **Beta**, lives on the `dev-frontend` branch, and is not part of the Docker Compose stack. Several of its screens — Batches and Knowledge Base — render static sample data and make no API calls. See [Dashboard (Beta)](../../developer/frontend/overview.md) and [Operating via the API](operating-via-api.md).

## Why is the database on port 27018?

The container listens on `27017`; the host mapping is `27018` so it cannot collide with a MongoDB you already run locally. From your machine use `27018`; inside the Compose network services use `mongodb:27017`.

It is FerretDB — the MongoDB wire protocol on top of PostgreSQL — not MongoDB. See [Data store](../concepts/data-store.md).

## Do I need a GPU?

Not with cloud model providers. The core stack runs on 2 CPU cores and 4 GB of RAM.

A GPU is only needed to self-host models with the [model server](../../developer/model-server/overview.md).

## Can I use only OpenAI?

Yes. `openai` registers speech-to-text, text-to-speech, and a language model, so one credential covers all three. `google` and `sarvam` do the same.

Mixing is common — Deepgram for STT, Cartesia for TTS, OpenAI for the LLM. The choice is per agent.

## Why are my provider dropdowns or catalogs empty?

Catalogs filter to providers you have stored credentials for. Store them first:

```bash
curl -X POST http://localhost:8000/api/v1/auth \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"provider": "openai", "auth": {"api_key": "sk-..."}}'

curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/auth/configured
```

## Why did my campaign pause itself?

The circuit breaker tripped — by default, more than 50% of calls failing within a 300-second window, over a minimum of 5 calls. It exists so a broken agent burns a handful of calls instead of the whole list.

Find out why the calls failed, fix it, then `POST /campaign/{id}/resume`. See [Troubleshooting campaigns](../troubleshooting/campaigns.md).

## Why does my browser test call have no transcript?

Browser sessions do produce transcripts and recordings — the runtime registers a `call_type: web` call log on connect. If one is missing, check the runtime log for `Registered web call call_id=` and confirm MinIO is reachable.

## Do I need a public URL?

Only for real phone calls. Your telephony provider fetches `/answer` over HTTPS and opens a WSS connection for audio — both inbound, so the runtime must be publicly reachable.

For evaluation, a `websocket` agent needs no public URL and no telephony account. See [Public voice URLs](../../guides/deployment/public-voice-urls.md).

## My agents stopped answering after I changed a setting. Why?

Almost certainly `VOICE_SERVER_BASE_URL`. The answer URL is baked into the provider application when an agent is **created**, so changing it later does not update existing agents — your provider keeps calling the old address, and nothing reaches Voicera to log.

`PATCH` each affected agent to re-provision, or recreate it.

## How do I change a default password?

Edit `.env` and recreate the affected containers. Change `MONGODB_PASSWORD`, `MINIO_ROOT_PASSWORD`, and `REDIS_PASSWORD` at minimum.

{% hint style="warning" %}
Changing `MONGODB_PASSWORD` after the volume exists does not update the PostgreSQL user — the same credentials serve both layers. Set it before the first start, or change it inside Postgres too.
{% endhint %}

See [Security hardening](../../guides/deployment/security-hardening.md).

## What happens if I lose PROVIDER_AUTH_ENCRYPTION_KEY?

Every stored provider credential becomes permanently undecryptable. There is no recovery path and no re-encryption tool — each organisation must re-enter every provider key.

Back it up with the same care as the database, and store it alongside your backups.

## Is there a default login?

No. The first `POST /users/signup` creates the user, an organisation, and a `super_admin` membership. Whoever signs up first owns the deployment, so do it immediately after starting.

## How many calls can run at once?

`DEFAULT_ORG_CONCURRENCY_LIMIT` caps simultaneous calls per organisation, default `10`. Campaigns can set a lower `max_concurrency`.

In practice your telephony account's channel limit or your model vendor's rate limits usually bind first. See [Call concurrency](../concepts/call-concurrency.md).

## Can I scale the services?

The API, runtime, and ARQ worker scale horizontally. The runtime needs session affinity, since each live call holds one WebSocket.

{% hint style="danger" %}
The campaign orchestrator must run as **exactly one** replica. Its state is in-memory and it uses Redis pub/sub, which fans out to every subscriber — two replicas would dial each campaign at twice its configured rate.
{% endhint %}

See [Production deployment](../../guides/deployment/production.md).

## Does Voicera switch language mid-call?

No. An agent declares a primary language and optional secondary ones, but nothing switches during a call. Choose a provider whose model covers the languages you expect, or run separate numbers per language.

## Where are recordings stored?

MinIO, under `voicera-calls/{org_id}/{call_id}/`. Fetch them through the authenticated API rather than the bucket:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/calls/$CALL_ID/recording -o recording.wav
```

Nothing expires automatically — set a retention policy.

## How do I back up?

Three stores together: PostgreSQL (via `pg_dump`), MinIO, and the Chroma volume. Redis is ephemeral. Store `PROVIDER_AUTH_ENCRYPTION_KEY` with the backup — credentials are useless without it.

{% hint style="danger" %}
`docker compose down -v` deletes all four volumes at once, irreversibly.
{% endhint %}

See [Daily operations](operations.md).

## Is there a CI pipeline?

No. There is no `.github/` directory. Run the test suites yourself before opening a pull request — see [Testing](../../developer/guides/testing.md).

## Related

* [Common issues](../troubleshooting/common-issues.md)
* [Operating via the API](operating-via-api.md)
* [Glossary](../concepts/glossary.md)
