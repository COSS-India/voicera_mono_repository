---
description: The failures people hit most often when starting or running Voicera.
---

# Common issues

Start here. If your problem is specifically about audio, telephony, campaigns, or deploying behind a proxy, the sibling pages go deeper.

## The stack will not start

### `SECRET_KEY must be set to a strong secret`

Compose refuses to start `api`, `arq-worker`, or `campaign-orchestrator` without it — the variable is declared `${SECRET_KEY:?...}`, which fails the whole command rather than starting a broken stack.

```bash
./scripts/start_docker.sh
```

The script generates `SECRET_KEY`, `INTERNAL_API_KEY`, and `PROVIDER_AUTH_ENCRYPTION_KEY` into the root `.env` if they are missing. Starting with a bare `docker compose up` on a fresh checkout is what produces this error.

To generate one by hand:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

### `docker-compose.yaml not found at repo root`

`start_docker.sh` must run from the repository root:

```bash
cd /path/to/voicera
./scripts/start_docker.sh
```

### A port is already in use

Every published port is overridable in `.env`:

| Conflict | Change |
| --- | --- |
| `8000` | `API_HOST_PORT` |
| `7860` | `RUNTIME_HOST_PORT` |
| `27018` | `FERRETDB_HOST_PORT` |
| `9000` / `9001` | `MINIO_API_PORT` / `MINIO_CONSOLE_PORT` |

Find the offender:

```bash
lsof -i :8000
```

See [Ports and defaults](../../developer/reference/ports-and-defaults.md).

## Database connection failures

### The API logs a connection error on first boot

Expected on a cold start, and self-correcting. Only `redis` and `minio` gate on healthchecks — `api` waits for `ferretdb` merely to *start*, so it can attempt a query before FerretDB is serving. `restart: unless-stopped` recovers it within a few seconds.

If it does not settle:

```bash
docker compose logs postgres | tail -30
docker compose ps
```

### `Authentication failed` against FerretDB

Almost always `MONGODB_AUTH_SOURCE`. It must be **empty** for FerretDB:

```bash
MONGODB_AUTH_SOURCE=
```

Setting it to `admin` — correct for real MongoDB, and what the old mono repo used — appends `?authSource=admin` and authentication fails. See [Data store](../concepts/data-store.md).

### Connection refused on port 27017

You are using the container port from outside Docker. From your host the port is **27018**; `27017` only works inside the Compose network:

```bash
mongosh "mongodb://admin:admin123@localhost:27018/voicera"
```

## Configuration is ignored

### An `.env` change had no effect

Two causes.

**Compose overrides some variables deliberately.** Values under a service's `environment:` beat `env_file`, so `MONGODB_HOST`, `API_BASE_URL`, and `MINIO_ENDPOINT` are pinned to in-network addresses regardless of `.env`. That is intended — see [Environment variables](../../developer/reference/environment-variables.md).

**Containers read `.env` at start.** Restart after editing:

```bash
docker compose up -d --force-recreate api runtime
```

### Tokens stop working after a restart

`SECRET_KEY` is unset. `apps/api/app/auth.py` logs a warning and generates a **temporary** key at import, so every restart invalidates every outstanding token — and separate replicas reject each other's.

```bash
grep '^SECRET_KEY=' .env
```

If it is blank, set it and restart. See [Security hardening](../deployment/security-hardening.md).

## Authentication and permissions

| Symptom | Cause |
| --- | --- |
| `401 Invalid authentication credentials` | Missing, malformed, or expired token. Tokens last 30 minutes by default — log in again. |
| `403` on a valid token | Your role is too low for that route. Role checks are explicit, so 403 means authenticated but not permitted. See [Multi-tenancy](../concepts/multi-tenancy.md). |
| `404` for a resource you know exists | It belongs to a different organisation. Existence is never leaked across tenants — switch with `POST /users/switch-organisation`. |
| `401 Missing API key` | An internal route needs `X-API-Key: $INTERNAL_API_KEY`. |
| `500 Internal API key not configured` | `INTERNAL_API_KEY` is empty in `.env`. |

## Provider and agent errors

### Provider dropdowns or catalogs come back empty

Catalogs are filtered by what you have configured. Store credentials first:

```bash
curl -X POST http://localhost:8000/api/v1/auth \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"provider": "openai", "auth": {"api_key": "sk-..."}}'

curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/auth/configured
```

### Creating a telephony agent fails

`VOICE_SERVER_BASE_URL` must be set before creating telephony agents — the API bakes the answer URL into the provider application at create time. You also need credentials stored for that telephony provider. See [Public voice URLs](../deployment/public-voice-urls.md).

### `422 Unprocessable Entity` on agent create

The config blob failed validation. The response body names the offending field path. Common causes: an unregistered `provider`, or **API keys placed in `config.models`** — those belong in `ProviderAuth`, and only non-secret settings go on the agent. See [Agent configuration](../../developer/reference/agent-configuration.md).

## Import errors when running from source

```
ModuleNotFoundError: No module named 'apps'
```

`apps/runtime` imports `apps.providers` and `apps.telephony`, so the repository root must be on `PYTHONPATH`:

```bash
cd /path/to/voicera
PYTHONPATH=. python -m apps.runtime.app
```

There is no `pip install -e .` — `pyproject.toml` is an empty placeholder. Install per app with `pip install -r apps/<app>/requirements.txt`. See [Local setup](../../developer/guides/local-setup.md).

## Checking health

```bash
curl -s localhost:8000/health   # API and its database
curl -s localhost:7860/health   # runtime
curl -s localhost:8100/health   # model server, if running
```

{% hint style="warning" %}
`GET /health` returns HTTP **200 even when the database is down** — the body says `"status": "degraded"`. A probe that only checks the status code will not notice. Parse the body.
{% endhint %}

## Where next

* [Voice and audio](voice-and-audio.md)
* [Telephony](telephony.md)
* [Campaigns](campaigns.md)
* [Deployment](deployment.md)
