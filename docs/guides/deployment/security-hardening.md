---
description: What to change before exposing a Voicera deployment.
---

# Security hardening

Voicera ships development defaults so the stack starts on the first try. Work through this page before anything is reachable beyond your laptop.

## Change every default

All of these are public knowledge — they are in `.env.example` and `docker-compose.yaml`.

| Variable | Ships as | Protects |
| --- | --- | --- |
| `MONGODB_PASSWORD` | `admin123` | The database, **and** the PostgreSQL superuser |
| `MINIO_ROOT_PASSWORD` / `MINIO_SECRET_KEY` | `minioadmin123` | Recordings and transcripts |
| `MINIO_ROOT_USER` / `MINIO_ACCESS_KEY` | `minioadmin` | Same |
| `REDIS_PASSWORD` | `redissecret` | Job queue, campaign events, concurrency slots |

{% hint style="warning" %}
`MONGODB_USER` and `MONGODB_PASSWORD` become the PostgreSQL credentials too. Changing them after the volume exists does **not** update the Postgres user — set them before the first start, or change the password inside Postgres as well.
{% endhint %}

## The three generated secrets

`./scripts/start_docker.sh` generates these when blank and never overwrites them.

| Secret | Protects | Rotation |
| --- | --- | --- |
| `SECRET_KEY` | JWT signatures (HS256) | Invalidates all tokens; users log in again |
| `INTERNAL_API_KEY` | Service-to-service auth | Update API and runtime together |
| `PROVIDER_AUTH_ENCRYPTION_KEY` | Fernet encryption of stored credentials | **One-way — see below** |

{% hint style="danger" %}
`PROVIDER_AUTH_ENCRYPTION_KEY` cannot be rotated. There is no re-encryption tool: changing or losing it makes every stored provider credential permanently undecryptable, and every organisation must re-enter every key. Back it up alongside the database, and store it with the same protection.
{% endhint %}

{% hint style="warning" %}
If `SECRET_KEY` is blank the API does **not** fail. `apps/api/app/auth.py` logs a warning and generates a temporary key at import — so tokens die on every restart and replicas reject each other's tokens. Verify it is set:

```bash
grep '^SECRET_KEY=' .env
```
{% endhint %}

Generate:

```bash
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Keep them out of files on disk in production — inject from a secret manager.

## CORS

`apps/api/app/main.py` sets:

```python
allow_origins=["*"], allow_credentials=True
```

Any origin can call the API with credentials. Restrict it to the origins that actually need it and rebuild.

## The internal API key

`INTERNAL_API_KEY` is a **single shared credential with organisation-wide reach**. `POST /users/bot/token` exchanges it plus an `org_id` for a token with role `admin` in that organisation — for any organisation.

| Rule | Why |
| --- | --- |
| Never send it from a browser | It is not a user credential |
| Never expose the routes that accept it publicly | They bypass user auth by design |
| Rotate on any suspicion | One value protects every tenant |

## The unauthenticated runtime endpoints

{% hint style="warning" %}
`GET|POST /answer` and `WS /agent/{org_id}/{agent_id}` have **no authentication**. They must be publicly reachable for telephony to work, and the runtime resolves everything from the path — so anyone who learns an org and agent id pair can open a pipeline session and spend your model credits.
{% endhint %}

Mitigations:

* Rate limit `/answer` and `/agent` at the proxy.
* IP-allowlist your telephony provider's published ranges.
* Treat org and agent ids as semi-secret — do not put them in public pages or client-side code.
* Monitor call volume for unexplained sessions.

## Network exposure

Publish only what must be public:

| Service | Expose |
| --- | --- |
| API `:8000` | Behind TLS, to your clients |
| Runtime `:7860` | Behind TLS, to your telephony provider |
| FerretDB `:27018` | **Never publicly.** Bind to localhost or drop the mapping. |
| MinIO `:9000` | Private |
| MinIO console `:9001` | **Never publicly.** It is an admin UI. |
| Model gateway `:8100` | Private only — it has no authentication |
| PostgreSQL, Redis | Already unpublished; keep it that way |

To stop publishing a port, remove its `ports:` entry or bind it to loopback:

```yaml
ports:
  - "127.0.0.1:27018:27017"
```

## TLS

Telephony providers require HTTPS for webhooks and WSS for audio, so TLS is mandatory rather than optional. Terminate at a reverse proxy — see [Production deployment](production.md) for a working nginx configuration, including the WebSocket upgrade headers and the long read timeouts calls need.

For Redis over TLS use a `rediss://` URL; the ARQ settings enable TLS when they see that scheme.

## Email enumeration

{% hint style="warning" %}
`GET /users/check/{email}` is **public and unauthenticated**, and confirms whether an account exists. Rate limit it at the proxy, or require authentication if you do not need the invite-flow convenience.
{% endhint %}

## Health probes

`GET /health` returns HTTP **200 even when the database is down** — only the body changes to `"status": "degraded"`. Configure probes to parse the body, or a broken API will look healthy.

## Images and dependencies

* `minio/minio:latest` is unpinned — pin a digest for reproducible deployments. The FerretDB, Postgres, and Redis images are already pinned.
* Rebuild periodically to pick up base-image security updates.
* Voicera has no CI, so nothing scans dependencies automatically. Run `pip-audit` or equivalent yourself.

## Log hygiene

Logs go to `json-file`, rotating at 10 MB with three files kept. Before shipping them anywhere central, confirm no provider keys or tokens appear — and note that `DEBUG=True` substantially increases what is logged. Keep it `False` in production.

## Data protection

You hold call recordings, transcripts, and contact lists. That is regulated data in most jurisdictions.

* Encrypt volumes at rest.
* Set a retention policy — nothing expires automatically.
* Restrict MinIO access; artifacts are served through the authenticated API proxy, so the bucket never needs to be public.
* Remember `docker compose down -v` destroys all of it irreversibly.

## Checklist

- [ ] `MONGODB_PASSWORD` changed
- [ ] `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` changed
- [ ] `REDIS_PASSWORD` changed
- [ ] `SECRET_KEY` set and non-empty
- [ ] `INTERNAL_API_KEY` set and non-empty
- [ ] `PROVIDER_AUTH_ENCRYPTION_KEY` set and backed up
- [ ] `SECRET_KEY` identical across API replicas
- [ ] CORS restricted
- [ ] TLS on the API and runtime
- [ ] FerretDB, MinIO console, and the model gateway not publicly reachable
- [ ] Rate limiting on `/answer`, `/agent`, and `/users/check`
- [ ] `DEBUG=False`
- [ ] Volumes encrypted, retention policy set
- [ ] Backups tested by restoring

## Related

* [Production deployment](production.md)
* [Generated secrets and defaults](../quickstart/secrets-and-defaults.md)
* [Provider credentials](../concepts/provider-auth.md)
* [Multi-tenancy and roles](../concepts/multi-tenancy.md)
