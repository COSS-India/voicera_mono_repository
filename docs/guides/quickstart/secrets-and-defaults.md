---
description: The secrets Voicera generates, the defaults it ships, and what you must change.
---

# Generated secrets and defaults

Voicera ships with working defaults so the stack starts on the first try. They are **development defaults**. This page lists all of them and what each one protects.

## There is no default user

No seeded account, no default login. The first `POST /users/signup` creates the user, an organisation, and a `super_admin` membership. Whoever signs up first owns the deployment — do it immediately after starting.

## Generated secrets

`./scripts/start_docker.sh` generates these into `.env` when blank, and never overwrites an existing value.

| Variable | Protects | If it changes |
| --- | --- | --- |
| `SECRET_KEY` | Signs JWTs (HS256) | Every outstanding token is invalidated; users log in again |
| `INTERNAL_API_KEY` | Service-to-service auth — the runtime uses it to mint org-scoped tokens | The runtime cannot reach the API until updated |
| `PROVIDER_AUTH_ENCRYPTION_KEY` | Fernet-encrypts stored provider credentials | **Every stored credential becomes permanently undecryptable** |

Generate by hand:

```bash
# SECRET_KEY, INTERNAL_API_KEY
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# PROVIDER_AUTH_ENCRYPTION_KEY
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

{% hint style="danger" %}
`PROVIDER_AUTH_ENCRYPTION_KEY` is one-way. There is no recovery path and no re-encryption tool: rotating it orphans every stored credential, and each organisation must re-enter every provider key. Back it up with the same care as the database.
{% endhint %}

{% hint style="warning" %}
If `SECRET_KEY` is blank, the API does **not** fail — `apps/api/app/auth.py` logs a warning and generates a temporary key at import. Tokens then die on every restart, and separate replicas reject each other's. Always set it.
{% endhint %}

## Shipped defaults

These are in `.env.example` and `docker-compose.yaml`. Change every one before exposing anything.

| Service | Variable | Default |
| --- | --- | --- |
| FerretDB / PostgreSQL | `MONGODB_USER` | `admin` |
| | `MONGODB_PASSWORD` | `admin123` |
| MinIO | `MINIO_ROOT_USER` / `MINIO_ACCESS_KEY` | `minioadmin` |
| | `MINIO_ROOT_PASSWORD` / `MINIO_SECRET_KEY` | `minioadmin123` |
| Redis | `REDIS_PASSWORD` | `redissecret` |

`MONGODB_USER` and `MONGODB_PASSWORD` are used **twice** — as the PostgreSQL superuser and as the FerretDB credentials. Changing them changes both.

{% hint style="warning" %}
Changing database credentials after the volume exists does not update the PostgreSQL user. Either set them before the first start, or change the password inside Postgres as well.
{% endhint %}

## Other defaults worth knowing

| Setting | Default | Meaning |
| --- | --- | --- |
| `ACCESS_TOKEN_EXPIRE_MINUTES` | `30` | Token lifetime |
| `SAMPLE_RATE` | `8000` | Telephony audio |
| `WEBSOCKET_SAMPLE_RATE` | `16000` | Browser audio |
| `CAMPAIGN_BATCH_SIZE` | `10` | Calls per campaign batch |
| `DEFAULT_ORG_CONCURRENCY_LIMIT` | `10` | Simultaneous calls per organisation |
| `MINIO_BUCKET` | `voicera-calls` | Recordings and transcripts |
| `DEBUG` | `False` | Verbose logging |

Full list in [Environment variables](../../developer/reference/environment-variables.md).

## Before you expose this

- [ ] Signed up the first user
- [ ] Changed `MONGODB_PASSWORD`
- [ ] Changed `MINIO_ROOT_PASSWORD` and the matching access keys
- [ ] Changed `REDIS_PASSWORD`
- [ ] Confirmed all three generated secrets are non-empty
- [ ] Backed up `PROVIDER_AUTH_ENCRYPTION_KEY` somewhere safe
- [ ] Stopped publishing the MinIO console publicly
- [ ] Put TLS in front of the API and runtime

Full guidance: [Security hardening](../deployment/security-hardening.md).

## Related

* [Environment variables](../../developer/reference/environment-variables.md)
* [Multi-tenancy and roles](../concepts/multi-tenancy.md)
* [Provider credentials](../concepts/provider-auth.md)
