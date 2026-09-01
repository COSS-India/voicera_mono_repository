---
description: Clone the repository, start the Docker stack, and verify every service is healthy.
---

# Install and run

Gets the whole stack running on one machine in a few minutes. Complete the [prerequisites](prerequisites.md) first.

## 1. Clone

```bash
git clone https://github.com/COSS-India/voicera.git
cd voicera
```

## 2. Start

```bash
./scripts/start_docker.sh
```

The script must run from the repository root. It:

1. Creates `.env` from `.env.example` if missing.
2. Generates `SECRET_KEY` and `INTERNAL_API_KEY` if either is blank.
3. Generates `PROVIDER_AUTH_ENCRYPTION_KEY` (a Fernet key) if blank.
4. Prints the URLs it is about to expose, and asks to confirm.
5. Runs `docker compose -f docker-compose.yaml up --build -d`.

Existing values are never overwritten — rerunning is safe.

{% hint style="warning" %}
Do not start with a bare `docker compose up` on a fresh checkout. Three services declare `${SECRET_KEY:?...}`, so Compose aborts without it. The script exists to generate these secrets.
{% endhint %}

The first build pulls images and compiles dependencies — expect several minutes. Subsequent starts are fast.

To skip the prompt, pipe from a non-interactive shell, or pass extra Compose arguments after `--`:

```bash
./scripts/start_docker.sh -- --no-build
```

## 3. What came up

```bash
docker compose ps
```

Nine containers:

| Container | Purpose | Published |
| --- | --- | --- |
| `voicera_oss_postgres` | Storage engine behind FerretDB | No |
| `voicera_oss_ferretdb` | MongoDB wire protocol | `27018` |
| `voicera_oss_api` | REST API | `8000` |
| `voicera_oss_runtime` | Voice pipeline | `7860` |
| `voicera_oss_arq_worker` | Campaign batches | No |
| `voicera_oss_campaign_orchestrator` | Campaign scheduling | No |
| `voicera_oss_redis` | Queue, events, concurrency | No |
| `voicera_oss_minio` | Recordings and transcripts | `9000`, `9001` |
| `voicera_oss_minio_init` | One-shot bucket creation, then exits | No |

`minio-init` exiting with code 0 is correct — it creates the bucket and stops.

## 4. Verify

```bash
curl -s localhost:8000/health
```

```json
{"status": "ok", "database": "up"}
```

```bash
curl -s localhost:7860/health
```

{% hint style="warning" %}
`/health` returns HTTP **200 even when the database is down** — only the body changes to `"status": "degraded"`. Check the body, not the status code.
{% endhint %}

Open the interactive API console at [http://localhost:8000/docs](http://localhost:8000/docs). With no dashboard in the core stack, this is your primary interface.

| What | Where |
| --- | --- |
| API | `http://localhost:8000` |
| OpenAPI console | `http://localhost:8000/docs` |
| ReDoc | `http://localhost:8000/redoc` |
| Runtime | `http://localhost:7860` |
| MinIO console | `http://localhost:9001` — `minioadmin` / `minioadmin123` |
| FerretDB | `mongodb://admin:admin123@localhost:27018/voicera` |

## 5. Create the first user

There is no seeded account. Signup creates a user, an organisation, and makes you its `super_admin`:

```bash
curl -X POST http://localhost:8000/api/v1/users/signup \
  -H "Content-Type: application/json" \
  -d '{
    "email": "you@example.com",
    "password": "change-me",
    "full_name": "Your Name",
    "organisation_name": "Your Org"
  }'
```

The response carries an `access_token`. Keep it:

```bash
export TOKEN="paste-the-token"
curl -H "Authorization: Bearer $TOKEN" http://localhost:8000/api/v1/users/me
```

Tokens expire after 30 minutes by default.

## Logs

```bash
docker compose logs -f api runtime
docker compose logs runtime --tail 100
```

Logs rotate at 10 MB with three files kept per container.

## Stop and reset

```bash
./scripts/stop_services.sh
```

That is `docker compose down` — containers stop, data survives.

{% hint style="danger" %}
`./scripts/stop_services.sh -- -v` (or `docker compose down -v`) also deletes the volumes: your database, recordings, RAG vectors, and queue. There is no undo. Back up first — see [Daily operations](../operator/operations.md).
{% endhint %}

## Running the API on your host instead

Useful for development with hot reload:

```bash
docker compose -f docker-compose.yaml up postgres ferretdb
```

Then, in `apps/api`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Point `.env` at the published database port — `MONGODB_HOST=localhost`, `MONGODB_PORT=27018`. See [Local setup](../../developer/guides/local-setup.md).

## Troubleshooting

| Symptom | Page |
| --- | --- |
| `SECRET_KEY must be set` | [Common issues](../troubleshooting/common-issues.md) |
| Port already in use | [Ports and defaults](../../developer/reference/ports-and-defaults.md) |
| Database connection errors on first boot | [Common issues](../troubleshooting/common-issues.md) |

## Next

[Create your first agent](first-agent.md)
