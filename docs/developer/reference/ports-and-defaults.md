---
description: Every port Voicera uses, its default, and how to change it.
---

# Ports and defaults

Which ports the stack binds, which of them reach your host, and the variable that changes each one. Extracted from `docker-compose.yaml`, `.env.example`, `model-server/compose.model-server.yml`, and `model-server/README.md`.

{% hint style="info" %}
Container port means the port a process binds inside its container. Host port means what Compose publishes on your machine. They differ for FerretDB and for the model-server gateway, and getting them the wrong way round is the most common connection failure in this stack.
{% endhint %}

## Main stack

From the `ports:` blocks in `docker-compose.yaml`. "Published?" means Compose maps it onto the host.

| Service | Host port | Container port | Published? | Override variable | Purpose |
|---|---|---|---|---|---|
| `api` | `8000` | `8000` | Yes | `API_HOST_PORT` | REST API, OpenAPI docs at `/docs`. |
| `runtime` | `7860` | `7860` | Yes | `RUNTIME_HOST_PORT` | Telephony `/answer` webhook and the media WebSocket. |
| `ferretdb` | `27018` | `27017` | Yes | `FERRETDB_HOST_PORT` | MongoDB wire protocol over PostgreSQL. |
| `minio` | `9000` | `9000` | Yes | `MINIO_API_PORT` | S3 API for recordings, transcripts, CSVs, PDFs. |
| `minio` (console) | `9001` | `9001` | Yes | `MINIO_CONSOLE_PORT` | MinIO web console. |
| `postgres` | — | `5432` | **No** | — | FerretDB's storage engine. Internal only. |
| `redis` | — | `6379` | **No** | — | ARQ job queue and campaign pub/sub. Internal only. |
| `arq-worker` | — | — | No | — | Background job worker. Binds no port. |
| `campaign-orchestrator` | — | — | No | — | Campaign batch scheduler. Binds no port. |
| `minio-init` | — | — | No | — | One-shot bucket creation, then exits. |

### FerretDB is 27018 on the host, 27017 in the container

The mapping is `"${FERRETDB_HOST_PORT:-27018}:27017"`. Which port you use depends on where you are connecting from:

* From your host — a `mongosh` session, a GUI client, or the API run outside Docker — use **27018**. That is what `.env.example` sets `MONGODB_PORT` to.
* From inside the stack, use **27017** against the hostname `mongodb`. `docker-compose.yaml` overrides `MONGODB_HOST` to `mongodb` and `MONGODB_PORT` to `27017` on `api`, `arq-worker`, and `campaign-orchestrator` for exactly this reason.

`mongodb` is a network alias on the `ferretdb` service, not a separate container. The container itself is `voicera_oss_ferretdb`.

```bash
mongosh "mongodb://admin:admin123@localhost:27018/voicera"
```

### Postgres and Redis publish nothing

Neither service has a `ports:` block. They are reachable only from `app-network`, by service name — `postgres:5432` and `redis:6379`. `redis://localhost:6379` in `.env.example` is for running the API on your host against a Redis you started yourself; it does not describe the Docker stack, where Compose rewrites `REDIS_URL` to `redis://:${REDIS_PASSWORD}@redis:6379`.

To reach either from your host for debugging, use `docker compose exec` rather than adding a `ports:` block:

```bash
docker compose exec redis redis-cli -a redissecret ping
docker compose exec postgres psql -U admin -d postgres
```

## Model-server stack

A separate compose project (`model-server/compose.model-server.yml`, project name `voicera-model-server`) with its own network, `model_net`. `ports:` appears exactly once in the whole file.

| Service | Host port | Container port | Published? | Override variable | Purpose |
|---|---|---|---|---|---|
| `gateway` | `8100` | `8000` | Yes | `GATEWAY_PORT` | The single entry point. Routes on modality and streams to the slots. |
| `stt` | — | `8001` | **No** | `PORT` (fixed in compose) | Speech-to-text slot. Reachable at `http://stt:8001`. |
| `tts` | — | `8002` | **No** | `PORT` (fixed in compose) | Text-to-speech slot. Reachable at `http://tts:8002`. |
| `llm` | — | `8003` | **No** | `PORT` (fixed in compose) | Language-model slot. Reachable at `http://llm:8003`. |

The gateway is the only published port. Model containers bind nothing on the host, which is why this stack can sit beside other stacks without competing for ports. The slot ports never change when you swap a model — a slot is one service on a fixed port, and which model fills it is a folder name in `model-server/.env`.

To reach a model container directly for debugging, `model-server/README.md` recommends `docker compose exec` or a temporary `ports:` mapping rather than a permanent one.

Override the upstream addresses only to point a slot at a different host entirely, with `STT_UPSTREAM`, `TTS_UPSTREAM`, or `LLM_UPSTREAM`. They are commented out in `model-server/.env.example`; unset means "use the Compose service name".

## Default credentials

These ship in `.env.example` and are what the stack comes up with if you change nothing.

| What | Username | Password | Where |
|---|---|---|---|
| MinIO root and application keys | `minioadmin` | `minioadmin123` | `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD`, `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` |
| FerretDB and PostgreSQL | `admin` | `admin123` | `MONGODB_USER` / `MONGODB_PASSWORD`, reused as `POSTGRES_USER` / `POSTGRES_PASSWORD` |
| Redis | — | `redissecret` | `REDIS_PASSWORD` |

{% hint style="warning" %}
Change all three before any deployment reachable from outside your machine. MinIO and FerretDB are both published on the host by default, so these credentials are live on `localhost:9000` and `localhost:27018` the moment the stack starts. Redis is not published, but its password is interpolated into `REDIS_URL` — changing `REDIS_PASSWORD` alone is enough, because Compose rebuilds the URL from it. See [Security hardening](../../guides/deployment/security-hardening.md).
{% endhint %}

There is no default Voicera login. The first `super_admin` is created by `POST /api/v1/users/signup` — see [Create your first agent](../../guides/quickstart/first-agent.md).

## Changing a host port

Set the variable in the root `.env` and recreate the stack. The container port is unaffected, so nothing inside the stack needs to change:

```bash
API_HOST_PORT=8080
```

```bash
docker compose up -d
```

If you move `FERRETDB_HOST_PORT`, also update `MONGODB_PORT` in `.env` — that value is what a host-side API process connects with. Inside the stack it stays `27017` regardless, because `docker-compose.yaml` overrides it.

## Related

* [Environment variables](environment-variables.md)
* [Endpoints cheatsheet](../../api-reference/endpoints-cheatsheet.md)
* [Docker Compose](../../guides/deployment/docker-compose.md)
* [Data store (FerretDB)](../../guides/concepts/data-store.md)
* [Overview](../model-server/overview.md)
