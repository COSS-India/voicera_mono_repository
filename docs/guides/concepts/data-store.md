---
description: Why Voicera speaks the MongoDB wire protocol to a PostgreSQL database.
---

# Data store

Voicera stores its documents in **FerretDB**, which speaks the MongoDB wire protocol on top of PostgreSQL. Application code uses `pymongo` and never knows the difference; operationally, your data lives in Postgres.

{% hint style="info" %}
This page explains the arrangement and the two things that surprise people: the port numbers and the empty `MONGODB_AUTH_SOURCE`. For field-level detail see [Data model](../../developer/reference/data-model.md).
{% endhint %}

## FerretDB in one paragraph

[FerretDB](https://www.ferretdb.com/) is a proxy that accepts MongoDB wire-protocol connections and translates them into SQL against a PostgreSQL server carrying the DocumentDB extension. You get MongoDB's document model and driver ecosystem with PostgreSQL's storage, backup, and operational tooling. Voicera never depends on MongoDB Inc. software, and the licence stays permissive.

## The container pair

```mermaid
flowchart LR
  API["api"]
  ARQ["arq-worker"]
  ORCH["campaign-orchestrator"]
  ALIAS["network alias<br/>mongodb"]
  FDB["ferretdb :27017<br/>wire protocol"]
  PG[("postgres :5432<br/>documentdb 17")]
  VOL[("voicera_oss_ferretdb<br/>_postgres_data")]

  API --> ALIAS
  ARQ --> ALIAS
  ORCH --> ALIAS
  ALIAS --> FDB
  FDB --> PG
  PG --> VOL
```

Two containers, pinned in `docker-compose.yaml`:

| Container | Image | Published |
| --- | --- | --- |
| `voicera_oss_postgres` | `ghcr.io/ferretdb/postgres-documentdb:17-0.107.0-ferretdb-2.7.0` | No — internal only |
| `voicera_oss_ferretdb` | `ghcr.io/ferretdb/ferretdb:2.7.0` | Yes — `27018` on the host |

FerretDB reaches Postgres over `FERRETDB_POSTGRESQL_URL`, and `postgres` must pass its `pg_isready` healthcheck before FerretDB starts.

## Ports: 27018 outside, 27017 inside

This is the detail that trips people up.

```
ports:
  - "${FERRETDB_HOST_PORT:-27018}:27017"
```

| Where you are | Host | Port |
| --- | --- | --- |
| On your machine, connecting from outside Docker | `localhost` | `27018` |
| Inside the Compose network | `mongodb` | `27017` |

The container always listens on `27017`. The host mapping is `27018` so it cannot collide with a MongoDB you already run locally. Change it with `FERRETDB_HOST_PORT`.

{% hint style="warning" %}
`.env.example` ships `MONGODB_PORT=27018` because the default assumes you are running the API **on your host** against the Dockerised database. Inside the stack, `docker-compose.yaml` overrides `MONGODB_HOST=mongodb` and `MONGODB_PORT=27017`. If you set these by hand, match them to where the process actually runs.
{% endhint %}

## The `mongodb` network alias

The FerretDB service publishes a network alias:

```yaml
networks:
  app-network:
    aliases:
      - mongodb
```

So in-stack services connect to `mongodb:27017`. The alias keeps connection strings readable and means nothing in the application refers to "ferretdb" by name.

## Authentication

Credentials are PostgreSQL users. FerretDB negotiates SCRAM against them, which is why the auth source is deliberately blank:

```bash
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_AUTH_SOURCE=          # empty for FerretDB
# MONGODB_AUTH_MECHANISM=SCRAM-SHA-256   # usually omit; let the driver negotiate
```

`apps/api/app/config.py` builds the URI and appends `authSource` or `authMechanism` **only when they are non-empty**:

```
mongodb://admin:admin123@mongodb:27017/voicera
```

{% hint style="warning" %}
Setting `MONGODB_AUTH_SOURCE=admin` — correct for real MongoDB, and what the old mono repo used — appends `?authSource=admin` and authentication fails. Leave it empty unless you have pointed Voicera at an actual MongoDB server.
{% endhint %}

The same values become `POSTGRES_USER` and `POSTGRES_PASSWORD`, so one credential pair covers both layers. Change them in [Security hardening](../deployment/security-hardening.md).

## Connecting by hand

{% tabs %}
{% tab title="mongosh" %}
```bash
mongosh "mongodb://admin:admin123@localhost:27018/voicera"
```

```javascript
show collections
db.Agents.countDocuments()
db.CallLogs.find().sort({ created_at: -1 }).limit(5)
```
{% endtab %}

{% tab title="psql" %}
```bash
docker exec -it voicera_oss_postgres psql -U admin -d postgres
```

Documents are stored in DocumentDB's internal schema — readable, but not a substitute for the Mongo view. Use `psql` for backups and vacuum, `mongosh` for data.
{% endtab %}

{% tab title="Compass" %}
Connection string:

```
mongodb://admin:admin123@localhost:27018/voicera
```

Leave the authentication database blank.
{% endtab %}
{% endtabs %}

## Collections and indexes

`apps/api/app/database_init.py` runs on every API start. It is idempotent: it creates any missing collection and ensures every index, so a fresh volume becomes a working database with no migration step.

Collections: `Organizations`, `Users`, `Memberships`, `ProviderAuth`, `Agents`, `PhoneNumbers`, `KnowledgeDocuments`, `CallLogs`, `Campaigns`, `QueuedRuns`.

Fields and indexes are documented in [Data model](../../developer/reference/data-model.md).

{% hint style="info" %}
There is no Alembic or migration tool. Schema is enforced by Pydantic models at the edge, and indexes are reconciled at startup.
{% endhint %}

## Backups

Back up **PostgreSQL**, not FerretDB — Postgres holds the bytes.

```bash
# Dump
docker exec voicera_oss_postgres pg_dump -U admin postgres | gzip > voicera-$(date +%F).sql.gz

# Restore into a fresh stack
gunzip -c voicera-2026-09-01.sql.gz | docker exec -i voicera_oss_postgres psql -U admin -d postgres
```

The volume `voicera_oss_ferretdb_postgres_data` is the other thing to snapshot. Note that a full backup also needs MinIO (recordings and transcripts) and the Chroma volume (RAG vectors) — see [Daily operations](../operator/operations.md).

## Differences from MongoDB you may hit

FerretDB implements most of the wire protocol, not all of it. Known limits relevant here:

| Area | Behaviour |
| --- | --- |
| Transactions | Multi-document ACID transactions are not fully supported. Voicera does not rely on them. |
| Aggregation | Common stages work; exotic operators may not. Campaign reporting stays within basic stages for this reason. |
| Change streams | Not available. Voicera uses Redis pub/sub for eventing instead — see [Campaigns](campaigns.md). |
| `$where`, server-side JS | Not supported. |
| Index types | Standard and compound indexes work; specialised types may differ. |

If you swap in real MongoDB, set `MONGODB_AUTH_SOURCE=admin` and point `MONGODB_HOST` and `MONGODB_PORT` at it. Nothing else in the application changes.

## Related

* [Data model](../../developer/reference/data-model.md) — collections, fields, enumerations
* [Environment variables](../../developer/reference/environment-variables.md) — every `MONGODB_*` setting
* [Ports and defaults](../../developer/reference/ports-and-defaults.md)
* [Architecture](architecture.md)
