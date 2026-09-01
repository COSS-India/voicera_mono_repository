---
description: Where every piece of data lives and how it moves between services.
---

# Data flow

This page traces the paths data actually takes: a call arriving, a call going out, a campaign running, a document being ingested, and credentials reaching the runtime. Each is a separate scenario because each touches a different set of stores.

{% hint style="info" %}
For the static picture — which container talks to which — see [Architecture](architecture.md). This page is about movement over time.
{% endhint %}

## Inbound call

A caller dials a number attached to a `telephony` agent.

```mermaid
sequenceDiagram
  participant C as Caller
  participant P as Telephony provider
  participant RT as Runtime :7860
  participant API as API :8000
  participant AI as STT · LLM · TTS
  participant MN as MinIO

  C->>P: Dials the number
  P->>RT: GET/POST /answer?agent_id=&org_id=
  RT->>API: Bot token, then GET agent + credentials
  API-->>RT: Agent config and decrypted keys
  RT-->>P: Stream XML with the wss:// URL
  P->>RT: Opens WSS, streams audio
  RT->>API: POST /calls (register inbound)
  API-->>RT: call_id
  loop Every turn
    RT->>AI: Audio to STT, text to LLM, reply to TTS
    AI-->>RT: Transcript, tokens, audio
    RT-->>P: Synthesised audio
  end
  C->>P: Hangs up
  RT->>MN: transcript.txt and recording.wav
  RT->>API: PATCH /calls/{call_id} with minio:// URIs
```

The runtime never holds long-lived credentials of its own. It mints a bot token with `INTERNAL_API_KEY`, and the API returns decrypted provider keys for that organisation only. See [Provider credentials](provider-auth.md).

Inbound registration is idempotent — `register_inbound_call` returns the existing record if the same `provider_call_sid` arrives twice, so a provider retry does not create a duplicate call log.

## Outbound call

Triggered by `POST /api/v1/calls/outbound`, or by a campaign batch.

```mermaid
sequenceDiagram
  participant CL as Caller of the API
  participant API as API :8000
  participant RD as Redis
  participant P as Telephony provider
  participant RT as Runtime :7860
  participant DB as FerretDB

  CL->>API: POST /calls {agent_id, to_number}
  API->>API: Resolve agent, validate telephony category
  API->>API: Resolve the from-number
  API->>RD: Acquire a concurrency slot
  RD-->>API: Slot granted
  API->>DB: Create CallLog (initiated)
  API->>P: Initiate call with answer_url
  P-->>API: provider_call_sid
  API->>DB: Store the SID on the CallLog
  P->>RT: GET/POST /answer when the callee answers
  Note over RT: Same pipeline as an inbound call
  RT->>API: PATCH /calls/{call_id} with the outcome
  API->>RD: Release the slot
```

The slot is taken **before** the provider is called, so a burst of requests cannot exceed the organisation's ceiling. If none is free the request fails fast rather than queueing. See [Call concurrency](call-concurrency.md).

## Campaign call

Campaigns add a queue, a worker, and an orchestrator between the request and the call.

```mermaid
sequenceDiagram
  participant OP as Operator
  participant API as API
  participant RD as Redis
  participant ORCH as Orchestrator
  participant ARQ as ARQ worker
  participant DB as FerretDB

  OP->>API: POST /campaign/{id}/start
  API->>DB: state = running
  API->>RD: publish to campaign_events
  RD-->>ORCH: event
  ORCH->>DB: Claim the next batch of QueuedRuns
  ORCH->>RD: Enqueue process_campaign_batch
  RD-->>ARQ: job
  loop Each contact in the batch
    ARQ->>RD: Acquire a concurrency slot
    ARQ->>API: Place the outbound call
    API->>DB: CallLog + QueuedRun updated
  end
  ARQ->>RD: publish BatchCompleted
  RD-->>ORCH: event
  ORCH->>ORCH: Schedule the next batch, or detect completion
```

Nothing polls the database in a loop: the orchestrator reacts to Redis events and only falls back to a timed sweep to catch stalls. See [Campaigns](campaigns.md).

## Knowledge ingest

```mermaid
flowchart LR
  PDF["PDF upload<br/>POST /knowledge"]
  TXT["pdf_to_text"]
  CH["chunk_text<br/>1000 chars, 200 overlap"]
  EM["embed_chunks<br/>batches of 100"]
  CR[("Chroma<br/>per-org collection")]
  MD[("FerretDB<br/>KnowledgeDocuments")]

  PDF --> TXT --> CH --> EM --> CR
  PDF --> MD
```

The document's metadata and status live in FerretDB; the vectors live in Chroma, on the `voicera_oss_chroma_data` volume. At call time the runtime retrieves chunks either as a tool the LLM can call or as prepended context. See [Knowledge base](knowledge-base-rag.md).

## Call artifacts

```mermaid
flowchart LR
  RT["Runtime"]
  MN[("MinIO<br/>voicera-calls")]
  API["API"]
  CL["Client"]

  RT -- "PUT at call end" --> MN
  RT -- "PATCH minio:// URIs" --> API
  CL -- "GET /calls/{id}/recording" --> API
  API -- "fetch object" --> MN
  API -- "stream bytes" --> CL
```

Layout inside the bucket:

```
voicera-calls/{org_id}/{call_id}/transcript.txt
voicera-calls/{org_id}/{call_id}/recording.wav
```

The CallLog stores `minio://` URIs, not signed URLs. Clients always fetch through the authenticated API proxy, so bucket access never has to be public.

{% hint style="warning" %}
Browser websocket sessions register a `call_type: web` CallLog on connect, so they produce transcripts and recordings under the same MinIO paths as telephony calls.
{% endhint %}

## Credentials

```mermaid
sequenceDiagram
  participant AD as Admin
  participant API as API
  participant DB as FerretDB
  participant RT as Runtime

  AD->>API: POST /auth {provider, auth}
  API->>API: Fernet encrypt with PROVIDER_AUTH_ENCRYPTION_KEY
  API->>DB: Store the encrypted blob
  RT->>API: POST /users/bot/token (X-API-Key)
  API-->>RT: Org-scoped JWT
  RT->>API: GET /auth/{provider}
  API->>DB: Read the blob
  API->>API: Decrypt
  API-->>RT: Plaintext credentials
  RT->>RT: Build the STT, TTS, and LLM services
```

Secrets are written once and read at call time. They are never stored on the agent document, and only secret fields are kept — non-secret settings live on the agent's config.

## Where everything lives

| Data | Store | Volume | Survives `docker compose down`? |
| --- | --- | --- | --- |
| Users, orgs, agents, numbers, call logs, campaigns | FerretDB on PostgreSQL | `voicera_oss_ferretdb_postgres_data` | Yes, unless `-v` |
| Provider credentials (encrypted) | FerretDB | same | Yes, unless `-v` |
| Recordings and transcripts | MinIO | `voicera_oss_minio_data` | Yes, unless `-v` |
| RAG vectors | Chroma | `voicera_oss_chroma_data` | Yes, unless `-v` |
| Job queue, campaign events, concurrency slots | Redis | `voicera_oss_redis_data` | Yes, but treat as ephemeral |
| Uploaded CSVs | MinIO | `voicera_oss_minio_data` | Yes, unless `-v` |

{% hint style="danger" %}
`docker compose down -v` deletes every volume — all four stores at once. A backup means Postgres, MinIO, and Chroma together; see [Daily operations](../operator/operations.md).
{% endhint %}

## Related

* [Architecture](architecture.md)
* [Voice pipeline](voice-pipeline.md)
* [Campaigns](campaigns.md)
* [Calls and call artifacts](calls.md)
