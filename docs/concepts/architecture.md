---
description: High-level architecture of VoicEra explained with the C4 model.
---

# Architecture

This page gives a high-level view of how VoicEra is structured, who interacts with it, and how the major services fit together. It is aimed at architects, operators, and developers approaching the platform for the first time.

{% hint style="info" %}
VoicEra is presented here using the [C4 model](https://c4model.com/): Level 1 (System Context) shows the system as a single box surrounded by users and external systems; Level 2 (Containers) zooms in to show each deployable service.
{% endhint %}

## Level 1 — System context

VoicEra as a single system with all external actors it interacts with.

```mermaid
flowchart TB
  Op(["Operator\n[Person]\nConfigures agents, reviews calls"])
  Caller(["End user\n[Person]\nPlaces or receives a call"])
  Tel["Telephony provider\n[External system]\nVobiz · Plivo"]
  AI["AI providers\n[External system]\nOpenAI · Anthropic · Groq · Deepgram\nCartesia · Sarvam · Bhashini · AI4Bharat · …"]

  SYS["VoicEra Platform\n[Software system]\nReal-time voice agents\nfor Indian-language telephony"]

  Op -- "Configures agents, reviews calls" --> SYS
  Caller -- "Dials number" --> Tel
  Tel -- "Webhooks + audio stream\n(HTTPS + WSS)" --> SYS
  SYS -- "LLM · STT · TTS · embeddings\n(HTTPS)" --> AI
```

| Actor | Role |
| --- | --- |
| End user | Person who places or receives a voice call routed through VoicEra. |
| Operator | Dashboard user who configures agents, links phone numbers, uploads knowledge documents, and reviews calls. |
| Telephony provider | Vobiz (or Plivo) — provides phone numbers and streams audio to the voice server over WebSocket. |
| AI providers | LLM, STT, and TTS vendors (OpenAI, Anthropic, Groq, Sarvam, Deepgram, ElevenLabs, Cartesia, Bhashini, AI4Bharat, etc.). |

## Level 2 — Containers

The deployable units that make up VoicEra, their technologies, and how they communicate.

```mermaid
flowchart TB
  Op(["Operator browser"])
  Tel["Telephony\nVobiz · Plivo"]
  AI["External AI\nLLM · STT · TTS"]

  subgraph platform ["VoicEra Platform"]
    FE["Frontend\n:3000"]
    BE["Backend\n:8000"]
    VS["Voice server\n:7860"]
    MONGO[("MongoDB\n:27017")]
    MINIO[("MinIO\n:9000")]
    CHROMA[("ChromaDB\nembedded in backend")]

    FE -- REST/HTTPS --> BE
    BE -- Mongo wire --> MONGO
    BE -- S3 API --> MINIO
    BE -. reads/writes .-> CHROMA
    VS -- "HTTPS · X-API-Key" --> BE
    VS -- "S3 recordings" --> MINIO
  end

  Op -- HTTPS --> FE
  Op -. "WSS · Test on Browser" .-> VS
  Tel -- "HTTPS /answer" --> VS
  Tel -- "WSS audio" --> VS
  VS -- HTTPS --> AI
  BE -- HTTPS --> AI
```

| Container | Technology | Responsibility |
| --- | --- | --- |
| Frontend | Next.js 16, React 19, TailwindCSS 4 | Operator dashboard, agent and campaign management, browser test client. |
| Backend | FastAPI, Python 3.10+ | REST API, auth, persistence, RAG ingest, integration management, MinIO storage orchestration. |
| Voice server | Pipecat, Python 3.11+, uvloop | Real-time audio pipeline (STT → LLM → TTS), telephony webhooks, browser audio, call recording. |
| MongoDB | NoSQL | Users, agents, campaigns, meetings, integrations, knowledge document metadata. |
| MinIO | S3-compatible object store | Recordings (`.wav`/`.mp3`), transcripts (`.txt`), uploaded PDFs. |
| ChromaDB | Embedded vector store | Per-organisation vector chunks for RAG. Runs inside the backend container, persisted to disk. |
| External AI | HTTPS APIs | LLM, STT, TTS, and embedding calls. |

{% hint style="success" %}
Backend and frontend are stateless and can scale horizontally. The voice server holds one WebSocket session per active call and should be scaled with sticky routing.
{% endhint %}

## Communication patterns

```mermaid
flowchart LR
  Caller((Caller)) --> Vobiz[Vobiz telephony]
  Op[Operator browser] -- REST/HTTPS --> FE[Frontend :3000]
  FE -- REST/HTTPS --> BE[Backend :8000]
  Op -. WSS test on browser .-> VS[Voice server :7860]
  Vobiz -- HTTPS /answer --> VS
  Vobiz -- WSS audio --> VS
  VS -- HTTPS X-API-Key --> BE
  VS -- HTTPS --> AI[(LLM / STT / TTS)]
  BE -- HTTPS --> AI
  BE -- Mongo wire --> Mongo[(MongoDB)]
  BE -- S3 --> Minio[(MinIO)]
  VS -- S3 recordings --> Minio
```

| Pattern | Used for |
| --- | --- |
| REST / HTTPS | Operator browser ↔ frontend ↔ backend; voice server ↔ backend (agent config, KB retrieval, meeting updates). |
| WebSocket (WSS) | Telephony audio stream from Vobiz/Plivo to voice server; browser-initiated audio for **Test on Browser**. |
| `X-API-Key` (HTTPS) | Voice server authenticates to backend for internal endpoints (agent config, integration keys, RAG retrieve). |

## Key design choices

| Choice | Rationale |
| --- | --- |
| Pluggable AI providers | Operators pick STT / LLM / TTS per agent. Provider keys live in **Integrations**, not in `.env`. |
| Pipeline-based voice runtime | Pipecat models the call as a frame pipeline so latency-critical stages (STT, LLM, TTS, VAD) can be swapped, tuned, and observed independently — see [voice-pipeline.md](voice-pipeline.md). |
| Per-org RAG isolation | Each organisation gets its own hashed Chroma subdirectory. Two orgs can never see each other's embeddings — see [knowledge-base-rag.md](knowledge-base-rag.md). |
| Object store for blobs | Audio, transcripts, and PDFs live in MinIO; only metadata in MongoDB. |
| Separate voice server | Telephony state and real-time audio are isolated from the CRUD backend so they can be scaled and restarted independently. |

## Deployment shape

{% tabs %}
{% tab title="Docker Compose (default)" %}
All services run on one host via `docker-compose.yml`. Suitable for single-host deployments from development through production. See [docker-compose.md](../guides/deployment/docker-compose.md).

```text
host
├── frontend         (:3000)
├── backend          (:8000)
├── voice_server     (:7860)
├── mongodb          (:27017)
└── minio            (:9000, :9001)
```
{% endtab %}

{% tab title="Production" %}
Frontend and backend behind a reverse proxy (TLS), voice server behind a TLS-terminating proxy with sticky sessions, MongoDB as a replica set, MinIO as a cluster (or replaced by managed S3). See [production.md](../guides/deployment/production.md) and [security-hardening.md](../guides/deployment/security-hardening.md).
{% endtab %}
{% endtabs %}

## Security boundaries

| Boundary | Mechanism |
| --- | --- |
| Dashboard ↔ backend | JWT bearer tokens, per-org RBAC. |
| Voice server ↔ backend | Shared `INTERNAL_API_KEY` over HTTPS. |
| Backend ↔ Mongo / MinIO | Auth credentials in env, network-restricted in production. |
| Telephony provider keys | Stored per-org under **Integrations**, never in repo `.env`. |
| In transit | TLS for all external API calls; WSS for browser and telephony audio. |

## Next steps

- [voice-pipeline.md](voice-pipeline.md) — how a call flows through Pipecat.
- [data-flow.md](data-flow.md) — where data lives and how it moves.
- [agents-campaigns-calls.md](agents-campaigns-calls.md) — the core data model.
- [../quickstart/install-and-run.md](../quickstart/install-and-run.md) — bring the stack up locally.
