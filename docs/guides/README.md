---
description: Open-source, self-hosted voice AI platform for real-time telephony agents in Indian languages.
---

# Welcome to Voicera

**Voicera** is an open-source platform for building real-time conversational phone agents. It wires speech-to-text, large language models, text-to-speech, and telephony into one self-hosted stack, and gives you a REST API to drive it.

Use Voicera to run inbound helplines, outbound calling campaigns, IVR replacements, and citizen-services hotlines — without building voice infrastructure yourself.

{% hint style="info" %}
New here? Read [What is Voicera](introduction/what-is-voicera.md) for the plain-language overview, or go straight to the [Quickstart](quickstart/README.md) if you have Docker ready.
{% endhint %}

## Browse the docs

Voicera's documentation is split into three tabs across the top of this site.

{% tabs %}
{% tab title="Guides" %}
**You are here.** Learn the system and run it day to day.

* [**Introduction**](introduction/README.md) — what Voicera is and what it is for
* [**Quickstart**](quickstart/README.md) — empty machine to working agent, in five steps
* [**Core concepts**](concepts/README.md) — architecture, pipeline, campaigns, providers
* [**Running Voicera**](operator/operating-via-api.md) — campaigns, documents, daily operations
* [**Deployment**](deployment/docker-compose.md) — Compose, production, hardening
* [**Troubleshooting**](troubleshooting/README.md) — symptom-first index
{% endtab %}

{% tab title="Developer" %}
Build on Voicera, or extend it.

* [**Services**](../developer/services/README.md) — the containers and the packages they run
* [**Model server**](../developer/model-server/README.md) — self-hosted STT, TTS, and LLM
* [**Clients**](../developer/clients/README.md) — the surfaces anything connects through
* [**Contributing**](../developer/guides/local-setup.md) — local setup, adding providers, testing
* [**Configuration reference**](../developer/reference/environment-variables.md) — variables, ports, data model
* [**Dashboard (Beta)**](../developer/frontend/README.md) — the Next.js console
{% endtab %}

{% tab title="API Reference" %}
Every route, extracted from source.

* [**Introduction**](../api-reference/overview.md) · [**Authentication**](../api-reference/authentication.md) · [**Errors**](../api-reference/errors.md)
* [**Agents**](../api-reference/agents.md) · [**Calls**](../api-reference/calls.md) · [**Campaigns**](../api-reference/campaigns.md)
* [**Phone numbers**](../api-reference/phone-numbers.md) · [**Knowledge and RAG**](../api-reference/knowledge-and-rag.md)
* [**Users and organisations**](../api-reference/users-and-orgs.md) · [**Provider credentials**](../api-reference/provider-auth.md)
* [**WebSocket API**](../api-reference/websocket-api.md) · [**Endpoints cheatsheet**](../api-reference/endpoints-cheatsheet.md)

A running API also serves an interactive console at `http://localhost:8000/docs`.
{% endtab %}
{% endtabs %}

## What you get

| Capability | What it does |
| --- | --- |
| **Real-time voice agents** | Sub-second STT → LLM → TTS loop built on [Pipecat](concepts/voice-pipeline.md). |
| **22 cloud providers** | OpenAI, Deepgram, Cartesia, ElevenLabs, Sarvam, Groq, Azure, Google and more, behind [one registry](concepts/provider-registry.md). |
| **Self-hosted models** | Run STT, TTS, and LLM on your own GPUs via the [model server](../developer/model-server/overview.md). |
| **Telephony** | Inbound and outbound calls through [Vobiz or Plivo](concepts/telephony-model.md), provider-agnostic. |
| **Outbound campaigns** | CSV-driven [campaigns](concepts/campaigns.md) with retries, scheduling, circuit breakers, and concurrency limits. |
| **Knowledge base (RAG)** | Ground answers in your own documents. See [Knowledge base](concepts/knowledge-base-rag.md). |
| **Multi-tenant** | Organisations, [roles, and scoped access](concepts/multi-tenancy.md) built in. |
| **Self-hosted** | One Docker Compose stack. Your data, your servers, your model keys. |

## Where to start

| If you are… | Start here |
| --- | --- |
| **Evaluating or demoing** | [Prerequisites](quickstart/prerequisites.md) → [Install and run](quickstart/install-and-run.md) → [Create your first agent](quickstart/first-agent.md) |
| **Running calls day to day** | [Operating via the API](operator/operating-via-api.md) → [Running a campaign](operator/running-a-campaign.md) |
| **Building or extending** | [Architecture](concepts/architecture.md) → [Local setup](../developer/guides/local-setup.md) → [REST API](../api-reference/overview.md) |
| **Deploying to production** | [Docker Compose](deployment/docker-compose.md) → [Public voice URLs](deployment/public-voice-urls.md) → [Security hardening](deployment/security-hardening.md) |

## The stack at a glance

```mermaid
flowchart LR
  Caller(["Caller"])
  Tel["Telephony<br/>Vobiz · Plivo"]
  RT["Runtime<br/>:7860"]
  API["API<br/>:8000"]
  DB[("FerretDB<br/>:27018")]
  S3[("MinIO<br/>:9000")]
  Q[("Redis<br/>queue")]
  AI["AI providers<br/>cloud or self-hosted"]

  Caller --> Tel
  Tel --> RT
  RT <--> API
  RT --> AI
  API --> DB
  API --> Q
  RT --> S3
  API --> S3
```

Two services, three stores, one job queue, and your choice of AI providers. The full picture is in [Architecture](concepts/architecture.md).

## What Voicera is not

Voicera is **API-first**. The core stack ships no user interface — you drive it with HTTP requests, and `/docs` gives you an interactive OpenAPI console.

{% hint style="warning" %}
A web dashboard exists but is **Beta** and lives on the separate `dev-frontend` branch. It is not part of the Docker Compose stack. See [Dashboard (Beta)](../developer/frontend/overview.md).
{% endhint %}

Voicera also does not resell telephony or model capacity. You bring your own provider accounts, or you run the models yourself.

## Need help?

* Start with [Troubleshooting](troubleshooting/common-issues.md).
* Look up unfamiliar terms in the [Glossary](concepts/glossary.md).
* Read the [Contributing guide](../developer/guides/contributing.md) before opening a pull request.
