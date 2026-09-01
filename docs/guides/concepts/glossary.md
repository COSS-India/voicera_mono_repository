---
description: Definitions of the terms used throughout the Voicera documentation.
---

# Glossary

## A

**Adapter**
A provider implemented first-party rather than through Pipecat, living under `apps/providers/adapters/`. Bhashini TTS is the only one today. Reported as `provider_type: adapter`. See [Provider registry](provider-registry.md).

**Agent**
A configured voice assistant: prompts, behaviour, language, and its STT, TTS, and LLM choices. Belongs to one organisation. See [Agents](agents.md).

**Agent category**
Either `telephony` (reached over the phone, provisioned with a provider application) or `websocket` (reached from a browser). Determines whether `/answer` works and which frame serializer the runtime uses.

**Application**
A configuration object created on a telephony provider's account that ties a phone number to Voicera's answer URL. The API provisions and deletes these for you.

**ARQ**
The Redis-backed job queue running campaign batches and CSV syncs off the request path. See [Workers and orchestrator](../../developer/services/workers.md).

## B

**Barge-in**
A caller interrupting the agent mid-sentence. The pipeline stops playback and starts listening. Tunable with `interruption_min_words`. See [Voice pipeline](voice-pipeline.md).

**Bot token**
A short-lived, organisation-scoped JWT minted by `POST /users/bot/token` against the `INTERNAL_API_KEY`. How the runtime authenticates to the API. See [Multi-tenancy](multi-tenancy.md).

## C

**Call log**
The record of one call: type, status, response, timings, and pointers to its transcript and recording. Stored in `CallLogs`.

**Campaign**
A CSV-driven outbound calling run with retry policy, scheduling, concurrency limits, and a circuit breaker. See [Campaigns](campaigns.md).

**Circuit breaker**
A guard that pauses a campaign when the failure rate in a rolling window crosses a threshold, so a broken configuration cannot burn an entire contact list.

**Chroma**
The embedded vector store holding per-organisation RAG chunks, persisted to the `voicera_oss_chroma_data` volume.

**Concurrency slot**
A Redis-held token representing one in-flight call. Bounded per organisation and per campaign. See [Call concurrency](call-concurrency.md).

## D

**Discriminated union**
The Pydantic pattern in `apps/providers/factory.py` that turns the registry into a single validated type, so adding a provider never means editing an if/elif chain.

## F

**FerretDB**
A proxy speaking the MongoDB wire protocol on top of PostgreSQL. Voicera's data store. Published on host port `27018`. See [Data store](data-store.md).

**Frame**
Pipecat's unit of data flowing through the pipeline: an audio chunk, a transcript, an LLM token, a control signal.

**Frame serializer**
The translator between a telephony provider's wire format and Pipecat frames. One per provider, registered in `apps/telephony`.

## G

**Gateway**
The single published entry point of the model server, on port `8100`. Routes to whichever STT, TTS, and LLM slots are deployed. See [Gateway API](../../developer/model-server/gateway-api.md).

## H

**Hold message**
Audio played when the agent needs time — typically while a tool call or retrieval runs. Configured in the agent's behaviour block.

## K

**Knowledge base**
Documents ingested, chunked, embedded, and retrieved to ground an agent's answers. Two modes, `tool` and `context`. See [Knowledge base](knowledge-base-rag.md).

## M

**Membership**
The join record between a user and an organisation, carrying the role. See [Multi-tenancy](multi-tenancy.md).

**MinIO**
The S3-compatible object store holding call recordings and transcripts.

**Model server**
The optional self-hosted stack running STT, TTS, and LLM on your own hardware behind one gateway. See [Model server](../../developer/model-server/overview.md).

## O

**Orchestrator**
The `campaign-orchestrator` container. Listens on Redis pub/sub, schedules the next campaign batch, and detects completion.

**Organisation**
The tenant boundary. Every agent, number, credential, campaign, and call belongs to exactly one.

## P

**Partial transcript**
Words returned while the speaker is still talking. Required for responsive telephony. Distinct from whether the model serves a streaming endpoint — see [STT models](../../developer/model-server/stt-models.md).

**Pipecat**
The async Python framework for real-time voice pipelines that the runtime is built on.

**ProviderAuth**
The Fernet-encrypted, provider-level credential record for an organisation. Replaces the old Integrations model. See [Provider credentials](provider-auth.md).

**Provider registry**
The self-describing catalogue of STT, TTS, and LLM vendors. Providers register themselves; the schema dump drives client forms. See [Provider registry](provider-registry.md).

## Q

**QueuedRun**
One queued outbound call in a campaign — a single contact and retry attempt. Stored in `QueuedRuns`.

## R

**RAG**
Retrieval-augmented generation: fetching relevant chunks from your documents and giving them to the LLM as grounding.

**Runtime**
The `apps/runtime` service on port `7860`. Serves the `/answer` webhook and runs one Pipecat pipeline per call. See [Runtime](../../developer/services/runtime.md).

**RTVI**
The real-time voice interaction protocol used by Pipecat browser clients over protobuf frames.

## S

**Slot**
One of the model server's three fixed positions — `stt`, `tts`, `llm` — each a container on a fixed internal port. Which model fills a slot is a folder name in `.env`. See [Slots and models](../../developer/model-server/slots-and-models.md).

**Stream XML**
The XML returned by `/answer` telling a telephony provider which WebSocket URL to stream the call audio to.

## T

**Telephony provider**
Vobiz or Plivo. Owns the phone numbers and streams call audio. Selected per agent, never global. See [Telephony model](telephony-model.md).

## V

**VAD**
Voice activity detection — deciding when the caller starts and stops speaking, which drives turn-taking and barge-in.

## Related

* [Architecture](architecture.md)
* [Voice pipeline](voice-pipeline.md)
* [Data model](../../developer/reference/data-model.md)
