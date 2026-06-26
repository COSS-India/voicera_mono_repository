---
description: Plain-language definitions for the terms used across the VoicEra docs.
---

# Glossary

A reference for terms used in the VoicEra docs and dashboard. Definitions are deliberately short — follow the link in each entry for the deep dive.

{% hint style="info" %}
Looking for the data model relationships behind these terms? See [agents-campaigns-calls.md](agents-campaigns-calls.md).
{% endhint %}

## A

| Term | Definition |
| --- | --- |
| **Agent** | A configured AI voice assistant (LLM/STT/TTS, prompt, language, telephony). **Not** a human employee. See [agents-campaigns-calls.md](agents-campaigns-calls.md). |
| **AI4Bharat** | Optional self-hosted Indian-language STT and TTS servers shipped with VoicEra. See [../services/ai4bharat-stt.md](../services/ai4bharat-stt.md). |
| **Answer URL** | Web address the telephony provider calls when a call is answered; points at the public voice server (`/answer?agent_id=...`). |
| **Assistants** | Dashboard page where operators manage agents. |

## B

| Term | Definition |
| --- | --- |
| **Backend** | FastAPI service that stores users, agents, meetings, integrations, and KB metadata; serves the dashboard. See [../services/backend.md](../services/backend.md). |
| **Barge-in** | Letting a caller interrupt the bot mid-sentence. Gated in VoicEra by VAD + minimum word count. |
| **Batch** | Synonym for **campaign** in some API routes (`/api/v1/batches`). |
| **Bhashini** | Government cloud speech API; optional STT/TTS when configured in **Integrations**. |
| **Bhili (`bhb`)** | Language code for Voice Bhili; uses dedicated STT route `/transcribe/bhili` on local AI4Bharat STT. |

## C

| Term | Definition |
| --- | --- |
| **C4 model** | Architecture-diagram convention (Context, Container, Component, Code). Used on [architecture.md](architecture.md). |
| **Call** | One conversation between a caller and an agent. Stored as a **meeting**. |
| **Campaign** | Outbound calling to many numbers from an uploaded list, driven by one agent. |
| **ChromaDB** | Vector store used for RAG embeddings; lives on disk inside the backend container. |
| **Container (C4)** | A deployable unit (backend, frontend, voice server, MongoDB, MinIO). |

## D

| Term | Definition |
| --- | --- |
| **Dashboard** | The VoicEra frontend where operators manage agents and view calls. See [../guides/operator/dashboard-tour.md](../guides/operator/dashboard-tour.md). |
| **Deepgram** | Cloud STT/TTS provider. |
| **DOWNSTREAM** | Pipecat frame direction toward the audio output. |

## F

| Term | Definition |
| --- | --- |
| **Frame** | The atomic unit of data in Pipecat — audio chunk, transcript, LLM token, control signal. See [voice-pipeline.md](voice-pipeline.md). |
| **FrameProcessor** | A single stage in the Pipecat pipeline. |

## G

| Term | Definition |
| --- | --- |
| **GPU** | Hardware used to accelerate local speech models (AI4Bharat, vLLM). |

## H

| Term | Definition |
| --- | --- |
| **Hangup cause** | Telephony-reported reason for call end (`USER_HANGUP`, `USER_BUSY`, …). |
| **Hold message** | Interim phrase played while waiting for a slow LLM (KenpathLLM). |

## I

| Term | Definition |
| --- | --- |
| **Integration** | Saved per-organisation API credentials (telephony and AI keys). See [../services/integrations.md](../services/integrations.md). |
| **Internal API key** | Shared secret used by the voice server to call backend RAG endpoints (`INTERNAL_API_KEY`). |

## J

| Term | Definition |
| --- | --- |
| **JWT** | JSON Web Token — bearer auth between dashboard and backend. |

## K

| Term | Definition |
| --- | --- |
| **Kenpath** | Indian-government LLM (Vistaar). Supports hold messages and Bhashini fast-turn. |
| **Knowledge base / RAG** | Uploaded PDFs the agent can search during a call. See [knowledge-base-rag.md](knowledge-base-rag.md). |

## L

| Term | Definition |
| --- | --- |
| **L16** | Linear PCM audio encoding (16-bit, signed). Used at 16 kHz on Vobiz when `SAMPLE_RATE=16000`. |
| **LLM** | Large language model — generates what the agent says. |

## M

| Term | Definition |
| --- | --- |
| **Meeting** | A record of one call session (times, transcript, metadata). |
| **MinIO** | S3-compatible object storage for recordings, transcripts, and uploaded PDFs. |
| **MongoDB** | Database for users, agents, campaigns, and meetings. |
| **μ-law (mulaw)** | 8 kHz telephony audio encoding. Default on Vobiz when `SAMPLE_RATE=8000`. |

## O

| Term | Definition |
| --- | --- |
| **Operator** | Dashboard user who configures agents and reviews calls. |
| **Organisation (org)** | The tenant scope. Agents, integrations, knowledge documents, and meetings all belong to an org. |
| **Outbound** | Calls placed by VoicEra to a destination (typically via a campaign). |

## P

| Term | Definition |
| --- | --- |
| **Pipecat** | The open-source voice-AI framework powering VoicEra's voice server. See [voice-pipeline.md](voice-pipeline.md). |
| **Pipeline** | The ordered list of processors that handle a call in real time. |
| **Plivo** | Telephony provider; supported alongside Vobiz. |
| **Public voice server URL** | The internet address of the voice server (`JOHNAIC_SERVER_URL` / `JOHNAIC_WEBSOCKET_URL`). |

## R

| Term | Definition |
| --- | --- |
| **RAG** | Retrieval-augmented generation — injecting KB excerpts into the LLM prompt at call time. |

## S

| Term | Definition |
| --- | --- |
| **Sarvam** | Indian-language STT/TTS provider (`saarika`, `bulbul`). |
| **Sentiment** | Per-call positive/neutral/negative tag derived from the transcript. |
| **Silero** | Neural VAD used by the voice server. |
| **STT** | Speech-to-text. |
| **Stream SID** | Vobiz/Plivo identifier for one audio stream within a call. |
| **System prompt** | The instruction text given to the LLM that defines the agent's behaviour. |

## T

| Term | Definition |
| --- | --- |
| **Telephony provider** | Company providing phone numbers (Vobiz, optionally Plivo). |
| **Test on Browser** | Talk to an agent via mic/speakers without making a phone call. |
| **Transcript** | Text of what was said during a call. |
| **TTFB** | Time-to-first-byte — latency from request to first response token/chunk. |
| **TTS** | Text-to-speech. |

## U

| Term | Definition |
| --- | --- |
| **UPSTREAM** | Pipecat frame direction toward the audio input (used by `UserOnlineDetectionFilter`). |

## V

| Term | Definition |
| --- | --- |
| **VAD** | Voice activity detection (Silero in VoicEra). Decides when the user is speaking. |
| **Vobiz** | Telephony provider linking Indian numbers to VoicEra; Plivo-protocol-compatible. |
| **Voice server** | The Pipecat-based service that runs the live conversation during each call. See [../services/voice-server.md](../services/voice-server.md). |
| **vLLM** | Self-hosted LLM runtime supported by VoicEra (`qwen` / `localqwen` / `vllm` providers). |

## W

| Term | Definition |
| --- | --- |
| **Webhook** | HTTP callback from a telephony provider on call events. |
| **WebSocket** | Live two-way streaming used for telephony audio and browser test. |
| **WSS** | Secure (TLS) WebSocket. |

## Next steps

- [architecture.md](architecture.md) — the C4 view of the system.
- [agents-campaigns-calls.md](agents-campaigns-calls.md) — how the data-model terms relate.
- [voice-pipeline.md](voice-pipeline.md) — the runtime terms (frame, processor, VAD, barge-in).
- [../introduction/what-is-voicera.md](../introduction/what-is-voicera.md) — the product overview.
