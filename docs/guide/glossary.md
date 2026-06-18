# Glossary

Plain-language definitions for operators and non-technical readers.

| Term | Definition |
|------|------------|
| **Agent** | A configured AI voice assistant for phone or browser testing — not a human employee. |
| **AI4Bharat** | Optional self-hosted Indian language STT/TTS servers in this repository. |
| **Answer URL** | Web address the phone company calls when a call is answered; points to your voice server. |
| **Backend** | Service that stores users, agents, and call records; APIs for the dashboard. |
| **Batch / campaign** | Outbound calling to many numbers from an uploaded list, using an agent. |
| **Bhashini** | Government cloud speech API; optional STT/TTS when configured in Integrations. |
| **Bhili (`bhb`)** | Language code for Voice Bhili; uses dedicated STT route `/transcribe/bhili` when using local AI4Bharat STT. |
| **Dashboard** | The VoicEra website where staff manage agents and view calls. |
| **Integration** | Saved API credentials per organization (telephony and AI keys). |
| **LLM** | Large language model — generates what the agent should say. |
| **Meeting** | A record of one call session (times, metadata). |
| **MinIO** | Object storage for recordings and uploads. |
| **MongoDB** | Database for users, agents, and call history. |
| **Pipeline** | Real-time steps during a call: hear → understand → reply → speak. |
| **Public voice server URL** | Internet address of the voice server (`JOHNAIC_*` env names today). |
| **RAG / Knowledge base** | Uploaded PDFs the agent can search during a call. |
| **STT** | Speech-to-text. |
| **TTS** | Text-to-speech. |
| **Telephony provider** | Company providing phone numbers (Vobiz). |
| **Test on Browser** | Talk to an agent via mic/speakers without a phone call. |
| **Transcript** | Text of what was said when logging is enabled. |
| **Voice server** | Runs the live conversation during each call. |
| **Vobiz** | Telephony provider linking Indian numbers to VoicEra. |
| **WebSocket** | Live two-way audio streaming during a call. |
| **Webhook** | HTTP callback from the phone company on call events. |

## Acronyms

| Acronym | Meaning |
|---------|---------|
| API | How software components communicate |
| HTTPS | Secure web connection |
| WSS | Secure WebSocket (encrypted live audio) |
| GPU | Hardware used to accelerate local speech models |
| RAG | Retrieval-augmented generation (knowledge base at call time) |
