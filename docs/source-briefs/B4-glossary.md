# Brief: Glossary (B4)

**Review gap:** No glossary for non-technical readers.

**Writer task:** Format A–Z (or grouped by topic). One clear sentence per term; avoid jargon in the definition itself.

---

| Term | Plain definition |
|------|------------------|
| **Agent** | A configured AI voice assistant for phone calls or browser testing — not a human employee. |
| **AI4Bharat** | Optional self-hosted Indian language speech engines (STT/TTS servers in this repo). |
| **Answer URL** | Web address the phone company calls when a call is answered; points to your voice server. |
| **Backend** | The service that stores users, agents, and call records and provides APIs to the dashboard. |
| **Batch / campaign** | Outbound calling to many phone numbers from a uploaded list, using an agent. |
| **Bhashini** | Government cloud speech API; can be used for STT/TTS when configured. |
| **Dashboard** | The VoicEra website where staff manage agents and view calls. |
| **Integration** | Saved API credentials in the dashboard for one organization (telephony, AI keys). |
| **LLM** | Large language model — the AI that generates what the agent should say in conversation. |
| **Meeting** | A record of one call session in the system (times, metadata). |
| **MinIO** | File storage system used for recordings and uploaded documents. |
| **MongoDB** | Database storing users, agents, call history, and settings. |
| **Pipeline** | The real-time steps during a call: hear → understand → reply → speak. |
| **Plivo** | A telephony provider alternative to Vobiz; connects phone numbers to VoicEra. |
| **Public voice server URL** | Internet address of the voice server (env names `JOHNAIC_*` today — see A8). |
| **Recording** | Audio file of a call stored after the call ends (when enabled). |
| **RAG / Knowledge base** | Uploaded PDFs the agent can search for answers during a call. |
| **STT** | Speech-to-text — converts the caller's voice into text the AI can read. |
| **TTS** | Text-to-speech — converts the AI's text reply into spoken audio. |
| **Telephony provider** | Company that provides phone numbers and connects calls (Vobiz or Plivo). |
| **Test on Browser** | Dashboard feature to talk to an agent via computer mic/speakers without a phone call. |
| **Transcript** | Text record of what was said (when STT/logging is enabled). |
| **Voice server** | Service that runs the live conversation during each call. |
| **Vobiz** | Telephony provider used to link Indian phone numbers to VoicEra. |
| **WebSocket** | Technology for live two-way audio streaming during a call. |
| **Webhook** | Automatic HTTP call from the phone company to your server when call events happen. |

---

## Optional acronyms section

| Acronym | Spells out |
|---------|------------|
| API | Application programming interface — how software components talk |
| HTTPS | Secure web connection |
| WSS | Secure WebSocket (live audio over encrypted connection) |
| GPU | Graphics processor used here for faster AI speech models |
| STT / TTS | See above |
