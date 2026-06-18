# Brief: Plain-language overview (B1)

**Review gap:** No plain-language overview for programme managers or district IT officers.

**Audience:** Non-technical implementors and operators. No Docker, Python, or YAML assumed.

---

## VoicEra in one sentence

VoicEra is a platform to run **AI phone agents** that can answer or make phone calls in Indian languages, with a **web dashboard** to configure agents and view call history and recordings in almost zero cost.

---

## What problem it solves

Government departments and partners need phone lines that can:

- Answer many calls automatically in local languages
- Follow a script or knowledge base
- Log who called and what was said
- Optionally place outbound calls (campaigns)

VoicEra provides the software stack; a **telephony provider** (Vobiz or Plivo) connects real phone numbers to the system.

---

## Main parts (plain language)

| Part | What it does | Analogy |
|------|--------------|---------|
| **Dashboard** (website) | Create agents, link phone numbers, view calls | Control panel |
| **Backend** | Saves users, agents, call history | Filing cabinet + rules |
| **Voice server** | Handles the live conversation on each call | The person on the phone (AI) |
| **Database** (MongoDB) | Stores settings and records | Filing cabinet storage |
| **File storage** (MinIO) | Stores recordings and uploaded files | Audio/file archive |
| **Speech servers** (optional) | Convert speech↔text on your own servers | In-house interpreter machines |

---

## How a phone call works (5 steps)

1. Someone dials your phone number (or receives an outbound call).
2. The **phone company** (Vobiz/Plivo) connects the call to your **voice server** over the internet.
3. The voice server **listens** (speech-to-text), **decides what to say** (AI), and **speaks** (text-to-speech).
4. Call details are saved to the **database** and appear in the **dashboard**.
5. Recordings (if enabled) are stored in **file storage**.

---

## What is an "agent"?

An **agent** is **not** a human employee. It is a **configured virtual call handler**:

- Which language to use
- Which AI voice and brain (STT/TTS/LLM settings)
- Instructions (system prompt)
- Which phone number(s) use this configuration

One deployment can have many agents (e.g. helpline in Hindi, outbound survey in Marathi).

---

## How parts connect (simplified)

```
Caller  →  Phone network (Vobiz/Plivo)  →  Voice server  →  AI services
                                              ↓
                                         Backend / Database
                                              ↓
                                         Dashboard (staff view)
```

**Writer:** Redraw as a simple diagram without port numbers for this document; put technical ports in B5 only.

---

## What staff do vs what hosting partner does

| Staff (operators) | Hosting partner (technical) |
|-------------------|----------------------------|
| Log into dashboard | Install server, Docker, HTTPS |
| Create agents | Set environment URLs |
| Enter API keys in Integrations | Start/stop services, backups |
| Link phone numbers | Firewall, DNS, logs |

---

## Related briefs

- Prerequisites: [B2-before-you-start-checklist.md](./B2-before-you-start-checklist.md)
- Terms: [B4-glossary.md](./B4-glossary.md)
- Telephony detail: [A3-vobiz-telephony.md](./A3-vobiz-telephony.md)
