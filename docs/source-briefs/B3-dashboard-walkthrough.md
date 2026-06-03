# Brief: Dashboard walkthrough / screenshots (B3)

**Review gap (original):** No screenshots or walkthrough of the frontend — what an "agent" means, how to create one, how to make a call.

**Merge status:** Written walkthrough is in [docs/guide/dashboard.md](../guide/dashboard.md). **Screenshots are explicitly deferred** until staging + test account are available; dashboard.md contains an acknowledgement callout and this brief remains the capture checklist.

**Requires:** Staging environment + test user account for documentation writer.

---

## Screenshots to capture (suggested order)

| # | Screen | Caption should explain |
|---|--------|------------------------|
| 1 | Login | How users access the system |
| 2 | Home / main dashboard after login | Landing view |
| 3 | **Assistants** list | Each card = one **agent** (virtual voice assistant) |
| 4 | **Create agent** — step 1 | Name, language |
| 5 | **Create agent** — telephony | Provider: Vobiz or Plivo |
| 6 | **Create agent** — AI settings | STT/TTS/LLM choices (simplify in caption) |
| 7 | **Create agent** — prompt / greeting | What the agent says and how it behaves |
| 8 | **Integrations** page | Where **Vobiz Auth ID/Token** are saved — **not in a server text file** |
| 9 | **Phone numbers** — list | Numbers available in account |
| 10 | **Phone numbers** — link to agent | Incoming calls to this number use this agent |
| 11 | **Test on Browser** on agent card | Test without a phone call |
| 12 | Test dialog open (orb/mic) | Microphone permission, connected state |
| 13 | **Meetings / calls** history | Past calls and status |
| 14 | **Recordings** (if shown in UI) | How to hear/download a call |
| 15 | **Campaigns / batches** (optional) | Outbound calling to many numbers |

---

## Key concepts for captions

### Agent

A saved configuration for an AI voice assistant: language, voice, AI providers, instructions, linked phone number. **Not a human staff member.**

### Integration

API credentials stored securely for your organization (telephony + AI). Operators with access use **Dashboard → Integrations**.

### Link number to agent

The telephony provider routes calls on that number to this agent's configuration.

### Test on Browser

Uses the same voice technology as phone calls but through the computer microphone and speakers — good for quick checks without spending phone minutes.

---

## How to make a call (writer narrative)

### Inbound

1. Ensure Integrations has Vobiz (or Plivo) credentials.
2. Create agent; note telephony provider.
3. On Numbers page, link a phone number to that agent.
4. Call the number from a mobile phone.
5. Confirm call appears under Meetings/calls.

### Outbound

Depends on product features enabled (API or campaigns). Reference technical `/outbound/call/` in appendix for integrators — see A2.

### Browser test (no phone)

1. Open Assistants.
2. Click **Test on Browser** on an agent.
3. Allow microphone; verify you hear the agent.

---

## Code references (for writer / engineering)

| Feature | File |
|---------|------|
| Assistants list & create | `voicera_frontend/app/(dashboard)/assistants/page.tsx` |
| Integrations | `voicera_frontend/app/(dashboard)/integrations/page.tsx` |
| Phone numbers | `voicera_frontend/app/(dashboard)/numbers/page.tsx` |
| Browser test | `voicera_frontend/components/assistants/test-browser-dialog.tsx` |

---

## Blur or redact in screenshots

- Real API keys and tokens
- Personal phone numbers if not test data
- Internal hostnames unless approved for publication
