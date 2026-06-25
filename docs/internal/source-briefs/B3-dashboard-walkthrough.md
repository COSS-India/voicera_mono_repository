# Brief: Dashboard walkthrough (B3)

**Purpose:** Writer context for the operator dashboard — what an "agent" means, how to create one, how to make a call.

**Published guide:** [docs/guide/dashboard.md](../guide/dashboard.md)

**Requires:** Staging environment + test user account for validation.

---

## Key concepts

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

## Sensitive data (do not publish in examples)

- Real API keys and tokens
- Personal phone numbers if not test data
- Internal hostnames unless approved for publication
