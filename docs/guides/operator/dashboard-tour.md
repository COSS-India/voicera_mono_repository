---
description: A tour of the VoicEra operator dashboard — agents, integrations, phone numbers, calls, and the browser test.
---

# Dashboard tour

This page walks through the VoicEra web dashboard for non-technical operators. Use it to learn where each setting lives and how to confirm that a real call worked.

{% hint style="info" %}
You only need a browser and your operator login. No SSH or server access is required for anything on this page.
{% endhint %}

## Signing in

Open your deployment URL — typically `https://your-voicera-host` or `http://your-host:3000` for local installs — and sign in with the account your administrator created.

If you do not have an account yet, ask your administrator to invite you under **Members**.

## Sidebar layout

The left sidebar groups everything by feature area.

| Section | What it is for |
|---------|----------------|
| **Assistants** | Create, edit, and test AI voice agents |
| **Integrations** | Store telephony and AI provider credentials |
| **Phone numbers** | List numbers and link them to agents |
| **Meetings** | Call history with time, duration, and status |
| **Knowledge base** | Upload PDFs that agents can read from |
| **Campaigns** | Run outbound calling batches |
| **Members** | Invite teammates and manage roles |

## Agents (Assistants)

An **agent** is a saved configuration for an AI voice assistant — language, voice, AI providers, instructions, and a linked phone number. It is not a human staff member.

### Create an agent

Click **Assistants** in the sidebar, then **New assistant**. Fill in the fields below.

| Field | What to enter |
|-------|---------------|
| Name | A label you will recognise, e.g. "Support agent — Hindi" |
| Language | Conversation language (e.g. Hindi `hi`) |
| Telephony | Provider for this agent (Vobiz or Plivo) |
| STT, TTS, LLM | Pick the speech-to-text, text-to-speech, and language model providers |
| Greeting | What the agent says when a call connects |
| System prompt | Instructions and persona for the agent |
| Knowledge base | Optional — attach uploaded PDFs for retrieval |

Click **Save**. The system registers the agent with the telephony provider so that calls can be routed to it.

{% hint style="info" %}
See [agents, campaigns, and calls](../../concepts/agents-campaigns-calls.md) for the underlying concepts and [knowledge base RAG](../../concepts/knowledge-base-rag.md) for how PDFs are used during a call.
{% endhint %}

## Integrations

**Dashboard → Integrations** is where your organization stores API credentials. Click **Integrations** in the sidebar.

| Integration type | Examples |
|------------------|----------|
| Telephony | Vobiz Auth ID and Vobiz Auth Token, Plivo keys |
| Speech-to-text | Deepgram, Bhashini, AI4Bharat |
| Text-to-speech | Cartesia, ElevenLabs, AI4Bharat |
| Large language model | OpenAI, Anthropic, custom |

{% hint style="warning" %}
Do not paste Vobiz Auth ID and Token into server `.env` files. The voice server reads telephony credentials from the database per organization. Enter them here.
{% endhint %}

See [integrations service](../../services/integrations.md) for a full provider list.

## Phone numbers

Click **Phone numbers** in the sidebar.

1. The page lists the numbers available in your telephony account.
2. Use **Link** to attach a number to an agent. Calls dialled to that number will use that agent's configuration.
3. Use **Unlink** to release a number from an agent.

## Test on Browser

Every agent card has a **Test on Browser** button. It uses the same voice pipeline as a phone call but routes audio through your computer microphone and speakers — useful for quick checks without spending phone minutes.

{% tabs %}
{% tab title="Run a browser test" %}
1. Open **Assistants** and find your agent.
2. Click **Test on Browser**.
3. Allow microphone access when prompted.
4. Wait for the connected state and speak.
5. You should hear the agent respond.
{% endtab %}

{% tab title="What it confirms" %}
Test on Browser proves that:

- The voice server is reachable.
- AI provider keys in **Integrations** are valid.
- The agent's STT, TTS, and LLM settings work.

It does **not** test telephony routing — phone calls require additional setup on the telephony provider.
{% endtab %}
{% endtabs %}

If the browser test fails, fix the voice server URL and AI keys before debugging telephony. See [common issues](../../troubleshooting/common-issues.md).

## Making your first inbound call

1. Confirm Vobiz credentials are saved under **Integrations**.
2. Create an agent and note its telephony settings.
3. Under **Phone numbers**, link a number to that agent.
4. Call the number from a mobile phone.
5. Check **Meetings** — a new row should appear with time, duration, and status.

See [first call walkthrough](../../quickstart/first-call.md) for a full step-by-step.

## Making outbound calls

VoicEra supports two outbound paths:

- **Campaigns** in the dashboard — upload a CSV of numbers and run a batch.
- **API call** to `POST /outbound/call/` on the voice server (for integrators — see [REST API](../../reference/rest-api.md)).

## Meetings and call history

Click **Meetings** to see every call that the system handled.

| Column | Meaning |
|--------|---------|
| Time | When the call started |
| Duration | How long the call lasted |
| Status | Completed, failed, or in progress |
| Recording | Audio playback link if recording is enabled |
| Transcript | Conversation text if generated |

Use this view to confirm that calls are happening and to spot failed or unusually short calls.

## Recordings

When recording is enabled, audio playback links appear under **Meetings**. Recordings are stored in object storage (MinIO) and served back to the dashboard. A hosting partner can access raw recordings directly — see the [operations guide](operations.md).

## Next steps

- [Day-to-day operations](operations.md)
- [Operator FAQ](faq.md)
- [First call walkthrough](../../quickstart/first-call.md)
- [Voice pipeline](../../concepts/voice-pipeline.md)
