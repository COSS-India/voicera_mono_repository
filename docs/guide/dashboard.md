# Dashboard walkthrough

This guide explains the VoicEra web dashboard for operators.

## Login

Open your deployment URL (typically port `3000` or your HTTPS domain). Sign in with the account provided by your administrator.

## Assistants (agents)

The **Assistants** page lists each **agent** — a saved configuration for an AI voice assistant (language, STT/TTS/LLM, instructions, telephony). This is **not** a human staff member.

### Create an agent

Typical steps when creating an agent:

1. **Name and language** — display name and conversation language (for example Hindi `hi`, Bhili `bhb` when supported).
2. **Telephony** — provider (Vobiz is the primary supported option on most deployments).
3. **AI settings** — STT, TTS, and LLM providers (see [Voice server](../services/voice-server.md) for provider list).
4. **Prompt / greeting** — what the agent says and how it behaves.
5. **Knowledge base** (optional) — attach uploaded PDFs for RAG — [Knowledge Base](../services/knowledge-base.md).

After creation, the system may register a Vobiz application with an **answer URL** pointing at your [public voice server](../deployment/public-voice-urls.md).

## Integrations

**Dashboard → Integrations** is where your organization stores API credentials:

| Integration type | Examples |
|------------------|----------|
| Telephony | **Vobiz Auth ID**, **Vobiz Auth Token** |
| AI / speech | OpenAI, Bhashini, Deepgram, Cartesia, etc. |

!!! warning
    Do **not** tell operators to put Vobiz Auth ID/Token only in server `.env` files. The voice server reads telephony credentials from the database per organization. See [Integrations](../services/integrations.md).

## Phone numbers

**Dashboard → Phone numbers**:

1. Lists numbers available in your telephony account.
2. **Link** a number to an agent so inbound calls on that number use that agent's configuration.

## Test on Browser

On an agent card, use **Test on Browser** to talk to the agent through your computer microphone and speakers — same voice pipeline as phone calls, without using phone minutes.

1. Click **Test on Browser**.
2. Allow microphone access when prompted.
3. Wait for a connected state, then speak.
4. You should hear the agent respond.

If this fails, fix voice server URLs and AI keys before debugging telephony. See [Verify it works](verification.md) and [FAQ](faq.md).

Technical protocol: `voice_2_voice_server/docs/talk-on-browser-feature.md` in the repository.

## Meetings / calls

After real calls, new rows appear under **Meetings** (or calls history) with time, duration, and status. Use this to confirm inbound/outbound traffic.

## Recordings

When recording is enabled, audio may be available from the dashboard or via MinIO (hosting partner). Confirm the exact UI path on your staging build.

## Campaigns and batches (optional)

If enabled, **Campaigns** / **Batches** support outbound calling to lists of numbers. Requires audience CSV upload and worker configuration — see backend Swagger for `/api/v1/batches`.

## How to make a call

### Inbound

1. Save **Vobiz** credentials under **Integrations**.
2. Create an agent and note its telephony settings.
3. On **Phone numbers**, link a number to that agent.
4. Call the number from a mobile phone.
5. Confirm the call appears under **Meetings**.

### Browser test (no phone)

Use **Test on Browser** on the agent card — see above.

### Outbound

May use campaigns in the dashboard or `POST /outbound/call/` on the voice server — confirm which features are enabled on your deployment. See [API quick reference](../api/endpoints.md).
