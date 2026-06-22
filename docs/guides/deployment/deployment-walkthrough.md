---
description: Step-by-step VoicEra deployment for a hosting partner, with what each step does and how to know it worked.
---

# Deployment walkthrough

A hosting partner uses this guide to bring VoicEra online with Docker. Operators oversee the result. Complete the [Prerequisites](../../quickstart/prerequisites.md) checklist before you begin.

Each step lists **what you do**, **what it means**, and **success looks like**.

## Step 1: Prepare the server

| | |
|---|---|
| **Do** | Install Linux, Docker 20.10+, Docker Compose v2; clone `voicera_mono_repository` |
| **Means** | The host can build images and run packaged services |
| **Success** | `docker --version` and `docker compose version` both work; the repo folder is present |

## Step 2: Configure environment files

| | |
|---|---|
| **Do** | Copy example env files into place and edit them |
| **Means** | Each service learns its database URL, secrets, and public endpoints |
| **Success** | The three `.env` files exist with non-default secrets |

```bash
cp voicera_backend/env.example          voicera_backend/.env
cp voice_2_voice_server/.env.example    voice_2_voice_server/.env
cp voicera_frontend/.env.example        voicera_frontend/.env.local
```

Production must set:

- [Public voice server URLs](public-voice-urls.md): `JOHNAIC_SERVER_URL`, `JOHNAIC_WEBSOCKET_URL`, `NEXT_PUBLIC_JOHNAIC_SERVER_URL`
- Strong random `SECRET_KEY` and `INTERNAL_API_KEY` (same value on backend and voice server)
- Non-default MongoDB and MinIO credentials

{% hint style="info" %}
Vobiz and Plivo auth ID and tokens go in **Dashboard → Integrations** after services are up, not in `.env`.
{% endhint %}

Generate strong secrets:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Full variable reference: [Environment variables](../../reference/environment-variables.md).

## Step 3: Build containers

| | |
|---|---|
| **Do** | `make build-all-services` |
| **Means** | Docker images build for MongoDB, MinIO, backend, voice server, frontend |
| **Success** | Command completes without errors |

## Step 4: Start services

| | |
|---|---|
| **Do** | `make start-all-services` |
| **Means** | All core containers run in the background |
| **Success** | `docker compose ps` shows every service `Up` and the database services `healthy` |

```bash
make start-all-services
docker compose ps
```

## Step 5: Open the dashboard

| | |
|---|---|
| **Do** | Browse to the frontend URL (port `3000`, or your HTTPS proxy domain) |
| **Means** | The Next.js dashboard is reachable |
| **Success** | The login or signup page renders |

Default development credentials are listed in [Default credentials](../../quickstart/default-credentials.md). Change them before exposing the dashboard externally.

## Step 6: Configure integrations

| | |
|---|---|
| **Do** | Log in → **Integrations** → enter Vobiz/Plivo and AI provider keys |
| **Means** | Telephony and AI providers can authenticate |
| **Success** | Save succeeds; a [browser test call](../../quickstart/first-call.md) reaches the agent |

## Step 7: Create a test agent and link a number

| | |
|---|---|
| **Do** | **Assistants** → create agent; **Phone numbers** → link a number |
| **Means** | Inbound calls route to your agent |
| **Success** | A test inbound call completes |

## Step 8: Telephony provider portal

| | |
|---|---|
| **Do** | In the Vobiz/Plivo portal, ensure the application **answer URL** matches `{JOHNAIC_SERVER_URL}/answer?agent_id=...` |
| **Means** | The provider knows where to send incoming call webhooks |
| **Success** | An inbound call reaches the agent's voice |

VoicEra usually sets this automatically when the agent is created from the dashboard. Confirm it manually if the provider portal allows.

## Stop and restart

| Action | Command |
|--------|---------|
| Stop all services | `make stop-all-services` |
| Start them again | `make start-all-services` |
| Free stale host ports | `make stop-all-ports` |

## Optional: local AI4Bharat speech

Only if agents use `indic-conformer-stt` or `indic-parler-tts`:

- [AI4Bharat STT](../../services/ai4bharat-stt.md) on port `8001`
- [AI4Bharat TTS](../../services/ai4bharat-tts.md) on port `8002`
- Development convenience: `make start-voice-only-services`
- Production: GPU required for acceptable latency

## Developer setup

For local development with hot reload, see [Local setup](../developer/local-setup.md) and the root `README.md`.

## Next steps

- [Security hardening](security-hardening.md)
- [Production deployment](production.md)
- [Operations](../operator/operations.md)
- [Troubleshooting: deployment](../../troubleshooting/deployment.md)
