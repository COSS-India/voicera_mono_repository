# Deployment walkthrough

Step-by-step guide for a **hosting partner** deploying VoicEra with Docker. Operators should complete the [Prerequisites](prerequisites.md) checklist first.

Each step includes: **what you do**, **what it means**, and **success looks like**.

## Step 1: Prepare the server

| | |
|---|---|
| **Do** | Install Linux, Docker, Docker Compose; clone the repository |
| **Means** | The machine can run packaged services |
| **Success** | `docker --version` and `docker compose version` work; repo folder present |

## Step 2: Configure environment files

| | |
|---|---|
| **Do** | Copy example env files to `.env` / `.env.local` (see [Configuration](../getting-started/configuration.md)) |
| **Means** | Each service knows database URLs and secrets |
| **Success** | Files exist with non-default `SECRET_KEY`, `INTERNAL_API_KEY`, MongoDB, MinIO |

**Production must set:**

- [Public voice server URLs](../deployment/public-voice-urls.md) (`JOHNAIC_SERVER_URL`, `JOHNAIC_WEBSOCKET_URL`, `NEXT_PUBLIC_JOHNAIC_SERVER_URL`)
- Strong `SECRET_KEY` and `INTERNAL_API_KEY` (same value on backend and voice server)
- Non-default MongoDB and MinIO credentials

!!! important
    **Vobiz Auth ID and Token** go in **Dashboard → Integrations** after services are running — not as the primary setup in `.env`.

## Step 3: Build containers

| | |
|---|---|
| **Do** | `make build-all-services` |
| **Means** | Docker images built for database, backend, MinIO, frontend, voice server |
| **Success** | Command completes without error |

## Step 4: Start services

| | |
|---|---|
| **Do** | `make start-all-services` |
| **Success** | `docker compose ps` shows services running |

## Step 5: Open dashboard

| | |
|---|---|
| **Do** | Browse to frontend URL (port 3000 or HTTPS proxy) |
| **Success** | Login or signup page loads |

## Step 6: Configure Integrations

| | |
|---|---|
| **Do** | Log in → **Integrations** → enter Vobiz and AI keys |
| **Success** | Saved without error; [Test on Browser](verification.md) works |

## Step 7: Create test agent and link number

| | |
|---|---|
| **Do** | **Assistants** → create agent; **Phone numbers** → link number |
| **Success** | Test inbound call completes — [Verify it works](verification.md) |

## Step 8: Telephony provider portal

| | |
|---|---|
| **Do** | In Vobiz portal, confirm application **answer URL** matches `{PUBLIC_URL}/answer?agent_id=...` (often set when creating the agent from the dashboard) |
| **Success** | Inbound call reaches agent voice |

## Stop and restart

| Action | Command |
|--------|---------|
| Stop all | `make stop-all-services` |
| Start again | `make start-all-services` |
| Free ports | `make stop-all-ports` |

## Optional: local AI4Bharat

Only if agents use `indic-conformer-stt` / `indic-parler-tts`:

- [AI4Bharat STT](../services/ai4bharat-stt.md) and [TTS](../services/ai4bharat-tts.md)
- Development: `make start-voice-only-services`
- Requires GPU for production speech quality

## Developer local setup

Technical readers: root `README.md`, [Local setup](../development/local-setup.md), `make start-dev`.

## Related

- [Verify it works](verification.md)
- [Security hardening](../deployment/security-hardening.md)
