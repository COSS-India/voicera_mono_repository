# Brief: Step-by-step deployment (B5)

**Review gap:** Instructions assume Docker/Python comfort; need plain explanations of each step and what success looks like.

**Audience:** Hosting partner implements; operator oversees using B2 checklist.

**Writer format for each step:**

1. **What you do** (command or action)
2. **What it means** (plain language)
3. **Success looks like** (observable outcome)

---

## Prerequisites

Complete [B2-before-you-start-checklist.md](./B2-before-you-start-checklist.md) first.

---

## Deployment steps (Docker / Makefile — standard path)

### Step 1: Prepare the server

| | |
|---|---|
| **Do** | Install Linux, Docker, Docker Compose; clone `voicera_mono_repository` |
| **Means** | The machine can run the packaged services |
| **Success** | `docker --version` and `docker compose version` work; repo folder present |

### Step 2: Configure environment files

| | |
|---|---|
| **Do** | Copy example env files to `.env` / `.env.local` (see root README Environment Configuration) |
| **Means** | Each service knows where database, URLs, and secrets are |
| **Success** | Files exist; hosting partner has filled non-default secrets |

**Important for operators:** Vobiz/Plivo **auth ID and token** go in **Dashboard → Integrations** after services are up — not as the primary setup method in `.env`.

**Must set for production:**

- `JOHNAIC_SERVER_URL` / `JOHNAIC_WEBSOCKET_URL` (public voice URLs — A8)
- `NEXT_PUBLIC_JOHNAIC_SERVER_URL` on frontend
- `SECRET_KEY`, `INTERNAL_API_KEY` (strong random values)
- MongoDB and MinIO credentials (non-default)

### Step 3: Build containers

| | |
|---|---|
| **Do** | `make build-all-services` |
| **Means** | Docker images are built for database, backend, MinIO, frontend, voice server |
| **Success** | Command completes without error |

### Step 4: Start services

| | |
|---|---|
| **Do** | `make start-all-services` |
| **Means** | All core containers run in background |
| **Success** | `docker compose ps` shows services running |

### Step 5: Open dashboard

| | |
|---|---|
| **Do** | Browse to frontend URL (port 3000 or HTTPS proxy) |
| **Means** | Web interface is live |
| **Success** | Login or signup page loads |

### Step 6: Configure Integrations

| | |
|---|---|
| **Do** | Log in → Integrations → enter Vobiz/Plivo and AI keys |
| **Means** | Telephony and AI can authenticate |
| **Success** | Saved without error; test call or browser test works (B6) |

### Step 7: Create test agent and link number

| | |
|---|---|
| **Do** | Assistants → create agent; Numbers → link phone number |
| **Means** | Inbound calls route to your agent |
| **Success** | Test inbound call completes (B6) |

### Step 8: Configure telephony provider portal

| | |
|---|---|
| **Do** | In Vobiz/Plivo portal, ensure application answer URL matches `{PUBLIC_URL}/answer?agent_id=...` (often set automatically when creating agent from dashboard) |
| **Means** | Provider knows where to send calls |
| **Success** | Inbound call reaches agent voice |

---

## Stop / restart

| Action | Command |
|--------|---------|
| Stop all | `make stop-all-services` |
| Start again | `make start-all-services` |

---

## Optional: local AI4Bharat servers

Only if agents use Indic local speech:

- See A5 for STT/TTS server setup
- `make start-voice-only-services` for dev-style local stack
- Requires GPU sizing per A5

---

## Appendix: developer local setup

Point technical readers to root `README.md` (Makefile, venv, `make start-dev`) — not required for operator-only documentation.

---

## Related

- Verify: [B6-how-to-know-its-working.md](./B6-how-to-know-its-working.md)
- Security: [A6-security-production-hardening.md](./A6-security-production-hardening.md)
