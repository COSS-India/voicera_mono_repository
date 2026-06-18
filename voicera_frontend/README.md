# VoicEra Frontend

**Next.js dashboard** for operators: sign-in, **Assistants** (agents), **Integrations**, **Phone numbers**, **Meetings** / call history, browser voice test, campaigns/batches (when enabled), and telemetry views.

## When you need it

| Scenario | Need this app? |
|----------|----------------|
| Operators configuring agents and telephony | **Yes** |
| API-only / scripted setup | No (backend + voice server suffice) |

Default URL: http://localhost:3000

## Run

### Monorepo (Docker)

```bash
# From repository root — includes backend; add voice_server for browser test
docker compose up -d frontend
# or:
make start-all-services
```

Compose service: **`frontend`** (port **3000**). Uses `voicera_frontend/.env.local` at build/runtime.

```bash
docker compose stop frontend
```

### Monorepo (local dev, hot reload)

```bash
make start-dev    # npm dev frontend + Docker backend/minio + local voice stack
# frontend only:
make start-frontend
```

### Local development

```bash
cd voicera_frontend
cp .env.example .env.local   # then edit — see below
npm install
npm run dev
```

Production build: `npm run build` then `npm run start`, or the repo `frontend` Docker image.

## Environment variables

| Variable | Where | Purpose |
|----------|--------|---------|
| `NEXT_PUBLIC_API_URL` | Client + server | Backend API base (browser calls; default `http://localhost:8000`) |
| `API_URL` | Server routes only | Docker internal URL, e.g. `http://backend:8000` |
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | Client | Public HTTPS base for Vobiz/Plivo answer URLs and browser test |
| `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL` | Client (optional) | WebSocket base for **Test on Browser**; falls back from server URL |
| `VOICE_SERVER_URL` | Server routes only | Outbound call / telemetry proxy (default `http://localhost:7860`) |

Example `.env.local` for local dev:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_JOHNAIC_SERVER_URL=https://your-public-voice-host
NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL=wss://your-public-voice-host
VOICE_SERVER_URL=http://localhost:7860
```

See [public voice URLs](../docs/deployment/public-voice-urls.md) for why the Johnaic URLs must be reachable from the browser and telephony provider.

## Dependencies

| Service | Required for |
|---------|----------------|
| **backend** (:8000) | Login, agents, integrations, meetings |
| **voice_server** (:7860) | **Test on Browser**, outbound-call API routes, GPU telemetry |

MongoDB and MinIO are used by the backend, not directly by the frontend.

## Documentation

- [Frontend service (MkDocs)](../docs/services/frontend.md)
- [Dashboard walkthrough](../docs/guide/dashboard.md)
- [Verify it works](../docs/guide/verification.md)
- [Source brief A1](../docs/source-briefs/A1-submodule-readmes.md) · [B3 walkthrough brief](../docs/source-briefs/B3-dashboard-walkthrough.md)
