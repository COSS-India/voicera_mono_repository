---
description: Set up VoicEra on a developer workstation — frontend, backend, and voice server with hot reload.
---

# Local development setup

This guide gets the three VoicEra services running on your machine with hot reload. Audience: engineers extending or debugging the code.

{% hint style="info" %}
If you only want to try VoicEra end-to-end, use the Docker Compose path in [install and run](../../quickstart/install-and-run.md). This page covers running the services directly against your editor.
{% endhint %}

> **GPU drivers**: If you plan to run local AI4Bharat STT/TTS containers during development, install the NVIDIA driver and `nvidia-container-toolkit` on your machine first. See [Prerequisites → Server and infrastructure](../../quickstart/prerequisites.md#server-and-infrastructure) for details.

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Docker + Docker Compose | latest stable |
| Git | latest |
| Editor | VS Code, PyCharm, or similar |

Clone the monorepo:

```bash
git clone https://github.com/COSS-India/voicera_mono_repository.git
cd voicera_mono_repository
```

## Shared infrastructure

The backend and voice server both need MongoDB and MinIO. Start them with Docker Compose from the repo root:

```bash
docker compose up -d mongodb minio
docker compose ps
```

Wait until both containers show `healthy`. See [Docker Compose deployment](../deployment/docker-compose.md) for the full stack.

## Frontend

```bash
cd voicera_frontend
npm install
cp .env.example .env.local
```

Edit `voicera_frontend/.env.local`:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_JOHNAIC_SERVER_URL=http://localhost:7860
NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL=ws://localhost:7860
```

Run the dev server:

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000). Saving a file refreshes the browser automatically.

{% hint style="success" %}
For faster builds on large branches, use Turbopack: `npm run dev -- --turbo`.
{% endhint %}

## Backend

```bash
cd voicera_backend
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp env.example .env
```

Edit `voicera_backend/.env`:

```env
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true

MONGODB_HOST=localhost
MONGODB_PORT=27017

SECRET_KEY=dev-secret-key-change-in-production
CORS_ORIGINS=["http://localhost:3000"]
```

Run with auto-reload:

```bash
uvicorn app.main:app --reload --port 8000
```

Swagger UI is at [http://localhost:8000/docs](http://localhost:8000/docs).

See [backend service](../../services/backend.md) and [environment variables](../../reference/environment-variables.md) for the full configuration surface.

## Voice server

```bash
cd voice_2_voice_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `voice_2_voice_server/.env`. Minimum keys for a working cloud-only pipeline:

```env
JOHNAIC_SERVER_URL=http://localhost:7860
JOHNAIC_WEBSOCKET_URL=ws://localhost:7860

VOICERA_BACKEND_URL=http://localhost:8000
INTERNAL_API_KEY=your-internal-api-key

MINIO_ENDPOINT=localhost:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
```

> **Provider API keys do not belong in `.env` for multi-tenant setups.** Store them in **Dashboard → Integrations** after the services are running — the voice server fetches them per-organisation at call time. The env vars (`OPENAI_API_KEY`, `DEEPGRAM_API_KEY`, etc.) are single-tenant dev fallbacks only; see [environment variables](../../reference/environment-variables.md#provider-api-keys-fallback).

Run the server:

```bash
python main.py
```

Swagger UI is at [http://localhost:7860/docs](http://localhost:7860/docs). See [voice server](../../services/voice-server.md) for the pipeline internals.

{% hint style="warning" %}
The voice server does not auto-reload. Restart it after Python changes.
{% endhint %}

## Running everything together

Open three terminals.

{% tabs %}
{% tab title="Terminal 1 — Frontend" %}
```bash
cd voicera_frontend && npm run dev
```
{% endtab %}

{% tab title="Terminal 2 — Backend" %}
```bash
cd voicera_backend && source venv/bin/activate
uvicorn app.main:app --reload --port 8000
```
{% endtab %}

{% tab title="Terminal 3 — Voice server" %}
```bash
cd voice_2_voice_server && source venv/bin/activate
python main.py
```
{% endtab %}
{% endtabs %}

Then walk through [your first call](../../quickstart/first-call.md) using Test on Browser.

## IDE setup (VS Code)

Recommended extensions:

- `ms-python.python`
- `ms-python.vscode-pylance`
- `ms-python.debugpy`
- `esbenp.prettier-vscode`
- `dbaeumer.vscode-eslint`
- `ms-vscode.makefile-tools`
- `ms-azuretools.vscode-docker`

Save the following as `.vscode/launch.json` to attach the debugger:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Backend",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/voicera_backend/app/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/voicera_backend",
        "RELOAD": "true"
      }
    },
    {
      "name": "Voice server",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/voice_2_voice_server/main.py",
      "console": "integratedTerminal"
    },
    {
      "type": "node",
      "request": "launch",
      "name": "Next.js",
      "skipFiles": ["<node_internals>/**"],
      "program": "${workspaceFolder}/voicera_frontend/node_modules/.bin/next",
      "args": ["dev"],
      "console": "integratedTerminal"
    }
  ]
}
```

Press `F5` to start a debug session.

## Debugging tips

| Where | How |
|-------|-----|
| Python | `import pdb; pdb.set_trace()` or VS Code breakpoints |
| TypeScript | Chrome DevTools, breakpoints in the Sources tab |
| Container logs | `docker compose logs -f <service>` |
| Filter logs | `docker compose logs <service> | grep error` |

## Common dev tasks

### Add an API endpoint

Create a router under `voicera_backend/app/routers/`:

```python
from fastapi import APIRouter

router = APIRouter(prefix="/feature", tags=["feature"])

@router.get("/")
async def list_items():
    return {"items": []}
```

Wire it up in `voicera_backend/app/main.py`:

```python
from app.routers import new_feature
app.include_router(new_feature.router)
```

### Add a frontend component

```bash
touch voicera_frontend/components/MyComponent.tsx
```

Import it from a page:

```tsx
import MyComponent from '@/components/MyComponent';

export default function Page() {
  return <MyComponent />;
}
```

### Update dependencies

```bash
# Backend
pip list --outdated
pip install --upgrade <package>

# Frontend
npm outdated
npm update
```

### Security checks

```bash
# Backend
safety check

# Frontend
npm audit
```

## Nginx reverse proxy (optional for local dev)

Local development typically runs without a reverse proxy — services bind to `localhost` ports directly. If you need to test webhook flows with a public URL, use ngrok (see [Install and run](../../quickstart/install-and-run.md#4-expose-the-voice-server-optional-dev-only)).

For a production nginx configuration that proxies to the backend (port 8000) and voice server (port 7860) with WebSocket and SSL support, see [Deployment walkthrough → Nginx reverse proxy](../deployment/deployment-walkthrough.md#nginx-reverse-proxy).

## Systemd service

A systemd unit is not required for development. If you need the development stack to start on boot, use the same unit described in [Docker Compose deployment → Systemd service](../deployment/docker-compose.md#systemd-service-auto-start-on-reboot), pointing `WorkingDirectory` at your local clone.

## Next steps

- [Testing guide](testing.md)
- [Contributing](contributing.md)
- [Architecture](../../concepts/architecture.md)
- [Environment variables](../../reference/environment-variables.md)
