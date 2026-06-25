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

## Nginx reverse proxy

In production, front the stack with nginx to handle TLS termination and proxy traffic to the backend and voice server. Install nginx and obtain a certificate:

```bash
sudo apt-get install -y nginx
sudo certbot certonly --standalone -d api.example.gov.in -d voice.example.gov.in
```

Create `/etc/nginx/sites-enabled/voicera.conf`:

```nginx
# Redirect HTTP → HTTPS
server {
    listen 80;
    server_name api.example.gov.in voice.example.gov.in;
    return 301 https://$server_name$request_uri;
}

# Backend API
server {
    listen 443 ssl http2;
    server_name api.example.gov.in;

    ssl_certificate     /etc/letsencrypt/live/api.example.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.example.gov.in/privkey.pem;

    location / {
        proxy_pass http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_set_header Upgrade           $http_upgrade;
        proxy_set_header Connection        "upgrade";
    }
}

# Voice server (HTTP + WebSocket)
server {
    listen 443 ssl http2;
    server_name voice.example.gov.in;

    ssl_certificate     /etc/letsencrypt/live/voice.example.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.example.gov.in/privkey.pem;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade         $http_upgrade;
        proxy_set_header Connection      "upgrade";
        proxy_set_header Host            $host;
        proxy_set_header X-Real-IP       $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

After the proxy is in place, set `JOHNAIC_SERVER_URL=https://voice.example.gov.in` and `JOHNAIC_WEBSOCKET_URL=wss://voice.example.gov.in` in `voice_2_voice_server/.env`. See [Production deployment](production.md) for a complete multi-domain nginx config.

## Verification / smoke test

After the stack starts, run these quick checks to confirm all services are healthy:

```bash
# 1. Check all containers are running
docker compose ps

# 2. Backend health
curl -fs http://localhost:8000/health && echo "Backend OK"

# 3. Voice server health
curl -fs http://localhost:7860/health && echo "Voice server OK"

# 4. Frontend reachable
curl -fsI http://localhost:3000 | head -1

# 5. Follow live logs for a few seconds
docker compose logs -f --tail 20
```

Expected results:
- `docker compose ps` — every service shows `Up` (no `Restarting` or `Exit`)
- Backend and voice server health endpoints return a 2xx response
- Frontend returns HTTP `200` or `302`
- No `ERROR` lines appear in the tail of the logs

If any service is unhealthy, check its logs:

```bash
docker compose logs -f <service-name>   # e.g. backend, voice_server, mongodb
```

See [Troubleshooting: deployment](../../troubleshooting/deployment.md) for remediation steps.

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

{% hint style="warning" %}
**GPU driver required for local AI4Bharat services.** Before starting these containers, verify `nvidia-smi` runs cleanly on the host. If the driver is not installed, see [Prerequisites → GPU for local AI4Bharat](../../quickstart/prerequisites.md) for NVIDIA driver and CUDA installation steps. Docker must also be configured for GPU access (`nvidia-container-toolkit`).
{% endhint %}

## Systemd service (auto-start on reboot)

To keep VoicEra running across server reboots without a process manager, create a systemd unit:

```bash
sudo tee /etc/systemd/system/voicera.service > /dev/null << 'EOF'
[Unit]
Description=VoicEra stack (Docker Compose)
Requires=docker.service
After=docker.service network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=/opt/voicera_mono_repository
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=300

[Install]
WantedBy=multi-user.target
EOF
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable voicera
sudo systemctl start voicera
sudo systemctl status voicera
```

View live logs via journald:

```bash
# Tail all service logs
journalctl -u voicera -f

# Last 200 lines
journalctl -u voicera -n 200 --no-pager
```

## Developer setup

For local development with hot reload, see [Local setup](../developer/local-setup.md) and the root `README.md`.

## Next steps

- [Security hardening](security-hardening.md)
- [Production deployment](production.md)
- [Operations](../operator/operations.md)
- [Troubleshooting: deployment](../../troubleshooting/deployment.md)
