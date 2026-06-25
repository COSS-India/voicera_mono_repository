---
description: Checklist of accounts, infrastructure, and network items needed before installing VoicEra.
---

# Prerequisites

Run through this checklist before installing VoicEra. It applies to operators, programme owners, and hosting partners preparing a server for development or production.

{% hint style="info" %}
Assign a responsible party (operator, programme owner, or hosting partner) for each item. Anything marked as production-only can be skipped for a local test install.
{% endhint %}

## People and accounts

- [ ] Decision owner who approves go-live
- [ ] Operator(s) who will use the dashboard daily
- [ ] Hosting partner who can run Docker and Linux commands
- [ ] Vobiz account with at least one phone number purchased
- [ ] AI provider accounts if using cloud speech or LLM (OpenAI, Bhashini, etc.), or plan for self-hosted AI4Bharat servers
- [ ] Email delivery for signup and password reset (production SMTP or API — not Mailtrap)

## Server and infrastructure

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Linux, macOS, or Windows (WSL2) | Ubuntu 20.04 LTS or newer |
| RAM | 8 GB | 16 GB+ |
| CPU | 2 cores | 4+ cores |
| Disk | 50 GB | 100 GB+ NVMe SSD |
| Docker | 20.10+ | latest stable |
| Docker Compose | 1.29+ | latest stable |
| GPU | not required for cloud AI | NVIDIA CUDA GPU if running local AI4Bharat STT/TTS |

{% hint style="warning" %}
Size CPU, RAM, and disk with your hosting partner based on expected concurrent calls. Load-test on staging before production.
{% endhint %}

## Network and DNS

- [ ] Public domain name for dashboard and voice server (recommended)
- [ ] HTTPS certificate for dashboard
- [ ] HTTPS and WSS for the public voice server URL
- [ ] Firewall allows the telephony provider to reach webhook URLs on port 443
- [ ] MongoDB (27017) and MinIO (9000/9001) are not exposed to the public internet
- [ ] Stable internet with inbound connections allowed to your public voice URL

## Software package

- [ ] Copy of `voicera_mono_repository` on the server
- [ ] Environment files prepared for backend, frontend, voice server, and (optional) AI4Bharat servers
- [ ] Vobiz credentials ready to enter in **Dashboard → Integrations** once services are up

## Optional tools

| Tool | When you need it |
|------|------------------|
| Node.js 18+ | Local frontend development outside Docker |
| Python 3.10+ | Local voice server or backend development |
| Make | Convenience commands from the repo Makefile |
| Ngrok | Exposing a local voice server during testing |

## Environment variable configuration

Each service reads its own `.env` file. Copy the example files before starting services:

```bash
cp voicera_backend/env.example          voicera_backend/.env
cp voice_2_voice_server/.env.example    voice_2_voice_server/.env
cp voicera_frontend/.env.example        voicera_frontend/.env.local

# Optional — only if running local AI4Bharat servers
cp ai4bharat_stt_server/.env.example    ai4bharat_stt_server/.env
cp ai4bharat_tts_server/.env.example    ai4bharat_tts_server/.env
```

Edit each file before going live. At minimum set a strong `SECRET_KEY`, a matching `INTERNAL_API_KEY` on both backend and voice server, and the public voice server URLs (`JOHNAIC_SERVER_URL`, `JOHNAIC_WEBSOCKET_URL`).

For the full variable reference see [Environment variables](../reference/environment-variables.md). Installation steps are in [Install and run](install-and-run.md).

## Nginx reverse proxy

In production, nginx terminates TLS and forwards traffic to the VoicEra containers. Install:

```bash
sudo apt-get install -y nginx certbot python3-certbot-nginx
sudo certbot certonly --standalone \
  -d api.example.gov.in \
  -d voice.example.gov.in
```

Minimal nginx server block with WebSocket support (`/etc/nginx/sites-enabled/voicera.conf`):

```nginx
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
        proxy_pass         http://localhost:8000;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade           $http_upgrade;
        proxy_set_header   Connection        "upgrade";
        proxy_set_header   Host              $host;
        proxy_set_header   X-Real-IP         $remote_addr;
        proxy_set_header   X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header   X-Forwarded-Proto $scheme;
    }
}

# Voice server (HTTP + WebSocket)
server {
    listen 443 ssl http2;
    server_name voice.example.gov.in;

    ssl_certificate     /etc/letsencrypt/live/voice.example.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.example.gov.in/privkey.pem;

    location / {
        proxy_pass         http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header   Upgrade         $http_upgrade;
        proxy_set_header   Connection      "upgrade";
        proxy_set_header   Host            $host;
        proxy_set_header   X-Real-IP       $remote_addr;
        proxy_set_header   X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

```bash
sudo nginx -t && sudo systemctl reload nginx
```

For a complete production nginx config including rate limiting and multiple domains see [Production deployment](../guides/deployment/production.md).

## Systemd service

To start VoicEra automatically on reboot, create a systemd unit that wraps `docker compose`:

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
Restart=on-failure
RestartSec=10
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

Change `WorkingDirectory` to the path where you cloned the repository.

## Verification / smoke test

After bringing up the stack, confirm every service is healthy:

```bash
# All containers should show "Up" (not "Restarting" or "Exit")
docker compose ps

# Backend health endpoint
curl -fs http://localhost:8000/health && echo "Backend OK"

# Voice server health endpoint
curl -fs http://localhost:7860/health && echo "Voice server OK"

# Tail logs for 30 seconds and look for errors
docker compose logs -f --tail 30
```

Then open the frontend (`http://localhost:3000`) and confirm the login page renders. For a guided first-call test see [First call](first-call.md).

## Known issues / troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| GPU not detected in AI4Bharat container | `--gpus all` flag missing or `nvidia-container-toolkit` not installed | Install `nvidia-container-toolkit` and confirm `nvidia-smi` works on the host; ensure `deploy.resources.reservations.devices` is present in `docker-compose.yml` |
| Port already in use on startup | Another process holds 8000, 7860, or 27017 | `lsof -i :<port>` to find it, or run `make stop-all-ports` |
| MongoDB connection refused | Backend starts before MongoDB is healthy | Wait for `docker compose ps` to show MongoDB `healthy`; check `MONGODB_HOST` is `mongodb` (not `localhost`) inside Docker |
| Backend returns 401 on voice server calls | `INTERNAL_API_KEY` mismatch | Ensure the same value is set in both `voicera_backend/.env` and `voice_2_voice_server/.env` |
| WebSocket handshake fails from telephony | `JOHNAIC_WEBSOCKET_URL` not reachable publicly | Confirm the URL is correct, the server is accessible on port 443, and the nginx config includes WebSocket upgrade headers |

More remedies: [Troubleshooting: deployment](../troubleshooting/deployment.md).

## Next steps

- [install-and-run.md](install-and-run.md)
- [default-credentials.md](default-credentials.md)
- [../reference/environment-variables.md](../reference/environment-variables.md)
- [../reference/ports-and-defaults.md](../reference/ports-and-defaults.md)
