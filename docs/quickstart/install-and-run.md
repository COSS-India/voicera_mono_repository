---
description: Install Docker, clone the repository, configure environment files, and start all VoicEra services.
---

# Install and run

End-to-end install path for operators and hosting partners running VoicEra with Docker Compose. Complete the [prerequisites](prerequisites.md) before starting.

> **GPU drivers**: If you plan to run local AI4Bharat STT/TTS containers, install the NVIDIA driver and `nvidia-container-toolkit` before proceeding. See [Prerequisites → Server and infrastructure](prerequisites.md#server-and-infrastructure) for GPU requirements.

## 1. Install Docker and Docker Compose

{% tabs %}
{% tab title="Ubuntu" %}
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose

# Optional: avoid sudo for docker
sudo usermod -aG docker $USER
newgrp docker
```
{% endtab %}

{% tab title="macOS" %}
```bash
brew install docker docker-compose
# Or download Docker Desktop from docker.com
```
{% endtab %}

{% tab title="Windows" %}
1. Download [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).
2. Install with the WSL2 backend.
3. Enable WSL2 in Windows Features and restart.
{% endtab %}
{% endtabs %}

Verify the install:

```bash
docker --version
docker-compose --version
docker run hello-world
```

## 2. Clone the repository

```bash
git clone https://github.com/COSS-India/voicera_mono_repository.git
cd voicera_mono_repository
```

## 3. Configure environment files

Copy each example file and edit it for your environment.

```bash
cp voicera_backend/env.example voicera_backend/.env
cp voicera_frontend/.env.example voicera_frontend/.env.local
cp voice_2_voice_server/.env.example voice_2_voice_server/.env

# Optional, only if running local AI4Bharat servers
cp ai4bharat_stt_server/.env.example ai4bharat_stt_server/.env
cp ai4bharat_tts_server/.env.example ai4bharat_tts_server/.env
```

### Backend (`voicera_backend/.env`)

```env
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin

DEBUG=False
SECRET_KEY=your-secret-key
INTERNAL_API_KEY=your-internal-api-key

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

FRONTEND_URL=http://localhost:3000
```

Generate strong secrets:

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

### Voice server (`voice_2_voice_server/.env`)

```env
VOBIZ_API_BASE=https://api.vobiz.in/v1
VOBIZ_CALLER_ID=+91XXXXXXXXXX

JOHNAIC_SERVER_URL=https://your-public-voice-url
JOHNAIC_WEBSOCKET_URL=wss://your-public-voice-url

VOICERA_BACKEND_URL=http://backend:8000
INTERNAL_API_KEY=your-internal-api-key

MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin
MINIO_SECURE=false
```

{% hint style="info" %}
`INTERNAL_API_KEY` must match between backend and voice server. Vobiz auth ID and token are entered in **Dashboard → Integrations** after services start, not in `.env`.
{% endhint %}

### Frontend (`voicera_frontend/.env.local`)

```env
NEXT_PUBLIC_API_URL=http://localhost:8000/api/v1
API_URL=http://backend:8000
VOICE_SERVER_URL=http://voice_server:7860
```

## 4. Expose the voice server (optional, dev only)

If you are running locally and need a public URL for telephony webhooks, use ngrok.

```bash
ngrok config add-authtoken <YOUR_AUTH_TOKEN>
ngrok http 7860
```

Copy the generated `https://` URL into `JOHNAIC_SERVER_URL` and the `wss://` form into `JOHNAIC_WEBSOCKET_URL`.

{% hint style="warning" %}
Ngrok tunnels are for testing only. Use a real public domain with TLS for production.
{% endhint %}

## 5. Build and start

```bash
make build-all-services
make start-all-services
```

Equivalent direct commands:

```bash
docker-compose build
docker-compose up -d
```

## 6. Verify the stack is up

| Service | URL | Expected |
|---------|-----|----------|
| Frontend | `http://localhost:3000` | Login or home page |
| Backend | `http://localhost:8000/docs` | Swagger UI |
| Voice server | `http://localhost:7860/health` | Healthy response |
| MinIO console | `http://localhost:9001` | Login page |

```bash
docker compose ps
docker images | grep voicera
```

## Nginx reverse proxy (production)

For production, place nginx in front of the stack to terminate TLS and proxy traffic. Install and obtain a certificate:

```bash
sudo apt-get install -y nginx
sudo certbot certonly --standalone \
  -d api.example.gov.in \
  -d voice.example.gov.in
```

Minimal `/etc/nginx/sites-enabled/voicera.conf` with WebSocket support:

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

After setting up nginx, update `JOHNAIC_SERVER_URL` and `JOHNAIC_WEBSOCKET_URL` in `voice_2_voice_server/.env` to use your public domain. For a full production-grade config see [Production deployment](../guides/deployment/production.md).

## Common issues

{% hint style="warning" %}
**Docker daemon not running** — on Linux, run `sudo systemctl start docker`. On macOS or Windows, open Docker Desktop.
{% endhint %}

{% hint style="warning" %}
**Port already in use** — find the process with `lsof -i :8000`, stop it, or change the port mapping in `docker-compose.yml`. `make stop-all-ports` force-frees 3000, 8000, 8001, 8002, 7860, and 27017.
{% endhint %}

{% hint style="warning" %}
**Permission denied on docker.sock** — run `sudo usermod -aG docker $USER && newgrp docker`, then log out and back in.
{% endhint %}

## Next steps

- [first-call.md](first-call.md)
- [default-credentials.md](default-credentials.md)
- [../guides/deployment/docker-compose.md](../guides/deployment/docker-compose.md)
- [../troubleshooting/deployment.md](../troubleshooting/deployment.md)
