---
description: Production security hardening for VoicEra covering credentials, MongoDB, MinIO, TLS, JWT secrets, Docker, firewall, secret storage, and recording retention.
---

# Security and production hardening

VoicEra ships with development defaults so a new operator can run the stack within minutes. All default values must be replaced before a deployment handles real traffic. This page is the hardening checklist to complete before any VoicEra deployment becomes reachable on the internet or processes real call data.

The intended audience is a hosting partner or platform engineer who has already completed the [Deployment walkthrough](deployment-walkthrough.md) and now needs to close gaps before go-live.

{% hint style="danger" %}
Do not connect a VoicEra deployment to live phone numbers, real users, or production telephony provider credentials until every item in the [Hardening checklist](#hardening-checklist) at the end of this page is complete.
{% endhint %}

## 1. Default credentials to change

Every value in this table is set somewhere in the bundled `docker-compose.yml` or in the example env files. Each one must change before production traffic.

| Component | Default | Where it lives | Risk if unchanged |
|-----------|---------|----------------|-------------------|
| MongoDB root | `admin` / `admin123` | `docker-compose.yml` env on `mongodb` | Full database read and write |
| MinIO root | `minioadmin` / `minioadmin` | `docker-compose.yml` env on `minio` and `voice_server` | All recordings, transcripts, uploads exposed |
| Backend `SECRET_KEY` | Example value | `voicera_backend/.env` | Session cookie forgery, JWT spoofing |
| `INTERNAL_API_KEY` | Example value | Backend and voice server `.env` | Service-to-service impersonation |
| CORS `allow_origins` | `["*"]` | `voicera_backend/app/main.py` | Any browser origin can call the API |
| Mailtrap SMTP | Sandbox creds | Backend `.env` | Password reset email is intercepted in a dev sandbox |

Steps to remediate:

1. Generate a strong random value for each secret (see [JWT and application secrets](#4-jwt-and-application-secrets)).
2. Update the env files on every host that runs the affected container.
3. Restart only the services whose env changed (`docker compose up -d --no-deps <service>`).
4. Tighten CORS in `voicera_backend/app/main.py` to the exact dashboard origin before deploying.

## 2. MongoDB hardening

The bundled MongoDB container starts with `mongod --bind_ip_all` and root credentials in plain environment variables. Harden it in three layers.

**Authentication**

- Rotate `MONGO_INITDB_ROOT_USERNAME` and `MONGO_INITDB_ROOT_PASSWORD` to a unique strong password (24+ chars).
- Add `--auth` to the `command:` so anonymous connections are rejected.
- Create a least-privilege application user instead of using root from the backend:

  ```javascript
  use voicera
  db.createUser({
    user: "voicera_app",
    pwd: "REPLACE_WITH_STRONG_PASSWORD",
    roles: [{ role: "readWrite", db: "voicera" }]
  })
  ```

  Then point the backend at `voicera_app`, not the root account.

**Network binding**

- Remove the `"27017:27017"` host port mapping in production so MongoDB is reachable only on the internal Docker network.
- If MongoDB must be reachable across hosts, restrict it to a private subnet (VPN or VPC) and never to the public internet.

**Durability and recovery**

- Enable a replica set (`--replSet rs0`) for write durability.
- Run daily `mongodump` backups, copy them off-host, and restore-test quarterly. See [Production deployment](production.md) for a backup script.

## 3. MinIO hardening

MinIO holds every call recording and transcript. Treat it like an S3 bucket containing PII.

**Root credentials**

- Replace `MINIO_ROOT_USER` and `MINIO_ROOT_PASSWORD` with strong values.
- Update `MINIO_ACCESS_KEY` and `MINIO_SECRET_KEY` on the voice server to match.
- Restart MinIO and the voice server together so the credentials stay in sync.

**Service accounts and IAM**

- Avoid using root credentials from application services. Create per-service IAM keys via the MinIO client:

  ```bash
  mc alias set local http://minio:9000 ROOT_USER ROOT_PASS
  mc admin user add local voicera_voice_server STRONG_SECRET
  mc admin policy attach local readwrite --user voicera_voice_server
  ```

- Rotate service-account secrets on a scheduled cadence (90 days is a reasonable starting point) and immediately on suspected compromise.

**Bucket policy and TLS**

- Set bucket policies to deny anonymous access. Buckets that hold recordings must be private.
- When MinIO sits behind a TLS-terminating reverse proxy, set `MINIO_SECURE=true` on every client.
- Never expose the MinIO console (`9001`) on the public internet. If operators need it, put it behind a VPN or HTTP auth.

## 4. JWT and application secrets

VoicEra signs session tokens and internal API calls with secrets read from env files. Weak or shared secrets break the whole trust model.

Generate strong, unrelated values:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"  # SECRET_KEY
python -c "import secrets; print(secrets.token_urlsafe(48))"  # INTERNAL_API_KEY
openssl rand -base64 48                                        # alternative generator
```

Rules of thumb:

- `SECRET_KEY` and `INTERNAL_API_KEY` must be different.
- `INTERNAL_API_KEY` must match on the backend and the voice server. If you rotate one, rotate both in the same change window.
- Never reuse a secret across staging and production.
- Treat any secret in chat, ticket comments, screenshots, or browser history as leaked.
- Rotate immediately on staff offboarding or any suspected leak.

## 5. TLS and HTTPS via reverse proxy

Terminate TLS at nginx, Traefik, or a cloud load balancer. The application containers should never speak plain HTTP to the internet.

Minimum production nginx for the voice server:

```nginx
server {
    listen 443 ssl http2;
    server_name voice.example.gov.in;

    ssl_certificate     /etc/letsencrypt/live/voice.example.gov.in/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/voice.example.gov.in/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;
    ssl_ciphers         HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://voice_server:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For  $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

Apply similar blocks for the dashboard and the backend API. Force HTTP → HTTPS redirects on port 80 and add HSTS once you are confident the certificate chain is stable. The same hostname must appear in `JOHNAIC_SERVER_URL` and `JOHNAIC_WEBSOCKET_URL` ([Public voice server URLs](public-voice-urls.md)).

{% hint style="warning" %}
Browsers refuse `ws://` connections from `https://` pages. If the dashboard is on HTTPS, every voice WebSocket must use `wss://` or **Test on Browser** silently fails.
{% endhint %}

## 6. Docker daemon and host security

- Keep the host kernel and Docker engine patched. Schedule monthly `apt update && apt upgrade`.
- Run containers as non-root where the Dockerfile allows; add `user:` to compose entries that ship as root.
- Drop unneeded Linux capabilities (`cap_drop: [ALL]`) and add back only what each service needs.
- Set `read_only: true` on containers that do not write to disk, with explicit `tmpfs` for ephemeral paths.
- Do not enable the Docker remote API on TCP. If you must, require mTLS.
- Restrict `docker.sock` access — anyone who can read it is effectively root on the host.
- Configure Docker log rotation so disks do not fill silently:

  ```yaml
  logging:
    driver: json-file
    options:
      max-size: "100m"
      max-file: "10"
  ```

## 7. Firewall rules

Block everything by default and allow only what the deployment needs. UFW example for a single-host install with the reverse proxy on the same machine:

```bash
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Administrative SSH (consider restricting to a jump host)
sudo ufw allow from 203.0.113.0/24 to any port 22 proto tcp

# Public web traffic
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Internal-only services from a private subnet
sudo ufw allow from 10.0.0.0/8 to any port 27017 proto tcp   # MongoDB
sudo ufw allow from 10.0.0.0/8 to any port 9000  proto tcp   # MinIO API
sudo ufw allow from 10.0.0.0/8 to any port 9001  proto tcp   # MinIO console

sudo ufw enable
sudo ufw status verbose
```

Do not open `3000`, `7860`, `8000`, `8001`, `8002`, `9000`, `9001`, or `27017` to the public internet. The reverse proxy on `443` is the only public entry point.

## 8. Secret storage practices

Env files are convenient but they are not a secret manager.

- Never commit `.env`, `.env.local`, `.env.prod`, `*.pem`, or `*.key` to git. Confirm a `.gitignore` rule exists for each.
- Restrict env-file permissions on the host: `chmod 600 voicera_backend/.env`.
- Use a dedicated secret manager for anything beyond a single host: HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager, Azure Key Vault, or sealed Kubernetes Secrets.
- Inject secrets at container start, not at image build. Secrets baked into an image are visible to anyone who pulls it.
- Audit who can read deployment secrets. Reduce the list to a named on-call group.
- Telephony auth IDs and tokens entered through **Dashboard → Integrations** are stored in MongoDB. They inherit MongoDB's protections, so hardening MongoDB is part of protecting them.

## 9. Logs, transcripts, and recording retention

Call recordings and live transcripts contain PII and sometimes sensitive content (health, finance, ID numbers).

**Redaction**

- Strip API keys, JWTs, and tokens from application logs. Confirm the backend and voice server do not log `Authorization` headers or request bodies that include `INTERNAL_API_KEY`.
- For transcript storage, mask obvious PII patterns (phone numbers, government IDs, card numbers) before persistence when the use case permits.
- Disable verbose debug logging in production.

**Retention**

- Define and document a retention policy: how long recordings, transcripts, and call logs are kept, and on what schedule they are deleted.
- Implement deletion as a scheduled job that removes both the MinIO object and any database reference.
- Publish the retention policy to end users where regulation requires it.

**Access control**

- Recordings are private bucket objects. Serve them only via short-lived presigned URLs issued by the backend after an authorization check, never as public links.
- Audit which org admins can download recordings and review the list periodically.

**API docs exposure**

- Decide whether `/docs` and `/redoc` on the backend and voice server should be public. For most deployments they should sit behind authentication or be disabled in production builds.

## Hardening checklist

Run through this list before flipping production traffic on. Tick every box.

| Area | Item |
|------|------|
| Credentials | MongoDB root password rotated; no defaults remain |
| Credentials | MinIO root credentials rotated; voice server updated to match |
| Credentials | `SECRET_KEY` and `INTERNAL_API_KEY` regenerated with `secrets.token_urlsafe(48)` |
| Credentials | CORS `allow_origins` restricted to the dashboard origin |
| MongoDB | `--auth` enabled; least-privilege app user created |
| MongoDB | Host port 27017 removed from public mapping |
| MongoDB | Daily backup job scheduled and restore-tested |
| MinIO | Per-service IAM key used instead of root |
| MinIO | `MINIO_SECURE=true` set when behind TLS |
| MinIO | Console (9001) not exposed publicly |
| TLS | Valid certificates on dashboard, API, voice domains |
| TLS | HTTP redirects to HTTPS; HSTS enabled |
| TLS | Voice server uses `wss://` for WebSockets |
| Docker | Containers patched; non-root where possible |
| Docker | Log rotation configured (`max-size`, `max-file`) |
| Network | UFW or equivalent deny-by-default policy active |
| Network | Only ports 80 and 443 reachable from the public internet |
| Secrets | `.env` files chmod 600 and excluded from git |
| Secrets | Production secrets stored in a dedicated secret manager |
| Logs | API keys and tokens absent from application logs |
| Data | Retention policy documented and enforced for recordings |
| Data | Recordings served via short-lived presigned URLs |
| Email | Mailtrap replaced with production SMTP/API |
| API docs | `/docs` and `/redoc` access decided and configured |

## Next steps

- [Production deployment](production.md)
- [Deployment walkthrough](deployment-walkthrough.md)
- [Operations](../operator/operations.md)
- [Troubleshooting: deployment](../../troubleshooting/deployment.md)
