# Brief: Security & production hardening (A6)

**Review gap:** Default credentials are documented but there is no guide for production hardening (MinIO, MongoDB, secret rotation, TLS/HTTPS).

---

## Default credentials in development (MUST change in production)

| Component | Default | Risk if unchanged |
|-----------|---------|-------------------|
| MongoDB | `admin` / `admin123` | Full database access |
| MinIO | `minioadmin` / `minioadmin` | All recordings and uploads |
| Backend | `SECRET_KEY`, `INTERNAL_API_KEY` from examples | Session forgery, service impersonation |
| CORS | `allow_origins=["*"]` in `voicera_backend/app/main.py` | Any origin can call API from browsers |

---

## Production checklist (writer expands each item)

### 1. Secrets

- Generate strong random `SECRET_KEY` and `INTERNAL_API_KEY` (e.g. `secrets.token_urlsafe(32)`).
- Never commit `.env` files to git.
- Rotate keys if leaked; plan rotation procedure for `INTERNAL_API_KEY` (voice server + backend must update together).

### 2. MongoDB

- Unique strong password; restrict network (VPN/private subnet only).
- Enable authentication; regular backups; restore tested.

### 3. MinIO

- Replace default access/secret keys.
- Set `MINIO_SECURE=true` when behind TLS terminator.
- Do not expose MinIO console (9001) to public internet without auth/VPN.

### 4. TLS / HTTPS

- Public **frontend** URL: HTTPS only.
- Public **voice server**: HTTPS for webhooks, **WSS** for WebSocket (`JOHNAIC_*` URLs — see A8).
- Terminate TLS at reverse proxy (nginx, traefik, cloud LB) with valid certificates.

### 5. Integrations / API keys

- Telephony (Vobiz, Plivo) and AI keys stored in dashboard Integrations — treat as secrets.
- Restrict who has org admin access on dashboard.

### 6. Network

- Expose only required ports (typically 443 to frontend and voice proxy).
- MongoDB (27017), MinIO (9000/9001) not reachable from internet.

### 7. Email

- Dev uses Mailtrap; production needs real SMTP/API for password reset (`MAILTRAP_*` → production provider).

### 8. API documentation exposure

- Consider disabling public access to `/docs` and `/redoc` on production backend/voice server, or protect with authentication.

### 9. Docker / host

- Run containers as non-root where possible; keep images updated; limit SSH to administrators.

---

## Engineering follow-ups (mention in internal appendix)

- Lock down CORS to known frontend origins.
- Rename `JOHNAIC_*` env vars to neutral names (`VOICERA_PUBLIC_URL`).
- Remove hardcoded fallback host `vobiz.johnaic.com` in code paths where present.
