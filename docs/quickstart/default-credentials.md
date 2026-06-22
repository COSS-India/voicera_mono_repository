---
description: Default development credentials shipped with VoicEra and how to log in for the first time.
---

# Default credentials

Reference for the development defaults baked into a fresh VoicEra install. Use these only for first-run access; rotate them before exposing the stack to anyone else.

{% hint style="danger" %}
Every credential on this page is a known public default. Change all of them before production go-live. See [security hardening](../guides/deployment/security-hardening.md).
{% endhint %}

## Default accounts and secrets

| Component | Username | Password | Where it lives |
|-----------|----------|----------|----------------|
| MongoDB | `admin` | `admin123` | `voicera_backend/.env` (`MONGODB_USER`, `MONGODB_PASSWORD`) |
| MinIO | `minioadmin` | `minioadmin` | `voicera_backend/.env` and `voice_2_voice_server/.env` (`MINIO_ACCESS_KEY`, `MINIO_SECRET_KEY`) |
| Backend `SECRET_KEY` | — | example value | `voicera_backend/.env` |
| Backend `INTERNAL_API_KEY` | — | example value | `voicera_backend/.env` and `voice_2_voice_server/.env` (must match) |
| CORS | — | `allow_origins=["*"]` | Backend default |

## First-time dashboard login

1. Open the frontend in a browser at `http://localhost:3000` (or your configured `FRONTEND_URL`).
2. Sign up to create the first user.
3. Confirm the verification email — in development, Mailtrap captures it; in production, a real SMTP/API provider sends it.
4. Log in and proceed to **Dashboard → Integrations** to enter Vobiz and AI provider keys.

{% hint style="info" %}
There is no pre-seeded admin user. The first account you create becomes the organisation owner.
{% endhint %}

## MinIO console login

- URL: `http://localhost:9001`
- Username: `minioadmin`
- Password: `minioadmin`

Used for inspecting stored recordings and uploads. Do not expose this port to the public internet.

## Generating replacements

```bash
python -c "import secrets; print(secrets.token_urlsafe(32))"
```

Use the output above for `SECRET_KEY` and `INTERNAL_API_KEY`. Replace MongoDB and MinIO defaults with strong unique passwords and update every `.env` that references them.

{% hint style="warning" %}
If `INTERNAL_API_KEY` is rotated, update the backend and voice server `.env` together — they must match for service-to-service calls to authenticate.
{% endhint %}

## Next steps

- [../guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md)
- [../reference/environment-variables.md](../reference/environment-variables.md)
- [first-call.md](first-call.md)
