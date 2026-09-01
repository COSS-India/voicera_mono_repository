---
description: Store and retrieve encrypted provider API keys.
---

# Provider credentials

`apps/api/app/routers/auth.py`, prefix `/api/v1/auth`. Credentials are **provider-level** — one key set per provider per organisation, shared across whichever of STT, TTS, and LLM that vendor serves. The whole `auth` object is Fernet-encrypted at rest with `PROVIDER_AUTH_ENCRYPTION_KEY`. See [Provider credentials (ProviderAuth)](../guides/concepts/provider-auth.md).

## `GET /auth/catalog`

Bearer. Auth field schemas for every provider that takes credentials.

## `GET /auth/catalog/{provider}`

Bearer. One provider's schema, with fields merged across the kinds it serves. Unknown provider returns `404`.

## `GET /auth/configured`

Bearer. A bare JSON array of provider ids the caller's organisation has stored credentials for:

```json
["deepgram", "cartesia", "openai", "vobiz"]
```

## `POST /auth`

Bearer, `admin` or `super_admin`. `201`. Upsert — creating and updating are the same call.

```json
{
  "provider": "deepgram",
  "auth": { "api_key": "YOUR_DEEPGRAM_KEY" }
}
```

Returns `ProviderAuthResponse`: `org_id`, `provider`, `auth`, `created_at`, `updated_at`. The field names inside `auth` come from `GET /auth/catalog/{provider}` — send only secrets here, never model settings.

## `GET /auth/{provider}`

Bearer. Returns `ProviderAuthResponse` with the decrypted `auth`. Secrets are **masked** for callers whose role is neither `admin` nor `super_admin`. No stored credentials returns `404 No auth stored for provider: {provider}`.

## `DELETE /auth/{provider}`

Bearer, `admin` or `super_admin`. Returns `SuccessResponse`. Nothing stored returns `404`.

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
