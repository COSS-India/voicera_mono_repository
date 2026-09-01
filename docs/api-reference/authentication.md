---
description: Tokens, headers, roles, and organisation scoping.
---

# Authentication

Voicera uses two credentials: a **JWT** for human and application callers, and a **shared API key** for service-to-service calls. Every route uses one or the other — none are open except `/`, `/health`, and the signup and login routes.

| Header | Where it comes from | Dependency |
| --- | --- | --- |
| `Authorization: Bearer <jwt>` | `POST /api/v1/users/login`, `/signup`, `/switch-organisation`, or `/users/bot/token` | `Depends(get_current_user)` |
| `X-API-Key: <INTERNAL_API_KEY>` | The root `.env`. Service-to-service only. | `Depends(verify_api_key)` |

The JWT carries `sub` (email), `email`, `org_id`, and `role`. `org_id` is the **active** organisation — a user in several organisations holds one token per organisation and swaps with `POST /api/v1/users/switch-organisation`. Tokens are signed with `SECRET_KEY` using `JWT_ALGORITHM` and expire after `ACCESS_TOKEN_EXPIRE_MINUTES`.

Role checks are not dependencies. They are explicit `HTTPException` raises inside the handler after the token has been verified, so an authenticated caller with the wrong role gets `403`, never `401`.

`X-API-Key` returns `401 Missing API key` when the header is absent, `401 Invalid API key` when it does not match, and `500 Internal API key not configured` when `INTERNAL_API_KEY` is unset on the server.

## Getting a token

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/users/login \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"…"}' | jq -r .access_token)

curl -s http://localhost:8000/api/v1/agents -H "Authorization: Bearer $TOKEN"
```

The first call to `POST /api/v1/users/signup` creates the user, an organisation, and a `super_admin` membership in one step. There is no seeded default account.

## Roles

| Role | Can do |
| --- | --- |
| `super_admin` | Everything, including deleting the organisation and assigning admins. |
| `admin` | Manage agents, provider credentials, numbers, campaigns, and invite members. |
| `member` | Read and operate. Sees provider secrets masked. |

## Organisation scoping

Every organisation-scoped route reads `org_id` from the token, not from the request. Routes that also take `org_id` in the path — `GET /api/v1/members/{org_id}`, `GET /api/v1/calls/org/{org_id}` — check membership separately and return `403` when you are not a member. A token with no `org_id` claim gets `400 No active organisation in token` from the routes that require one.

## Related

* [Users and organisations](users-and-orgs.md) — the routes that issue and swap tokens
* [Multi-tenancy and roles](../guides/concepts/multi-tenancy.md) — the model behind this
* [Errors](errors.md)
