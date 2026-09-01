---
description: Signup, login, membership, roles, and bot tokens.
---

# Users and organisations

# Users

`apps/api/app/routers/users.py`, prefix `/api/v1/users`.

## `POST /users/signup`

Public. `201`. Creates a user, a new organisation, and a `super_admin` membership in one call. This is the only way an organisation comes into existence.

```json
{
  "email": "you@example.com",
  "password": "a-strong-password",
  "organisation_name": "Acme Support"
}
```

Returns `UserLoginResponse`: `status`, `message`, `access_token`, `token_type`, `org_id`, `role`, and `organisations` (a list of `{org_id, name, role}`). A duplicate email returns `400`.

## `POST /users/login`

Public. Exchanges credentials for a JWT scoped to the user's default organisation.

```json
{ "email": "you@example.com", "password": "a-strong-password" }
```

Returns `UserLoginResponse`. Bad credentials return `401`.

## `POST /users/bot/token`

`X-API-Key`. The one route that trades the internal key for a JWT. The runtime and the campaign workers use it to call everything else.

```json
{ "org_id": "org_abc123" }
```

Returns `BotTokenResponse`: `access_token`, `token_type`, `org_id`, `role`. The role is always `admin` and the token's subject is the synthetic `bot@voicera.internal`. An unknown `org_id` returns `404`.

## `POST /users/switch-organisation`

Bearer. Reissues the token against another organisation you belong to, and persists that choice as your default for the next login.

```json
{ "org_id": "org_other" }
```

Returns `UserLoginResponse` with the new token. An organisation you are not a member of returns `400`.

## `GET /users/organisations`

Bearer. Organisations the caller is a member of, as an object with a `status` and the list.

## `GET /users/me`

Bearer. Returns `UserResponse` for the **active** organisation: `email`, `org_id`, `role`, `organisation_name`, `organisations` (every membership as `{org_id, name, role}`), and `created_at`. The joined fields are computed at read time, not stored on the user document.

## `GET /users/check/{email}`

Public. Invite helper. Optional `org_id` query parameter. Returns `CheckEmailResponse`: `exists`, `already_in_org`, `can_join`.

## `GET /users/{email}`

Bearer, self only. Any email other than the token's own returns `403 Not authorized to access this user's data`. Returns `UserResponse`.

{% hint style="warning" %}
`/users/check/{email}` is public and unauthenticated. It confirms whether an email has an account. That is deliberate — the invite form needs it — but it is an enumeration surface. Rate-limit it at your proxy.
{% endhint %}

## `POST /users/forgot-password`

Public. Body `{ "email": "you@example.com" }`. Sends a reset email. Returns `{status, message}`; a failure is `400`.

## `POST /users/reset-password`

Public. Body `{ "token": "…", "new_password": "…" }`. Returns `{status, message}`; an invalid or expired token is `400`.

# Members

`apps/api/app/routers/members.py`, prefix `/api/v1/members`. Members are the join between a user and an organisation, carrying a role. See [Multi-tenancy and roles](../guides/concepts/multi-tenancy.md).

## `POST /members/invite`

Bearer, `admin` or `super_admin`. `201`. Adds a user to the caller's **active** organisation, creating the account when it does not exist.

```json
{ "email": "colleague@example.com", "password": "their-initial-password" }
```

Returns `{status, message}`. A failure — already a member, bad role — returns `400`.

## `GET /members/{org_id}`

Bearer, must hold a membership in that organisation. Returns `{status, members: [{email, role, created_at}]}`. A non-member gets `403`.

## `POST /members/assign-admin`

Bearer, `super_admin` only. Body `{ "email": "…" }`. Promotes an existing `member` to `admin` in the active organisation.

## `POST /members/remove`

Bearer, `super_admin` only. Body `{ "email": "…" }`. Removes the membership. The `Users` document survives — a user with memberships elsewhere keeps them.

# Organisations

`apps/api/app/routers/organisations.py`, prefix `/api/v1/organisations`. One route.

## `DELETE /organisations/{org_id}`

Bearer, `super_admin`, and `org_id` must equal the token's active organisation. Deletes the organisation and its memberships.

{% hint style="danger" %}
There is no create-organisation endpoint — only `POST /users/signup` makes one. Deleting an organisation is not reversible and does not clean up the agents, call logs, campaigns, or MinIO objects that referenced it.
{% endhint %}

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
