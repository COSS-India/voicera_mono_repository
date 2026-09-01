---
description: Place calls, register them, and fetch recordings and transcripts.
---

# Calls

`apps/api/app/routers/calls.py`, prefix `/api/v1/calls`. See [Calls and call artifacts](../guides/concepts/calls.md).

## `POST /calls/outbound`

Bearer. `201`. Places a call and registers a `CallLog` with status `initiated`.

```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "to_number": "+14155551234",
  "from_number": "+14155559999",
  "custom_variables": {
    "customer_name": "Jane Doe",
    "account_id": "ACC-123"
  }
}
```

`from_number` is an optional caller-ID override. `custom_variables` are merged over the agent's defaults at call time and substituted into the prompts — see [Agent configuration](../developer/reference/agent-configuration.md).

Returns `OutboundCallResponse`: `call_id`, `status`, `provider_call_sid`, `from_number`, `to_number`, `agent_id`, `custom_variables`.

## `POST /calls/inbound`

Bearer. `201`. The runtime calls this from the answer webhook, using a bot JWT.

```json
{
  "agent_id": "…",
  "provider_call_sid": "…",
  "from_number": "+14155551234",
  "to_number": "+14155559999"
}
```

Returns `InboundCallRegisterResponse`: `call_id`, `status`, `provider_call_sid`, `call_type`, `from_number`, `to_number`, `agent_id`.

## `POST /calls/web`

Bearer. `201`. Registers a browser websocket session so it gets a `CallLog` and artifacts. The agent must be `agent_category: websocket`.

```json
{
  "agent_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "custom_variables": {
    "customer_name": "Jane Doe"
  }
}
```

Returns `WebCallRegisterResponse`: `call_id`, `status`, `call_type` (always `web`), `agent_id`, `custom_variables`.

Calling this is optional. If a browser connects to `WS /agent/{org_id}/{agent_id}` with no `call_id`, the runtime registers one itself. Pre-register only when you need the `call_id` before the session starts — then pass it as `?call_id=` on the WebSocket URL. The runtime discards a `call_id` whose `call_type` is not `web` or whose `agent_id` does not match, and falls back to auto-creating one.

## `PATCH /calls/{call_id}` and `PATCH /calls/by-provider-sid/{provider_call_sid}`

Bearer. Both take the same partial body and return the same `CallLogResponse`. The by-SID variant exists for provider hangup callbacks, which know the SID and not the `call_id`.

```json
{
  "transcript_url": "minio://…",
  "recording_url": "minio://…",
  "end_time_utc": "2026-01-01T12:34:56+00:00",
  "status": "completed",
  "call_response": "answered"
}
```

Every field is optional, but an empty body returns `400 No fields to update` — only fields you actually set are applied.

`duration` is derived, never sent. When a patch carries `end_time_utc`, the service computes the duration from `start_time_utc` and clamps it at zero; a patch without `end_time_utc` has any `duration` stripped. The `minio://` URIs you write are rewritten in the response to the two proxy routes below.

## `GET /calls/{call_id}/recording` and `GET /calls/{call_id}/transcript`

Bearer. Stream the artifact out of MinIO through the API, so clients never need MinIO credentials. `404` when the call or the object is missing, `400` when the stored URL is not a usable object key.

## `GET /calls/org/{org_id}`

Bearer, must be a member of that organisation. Query parameters `limit` (default `50`, `1`–`500`) and `offset` (default `0`). Returns `CallLogListResponse`: `{calls: [...], limit, offset, total}`.

## `GET /calls/{call_id}`

Bearer. One `CallLogResponse`, active organisation only. Bot JWTs work here.

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
