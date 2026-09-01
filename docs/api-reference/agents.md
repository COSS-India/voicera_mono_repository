---
description: Create, read, update, and delete voice agents.
---

# Agents

`apps/api/app/routers/agents.py`, prefix `/api/v1/agents`. The full shape of the nested `config` object is documented in [Agent configuration](../developer/reference/agent-configuration.md); this section covers the routes around it.

## `POST /agents`

Bearer, any org member. `201`. Creates an agent in the token's active organisation and sets `created_by` from the token email.

```json
{
  "name": "Support Agent",
  "agent_category": "telephony",
  "telephony_provider": "vobiz",
  "config": {
    "schema_version": 1,
    "prompts": {
      "system_prompt": "You are a helpful phone support agent.",
      "greeting_message": "Hello! How can I help you today?"
    },
    "behaviour": { "interruption_min_words": 2 },
    "language": { "primary": "en", "secondary": [] },
    "models": {
      "stt_config": { "provider": "deepgram", "model": "nova-3-general", "language": "en" },
      "tts_config": { "provider": "cartesia", "model": "sonic-3.5", "language": "en" },
      "llm_config": { "provider": "openai", "model": "gpt-4.1" }
    },
    "knowledge_base": { "enabled": false },
    "custom_variables": { "customer_name": "" }
  }
}
```

Returns `AgentResponse`: `agent_id`, `org_id`, `name`, `status`, `agent_category`, `created_by`, `linked_phone_number`, `telephony`, `config`, `created_at`, `updated_at`.

`agent_category` decides what else happens:

| Category | `telephony_provider` | Side effects on create |
| --- | --- | --- |
| `telephony` | Required, must be a registered id with stored credentials | The API provisions a provider application and stores `telephony.{provider, application_id, answer_url, hangup_url}` on the agent. |
| `websocket` | Must be omitted | None. `telephony` stays `null`. |

Set `VOICE_SERVER_BASE_URL` before creating a telephony agent — the answer URL is built from it as `{VOICE_SERVER_BASE_URL}/answer?agent_id={agent_id}&org_id={org_id}`.

Failure codes: `422` for a config validation error (with the message as a plain string in `detail`), `409` for a duplicate name in the organisation, and whatever the telephony provider returned when provisioning fails.

## `GET /agents`

Bearer. A JSON array of `AgentResponse` for the active organisation.

## `GET /agents/by-phone/{phone_number}`

`X-API-Key`. Resolves an agent from an inbound number. This is the runtime's route, not yours. Returns `AgentResponse`.

## `GET /agents/{agent_id}`

Bearer. One `AgentResponse`, same organisation only. `404` otherwise.

## `PATCH /agents/{agent_id}`

Bearer, any org member. Every field is optional: `name`, `agent_category`, `telephony_provider`, `config`. `config` is **not** merged — send the whole object.

Changing `agent_category` away from `telephony`, or changing `telephony_provider`, detaches any attached phone number and removes the old provider application. Returns the updated `AgentResponse`.

## `DELETE /agents/{agent_id}`

Bearer, `admin` or `super_admin`. Unlinks any attached number, removes the provider application, and deletes the agent. Returns `SuccessResponse`.

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
