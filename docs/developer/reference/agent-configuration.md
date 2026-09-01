---
description: Every field of an agent's configuration blob.
---

# Agent configuration

An agent's behaviour, language, models, and knowledge attachment all live in one nested `config` object on the agent document. This page documents every field of it: type, default, bounds, and what the runtime actually does with it.

Field names, defaults, and bounds come from `AgentConfigPayload` and its nested models in `apps/api/app/models/schemas.py`. What the runtime reads comes from `apps/runtime/services/pipecat/config.py` and its neighbours.

{% hint style="info" %}
`config` is validated on both `POST /api/v1/agents` and `PATCH /api/v1/agents/{agent_id}` by `validate_agent_config()` in `apps/api/app/services/agent_config_validation.py`. A failure returns `422`. The whole page is also live on a running API at `/docs`.
{% endhint %}

## The config object

`AgentConfigPayload` has seven fields.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `schema_version` | int | `1` | Reserved for future migrations. Nothing branches on it today. |
| `prompts` | `AgentPrompts` | required | System prompt and greeting. |
| `behaviour` | `AgentBehaviour` | all defaults | Turn-taking, silence, hold, and call-ending knobs. |
| `language` | `AgentLanguage` | required | Primary and secondary language ids. |
| `models` | `AgentModels` | required | STT, TTS, and LLM provider configs. |
| `knowledge_base` | `AgentKnowledgeBase` | disabled | Optional RAG attachment. |
| `custom_variables` | object | `{}` | Named defaults, substituted into prompts at call time. |

`behaviour` and `knowledge_base` may be omitted entirely and take their model defaults. `prompts`, `language`, and `models` are required.

## `prompts`

`AgentPrompts`. Two fields.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `system_prompt` | string | `""` | Becomes the `system` message seeded into the LLM context. Omit it and the context starts empty. |
| `greeting_message` | string | required | Spoken as soon as the transport connects, before any caller audio. |

`validate_agent_config()` strips whitespace from `greeting_message` and rejects an empty result with `prompts.greeting_message is required`. Both strings pass through variable substitution — see [Custom variables](#custom-variables-and-prompt-substitution).

## `behaviour`

`AgentBehaviour`. Every field is optional and every field has a default, so `"behaviour": {}` is valid. The `description=` column is the text from the model itself.

| Field | Type | Default | Bounds | Description |
| --- | --- | --- | --- | --- |
| `interruption_min_words` | int | `0` | `>= 0` | Minimum words before the caller can interrupt the agent. |
| `user_silence_hangup_seconds` | float or null | `null` | `>= 0` | Hang up after this many seconds of user silence (null = disabled). |
| `call_timeout_seconds` | float or null | `null` | `>= 0` | Maximum call duration in seconds (null = no hard limit). |
| `ignore_user_speech_before_greeting` | bool | `false` | — | Ignore caller speech until the greeting has finished playing. |
| `hold_messages` | list of string | `[]` | — | Messages played while the agent is on hold / thinking. |
| `hold_message_timeout_seconds` | float or null | `null` | `>= 0` | Seconds to wait after LLM inference starts before playing a single hold message (null = disabled). |
| `user_online_detection_enabled` | bool | `false` | — | Prompt the caller after silence following bot speech. |
| `user_online_detection_message` | string | `""` | — | Prompt played when checking if the user is still online. |
| `user_online_detection_seconds` | float or null | `null` | `>= 0` | Seconds of silence after bot speech before the online-detection prompt. |
| `user_online_detection_repeats` | int or null | `null` | `>= 1` | How many times to speak the online-detection prompt in one silence cycle. |
| `user_online_detection_closing_message` | string | `""` | — | Spoken after the last online-detection prompt, before hangup. |
| `automatic_call_ending` | `AutomaticCallEnding` | `{enabled: false, graceful_llm_call_ending: false}` | — | Graceful call ending via LLM tool. |

### How the runtime reads them

The Pydantic defaults are not always the effective defaults. `pipeline_config_from_behaviour()` and `online_detection_from_behaviour()` coerce nulls when they build the pipeline:

| Field | Effective value when null or absent |
| --- | --- |
| `interruption_min_words` | `0` — no minimum-words turn strategy is installed at all when this is zero. |
| `user_silence_hangup_seconds` | `0` |
| `user_online_detection_seconds` | `10` |
| `user_online_detection_repeats` | `1` |

The idle timeout the pipeline uses is `user_online_detection_seconds` when online detection is enabled, and `user_silence_hangup_seconds` otherwise (`PipelineConfig.user_idle_timeout`). The two settings share one timer — you cannot have both.

`ignore_user_speech_before_greeting` installs Pipecat's `MuteUntilFirstBotCompleteUserMuteStrategy`. `interruption_min_words` above zero installs a `MinWordsUserTurnStartStrategy`.

Hold messages need **both** halves: `hold_from_behaviour()` returns nothing unless `hold_message_timeout_seconds` is non-null and greater than zero **and** `hold_messages` contains at least one non-blank string. One message is chosen at random per inference, and only one is played per turn.

Online detection speaks `user_online_detection_message` up to `user_online_detection_repeats` times; on the next idle it speaks `user_online_detection_closing_message` and ends the call. With detection disabled, a single idle timeout speaks the closing message and ends the call directly.

`automatic_call_ending` registers an `end_conversation` function tool on the LLM context, but only when **both** `enabled` and `graceful_llm_call_ending` are true — `_call_ending_enabled()` in `apps/runtime/services/pipecat/call_ending.py` requires the pair. The tool ends the call when the model calls it.

{% hint style="warning" %}
`call_timeout_seconds` is accepted, validated, and stored, but nothing in `apps/runtime` reads it. There is no hard call-duration limit in the pipeline. Enforce a ceiling at your telephony provider if you need one.
{% endhint %}

### Example from the model

`AgentBehaviour` carries a `json_schema_extra` example, reproduced here verbatim from `apps/api/app/models/schemas.py`:

```json
{
  "interruption_min_words": 2,
  "user_silence_hangup_seconds": 30,
  "call_timeout_seconds": 600,
  "ignore_user_speech_before_greeting": true,
  "hold_messages": ["One moment please."],
  "hold_message_timeout_seconds": 0.6,
  "user_online_detection_enabled": false,
  "user_online_detection_message": "",
  "user_online_detection_seconds": 10,
  "user_online_detection_repeats": 1,
  "user_online_detection_closing_message": "",
  "automatic_call_ending": {
    "enabled": true,
    "graceful_llm_call_ending": true
  }
}
```

## `language`

`AgentLanguage`. Two fields.

| Field | Type | Default | Notes |
| --- | --- | --- | --- |
| `primary` | string | required | Canonical language id. Validation strips it and rejects an empty result. |
| `secondary` | list of string | `[]` | Additional declared languages. |

Valid ids come from `GET /api/v1/languages`, which returns the canonical id → label map the agent builder uses. The per-provider language filter is `GET /api/v1/configuration/stt?languages=` and its TTS equivalent.

{% hint style="warning" %}
`secondary` is stored on the agent and returned by the API, but **no code reads it**. There is no mid-call language switching in Voicera: nothing in `apps/runtime` inspects `language.secondary`, and no processor swaps the STT or TTS language once a session is running. The language a call runs in is whatever the `stt_config` and `tts_config` were built with. Treat `secondary` as documentation of intent, not as behaviour.
{% endhint %}

## `models`

`AgentModels`. Three required objects, one per pipeline stage.

| Field | Type | Notes |
| --- | --- | --- |
| `stt_config` | object | Speech-to-text provider and settings. |
| `tts_config` | object | Text-to-speech provider and settings. |
| `llm_config` | object | Large-language-model provider and settings. |

Two rules apply to all three, enforced by `validate_persisted_model_config()`:

1. **`provider` is required and must be registered.** It is matched against the provider registry for that kind; an unknown id fails validation with the registry's own error. Enumerate the valid ids with `GET /api/v1/configuration/stt`, `/tts`, and `/llm`, and fetch one provider's setting schema from `GET /api/v1/configuration/{kind}/setting/{provider}`.
2. **Non-secret settings only.** Every auth and secret field name declared by the provider's config class is forbidden. Sending one fails with `{kind}_config must not include secret/auth fields: …`. API keys live in `ProviderAuth`, stored once per organisation through `POST /api/v1/auth` and Fernet-encrypted at rest. See [Provider credentials (ProviderAuth)](../../guides/concepts/provider-auth.md).

The remaining fields are whatever that provider's config class declares — model name, voice, language, speed, base URL and so on. Validation runs the payload through the provider's own Pydantic model, so an unknown or malformed field is rejected there. The stored result is the validated dump with secrets excluded and `null` values dropped.

## `knowledge_base`

`AgentKnowledgeBase`. Optional; the default is disabled.

| Field | Type | Default | Bounds | Notes |
| --- | --- | --- | --- | --- |
| `enabled` | bool | `false` | — | Master switch. |
| `mode` | `tool` or `context` | `context` | — | How retrieved chunks reach the LLM. |
| `document_ids` | list of string | `[]` | — | `KnowledgeDocuments` ids. Must be non-empty when enabled. |
| `top_k` | int | `5` | `1`–`10` | Chunks retrieved per query. |

When `enabled` is true, validation requires at least one non-blank `document_ids` entry, and — when the request carries an organisation — asserts every named document is `ready`, not still `processing` or `failed`.

`mode: "tool"` additionally requires an LLM provider that supports function calling. The allowed set is `KB_TOOL_LLM_PROVIDERS` in `apps/api/app/services/agent_config_validation.py`: `anthropic`, `azure_openai`, `groq`, `openai`. Any other provider is rejected.

The runtime re-parses this blob in `apps/runtime/services/knowledge/config.py` and is more forgiving than the API: an unrecognised `mode` falls back to `context`, and `top_k` is clamped into `1`–`10`. Knowledge is skipped entirely when `enabled` is false or `document_ids` is empty. See [Knowledge base (RAG)](../../guides/concepts/knowledge-base-rag.md).

## Custom variables and prompt substitution

`custom_variables` is a free-form object of named defaults. Keys must be non-empty strings; values are arbitrary.

At call time the runtime merges two layers, with the call winning:

```text
agent.config.custom_variables   ←  defaults
CallLogs.custom_variables       ←  per-call overrides
```

That merge is `resolve_custom_variables()` in `apps/runtime/services/pipecat/audio.py`. Per-call values arrive from the `custom_variables` field on `POST /api/v1/calls/outbound`, or from a campaign contact row's `context_variables`.

Substitution is applied to **both** `system_prompt` and `greeting_message` before the pipeline starts. The syntax is `{{variable_name}}`, matched by the regular expression `\{\{(\w+)\}\}` — word characters only, so no dots, dashes, or spaces in a name.

```json
{
  "prompts": {
    "system_prompt": "You are calling on behalf of {{company}}. The account id is {{account_id}}.",
    "greeting_message": "Hello {{customer_name}}, this is a call about your order."
  },
  "custom_variables": {
    "company": "Acme",
    "customer_name": "",
    "account_id": "unknown"
  }
}
```

Two rules from `substitute_variables()`, both covered by `apps/runtime/tests/test_prompt_substitution.py`:

* **A missing key becomes an empty string**, not the literal placeholder. `"Hi {{name}}"` with no `name` renders as `"Hi "`. Declare a default in `custom_variables` for every placeholder you use.
* **Non-string values are coerced** with `str()`. Numbers and booleans render as their Python representation.

Only the two prompt strings are substituted. Hold messages, online-detection messages, and the closing message are used verbatim.

## A complete example

The example below is `AgentCreateRequest.config` as declared in `apps/api/app/models/schemas.py`, unaltered:

```json
{
  "schema_version": 1,
  "prompts": {
    "system_prompt": "You are a helpful phone support agent.",
    "greeting_message": "Hello! How can I help you today?"
  },
  "behaviour": {
    "interruption_min_words": 2,
    "user_silence_hangup_seconds": 30,
    "call_timeout_seconds": 600,
    "ignore_user_speech_before_greeting": true,
    "hold_messages": ["One moment please."],
    "hold_message_timeout_seconds": 0.6,
    "user_online_detection_enabled": false,
    "user_online_detection_message": "",
    "user_online_detection_seconds": 10,
    "user_online_detection_repeats": 1,
    "user_online_detection_closing_message": "",
    "automatic_call_ending": {
      "enabled": true,
      "graceful_llm_call_ending": true
    }
  },
  "language": { "primary": "en", "secondary": [] },
  "models": {
    "stt_config": {
      "provider": "deepgram",
      "model": "nova-3-general",
      "language": "en"
    },
    "tts_config": {
      "provider": "cartesia",
      "model": "sonic-3.5",
      "language": "en",
      "voice": "3faa81ae-d3d8-4ab1-9e44-e50e46d33c30",
      "speed": 1.0,
      "volume": 1.0
    },
    "llm_config": {
      "provider": "openai",
      "model": "gpt-4.1",
      "base_url": "https://api.openai.com/v1"
    }
  },
  "knowledge_base": {
    "enabled": false,
    "mode": "context",
    "document_ids": [],
    "top_k": 5
  },
  "custom_variables": {
    "customer_name": "",
    "account_id": "unknown"
  }
}
```

Wrap it in a create request and the provider credentials must already be stored:

```bash
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer YOUR_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support Agent",
    "agent_category": "telephony",
    "telephony_provider": "vobiz",
    "config": { "...": "the object above" }
  }'
```

## Related

* [Data model](data-model.md)
* [REST API](../../api-reference/overview.md)
* [Agents and agent categories](../../guides/concepts/agents.md)
* [Provider registry](../../guides/concepts/provider-registry.md)
* [Provider credentials (ProviderAuth)](../../guides/concepts/provider-auth.md)
* [Knowledge base (RAG)](../../guides/concepts/knowledge-base-rag.md)
