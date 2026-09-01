---
description: The multi-step agent creation wizard.
---

# Agent creation wizard

The wizard at `/agent-creation` walks you through building an agent config and ends by creating it with `POST /agents`. It is the dashboard's main reason to exist: the agent config is a deeply nested object, and the wizard assembles it from provider catalogs the API serves rather than making you type it.

{% hint style="warning" %}
The dashboard is **Beta**. It lives on the `dev-frontend` branch, is not merged into `dev`, and is not part of the Docker Compose stack. You run it separately against a running API.
{% endhint %}

## The steps

Six steps, declared as `STEPS` in `frontend/src/app/(app)/agent-creation/page.tsx`. The stepper is clickable, so you can jump backwards and forwards freely; the Next button is what enforces the gates.

```mermaid
flowchart LR
  S["Setup<br/>StartStep"]
  T["Template"]
  K["Language and stack<br/>StackStep"]
  I["Instructions<br/>AgentFieldsStep"]
  M["Call manners<br/>AgentFieldsStep"]
  R["Review<br/>ReviewStep"]
  C["Test call<br/>TestCallStep"]

  S -->|"from scratch"| K
  S -->|"from a template"| T
  T --> R
  K --> I
  I --> M
  M --> R
  R --> C
```

| Step | Component | What it does |
| --- | --- | --- |
| Setup | `StartStep.tsx` | Choose "start from scratch" or browse the template gallery. |
| Language & stack | `StackStep.tsx` + `AgentStackFields.tsx` | Pick languages, then STT, TTS and LLM providers, a voice, a model, and a delivery mode. |
| Instructions | `AgentFieldsStep.tsx` | Name, greeting, system prompt, LLM settings, knowledge base toggle. |
| Call manners | `AgentFieldsStep.tsx` | Interruption threshold, silence hang-up, call limit, hold phrases, still-there check. |
| Review | `ReviewStep.tsx` | Read the whole config back; each section links to the step that owns it. |
| Test call | `TestCallStep.tsx` | Creates the agent, then opens a live browser call. |

Templates short-circuit the middle: picking one merges its `set` object into the form and jumps straight to Review. There are nine templates across eight categories (`TEMPLATES` and `TPL_CATS` in `frontend/src/lib/wizard-data.ts`) — mandi rates, pension status, grievance intake, and similar Indic public-service starting points. They set names, greetings, languages, and prompts; they do not set providers, so you still need the stack step's choices before the agent can be created.

The Next button is disabled until:

* **Language & stack** — at least one language plus an STT, TTS, LLM provider and an LLM model are chosen (`stackReady`).
* **Instructions** — the agent has a name.
* **Review** — both of the above, checked again before Test call.

## How catalogs populate

Nothing in the form's provider dropdowns is hardcoded. `frontend/src/lib/use-wizard-catalogs.ts` fetches everything from the API and re-fetches as your selections change.

On mount, and again whenever the language selection changes, it issues six requests in parallel:

| Request | Purpose |
| --- | --- |
| `GET /languages` | The full language id → label map. |
| `GET /configuration/stt?languages=…` | STT providers that support the selected languages. |
| `GET /configuration/tts?languages=…` | TTS providers that support the selected languages. |
| `GET /configuration/llm` | LLM providers (not language-filtered). |
| `GET /configuration/telephony` | Telephony providers, for the delivery dropdown. |
| `GET /auth/configured` | Which providers this organisation has credentials for. |

The last one is the important filter. `filterToConfigured()` intersects each provider list with `GET /auth/configured`, so **a provider you have not connected under Integrations never appears in the wizard**. If your stack step shows an empty dropdown, the fix is to add credentials — see [Provider credentials (ProviderAuth)](../../guides/concepts/provider-auth.md).

Once you pick a provider, a second round of requests fetches its field schema:

* `GET /configuration/stt/setting/{provider}?languages=…`
* `GET /configuration/tts/setting/{provider}?languages=…`
* `GET /configuration/llm/setting/{provider}`

These return a `fields` map — each field carrying a `type`, `default`, `examples`, and a `secret` flag (`ProviderSettingsCatalog` in `frontend/src/lib/catalog-types.ts`). The wizard turns them into UI directly:

* `modelOptionsFromSettings()` builds the LLM model dropdown from the `model` field's `examples`.
* `voiceOptionsFromSettings()` builds the voice picker from the `voice` field's `examples`.
* `voiceFieldIsFreeText()` switches the voice picker to a text input when a provider has no enumerated voices — Cartesia's voice UUIDs, for example.

The language list is narrowed the same way. A separate effect fetches the settings for every *configured* STT and TTS provider and unions their `language` field `examples` into `availableLanguages`, so the language chips only offer languages your connected providers can actually handle. If that lookup yields nothing, it falls back to the full `GET /languages` map rather than showing an empty list.

When you change languages, providers that no longer appear in the filtered catalogs are dropped and replaced with the first still-valid option. Model and voice selections are reconciled the same way.

`frontend/src/lib/wizard-data.ts` still contains hardcoded `LLM_PROVIDERS`, `STT_PROVIDERS` and `TTS_PROVIDERS` arrays with provider and model names. Nothing imports them — they are leftovers from the pre-API prototype. The live wizard reads catalogs only. Do not use those arrays as a provider inventory.

## The prompt library

The Instructions step has an "insert from library" action that opens `PromptLibraryDialog.tsx`. It offers nine reusable prompt fragments from `frontend/src/lib/prompt-modules.ts`, filed under five categories — Tone, Escalation, Compliance, Language, and Verification.

Each module is static text with a short rationale: a warm rural-helpline tone, an escalation phrasing, a verification script. Selecting one appends its text to your system prompt with a blank line between, and the dialog marks it "Added" if the text is already present so you cannot insert it twice.

The same nine modules are also browsable on their own at `/library`, which is a standalone copy of the picker with no insert target.

These are text snippets shipped with the dashboard, not API data. The API has no prompt-library endpoint.

## What it posts to the API

`formToAgentCreatePayload()` in `frontend/src/lib/agent-mapper.ts` turns the flat form into the nested agent body. The same function is used by the edit page, which sends it as `PATCH /agents/{agent_id}`.

**Agent category is derived, not chosen directly.** If the delivery dropdown has a telephony provider selected, `agent_category` is `"telephony"` and `telephony_provider` is included. If it is empty, the agent is `"websocket"` — the default, and the only kind that can take a browser test call. See [Agents and agent categories](../../guides/concepts/agents.md).

**Languages split by position.** The first language chip becomes `language.primary`; the rest become `language.secondary`.

**Model configs come from the catalogs, not the form.** `buildModelConfigsFromCatalogs()` calls `settingsToModelConfig()` (`frontend/src/lib/catalog-utils.ts`), which walks the provider's field schema and, for each non-secret field, takes your override if you set one, else the field's `default`, else its first `example`. Secret fields and `provider` / `kind` / `name` are skipped — credentials live in ProviderAuth, never in the agent config. Only three overrides are passed: the primary language, the chosen voice, and the chosen LLM model.

The resulting body:

```json
{
  "name": "Kisan Sahayak",
  "agent_category": "websocket",
  "config": {
    "schema_version": 1,
    "prompts": {
      "system_prompt": "...",
      "greeting_message": "..."
    },
    "behaviour": {
      "interruption_min_words": 3,
      "user_silence_hangup_seconds": 30,
      "call_timeout_seconds": 600,
      "ignore_user_speech_before_greeting": true,
      "hold_messages": ["Ek minute dekh raha hoon"],
      "hold_message_timeout_seconds": 5,
      "user_online_detection_enabled": true,
      "user_online_detection_message": "Are you still there?",
      "user_online_detection_seconds": 90,
      "user_online_detection_repeats": 1,
      "user_online_detection_closing_message": "I'll end the call now. Goodbye."
    },
    "language": { "primary": "hi", "secondary": [] },
    "models": {
      "stt_config": { "provider": "..." },
      "tts_config": { "provider": "...", "voice": "..." },
      "llm_config": { "provider": "...", "model": "..." }
    },
    "knowledge_base": { "enabled": false, "document_ids": [], "top_k": 5 }
  }
}
```

Several values in `behaviour` and `knowledge_base` are **constants the mapper hardcodes**, not wizard inputs: `hold_message_timeout_seconds` (5), all four `user_online_detection_*` values apart from the enabled toggle, and `top_k` (5). To change them, edit the agent over the API. Field meanings are in [Agent configuration](../reference/agent-configuration.md).

{% hint style="warning" %}
The Instructions and Call manners steps render controls for **Tokens per reply**, **Temperature**, and **Audio buffer (ms)** (`frontend/src/lib/agent-form-cards.ts`), but `formToAgentCreatePayload()` never reads `tokens`, `temperature`, or `bufferMs`. Setting them has no effect on the created agent. Set LLM sampling parameters over the API instead.
{% endhint %}

If the provider settings have not finished loading, the mapper throws rather than posting a half-built config — the wizard surfaces this as "Provider settings are still loading."

On success, the Test call step holds the returned `agent_id` and immediately opens a live browser call against it. See [Browser test calls](test-calls.md).

## Related

* [Browser test calls](test-calls.md)
* [Dashboard tour](dashboard-tour.md)
* [Agents and agent categories](../../guides/concepts/agents.md)
* [Agent configuration](../reference/agent-configuration.md)
* [Create your first agent](../../guides/quickstart/first-agent.md)
