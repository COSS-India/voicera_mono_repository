---
description: How VoicEra stores per-organisation provider API keys and Vobiz telephony credentials.
---

# Integrations

Integrations let each organisation store API keys for **AI providers** (LLM, STT, TTS) and **telephony** (Vobiz) inside the Backend, scoped by `org_id`. The Voice Server fetches these at call time via an internal endpoint, so production deployments do not put per-org provider keys in `voice_2_voice_server/.env`.

## Responsibilities

- Persist provider API keys per `org_id` in the `Integrations` MongoDB collection
- Persist multiple **Custom LLM** endpoints per org in `CustomLLMIntegrations`
- Serve decrypted keys to the Voice Server via an `X-API-Key`-protected internal endpoint
- Provide JWT-protected CRUD for the dashboard

## Telephony credentials (Vobiz)

{% hint style="warning" %}
**Vobiz Auth ID** and **Vobiz Auth Token** must be added in **Dashboard -> Integrations**, not in `voice_2_voice_server/.env`, for any multi-tenant or production deployment. The Voice Server reads them from MongoDB per `org_id` when placing outbound calls and when the Backend manages Vobiz applications.
{% endhint %}

| Integration key | Used for |
|-----------------|----------|
| Vobiz Auth ID | Vobiz API account id |
| Vobiz Auth Token | Vobiz API token |

For the full telephony model, see [concepts/telephony-model.md](../concepts/telephony-model.md).

## How AI integrations work

```
Dashboard user
    | POST /api/v1/integrations  (JWT)
    v
Backend -> MongoDB Integrations collection (per org_id)

Voice Server
    | POST /api/v1/integrations/bot/get-api-key  (X-API-Key)
    v
Backend -> returns stored API key for requested model
```

When an agent uses OpenAI (LLM or knowledge base embeddings), the Backend looks up the org's OpenAI integration key automatically. No environment variable is required on the Voice Server.

## Supported providers

The following provider keys can be stored as Integrations.

### LLM

| Provider | `model` value | Notes |
|----------|---------------|-------|
| OpenAI | `openai` | GPT models as LLM + KB embeddings |
| Anthropic | `anthropic` | Claude models |
| Grok / xAI | `grok` | Grok models |
| Custom LLM | _(see below)_ | OpenAI-compatible endpoints in `CustomLLMIntegrations` |

### STT

| Provider | `model` value |
|----------|---------------|
| Deepgram | `deepgram` |
| ElevenLabs | `elevenlabs` |
| Sarvam | `sarvam` |
| Bhashini | `bhashini` |

### TTS

| Provider | `model` value |
|----------|---------------|
| Cartesia | `cartesia` |
| Deepgram | `deepgram` |
| ElevenLabs | `elevenlabs` |
| Sarvam | `sarvam` |

### Telephony

| Provider | Integration keys |
|----------|------------------|
| Vobiz | `Vobiz Auth ID`, `Vobiz Auth Token` |

## Custom LLM integrations

Custom LLMs let each organisation register **multiple** OpenAI Chat Completions v1-compatible endpoints (for example NVIDIA NIM, vLLM, or hosted inference). Configuration lives in the `CustomLLMIntegrations` collection, not in the flat `Integrations` table.

```
Dashboard -> Integrations -> Custom LLM
    | POST /api/v1/custom-llm-integrations  (JWT)
    v
Backend -> MongoDB CustomLLMIntegrations (per org_id, multiple rows)

Assistant create/edit
    | llm_model.name = "Custom LLM"
    | llm_model.custom_llm_id = "<mongo_id>"
    v
Voice Server
    | POST /api/v1/custom-llm-integrations/bot/get-config  (X-API-Key)
    v
Pipecat OpenAILLMService -> POST {base_url}/chat/completions
```

Each Custom LLM record stores:

| Field | Description |
|-------|-------------|
| `name` | Display label in the dashboard |
| `base_url` | Normalised OpenAI base URL (ends with `/v1`) |
| `api_key` | Bearer token for the endpoint |
| `model` | Model id sent in the chat completion request body |

Agent config example:

```json
{
  "llm_model": {
    "name": "Custom LLM",
    "custom_llm_id": "507f1f77bcf86cd799439011",
    "model": "google/gemma-4-26B-A4B-it"
  }
}
```

## API surface

| Method | Path | Auth |
|--------|------|------|
| `GET` | `/api/v1/integrations` | JWT |
| `GET` | `/api/v1/integrations/{model}` | JWT |
| `POST` | `/api/v1/integrations` | JWT |
| `DELETE` | `/api/v1/integrations/{model}` | JWT |
| `POST` | `/api/v1/integrations/bot/get-api-key` | `X-API-Key` |
| `GET` | `/api/v1/custom-llm-integrations` | JWT |
| `POST` | `/api/v1/custom-llm-integrations` | JWT |
| `PUT` | `/api/v1/custom-llm-integrations/{id}` | JWT |
| `DELETE` | `/api/v1/custom-llm-integrations/{id}` | JWT |
| `POST` | `/api/v1/custom-llm-integrations/bot/get-config` | `X-API-Key` |

Create example:

```bash
curl -X POST http://localhost:8000/api/v1/integrations \
  -H "Authorization: Bearer $JWT" \
  -H "Content-Type: application/json" \
  -d '{ "model": "openai", "api_key": "sk-...", "org_id": "your-org-id" }'
```

Voice Server internal lookup:

```bash
curl -X POST http://localhost:8000/api/v1/integrations/bot/get-api-key \
  -H "X-API-Key: $INTERNAL_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{ "org_id": "your-org-id", "model": "openai" }'
# -> { "api_key": "sk-..." }
```

## Integration vs. environment variable

Both approaches work. The Integration record takes precedence for OpenAI when the agent config carries an `org_id`.

| Approach | When to use |
|----------|-------------|
| **Integration (database)** | Multi-tenant: different orgs, different keys, managed from the dashboard |
| **Environment variable** | Single-tenant or local dev: one global key for all orgs on this instance |

{% hint style="warning" %}
For knowledge base (RAG) embeddings, the org's **OpenAI integration must be configured**. The global `OPENAI_API_KEY` environment variable is not used for KB ingest or retrieval.
{% endhint %}

## Security

- Keys are stored in MongoDB, scoped by `org_id`.
- The dashboard never returns full key material after save; only a masked preview.
- Internal lookup (`bot/get-api-key`, `bot/get-config`) requires the `INTERNAL_API_KEY` header.
- In production, `INTERNAL_API_KEY` should be a long random string and must match between Backend and Voice Server.

See [guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md).

## Troubleshooting

- [troubleshooting/common-issues.md](../troubleshooting/common-issues.md)
- Voice Server logs `integration not found` -> the agent's org has no Integration for the configured provider, or `INTERNAL_API_KEY` does not match.

## Next steps

- [services/backend.md](backend.md)
- [services/voice-server.md](voice-server.md)
- [concepts/telephony-model.md](../concepts/telephony-model.md)
- [reference/rest-api.md](../reference/rest-api.md)
