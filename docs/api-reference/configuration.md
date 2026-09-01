---
description: Discover which providers exist and what each one accepts.
---

# Configuration catalogs

# Configuration

`apps/api/app/routers/configuration.py`, prefix `/api/v1/configuration`. The self-describing provider catalogue. Everything here is generated from the registry at runtime, so a newly added provider appears without a code change here. See [Provider registry](../guides/concepts/provider-registry.md).

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| GET | `/configuration/stt` | Bearer | Registered STT providers. Optional `languages` query. |
| GET | `/configuration/tts` | Bearer | Registered TTS providers. Optional `languages` query. |
| GET | `/configuration/llm` | Bearer | Registered LLM providers. |
| GET | `/configuration/telephony` | Bearer | Registered telephony providers. |
| GET | `/configuration/stt/setting/{provider}` | Bearer | Setting schema for one STT provider. Optional `languages`. |
| GET | `/configuration/tts/setting/{provider}` | Bearer | Setting schema for one TTS provider. Optional `languages`. |
| GET | `/configuration/llm/setting/{provider}` | Bearer | Setting schema for one LLM provider. |
| GET | `/configuration/telephony/setting/{provider}` | Bearer | Setting schema for one telephony provider. |

`languages` is a comma-separated list of canonical language ids and acts as an **AND** filter: `?languages=en,hi` returns only providers supporting both.

The `setting` routes return the field catalogue for that provider — enough to render a configuration form and to know which keys `config.models.{stt,tts,llm}_config` will accept. An unknown provider returns `404`; a malformed request returns `400`.

# Languages

`apps/api/app/routers/languages.py`.

## `GET /languages`

Bearer. Returns the canonical language id → label map that the agent builder's picker uses:

```json
{ "languages": { "en": "English", "hi": "Hindi" } }
```

These ids are what `config.language.primary` accepts. Which of them a given provider actually supports is a separate question — filter with `GET /configuration/stt?languages=`.

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
