# Brief: Integration, pipeline, languages, fallback (A4)

**Review gap:** No documentation on which pipeline is used, supported language codes, or fallback between cloud and local AI4Bharat servers.

---

## How the pipeline is chosen

- **Per agent** in MongoDB (configured in dashboard when creating/editing an agent).
- Config fields: `llm_model`, `stt_model`, `tts_model` (provider + model args + language).
- At call start, voice server loads config from backend: `fetch_agent_config_from_backend`.
- Services are created in: `voice_2_voice_server/api/services.py`

There is **no single global pipeline** for the whole deployment — each agent can differ.

---

## Supported providers (summary)

Copy and simplify from `voice_2_voice_server/README.md` "Supported Providers" section.

### LLM

| Provider key | Notes |
|--------------|-------|
| `openai` | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| `kenpath` | Vistaar streaming API; agent language `bhb` uses Voice Bhili GET API |

### STT

| Provider key | Notes |
|--------------|-------|
| `deepgram` | nova-3, nova-2, flux-general-en |
| `google` | chirp models |
| `openai` | whisper-1 |
| `indic-conformer-stt` | AI4Bharat local server |
| Bhashini | via `bhashini` provider / Integrations key |

### TTS

| Provider key | Notes |
|--------------|-------|
| `deepgram` | Aura voices |
| `cartesia` | sonic models |
| `google` | various voices |
| `openai` | alloy, echo, etc. |
| `indic-parler-tts` | AI4Bharat local server |
| Bhashini | separate TTS env if used |

---

## Language codes

- Agent has a `language` field (e.g. `hi`, `en`, `bhb` for Bhili).
- Provider-specific mappings: `voice_2_voice_server/config/stt_mappings.py`, `tts_mappings.py`
- **Bhili (`bhb`):** STT uses `POST /transcribe/bhili` on local STT server when agent uses AI4Bharat STT.

**Writer:** Publish a table of supported agent languages and which STT/TTS providers support each (extract from mapping files).

---

## Cloud vs local — is there automatic fallback?

**No.** The codebase does not automatically switch from Bhashini/cloud to local AI4Bharat (or vice versa) on failure.

Whatever provider is configured on the agent is used.

| Provider type | Credential source |
|---------------|-------------------|
| Bhashini STT | Integrations `Bhashini` or env `BHASHINI_API_KEY` |
| AI4Bharat STT/TTS | Agent must use `indic-conformer-stt` / `indic-parler-tts`; deployment sets `AI4BHARAT_STT_URL`, `AI4BHARAT_TTS_URL` (or `INDIC_STT_SERVER_URL` / `INDIC_TTS_SERVER_URL` in voice server docs) |
| OpenAI, Deepgram, etc. | Integrations and/or env API keys |

**Writer:** State clearly: "To use local speech engines, create agents with AI4Bharat providers and run the optional GPU servers (see A5)."

---

## Pipecat pipeline (technical one-liner)

During a call: audio in → STT → LLM → TTS → audio out. Implemented in `voice_2_voice_server/api/bot.py` using Pipecat `Pipeline` / `PipelineRunner`.
