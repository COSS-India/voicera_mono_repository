# Live Translation Agent - Technical Design

**Status:** Implemented (see §11 for the as-built checklist and §13 for deltas from the original plan)
**Scope:** New agent type for real-time, one-to-many live translation with public shareable listener links. Opt-in public sharing for any agent as a secondary, generic capability.
**Guiding constraint:** Minimal, standard, non-breaking. All heavy new logic is additive (new modules, new routes, new optional fields). No existing flow is modified in a way that changes its behaviour.

**Follow-up design:** [NMT_ENGINE_PLAN.md](NMT_ENGINE_PLAN.md) — pluggable translation engine, so a broadcast uses either the streaming LLM (as described here) or an on-prem NMT model, never both.

---

## 1. Product summary

Create an agent whose job is *live interpretation*:

- **One presenter** (host) speaks into a mic. Their speech is transcribed (STT), translated per target language (LLM), synthesised (TTS), and streamed out.
- **Many listeners** open a **public shareable link**, pick a language from an allowed set, and hear the live translation in that language. Multiple listeners are supported; listeners on the same language share one translation stream.
- The presenter's own connection receives **no bot audio** ("mute bot response"): the presenter just talks, translations go only to listeners.

Secondary generic capability: an **opt-in "public share" flag** on *any* agent, so a normal (conversational) agent can also be exposed at a public link as an independent 1:1 session. This reuses the existing browser voice path with almost no new code.

---

## 2. Current-state facts this design builds on

Established by codebase survey (file:line references are load-bearing anchors for implementation):

**Agent model (backend - FastAPI + MongoDB, collection `AgentConfig`)**
- Pydantic models: `voicera_backend/app/models/schemas.py:49-94` (`AgentConfigCreate` / `Response` / `Update`).
- Type enum lives in `agent_config.interaction_mode`: `voicera_backend/app/services/agent_service.py:19` → `VALID_INTERACTION_MODES = {"conversational", "non_conversational"}`.
- Per-mode validation: `_validate_agent_config_for_mode` `agent_service.py:64-73`. Mode is **immutable after create**: `agent_service.py:404-409`.
- `agent_category` is a free-text tag (only value in use: `"voicera_telephony"`, set at `agent_service.py:160` and frontend `assistants/page.tsx:1048`).
- System prompt: nested `agent_config.system_prompt` (`agent_service.py:161`, frontend `page.tsx:288,1061`).
- `telephony_provider` optional; only written when truthy (`agent_service.py:273-274, 430-431`). Empty = no telephony (recent commits `29516b6`, `bc9cf63`).
- CRUD service: `agent_service.create_agent` (`agent_service.py:220-296`), `update_agent_config` (`agent_service.py:372-484`).
- Router: `voicera_backend/app/routers/agents.py` (prefix `/agents`, mounted under `/api/v1`).

**Voice engine (`voice_2_voice_server` - FastAPI + Pipecat)**
- WS routes in `voice_2_voice_server/api/server.py`: `/agent/{agent_id}` (Vobiz, `:430`), `/plivo/agent/{agent_id}` (`:514`), `/browser/agent/{agent_id}` (browser, `:545`). **All unauthenticated** - `websocket.accept()` immediately.
- Pipeline assembled in `api/bot.py:run_bot` (`:302-335`): `transport.input → stt → … → llm → … → tts → … → transport.output`. Strictly **1:1**, no fan-out, no session registry.
- Service factories in `api/services.py`: `create_stt_service` (`:373-567`), `create_llm_service` (`:120-370`), `create_tts_service` (`:570-785`). Providers include OpenAI, Deepgram, Sarvam, ElevenLabs, AI4Bharat, Bhashini.
- Config fetch: `utils/backend_utils.py:fetch_agent_config_from_backend` (`:156-199`) → `GET /api/v1/agents/config/id/{agent_id}` with `X-API-Key`.
- Language: per-agent, seeded into `stt_config`/`tts_config` (`bot.py:157-165`). Display→provider-code maps in `config/stt_mappings.py`, `config/tts_mappings.py`. Mid-call switch tool exists but is provider-limited (`utils/language_switching.py`).
- Outbound audio serialisation choke point: `serializer/vobiz_serializer.py:serialize` (`:41-62`) - returning `None` suppresses an outbound audio frame.
- Multi-worker: `VOICE_SERVER_NUM_WORKERS` (`server.py:623`). **Per-process state is not shared across workers** - critical for the room registry (see §7).

**Auth / sharing / frontend**
- Backend auth: user JWT `get_current_user` (`voicera_backend/app/auth.py:131-168`) and internal `X-API-Key` `verify_api_key` (`auth.py:171-205`). WebSocket layer has **no auth**.
- No existing share-token model, no public agent flag, no unauthenticated page route.
- Platform-key fallback: `services.py:54-59` (`ALLOW_PLATFORM_KEY_FALLBACK`).
- Frontend: Next.js App Router (`voicera_frontend/app/`). Route groups `(auth)` and `(dashboard)`; **auth is client-side only** (localStorage token, redirect on 401 in `lib/api.ts:70-76`). New public page = new top-level segment outside both groups, not using `fetchWithAuth`.
- Browser voice client: `voicera_frontend/components/assistants/test-browser-dialog.tsx` - mic capture + 16 kHz PCM + `playAudio` playback + transcript rendering. WS URL: `lib/johnaic-config.ts:getBrowserAgentWebSocketUrl`.
- Create/edit wizard: `voicera_frontend/app/(dashboard)/assistants/page.tsx`; step keys `WizardStepKey` (`:205`), `getWizardStepKeys` (`:220-226`); type cards (`:1466-1507`); payload build (`:1046-1096`). TS types `lib/api.ts:362-447`.

---

## 3. Architecture overview

```
                       ┌──────────────────────────── voice_2_voice_server ───────────────────────────┐
 Presenter (host)      │                                                                              │
  mic ─ WS ────────────┼─▶  /translate/publish/{agent_id}                                             │
  (authed host token)  │        └─▶ Publisher pipeline:  transport.input → VAD → STT                  │
                       │                 └─▶ final transcript ──▶ Room bus (asyncio pub/sub)          │
                       │                                              │                               │
                       │                        ┌─────────────────────┼──────────────────────┐        │
                       │                        ▼                     ▼                      ▼        │
                       │               LangWorker[hi]        LangWorker[ta]         LangWorker[en]    │
                       │             translate→TTS→fanout   translate→TTS→fanout  translate→TTS→fanout│
                       │                    │                     │                      │            │
 Listeners (public)    │   /translate/listen/{share_token}?lang=hi │  ?lang=ta            │ ?lang=en   │
  browser playback ◀───┼───(subscribers[hi] set) ◀────────────────┘   ◀──(subscribers[ta])◀──────────┘
                       └──────────────────────────────────────────────────────────────────────────────┘
```

- **Publisher leg** reuses the existing transport + STT stack, but ends at STT (no LLM/TTS/output) → presenter inherently receives no bot audio.
- **LangWorker** is created lazily when the first listener selects a language, and torn down when the last listener for that language leaves. Cost scales with *distinct active languages*, not listener count.
- **Fan-out** is a small in-memory broadcast to all listener sockets in a language group - the one capability Pipecat lacks today, added as a new module without touching the 1:1 path.

---

## 4. Data model changes (backend)

### 4.1 New interaction mode

`agent_service.py:19`:
```python
VALID_INTERACTION_MODES = {"conversational", "non_conversational", "translation"}
```

Add a `translation` branch to `_validate_agent_config_for_mode` (`agent_service.py:64-73`). Requirements for a translation agent:
- `agent_config.source_language` - non-empty (what the presenter speaks).
- `agent_config.target_languages` - non-empty list, each resolvable in `TTS_LANGUAGE_MAP` for the chosen `tts_model`.
- `agent_config.stt_model.name` and `agent_config.tts_model.name` present.
- `system_prompt` optional; if empty, use a built-in translation prompt (§6.2).

**Branch-point audit (must review - additive value can be silently mis-handled by code that assumes "not conversational ⇒ non_conversational"):**
- `_validate_agent_config_for_mode` - add branch.
- Immutability rule `agent_service.py:404-409` - keep translation immutable too.
- Frontend `getWizardStepKeys` (`page.tsx:220-226`) - add translation step set.
- Any `if mode == "non_conversational"` special-casing (e.g. greeting-required, `agent-card.tsx:66` `isAlertAgent`) - ensure translation is not accidentally treated as an alert agent.

### 4.2 New agent_config fields (all optional, additive)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `source_language` | str | - | Presenter's spoken language (display name, mapped via existing STT map). |
| `target_languages` | list[str] | `[]` | Allowed listener output languages. |
| `mute_publisher_playback` | bool | `true` | The "mute bot response" toggle. For translation, publisher leg never builds an output path (so this is inherently satisfied); the flag is retained so the behaviour is explicit and reusable. |

### 4.3 New top-level agent fields (opt-in public sharing - generic, all types)

| Field | Type | Default | Meaning |
|---|---|---|---|
| `public_share_enabled` | bool | `false` | Opt-in. When false, no public access whatsoever. |
| `share_token` | str \| null | `null` | Random URL-safe token (`secrets.token_urlsafe(16)`), generated on create/update when `public_share_enabled` flips true and no token exists. Never regenerated silently. Rotatable via an explicit endpoint. |

Add to `AgentConfigCreate` / `AgentConfigUpdate` / `AgentConfigResponse` (`schemas.py:49-94`) and write them in `create_agent` / `update_agent_config` following the existing "only write when truthy" idiom.

Migration: none required (MongoDB, absent fields read as defaults). Existing agents are unaffected - `public_share_enabled` absent ⇒ treated as false.

---

## 5. Public share endpoints (backend)

New router `voicera_backend/app/routers/public.py` (prefix `/public`, **no auth dependency**), mounted under `/api/v1`. Keeps public surface isolated from authed routers.

- `GET /public/agents/{share_token}` → resolve token → agent. **404 if not found or `public_share_enabled` is false.** Returns a *secret-stripped* public projection (reuse the pattern of `IntegrationPublicResponse`, `integration_service.py:17-22`):
  ```json
  {
    "agent_id": "...",
    "display_name": "...",
    "interaction_mode": "translation",
    "source_language": "English",
    "target_languages": ["Hindi", "Tamil"],
    "greeting_message": "..."
  }
  ```
  Never returns `org_id`, keys, prompts, or telephony data.

Token-rotation / host access (authed, JWT):
- `POST /agents/{agent_type}/share/rotate` → regenerate `share_token`.
- Host connect to the voice server is authorised with a short-lived **host token** minted by the backend for the agent owner (see §7.1) - not the public `share_token`.

Voice-server → backend lookup for the listener leg: add `GET /agents/public/by-token/{share_token}` behind `X-API-Key` (same style as `/agents/config/id/{agent_id}`, `agents.py:39-43`) so the voice server can validate a listener's token and read `agent_id` + allowed `target_languages` without exposing secrets publicly.

---

## 6. Voice server - translation module (new, self-contained)

New file `voice_2_voice_server/api/translation_room.py`. New routes added to `server.py`. **No change to `bot.py` / existing WS routes.**

### 6.1 Room registry

```python
# process-local registry
_ROOMS: dict[str, TranslationRoom] = {}   # keyed by agent_id

class TranslationRoom:
    agent_id: str
    config: dict                      # fetched via existing backend_utils
    publisher: WebSocket | None
    subscribers: dict[str, set[WebSocket]]      # lang_code -> sockets
    workers: dict[str, LangWorker]              # lang_code -> worker
    bus: "SourceSegmentBus"                     # fan-in of final transcripts
```

- `bus` is an asyncio pub/sub: publisher pushes `{seq, text, ts}`; each `LangWorker` subscribes. Use per-worker `asyncio.Queue` so a slow language cannot back-pressure others (bounded queue, drop-oldest on overflow with a log - no silent unbounded growth).

### 6.2 Publisher WS `/translate/publish/{agent_id}`

Authorised by host token (§7.1). Lifecycle:
1. Validate host token → agent_id + org.
2. Register as `room.publisher` (reject if one already active - single presenter per room; return a clear close code).
3. Build a **minimal Pipecat pipeline**: `transport.input() → SileroVAD → stt → TranscriptCollector`.
   - Reuse `create_stt_service` (`services.py:373`) and the existing `FastAPIWebsocketTransport` + `VobizFrameSerializer` setup from `bot.py:514-526` (audio-in only; `audio_out_enabled=False`).
   - `TranscriptCollector` is a tiny `FrameProcessor` that captures final `TranscriptionFrame`s and calls `room.bus.publish(...)`. No LLM, no TTS, no `transport.output()` ⇒ presenter receives nothing back (the "mute" requirement, satisfied structurally).
4. On disconnect: mark `publisher = None`; optionally keep workers warm for a grace period, then tear down room if no publisher and no subscribers.

### 6.3 LangWorker (translate → TTS → fan-out)

One per active target language. Driven directly (no fake transport - no VAD needed for text-in):
```
loop:
    seg = await queue.get()                      # {seq, text}
    translated = await translate(seg.text, source_language, target_language)
    async for audio_frame in tts.run_tts(translated):     # reuse create_tts_service(target_lang)
        payload = serialize_playAudio(audio_frame)         # reuse serializer framing
        await room.fanout(target_language, payload)        # send_text to all subscribers[lang]
    await room.fanout_transcript(target_language, seg, translated)   # optional text events
```
- `translate(...)`: reuse `create_llm_service` (OpenAI etc.) with a fixed instruction - *"Translate the text from {source} to {target}. Output only the translation, no commentary."* Falls back to platform key via existing `platform_key_fallback_enabled()`. (If a dedicated translation API is later preferred, it swaps in behind this one function.)
- `tts`: `create_tts_service` with `tts_config.language = target` (reuse `TTS_LANGUAGE_MAP`).
- **Ordering:** process segments strictly by `seq` per worker to keep translated audio in order.
- Lazily started on first subscriber for that language; stopped when `subscribers[lang]` becomes empty.

### 6.4 Listener WS `/translate/listen/{share_token}?lang=<display>`

Public, no user auth. Lifecycle:
1. `websocket.accept()`.
2. Validate `share_token` via backend `GET /agents/public/by-token/{share_token}` (`X-API-Key`). Reject if not public or `interaction_mode != "translation"`.
3. Validate `lang ∈ target_languages`. Reject otherwise (close code + reason).
4. Add socket to `room.subscribers[lang]`; lazily start `LangWorker[lang]`.
5. Stream `playAudio` (and optional `transcript`) frames to the socket. Listener sends no audio (playback-only); ignore/soft-close any inbound audio.
6. On disconnect: remove from set; if set empty, stop the worker.

### 6.5 Fan-out

`room.fanout(lang, payload)` iterates `subscribers[lang]` and `await ws.send_text(payload)`, guarding each send in try/except to evict dead sockets. This is the only genuinely new capability vs. the current 1:1 engine, and it is ~30 lines.

---

## 7. Auth, capacity, and the multi-worker caveat

### 7.1 Host authorisation
The presenter must be the agent owner (or delegated). The public `share_token` must **not** grant publish rights. Approach: backend endpoint `POST /agents/{agent_type}/broadcast-token` (JWT-authed) mints a short-lived signed **host token** (HS256, same `SECRET_KEY`, `exp` a few minutes, claims `agent_id`+`org_id`+`role=host`). The `/translate/publish` route verifies it. Reuses existing JWT machinery (`auth.py:93-112`).

### 7.2 Capacity / cost policy
- Slots: count **1 slot for the publisher + 1 slot per active target language** (not per listener). Wire into the existing `try_acquire_call_slot` / `peek_capacity_available` accounting so translation rooms respect org limits like calls do.
- Cost driver: distinct active languages × presenter speech volume. Document this explicitly so operators size correctly. Cap `len(target_languages)` (e.g. ≤ 8) at validation time.

### 7.3 Multi-worker caveat (must decide before ship)
`_ROOMS` is process-local. With `VOICE_SERVER_NUM_WORKERS > 1`, the presenter and a listener can land on different workers and never see each other. Options, cheapest first:
1. **MVP:** run the translation routes affinity-pinned - deploy translation on a single-worker instance, or use sticky routing keyed by `agent_id`/`share_token` at the load balancer. Document the constraint. *(Recommended for first ship - zero extra infra.)*
2. **Scale-out:** back the room bus + subscriber registry with Redis pub/sub so any worker can serve any leg. Deferred; the `SourceSegmentBus` and `fanout` interfaces are designed so this is a drop-in later.

This caveat is called out loudly rather than hidden - it is the one real operational constraint of the minimal design.

---

## 8. Frontend changes

### 8.1 Create/edit wizard (`app/(dashboard)/assistants/page.tsx`)
- Add a third type card "Live Translation" alongside conversational/non-conversational (`:1466-1507`).
- `getWizardStepKeys` (`:220-226`): translation step set = `type → agent → llm → audio → share → review` (skip `telephony`, `call_mgmt`).
- `audio` step: single-select **source language**, multi-select **target languages** (reuse `components/assistants/language-selection-section.tsx` patterns).
- `share` step: **Enable public link** toggle (`public_share_enabled`); when on, show the generated listener URL after save.
- `agent` step: prompt prefilled with the translation instruction; editable but optional.
- Payload build (`:1046-1096`): include the new fields; `agent_category` can stay `"voicera_telephony"` or a new `"voicera_translation"` (cosmetic).
- TS types: extend `InteractionMode` (`lib/api.ts:379`), `AgentConfig` (`:381-432`), `CreateAgentRequest` (`:434-447`).

### 8.2 Agent detail page
- If `public_share_enabled`: show **listener share link** `${APP_URL}/live/{share_token}` with copy button, plus a **"Start broadcasting"** action (host page/dialog).

### 8.3 New public listener page - `app/live/[token]/page.tsx`
- Top-level segment, outside `(auth)`/`(dashboard)`. Must not import `fetchWithAuth`.
- On load: `GET /api/public/agents/{token}` (new BFF passthrough → backend `/public/agents/{token}`) to fetch display name + allowed target languages.
- Language picker → connect WS `${JOHNAIC_WS}/translate/listen/{token}?lang=<chosen>`.
- Playback + transcript rendering: extract the playback half of `test-browser-dialog.tsx` into a reusable `useAudioPlayback` hook (no mic). Reuse `Orb` visualiser.
- Language can be changed live by reconnecting with a new `lang`.

### 8.4 Host/broadcast page (authed) - dialog or `app/(dashboard)/assistants/[id]/broadcast`
- Fetch a host token (`POST /api/agents/{id}/broadcast-token`), connect WS `${JOHNAIC_WS}/translate/publish/{agent_id}` with it, capture mic (reuse mic-capture half of `test-browser-dialog.tsx`), show live source transcript. No playback.

### 8.5 Generic share (tier 1, any agent)
- The same `/live/{token}` page branches on `interaction_mode`: `translation` → listener flow above; anything else → the **existing 1:1 browser flow** (reuse `TestBrowserDialog` internals, connect to existing `/browser/agent/{agent_id}`). This makes "share any agent" a few lines, not a new engine - each visitor gets an independent session.

---

## 9. Why not "blanket-enable share links for all agents"

The voice WS already has zero transport auth, so a raw `agent_id` is *already* a de-facto key - blanket-enabling would only widen an existing exposure. Unauthenticated sessions consume the org's LLM/STT/TTS keys and call slots (real money + abuse surface). The **opt-in flag + `share_token`** is the minimal fix that (a) makes `agent_id` no longer the key, (b) keeps every agent private by default, and (c) still delivers the generic "share any agent" capability for those who opt in - one boolean, one token, one public endpoint, one page.

---

## 10. Non-breaking guarantees

- New `interaction_mode` value is additive; the branch-point audit (§4.1) is the only place existing logic is touched, and only to add a case.
- New agent fields are optional; absent ⇒ safe defaults; no migration.
- New public router and new voice-server module/routes are fully separate; `/agent`, `/plivo/agent`, `/browser/agent`, `bot.py`, and all existing services are untouched.
- Public access is impossible unless an owner explicitly flips `public_share_enabled`.
- Multi-worker constraint is documented and pinned for MVP (§7.3), not silently assumed away.

---

## 11. Implementation checklist (as built)

Backend
- [x] `"translation"` added to `VALID_INTERACTION_MODES` (`agent_service.py:19`); new `IMMUTABLE_INTERACTION_MODES = {"non_conversational", "translation"}` replaces the hard-coded non-conversational immutability check.
- [x] Validation branch for `translation` in `_validate_agent_config_for_mode` (requires `source_language`, non-empty `target_languages`, `stt_model.name`, `tts_model.name`).
- [x] `public_share_enabled` + `share_token` on `AgentConfigCreate/Update/Response`; token minted by `generate_share_token()` on first enable and preserved across toggles.
- [x] `fetch_agent_by_share_token` (only resolves when `public_share_enabled` is true), `rotate_share_token`, `build_public_agent_projection`.
- [x] New `routers/public.py`: `GET /public/agents/{share_token}` (unauthenticated, secret-stripped, 404 when not shared); mounted in `main.py`.
- [x] `GET /agents/public/by-token/{share_token}` and `POST /agents/broadcast/resolve` (both X-API-Key) for the voice server.
- [x] `POST /agents/{agent_type}/broadcast-token` and `POST /agents/{agent_type}/share/rotate` (JWT).

Voice server
- [x] `api/translation_room.py`: `TranslationRoom`, `LangWorker`, process-local registry, fan-out, `TranscriptCollector`, `FanOutSink`.
- [x] Routes `/translate/publish/{agent_id}` (host token via `?token=`) and `/translate/listen/{share_token}?lang=` (public) in `server.py`.
- [x] Publisher pipeline is `transport.input() → VAD → STT → TranscriptCollector` with `audio_out_enabled=False` - the presenter structurally receives no bot audio.
- [x] Capacity: one call slot for the publisher plus one per active language, via the existing `try_acquire_call_slot`/`release_call_slot` accounting.
- [x] Bounded per-language backlog (`MAX_SEGMENT_BACKLOG`, drop-oldest with a log) so one slow language cannot grow memory or stall the others.
- [x] Startup warning when `VOICE_SERVER_NUM_WORKERS > 1` (rooms are process-local - see §7.3).

Frontend
- [x] "Live Translation" type card, `translation` step set (`type → agent → audio → share → review`), and config fields in the create wizard.
- [x] `InteractionMode`/`AgentConfig`/`CreateAgentRequest` type extensions plus `PublicAgent` and `BroadcastToken`.
- [x] Shared `components/assistants/translation-languages-section.tsx` (listener languages + per-language voices) used by **both** the create wizard and the edit page.
- [x] Edit page round-trips translation agents: loads the mode, edits listener languages/voices, and manages the share link (enable/disable, copy, rotate).
- [x] Public `app/live/[token]/page.tsx` - language picker + playback + translated transcript, no auth.
- [x] `components/assistants/broadcast-dialog.tsx` - host mic capture → `/translate/publish`, starts muted, shows the listener link.
- [x] Agent card: "Live Translation" badge, link-active chip, Broadcast + Copy Link actions in place of Test Call / Test on Browser.
- [x] BFF passthrough routes: `app/api/public/agents/[token]`, `app/api/agents/broadcast-token`, `app/api/agents/share-rotate`.

Verification performed: `python3 -m py_compile` on all changed Python; `tsc --noEmit` (error count unchanged from the pre-existing baseline of 20, none in new or changed code); `eslint` clean on all new files; `next build` succeeds with `/live/[token]` and the three new API routes registered.

---

## 13. Deltas from the original plan

Two design points changed during implementation, both for correctness:

1. **Per-language voices (`agent_config.target_voices`).** The plan assumed one global TTS voice. In practice `tts.json` voices are language-specific and disjoint for the on-prem providers - AI4Bharat offers `Rohit/Divya/Aman/Rani` for Hindi but `Kavitha/Jaya` for Tamil - so a single speaker cannot serve multiple listener languages. A translation agent therefore stores `target_voices: {language: voice}`, and `LangWorker.start()` applies the matching voice when it builds that language's TTS service. The UI requires a voice only for languages whose provider exposes a fixed list; free-form voice-ID providers keep it optional.
2. **Host-token verification lives in the backend.** Rather than duplicating JWT decoding (and the `SECRET_KEY`) in the voice server, the presenter's token is verified by `POST /agents/broadcast/resolve` over the existing internal X-API-Key channel. The voice server holds no JWT logic.

Also worth noting: the agent's own prompt is **not** ignored. `TranslationRoom.translate()` sends a fixed "translate only, no commentary" system instruction and appends the agent's `system_prompt` as additional style/domain guidance, so the authored prompt shapes tone and terminology without being able to break the output contract. The create wizard labels this field "Translation Guidance" and prefills a sensible default.

### 13.1 Hardening applied during post-implementation review

Security
- **`agent_id` is not exposed publicly.** `build_public_agent_projection()` withholds it unless the caller passes `include_agent_id=True`, which only the internal X-API-Key endpoint does. Because the voice server's WebSocket routes are unauthenticated, a public `agent_id` would itself be a usable credential for `/browser/agent/{agent_id}` - which would have defeated the whole point of the share token (§9).
- **Host broadcast tokens cannot be used as session tokens.** They travel in a WebSocket URL query string, which proxies routinely log. `get_current_user` now rejects any JWT carrying a `role` claim, so a leaked broadcast URL cannot be replayed against the REST API. (Verified safe: real login tokens carry `is_member`, never `role`.)

Resource safety
- **Language workers are released when the presenter leaves.** `TranslationRoom.end_broadcast()` runs in the publisher's `finally`: it notifies listeners (`session_ended`), disconnects them, and stops every worker. Previously a listener leaving a tab open would pin one call slot per active language indefinitely, starving the org's real call concurrency.
- **Empty rooms no longer leak** on the capacity-rejection, missing-credential, and worker-start-failure paths.
- **Lock ordering** is consistently `room.lock → _ROOMS_LOCK`, never the reverse.

Failure behaviour
- **Missing translation credential fails at connect, not mid-talk.** The publisher resolves the OpenAI client before accepting audio and closes with code `4402` if the org has no OpenAI integration and platform fallback is off. This also moves the blocking integration lookup out of the audio path.
- **A bad segment can no longer mute a language.** The per-language consumer loop catches and logs per-segment errors and continues (matching the convention of commit `497c67d`, "keep batch worker alive on inference failure"), and a dead TTS pipeline is now logged via a task done-callback instead of vanishing.
- **Backlog overflow drops the oldest segment**, not the newest, so a lagging language catches up to live speech instead of falling further behind - the code now matches the documented intent.
- **`audio_out_sample_rate` is pinned** to 16 kHz. Providers constructed without an explicit rate would otherwise inherit pipecat's 24 kHz default; playback was already correct (the listener honours `media.sampleRate`) but payloads were larger than necessary.

UX correctness
- Listeners receive a `status` event on join and `presenter_live` when the talk starts, so opening the link early reads as "waiting for the presenter" rather than an apparent fault.
- The presenter-language picker is constrained to a single language in translation mode; extra selections were previously accepted and then silently ignored (STT only ever uses the first).

### 13.2 Audio continuity, latency and stall handling

A second pass after listening to real broadcasts. Every audible "chop" traced to one of five causes, none of them the TTS model itself:

Playback (`app/live/[token]/page.tsx`)
- **The listener was cutting itself off mid-word.** Translated speech runs longer than the source it came from, so a listener's buffered lead grows for as long as the presenter talks. On crossing the lead ceiling the client called `stopScheduledSources()`, which stopped the buffer *currently sounding* — an instant amplitude jump mid-word, discarding up to 8 s of speech, on a period set by how fast the lead accrued. Replaced with two graded mechanisms: above `SOFT_LEAD_SECS` playback runs at `SOFT_DRAIN_RATE` (5 % fast, no audio discarded), and above `MAX_LEAD_SECS` it skips forward — but only ever dropping frames *still queued*, cutting at a sentence end when one is within `BOUNDARY_CUT_WAIT_SECS` and otherwise at the end of the 20 ms frame that is playing. Sentence ends are known because the server now emits an `audio_boundary` event after each synthesised chunk (unknown events are ignored by older clients).
- **The audio context now opens at the stream's 16 kHz.** At the device rate every 20 ms buffer was resampled in isolation with no filter state carried across buffers, putting a discontinuity at each frame edge — a 50 Hz buzz layered over the speech. Falls back to the default rate on browsers that ignore the hint.

Fan-out (`translation_room.py`)
- **Per-listener send buffer raised to ~10 s and now drops in blocks.** The queue was sized in audio seconds (2 s) but filled in bursts — a language worker hands over a whole sentence as fast as TTS yields it — so ordinary sentences overflowed it and the drop-oldest policy ate the *head* of each one. Overflow now sheds one contiguous block (one clean skip) instead of a frame per push (a burst of holes through a word), and logs how much was skipped.
- **A dead listener writer no longer means silent forever.** The writer task swallowed its exception and exited, leaving the socket open and the listener watching a "live" indicator with nothing playing. It now logs, marks itself dead and closes the socket so the client reconnects.
- **A TTS failure before any audio retries once.** Previously any `ErrorFrame` returned mid-chunk, so a provider that died at connect dropped the sentence outright. Retry only happens when nothing has been spoken yet — re-running a chunk that was half spoken would repeat it.

Latency
- **Translate and synthesise now overlap.** `_consume` was awaiting `_synthesize` before pulling the next segment, so every segment paid the LLM's full time-to-first-token as dead air. The worker is now two stages joined by a bounded `_synth_queue`: translation of segment *n+1* proceeds while segment *n* is still being spoken. Order is preserved by a single synth task draining a FIFO, and the bounded queue keeps the original back-pressure path (source queue fills → oldest segment dropped) intact.
- **Every segment logs its own breakdown** (`llm_ttft`, `tts_ttfa`, audio duration, synth wall-clock, and `rtf` — above 1 means this language synthesises slower than it is spoken, so listener lag will grow without bound). Previously the only way to attribute lag to a stage was to guess from the audio.

Stall handling — a hung provider must not mute a language for the rest of the broadcast. All three are *inactivity* limits, so a legitimately long sentence is never cut short: `TRANSLATION_LLM_TIMEOUT_SECS` (10 s, per token, also passed to the OpenAI client), `TRANSLATION_TTS_STALL_SECS` (20 s, per audio frame), and `INDIC_TTS_SOCK_READ_SECS` (30 s, was a hard-coded 600).

On-prem Parler backend (`ai4bharat_tts_server`)
- **`done` was racing the final audio tail.** DAC of an EOS tail is capped at `_dac_max_finals_per_tick` (2) per tick and the remainder stays queued, but `done` was sent for every evicted pid in the same iteration. A client that saw `done` stopped reading and its pid was dropped from the dispatch table, so that tail — including the ~93 ms every live window holds back as `_audio_stride` — was decoded and thrown away. Clipped sentence endings, sporadic, worse under concurrency (and translation mode issues many short requests, so EOS collisions are frequent). `done` is now held in `done_pending` until `runner.pending_final_pids()` no longer lists the pid; the decode also runs whenever tails are queued, including after the batch has fully drained, so nothing waits on a tick that never comes.

Deliberately unchanged: whole-segment drops at `MAX_SEGMENT_BACKLOG` (correct policy for live interpretation — stay near live speech; already logged), and the single-worker room constraint of §7.3.

Not fixed here: `test-browser-dialog.tsx` schedules playback the same way as the listener page did and has the same per-frame resampling artefact on the 1:1 path.

Not implemented (deliberately out of scope): the generic 1:1 public share for *non*-translation agents. The opt-in flag, `share_token`, public endpoint and public page all exist and are type-agnostic, but `/live/[token]` currently renders an explanatory message for a non-translation agent instead of a microphone session. Wiring it to the existing `/browser/agent/{agent_id}` flow is a small, additive follow-up.

---

## 12. Sequence (happy path)

```
Owner: create translation agent (source=English, targets=[Hindi,Tamil]), enable public link
  → backend stores agent, generates share_token
Owner: open broadcast page → POST broadcast-token → WS /translate/publish/{agent_id}
  → room created, publisher pipeline (VAD+STT) running
Listener A: open /live/{token}, pick Hindi → WS /translate/listen/{token}?lang=Hindi
  → LangWorker[Hindi] started; A added to subscribers[Hindi]
Listener B: open /live/{token}, pick Hindi → added to subscribers[Hindi] (shares worker)
Listener C: open /live/{token}, pick Tamil → LangWorker[Tamil] started
Owner speaks: STT → "Good morning" → bus
  → LangWorker[Hindi]: translate+TTS → fanout to A,B
  → LangWorker[Tamil]: translate+TTS → fanout to C
Listener C leaves → subscribers[Tamil] empty → LangWorker[Tamil] stopped
Owner ends broadcast → room torn down after grace period
```
