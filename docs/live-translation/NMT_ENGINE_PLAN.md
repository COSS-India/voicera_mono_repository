# Live Translation — Pluggable Translation Engine (LLM | NMT)

**Status:** Implemented (see §11 checklist). Konkani confirmed as `gom` on the hosted model.
**Goal:** A translation agent picks **one** translation engine — streaming **LLM** (today) or on-prem **NMT** (Triton / IndicTrans2). The unselected engine is never constructed, never imported at call time, never credential-checked. Each path is tuned for its own shape; neither pays for the other's assumptions.
**Guiding constraint:** Same as the original design — additive, minimal, non-breaking. `LangWorker`, `_Listener`, the fan-out and the TTS stage are **untouched**. All new logic lands behind one narrow interface plus one new service module.

Companion to [TECHNICAL.md](TECHNICAL.md). Anchors below are load-bearing.

---

## 1. Measured facts about the NMT backend

Probed directly against the hosted Triton endpoint (2026-08-25). The address is private and lives only in the gitignored `.env.translate` — never commit it. These numbers drive every design choice below.

**Model config** (`GET /v2/models/nmt/config`):

| Property | Value | Consequence |
|---|---|---|
| `backend` | `python` (IndicTrans2 `paragraphs_batch_translate__multilingual`) | Server splits paragraphs into sentences itself → we send whole segments, not sentences |
| `max_batch_size` | 512 | One request can carry every language of every concurrent room |
| `dynamic_batching.max_queue_delay_microseconds` | **0** | Triton batches only what is *already* queued → **client-side coalescing is what actually creates batches** |
| `instance_group` | 1 × KIND_GPU, count 1 | Single GPU worker; throughput comes from batch width, not concurrency |
| Health | `GET /v2/health/ready` → 200, `GET /v2/models/nmt/ready` → 200 | Cheap pre-flight check at presenter connect |

**Latency** (incl. ~50 ms RTT to ap-south-1):

| Request | Wall clock |
|---|---|
| 1 row, 1 short sentence | **0.20 s** |
| 3 rows (same text → hi, ta, ml) | **0.38 s** |
| 1 row, 4-sentence paragraph (260 chars) | **0.43 s** |

Compare: the LLM path's `llm_ttft` is the *first token* and a full segment completes well after. NMT returns the **entire segment for every language in ~0.2–0.4 s**, which is below the LLM's typical time-to-first-token. So NMT needs no streaming to win; chasing sentence-level streaming on the NMT path would only add rows and lose context.

**Batching semantics** — verified:
```jsonc
// one request, three target languages, one GPU pass
"INPUT_TEXT":        ["<same text>", "<same text>", "<same text>"]
"INPUT_LANGUAGE_ID": ["en", "en", "en"]
"OUTPUT_LANGUAGE_ID":["hi", "ta", "ml"]     // shape [3,1] on every input
// → OUTPUT_TEXT shape [3,1], index-aligned with the request
```

**Failure semantics** — verified, and the reason for §4.4:

| Input | Result |
|---|---|
| One unsupported pair in a 3-row batch (`en-kok`) | **The whole batch fails** with a single `{"error": ...}` — the other two rows are lost |
| Blank text | Whole batch fails (`AssertionError: blank lines are not allowed`) |
| `en → en` | Returns **Hindi** (garbage). Same-language pairs must never be sent |

Two hard rules follow: **validate every row before it enters a batch**, and **never let one poisoned row mute other rooms** (split-retry).

**Supported target codes** — probed one by one, `en → X`:

```
as bn brx doi gom gu hi kn ks mai ml mni mr ne or pa sa sat sd ta te ur   ✅
kok ✗ (Konkani is gom, NOT kok)   bhb ✗ (Bhili)   en→en ✗ (garbage)
```
`X → en` and Indic→Indic (`hi → ta`) both verified working.

⚠️ `TTS_LANGUAGE_MAP["AI4Bharat"]["Konkani"] == "kok"` (`config/tts_mappings.py:158`). Copying that map for NMT would silently kill every batch containing Konkani. NMT needs **its own** map.

---

## 2. Why the two engines cannot share one code path

| | LLM engine | NMT engine |
|---|---|---|
| Unit of work | token → sentence, streamed | whole segment, one shot |
| Fan-out cost | **N languages = N concurrent API calls** | **N languages = N rows in 1 request** |
| Chunking for TTS | on the *streamed output*, as it arrives | on the *returned output*, same function |
| Credential | org OpenAI key / platform fallback | none (on-prem URL) |
| Style/domain prompt | honoured (`system_prompt`) | **no effect** — must be hidden in UI |
| Timeout meaning | per-token inactivity (`TRANSLATION_LLM_TIMEOUT_SECS`) | per-request deadline (`NMT_TIMEOUT_SECS`) |
| Retry | once, only before first emit | split-retry a failed batch, once |
| Scaling lever | provider concurrency | **batch width** |

Forcing one implementation to serve both is where efficiency leaks: sentence-splitting the source for NMT costs context *and* rows; per-language calls on NMT waste the 512-wide batch; batching the LLM path would destroy streaming. Hence: one narrow interface, two implementations that share **nothing but the chunker**.

---

## 3. Design

### 3.1 Shape

```
presenter → STT → segment text
                     │  room.translate_stream(text, lang)   ← unchanged call site (translation_room.py:420)
                     ▼
              room._engine  ─┬─ LlmTranslator  → OpenAI stream → sentences (as today)
                             │
                             └─ NmtTranslator  → submit(text, src, tgt) ─┐
                                                                          │
   ┌──────────────────────── process-global micro-batcher ────────────────┘
   │  8 ms coalescing window, ≤64 rows, ≤4 in-flight, shared aiohttp session
   │  rows from EVERY room and EVERY language merge into one Triton request
   └──▶ POST /v2/models/nmt/infer  →  index-aligned OUTPUT_TEXT  →  resolve futures
                             │
                             ▼  _next_chunk_end() on the translated text (shared with LLM path)
                        sentences → existing synth queue → TTS → fan-out   (unchanged)
```

**The key optimisation:** the batcher is *transparent*. Every `LangWorker` still calls translate independently — the fan-in happens one layer down, in the client. A 5-language room fires 5 `submit()` calls within microseconds of each other; they land in the same 8 ms window and leave as **one** GPU pass. Cross-room merging comes free: 20 concurrent broadcasts × 5 languages = 100 rows → 2 requests, not 100.

This is why the batcher is process-global rather than per-room, and why `LangWorker` needs no change at all.

### 3.2 Whole segment in, chunk the output

IndicTrans2 splits paragraphs into sentences server-side (`engine.py:split_sentences`, confirmed in the blank-text traceback). So the NMT engine sends **one row per (segment, language)** — best context, fewest rows — and then runs the *existing* `_next_chunk_end()` (`translation_room.py:127`) over the returned text to feed the TTS stage in exactly the same chunk shapes the LLM path produces. Downstream code cannot tell the engines apart.

Guard: segments longer than `NMT_MAX_SEGMENT_CHARS` (default 1200) are split at sentence boundaries into ≤k rows **submitted in the same batch** and re-joined in order. A VAD segment (`TRANSLATION_VAD_STOP_SECS=0.4`) essentially never reaches this; it exists so a pathological run-on cannot blow the request.

### 3.3 Engine interface

New, tiny, in `voice_2_voice_server/api/translation_engines.py`:

```python
class TranslationEngine(Protocol):
    name: str

    async def prepare(self) -> Optional[str]:
        """Pre-flight at presenter connect. Returns an error string, or None if ready."""

    def unsupported(self, target_language: str) -> Optional[str]:
        """Error string if this engine cannot serve the language; None if it can."""

    def stream(self, text: str, target_language: str, on_token=None) -> AsyncIterator[str]:
        """Yield TTS-ready chunks, in order."""

    async def aclose(self) -> None: ...
```

`TranslationRoom.translate_stream` (`translation_room.py:712-804`) becomes a two-line delegate to `self._engine.stream(...)`. Its current body moves verbatim into `LlmTranslator` — **no behaviour change on the LLM path**, so the existing deployment is bit-for-bit identical when `translation_engine` is absent or `"llm"`.

### 3.4 Selection, and keeping the LLM out of the picture

`_derive_from_config` (`translation_room.py:633`) reads `config["translation_engine"]`, default `"llm"`, and constructs exactly one engine. Then, concretely:

1. **`run_publisher`'s hard OpenAI gate (`translation_room.py:898-904`) becomes engine-dispatched.** Today it does a blocking `fetch_integration_key(org, "OpenAI")` and closes 4402 if missing. On the NMT path it must not run at all — instead `await engine.prepare()` does `GET {NMT_SERVER_URL}/v2/models/{NMT_MODEL_NAME}/ready` with a 2 s timeout and closes 4402 with `"translation backend not ready"` on failure. An NMT-only deployment then needs **no OpenAI key anywhere**.
2. `get_openai_client` / `_openai` / `_model` / `TRANSLATION_MODEL` / `TRANSLATION_LLM_TIMEOUT_SECS` move into `LlmTranslator` and are unreachable from the NMT path. `from openai import AsyncOpenAI` stays where it already is — inside the method — so the import never executes.
3. `system_prompt` ("Translation Guidance") is consumed only by `LlmTranslator`. The UI hides it for NMT (§7) rather than letting a user write guidance that is silently dropped.
4. `LangWorker.start` (`translation_room.py:352`) calls `room._engine.unsupported(self.language)` before acquiring the call slot, and refuses with a specific reason (WS 1013 → `"language not supported by NMT engine"`) instead of starting a worker that will emit silence.

---

## 4. The NMT client — `voice_2_voice_server/services/nmt/triton_nmt.py`

New package `services/nmt/` (`__init__.py`, `triton_nmt.py`), following the shape of `services/ai4bharat/tts.py` (env-driven URL, shared `aiohttp` session, bounded timeouts).

### 4.1 Shared session

One `aiohttp.ClientSession` per process, lazily created on the running loop:
```python
connector = aiohttp.TCPConnector(limit=NMT_MAX_INFLIGHT * 2, ttl_dns_cache=300, keepalive_timeout=60)
timeout   = aiohttp.ClientTimeout(total=None, connect=2, sock_read=NMT_TIMEOUT_SECS)
```
Keep-alive matters: at 0.2 s per request a fresh TCP+DNS handshake per call would be a third of the budget.

### 4.2 Coalescer

```
submit(text, src, tgt) -> Future        # per row, resolved with translated text

collector loop:
    row = await queue.get()                  # block until there is work (no idle polling)
    batch = [row]
    deadline = now + NMT_BATCH_WINDOW_MS
    while len(batch) < NMT_MAX_BATCH and now < deadline:
        drain queue without blocking; if empty, await with the remaining window
    dispatch(batch)                          # under a semaphore of NMT_MAX_INFLIGHT
```
`dispatch` is a task, not an await — the collector immediately opens the next window, so a slow GPU pass never blocks accumulation.

**Window sizing.** 8 ms is chosen deliberately: it is ~2 % of the 400 ms translation budget (invisible), and it is wider than the microseconds between a room's N `submit()` calls, so a room's languages *always* batch together. Raising it to 20–30 ms buys cross-room width under heavy load at a still-invisible cost; it is env-tunable for exactly that reason.

### 4.3 Request / response

Build the three `[N,1]` BYTES inputs; map `OUTPUT_TEXT.data[i] → futures[i]`. Reject a response whose length ≠ batch length (fail all rows with a clear error rather than mis-assign translations to languages — a silent index shift would send Tamil audio to Hindi listeners).

### 4.4 Poison-row containment

Because one bad row kills the batch (§1), two layers:

- **Prevention:** `submit()` refuses before enqueue — blank/whitespace text, `src == tgt` (returns the source unchanged, no request), unmapped language. These are the three failure modes actually observed.
- **Containment:** on a batch-level failure with `len(batch) > 1`, re-submit each row **individually, once**, with a `no_split_retry` flag. One tainted row then fails alone; every other room's language is unaffected. `len(batch) == 1` → resolve that future with the error. Log `nmt: batch of N failed (<error>), split-retrying`.

Without this, a single malformed segment in one broadcast would mute every other live broadcast on the box for that segment.

### 4.5 Language map — `voice_2_voice_server/config/nmt_mappings.py`

Display name → NMT code, built from the §1 probe. **Not** derived from `TTS_LANGUAGE_MAP` (Konkani = `gom`, not `kok`; Bhili absent). `NMT_SUPPORTED = set(NMT_LANGUAGE_MAP.values())` backs `unsupported()` and the frontend's disabled-option list.

---

## 5. Concurrency and capacity

Per presenter: one segment per ~3–5 s of speech (VAD stop 0.4 s + speaking cadence). Rows/s ≈ `presenters × languages / 4`.

| Concurrent broadcasts | Languages each | Rows/s | Requests/s @ 8 ms window | GPU time/s (0.4 s per pass) |
|---|---|---|---|---|
| 5 | 3 | ~4 | ~1–2 | ~0.5 s |
| 20 | 5 | ~25 | ~3–5 | ~1.6 s ⚠ needs ≥2 in-flight |
| 50 | 5 | ~63 | ~4–8 (batches of ~16) | saturating; raise window to 25 ms |

Read: throughput is **batch width × in-flight**, not request count. `NMT_MAX_INFLIGHT=4` with a single GPU instance keeps the GPU fed while the queue absorbs bursts. When the box saturates, widening `NMT_BATCH_WINDOW_MS` (more rows per pass) is the correct lever — not more in-flight requests.

Back-pressure is already handled upstream and unchanged: `MAX_SEGMENT_BACKLOG=50` drops the oldest segment, `MAX_SENTENCE_BACKLOG=8` back-pressures translation, and each listener's queue sheds its own frames. The NMT queue itself is bounded (`NMT_MAX_QUEUE`, default 2048); on overflow `submit()` fails fast so the caller drops that segment rather than growing memory.

Note the existing single-worker constraint (`VOICE_SERVER_NUM_WORKERS=1`, `docker-compose.translate.yml:66`) is unchanged — batching is per process, and one process already hosts every room.

---

## 6. Backend changes (`voicera_backend`)

Minimal — the backend stays out of the language business.

| File | Change |
|---|---|
| `app/services/agent_service.py:74` `_validate_agent_config_for_mode` | In the `translation` branch: `translation_engine` must be `"llm"` or `"nmt"` (absent → `"llm"`). **Do not** validate languages here — that would duplicate the NMT map in a second service and drift. |
| `app/models/schemas.py` | No change needed (`agent_config` is a free-form dict); optionally document the field. |
| Public projection (`agent_service.py:~434`) | No change — the engine is not a listener-visible fact. |

Existing translation agents have no `translation_engine` key → default `"llm"` → unchanged behaviour.

---

## 7. Frontend changes (`voicera_frontend`)

| File | Change |
|---|---|
| `lib/api.ts:~382` (`AgentConfig`) | `translation_engine?: "llm" \| "nmt"` |
| `lib/nmt-languages.ts` *(new)* | `NMT_SUPPORTED_LANGUAGES` — the §1 list, one flat set. Single source of truth for the UI. |
| `app/(dashboard)/assistants/page.tsx:~1202` (create wizard, translation payload) | Emit `translation_engine` |
| `app/(dashboard)/assistants/[id]/page.tsx:~1029` and `~1177` (both translation payload builders) | Emit `translation_engine`; load it at `~674` alongside `interaction_mode` |
| Audio step, next to `TranslationLanguagesSection` (`[id]/page.tsx:1753`, `page.tsx:2109`) | Segmented control: **NMT** (`fast · on-prem · 23 Indic languages`) vs **LLM** (`context-aware · follows guidance · needs OpenAI key`) |
| `TranslationLanguagesSection` | When `nmt`, pass NMT-unsupported targets through the **existing** `unsupportedLanguages` prop — no new component, the disabled-with-reason UI already exists |
| Source language | When `nmt`, warn if the presenter language is outside `NMT_SUPPORTED_LANGUAGES` |
| Prompt block (`[id]/page.tsx:1815-1828`, "Translation Guidance") | Hidden when `nmt` — it has no effect on NMT and showing it invites silent misconfiguration |

The wizard step list (`page.tsx:242-246`) already excludes `llm` for translation agents, so nothing there changes.

---

## 8. Configuration

New env, `voice_2_voice_server/.env.example` (+ the `voice_server` block of `deploy/compose/docker-compose.translate.yml`, which already pins translation-specific overrides):

```bash
# --- NMT translation engine (agent_config.translation_engine = "nmt") ---
# NMT_SERVER_URL=http://<nmt-host>:8000        # Triton root; REQUIRED for nmt; private — gitignored .env only
# NMT_MODEL_NAME=nmt
# NMT_BATCH_WINDOW_MS=8        # coalescing window; widen (20-30) before adding in-flight
# NMT_MAX_BATCH=64             # rows per request; server cap is 512
# NMT_MAX_INFLIGHT=4           # concurrent requests; 1 GPU instance behind it
# NMT_TIMEOUT_SECS=8           # per-request read deadline (measured p100 ~0.45 s)
# NMT_MAX_QUEUE=2048           # rows waiting to batch; overflow fails fast
# NMT_MAX_SEGMENT_CHARS=1200   # above this, split at sentence bounds into same-batch rows
```

`TRANSLATION_MODEL` and `TRANSLATION_LLM_TIMEOUT_SECS` keep their meaning and apply **only** to the LLM engine.

---

## 9. Observability

`_SegmentTiming.summary()` (`translation_room.py:305`) hardcodes `llm_ttft=`. Change to `{engine}_ttft=` so a log line states which engine produced it. For NMT, `note_token()` fires when the row resolves, so `nmt_ttft` reads as full translation latency — directly comparable to the LLM's time-to-first-token.

Add one batcher log line per 100 batches (not per batch): `nmt: 100 batches, avg rows=12.4, avg latency=0.31s, split-retries=2`. That is the number that tells you whether to widen the window.

---

## 10. Non-breaking guarantees

1. `translation_engine` absent → `"llm"` → the current code path, moved but not modified.
2. No change to `LangWorker`, `_Listener`, the synth stage, the serializer, the fan-out, or any listener-facing WS event.
3. No change to the 1:1 voice pipeline, `bot.py`, or `create_llm_service`.
4. `services/nmt/` is only imported when an NMT room is constructed.
5. An NMT-only deployment needs no OpenAI credential; an LLM-only deployment needs no `NMT_SERVER_URL`.

---

## 11. Implementation checklist

**voice_2_voice_server**
- [x] `config/nmt_mappings.py` — display → NMT code (Konkani = `gom`), `NMT_SUPPORTED_CODES`, `to_nmt_code()`
- [x] `services/nmt/__init__.py`, `services/nmt/triton_nmt.py` — shared session, coalescer, split-retry, `ready()` probe
- [x] `utils/text_chunking.py` — chunker moved out of the room (shared by both engines, no import cycle)
- [x] `api/translation_engines.py` — `TranslationEngine` protocol; `LlmTranslator` (verbatim move of `translate_stream`); `NmtTranslator`; `create_translation_engine` factory
- [x] `api/translation_room.py` — engine construction in `_derive_from_config`; `_consume` delegates to `engine.stream`; `run_publisher` pre-flight via `engine.prepare()`; `LangWorker.start` calls `engine.unsupported()`; `summary()` engine tag
- [x] `.env.example` — the block in §8

**voicera_backend**
- [x] `_validate_agent_config_for_mode` — `translation_engine` enum check

**voicera_frontend**
- [x] `lib/api.ts` field + `TranslationEngine` type, `lib/nmt-languages.ts`, engine selector (both create + edit), unsupported-target wiring, prompt hidden for NMT, payload in all three builders, edit-page original backfill + change-detection dep

**deploy**
- [x] `docker-compose.translate.yml` — `NMT_SERVER_URL` (+ overrides) on `voice_server`

---

## 12. Test plan

| # | Test | Expect |
|---|---|---|
| 1 | Existing LLM agent, no `translation_engine` | Byte-identical behaviour; `llm_ttft` in logs |
| 2 | NMT agent, 3 targets, one presenter | **One** Triton request per segment (3 rows); `nmt_ttft` ≈ 0.2–0.4 s |
| 3 | NMT agent with `OPENAI_API_KEY` unset and no org integration | Broadcast starts normally; zero OpenAI calls |
| 4 | Two NMT rooms speaking simultaneously | Rows from both merge into shared batches (log `avg rows`) |
| 5 | Konkani target on NMT | Rejected at worker start with a clear reason — **not** a failed batch |
| 6 | Forced bad row mid-batch (blank text injected) | Split-retry; other rooms' languages unaffected |
| 7 | `NMT_SERVER_URL` pointing at a dead host | Presenter closed 4402 at connect, not silence mid-talk |
| 8 | Source == target language | Source text passed through, no request issued |
| 9 | 20 rooms × 5 languages, 10 min soak | No queue growth; listener backlog stable; GPU not saturated |
| 10 | Switch an agent llm ↔ nmt and re-broadcast | New engine used; no stale client or key cached |

---

## 13. Rejected alternatives

- **Room-level fan-in** (room collects its languages, issues one request). Simpler-looking, but batches only *within* a room — 20 rooms still means 20 requests. The transparent process-global batcher is the same amount of code and merges across rooms.
- **Sentence-level source splitting for NMT.** Costs context (IndicTrans2 already splits internally), multiplies rows, and buys streaming that a 0.3 s round-trip does not need.
- **Reusing Triton's dynamic batching alone.** `max_queue_delay_microseconds: 0` means it batches only what is coincidentally in flight. Client-side coalescing is what creates the width.
- **One engine with an `if` per call site.** Spreads engine-specific timeouts, retry policy and credential handling through `LangWorker` — precisely the mismatching the goal forbids.
- **Deriving NMT codes from `TTS_LANGUAGE_MAP`.** Konkani `kok` vs `gom` would fail entire batches at runtime; the maps serve different backends and must stay separate.
