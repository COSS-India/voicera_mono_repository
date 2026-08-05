# Indic-Mio TTS - Integration Plan

Status: IMPLEMENTED (see docs/services/indic-mio-tts.md for the as-built doc)
Author: (draft)
Target: add `SPRINGLab/Indic-Mio` as a second on-prem TTS provider, running in
parallel to the existing AI4Bharat (Indic Parler) TTS server.

---

## 1. Goal

Make Indic-Mio a first-class, selectable TTS provider in Voicera - same way
AI4Bharat is today - with production-grade concurrency, scalability and quality.
Both engines coexist; users pick per agent.

Model facts that drive the design:

- Indic-Mio is **not** a classic acoustic model. It is a **Qwen3-0.6B LLM
  fine-tune**. TTS happens in **two stages**:
  1. **Token generation** - LLM autoregressively emits *speech tokens*.
  2. **Codec decode** - `MioCodec` (`Aratako/MioCodec-25Hz-24kHz`) turns those
     tokens into a waveform.
- Model card's **recommended serving path is vLLM** (`vllm serve
  SPRINGLab/Indic-Mio`). vLLM gives us continuous batching + paged KV-cache +
  high throughput natively - so **we do not hand-roll a batching engine** the
  way the Parler server (`ai4bharat_tts_server/`) had to.
- 22 scheduled Indian languages + English, code-mixed OK, <0.1 RTF, zero-shot
  voice cloning via speaker embeddings.
- Emotion/style tags at sentence end (`<happy>`, `<sad>`, `<angry>`, …) and
  `*word*` stress markers are part of the input contract.

Decisions locked with the team:

- **Backend for stage 1: vLLM** (recommended path).
- **GPU layout: same GPU, split memory** - `vllm --gpu-memory-utilization 0.5`
  leaves room for MioCodec on the same card (the card's own suggestion).

---

## 2. Reuse existing patterns, don't reinvent

The AI4Bharat integration already defines every seam we need. We mirror it
exactly so ops, config and frontend behave identically.

| Layer | AI4Bharat (existing) | Indic-Mio (new, mirrors it) |
|---|---|---|
| Model server dir | `ai4bharat_tts_server/` | `indic_mio_tts_server/` |
| Inference engine | hand-written continuous batching | **vLLM** (token gen) + MioCodec (decode) |
| Transport to voice server | WS, float32 PCM frames | **same WS contract** |
| Voice-server adapter | `voice_2_voice_server/services/ai4bharat/tts.py` (`IndicParlerRESTTTSService`) | `services/indic_mio/tts.py` (`IndicMioRESTTTSService`) |
| Provider factory | `voice_2_voice_server/api/services.py` `create_tts_service` → `provider_map` + `elif "AI4Bharat"` | add `"IndicMio"` entry + branch |
| Language map | `config/tts_mappings.py` `TTS_LANGUAGE_MAP["AI4Bharat"]` | add `TTS_LANGUAGE_MAP["IndicMio"]` |
| Backend agent config | `voicera_backend/app/config/default_agents.json` `tts_model` | selectable `{"name":"IndicMio","model":"indic-mio"}` |
| Frontend surfacing | `voicera_frontend/tts.json`, `lib/languageModelSupport.ts` | add provider/model rows |
| nginx LB | `location /ttslb` round-robins over `tts:8002` replicas | add `location /ttslb2` over `mio-tts:8003` replicas |
| Compose | `prod/docker-compose.prod.yml` `tts:` (replicas: 3) | add `vllm-mio:` + `mio-tts:` services |

**Key contract to preserve** (from `ai4bharat_tts_server/server.py` +
`services/ai4bharat/tts.py`):

- Client → server, one JSON per utterance:
  `{"prompt": "...", "description": "...", "language": "..."}`
- Server → client, in order:
  1. `{"type":"meta","pid":...,"sample_rate":<SR>,"dtype":"float32","channels":1}`
  2. binary frames - raw **float32 mono PCM**
  3. `{"type":"done","pid":...}`; errors as `{"type":"error","message":...}`
- The adapter converts float32 → int16, applies gain, emits Pipecat
  `TTSAudioRawFrame` / `TTSStartedFrame` / `TTSStoppedFrame`.

If the new server speaks this exact protocol, the voice-server adapter is a
near-verbatim copy of `ai4bharat/tts.py` with a different env var name.

---

## 3. Target architecture

```
 voice_2_voice_server (Pipecat)
    └─ services/indic_mio/tts.py   (IndicMioRESTTTSService)   [NEW]
         │  WS  {prompt,description,language} → meta / float32 PCM / done
         ▼
 nginx  location /ttslb2  ── round-robin over N replicas ──┐
                                                           ▼
 ┌──────────────── replica (1 GPU) ─────────────────────────────┐
 │  mio-tts  (port 8003)                                [NEW]    │
 │    • WS front, voicera contract                              │
 │    • builds chat-template prompt (+ emotion/stress tags)     │
 │    • streams token gen from vLLM over HTTP                   │
 │    • filters speech tokens (offset 151669, range 12800)     │
 │    • MioCodec.decode(chunk) → float32 PCM  (on this GPU)     │
 │         │ HTTP /v1/completions (stream)                       │
 │         ▼                                                     │
 │  vllm-mio  (port 8100)  vllm serve SPRINGLab/Indic-Mio        │
 │    --gpu-memory-utilization 0.5   ← leaves VRAM for codec    │
 │    (continuous batching, paged KV - the concurrency engine)  │
 └──────────────────────────────────────────────────────────────┘
```

- **vLLM = the concurrency/scale lever.** `--max-num-seqs` controls in-flight
  requests per replica; horizontal scale = more replicas behind `/ttslb2`.
- **mio-tts is thin and mostly I/O-bound** (proxy vLLM stream + light codec
  decode). No custom KV paging, no CUDA-graph capture, no multi-process model
  replicas. Codec decode runs off the event loop (`asyncio.to_thread` or a
  small process/thread pool) so it never blocks WS accept.
- **Same-GPU split**: vLLM reserves ~50% VRAM; MioCodec + decode buffers live in
  the remainder. Verify headroom on the actual card before raising
  `--max-num-seqs` or `--gpu-memory-utilization`.

---

## 4. The one real technical unknown - solve with the official repo

Getting **raw speech token IDs** out of a vLLM OpenAI-compatible server is the
only non-trivial bit. Speech tokens are IDs `>= 151669` (offset), valid range
`+12800` (per the model card's Approach-2). The `/v1/chat/completions` endpoint
returns *text*, not token IDs, so we need one of:

- vLLM `--return-tokens-as-token-ids` + `/v1/completions` with logprobs, parsing
  `token_id:NNN` strings from the stream, **or**
- the speech tokens detokenize to recoverable marker strings (e.g.
  `<custom_token_N>`) we map back to IDs.

**Do not design this from scratch.** The model card ships an official inference
server: `MioTTS-Inference/run_server.py` (Approach 1). It already implements the
vLLM → speech-token-extraction → MioCodec decode bridge. Plan:

1. Read `MioTTS-Inference` (`run_server.py`, plus its vLLM launch args and codec
   handling). Confirm the exact token-extraction mechanism and streaming cadence.
2. Lift that bridge logic into `indic_mio_tts_server/tts_engine.py`.
3. Wrap it in our WS server (`server.py`) so it speaks the voicera contract
   instead of Gradio/whatever run_server.py exposes.

This de-risks stage 1↔2 entirely - we adapt a working reference rather than
reverse-engineer vLLM token output.

---

## 5. New server: `indic_mio_tts_server/`

```
indic_mio_tts_server/
  Dockerfile            # vLLM + miocodec + websockets + soundfile
  requirements.txt
  server.py             # WS front - copy the shell of ai4bharat server.py
  tts_engine.py         # vLLM client + speech-token filter + MioCodec decode
  config.py             # ports, offsets, decode cadence, SR
  README.md
  .env.example
  tests/ws_smoke.py     # copy ai4bharat smoke test, point at 8003
```

`server.py` - reuse the **outer shell** of `ai4bharat_tts_server/server.py`
(argparse, `websockets.serve`, `handle_client` meta/binary/done framing). Drop
the multi-process model workers and the Parler runner. Replace the per-request
body with: build prompt → open vLLM stream → decode chunks → send PCM.

`tts_engine.py` responsibilities:

- Apply chat template (`{"role":"user","content": text}`), append emotion tag /
  keep `*stress*` markers as-is.
- Stream generation from vLLM (`temperature=0.9, top_p=0.9, max_new_tokens` per
  card).
- Filter speech tokens: keep `t` where `151669 <= t < 151669+12800`, subtract
  offset. **Constants live in `config.py`, not inline.**
- Chunked decode: MioCodec is 25 Hz → 25 tokens ≈ 1 s audio. Decode every
  ~25–50 tokens → stream ~1–2 s PCM chunks for low TTFB (mirrors Parler's
  `decode_every`). Emit float32 PCM per chunk.

### ⚠ Sample-rate discrepancy - resolve during build

The model card is internally inconsistent: prose says **44 kHz**, the codec is
named **`MioCodec-25Hz-24kHz`**, and Approach-2 code writes
`sf.write(..., 44100)`. **Confirm the codec's true output sample rate** at
implementation time and set `meta.sample_rate` to the real value. The adapter
already trusts the server's `meta.sample_rate`, so as long as the server reports
the truth, downstream resampling is correct. Do not hardcode 44100 on faith.

---

## 6. Voice-server adapter: `services/indic_mio/tts.py`

Near-verbatim copy of `services/ai4bharat/tts.py`:

- Rename class → `IndicMioRESTTTSService`.
- New env vars: `INDIC_MIO_SERVER_URL` (required), `INDIC_MIO_GAIN` (optional).
- Same `_ws_url()` http→ws conversion, same meta/binary/done consumption, same
  float32→int16 conversion.
- Emotion/style: expose an optional `emotion` param (maps to `<happy>` etc.,
  appended to prompt server-side) - additive, defaults to none.

## 7. Factory + maps + config

- `voice_2_voice_server/api/services.py`:
  - `provider_map`: add `"indic-mio" → "IndicMio"`.
  - `create_tts_service`: add `elif provider == "IndicMio":` requiring model
    `indic-mio`, instantiate `IndicMioRESTTTSService(...)`.
  - Import the new adapter at top.
- `voice_2_voice_server/config/tts_mappings.py`: add
  `TTS_LANGUAGE_MAP["IndicMio"]` covering the 22 langs + English.
- `voicera_backend/app/config/default_agents.json`: make
  `{"name":"IndicMio","model":"indic-mio"}` a valid selection (no default flip
  unless requested).
- `voicera_frontend/tts.json` + `lib/languageModelSupport.ts`: add provider
  `indic-mio` → model `indic-mio` per supported language, mirroring the
  `ai4bharat` gate at `languageModelSupport.ts:121`.

## 8. Deployment

Add to `prod/docker-compose.prod.yml` **and** `deploy/compose/docker-compose.prod.yml`:

- `vllm-mio` service: `vllm/vllm-openai` image, `vllm serve SPRINGLab/Indic-Mio
  --gpu-memory-utilization 0.5 --port 8100`, GPU reservation, HF cache mount.
- `mio-tts` service: builds `../indic_mio_tts_server`, `EXPOSE 8003`,
  `env INDIC_MIO_VLLM_URL=http://vllm-mio:8100`, same GPU reservation (shares
  the card), `ipc: host` / `memlock` / `stack` ulimits like `tts:`,
  `deploy.replicas` (start at 1–2, raise after VRAM observed).
- nginx (`prod/nginx.prod.conf`, `deploy/nginx/nginx.conf`): add

  ```nginx
  location /ttslb2 {
      set $mio_up mio-tts:8003;
      proxy_pass http://$mio_up;
      # WS upgrade headers + docker resolver 127.0.0.11, same as /ttslb
  }
  ```

- Voice server env: `INDIC_MIO_SERVER_URL=http://nginx:8080/ttslb2/`
  (route via nginx, **not** `mio-tts:8003` directly - same aiohttp DNS-cache
  pinning gotcha documented for AI4Bharat).

## 9. Docs & env

- `docs/services/indic-mio-tts.md` - protocol, GPU/VRAM, env vars, emotion tags
  (mirror `docs/services/ai4bharat-tts.md`).
- `.env.example` additions in `indic_mio_tts_server/` and
  `voice_2_voice_server/`: `INDIC_MIO_SERVER_URL`, `INDIC_MIO_GAIN`,
  `INDIC_MIO_VLLM_URL`. Use the **code-read** names, not decorative ones (the
  AI4Bharat `.env.example` had a dead `AI4BHARAT_TTS_URL` - don't repeat that).

---

## 10. Build order (each step independently testable)

1. **Server, standalone.** Build `indic_mio_tts_server/` against the
   MioTTS-Inference reference. Verify with `tests/ws_smoke.py` → get a WAV.
   Confirm real sample rate here.
2. **Adapter + factory.** Wire `IndicMioRESTTTSService` into the factory; select
   it in one dev agent; run a call through the voice pipeline.
3. **Maps + frontend.** Language map + `tts.json` + gate; verify the provider
   shows and works per language.
4. **Deploy.** Compose services + nginx `/ttslb2` + env; scale replicas after
   observing VRAM under load.

## 11. Open items to confirm before/while building

- [ ] True output **sample rate** of MioCodec (44.1k vs 24k) - §5.
- [ ] Exact **speech-token extraction** mechanism from vLLM - read
      MioTTS-Inference - §4.
- [ ] `miocodec` package install/deps in the Dockerfile (GPU wheel, torch ver).
- [ ] **Speaker/voice-cloning**: card mentions speaker embeddings in the codec
      for zero-shot cloning - decide whether to expose now or defer.
- [ ] VRAM budget on the target card for vLLM 0.5 + codec + `--max-num-seqs`.
- [ ] Emotion tag surfacing in frontend (defer to
      [[platform-key-frontend-ux-todo]] batch or ship minimal now?).
```
