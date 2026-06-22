---
description: Diagnose STT, TTS, audio quality, language and voice-ID issues, latency, barge-in, and AI4Bharat GPU problems.
---

# Voice and Audio

Use this page when the call connects but the conversation is broken: the agent does not hear the caller, does not speak, speaks in the wrong language, lags, or talks over the caller. If the call itself never connects, start with [telephony.md](telephony.md).

A healthy voice path looks like this:

1. Caller audio reaches the voice server (WebSocket open).
2. STT transcribes audio into text.
3. The LLM produces a response.
4. TTS synthesizes speech and streams it back.

Each issue below is grouped by which of those stages is failing. For the full pipeline, see [../concepts/voice-pipeline.md](../concepts/voice-pipeline.md).

---

## WebSocket and session-level

### Voice WebSocket won't connect

**Symptom:** The browser "Test on Browser" dialog spins forever, or telephony reports the websocket dropped immediately. Voice server logs show no incoming session.

**Cause:** Voice server is not running, port 7860 is blocked, or the public voice URL is misconfigured.

**Fix:**

```bash
# Is the voice server up?
docker-compose ps voice_server
docker-compose logs voice_server

# Probe the health endpoint
curl http://localhost:7860/health

# Can the backend reach it?
docker-compose exec voice_server curl http://backend:8000/health

# Open the firewall if needed
sudo ufw allow 7860/tcp
```

For external callers (Vobiz, browser test from a remote machine), the public WSS URL must terminate TLS and forward to 7860. See [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md).

### Voice server CPU pinned at 100%

**Symptom:** `docker stats voicera_voice_server` shows CPU at 100% and audio gets choppy across all calls.

**Cause:** Stuck sessions, oversized audio buffers, or batch size too high for the host.

**Fix:**

```bash
# Inspect active sessions
docker-compose exec voice_server curl http://localhost:7860/health | grep sessions

# Restart to clear stuck sessions
docker-compose restart voice_server
```

Tune buffers in the voice server `.env`:

```bash
AUDIO_CHUNK_SIZE=2048
BATCH_SIZE=32
```

Lower both values on smaller hosts. See [../services/voice-server.md](../services/voice-server.md) for tuning guidance.

---

## STT (speech-to-text)

### Agent doesn't react to the caller

**Symptom:** Caller speaks; agent stays silent or only responds to the greeting. Voice server logs show no transcripts.

**Cause:** STT provider is not configured, the API key is wrong, or the local AI4Bharat STT server is down.

**Fix:**

```bash
# What STT provider is configured?
docker-compose exec voice_server cat .env | grep STT

# Cloud STT (Deepgram example): verify the key works
docker-compose exec voice_server python -c \
  "from deepgram import DeepgramClient; print('OK')"

# Local AI4Bharat STT: is it up?
docker-compose ps ai4bharat_stt_server
docker-compose exec voice_server curl http://ai4bharat_stt_server:8001/health
```

API keys for cloud STT live in Dashboard → Integrations per organization, not in `.env`. See [../services/ai4bharat-stt.md](../services/ai4bharat-stt.md) for the local server and [../guides/operator/operations.md](../guides/operator/operations.md) for the Integrations workflow.

### Wrong language detected / transcripts are gibberish

**Symptom:** Transcripts come back as nonsense, or in the wrong script (e.g., Latin instead of Devanagari).

**Cause:** The agent's language code does not match the STT provider's expected code, or the model selected does not support the chosen language.

**Fix:** Open Dashboard → Assistants → the agent, and verify the language code. Common pitfalls:

- AI4Bharat expects ISO codes like `hi`, `ta`, `te`, `bn` — not `hi-IN`.
- Deepgram uses `en-IN`, `hi`, etc.
- Multilingual models need both the language code and a model variant that supports it.

{% hint style="warning" %}
Changing language on an active agent does not retroactively re-transcribe past calls. Only new calls use the new setting.
{% endhint %}

---

## TTS (text-to-speech)

### Agent thinks but never speaks

**Symptom:** Call connects, STT picks up the caller (visible in logs), but the agent never produces audio.

**Cause:** TTS provider is misconfigured, the voice ID is invalid for the chosen provider, or the synthesis endpoint is unreachable.

**Fix:**

```bash
# What TTS provider is configured?
docker-compose exec voice_server cat .env | grep TTS

# Direct test of the TTS endpoint
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en"}'

# Local AI4Bharat TTS
docker-compose ps ai4bharat_tts_server
docker-compose logs ai4bharat_tts_server
```

### "Invalid voice ID" or "voice not found"

**Symptom:** Voice server logs show `voice_id not found` or the TTS provider returns 400 for synthesis requests.

**Cause:** The voice ID set on the agent belongs to a different provider, or the voice was removed/renamed by the provider.

**Fix:** In Dashboard → Assistants, re-select the voice from the dropdown rather than typing an ID. The dropdown is filtered to voices valid for the agent's TTS provider and language. See [../services/ai4bharat-tts.md](../services/ai4bharat-tts.md) for the supported voice catalog.

### Distorted, robotic, or chopped audio

**Symptom:** TTS audio sounds garbled, clipped, or repeatedly stutters.

**Cause:** Sample-rate mismatch between TTS output and the telephony codec, an overloaded TTS server, or packet loss on the WSS leg.

**Fix:**

1. Confirm the TTS output sample rate matches what telephony expects (usually 8 kHz µ-law for PSTN, 16 kHz for browser).
2. Check TTS server resource usage: `docker stats ai4bharat_tts_server`.
3. For local TTS, GPU thermal throttling shows up as audio that degrades after sustained load — see GPU OOM below.

---

## Latency and conversational quality

### Long pauses before the agent responds

**Symptom:** 2–5 second silence after the caller stops speaking before the agent replies.

**Cause:** End-of-utterance (VAD) timeout too high, slow LLM, cold-start on a serverless TTS, or network RTT to the cloud STT/TTS.

**Fix:**

- Tune VAD silence threshold on the voice server `.env` (lower = more responsive but more false cuts).
- Switch STT/TTS to a provider in a closer region.
- For local AI4Bharat, confirm the model is loaded into GPU memory — first call after a restart is always slow.
- Profile per stage in voice server logs (each stage logs its duration).

See [../concepts/voice-pipeline.md](../concepts/voice-pipeline.md) for the per-stage budget.

### Barge-in not working (caller can't interrupt)

**Symptom:** The caller speaks while the agent is talking and the agent keeps talking over them.

**Cause:** Barge-in is disabled on the agent, the VAD threshold on the inbound stream is too high, or the TTS playback is buffered too far ahead.

**Fix:** In Dashboard → Assistants → the agent, enable "Interruptions" (barge-in). On the voice server side, reduce the playback buffer and confirm VAD sensitivity is high enough to detect speech during TTS playback. See [../concepts/voice-pipeline.md](../concepts/voice-pipeline.md) for the barge-in design.

---

## AI4Bharat GPU

### CUDA out of memory on STT/TTS server

**Symptom:** AI4Bharat STT or TTS container logs show `CUDA out of memory` or `RuntimeError: CUDA error: out of memory`, and synthesis/transcription fails.

**Cause:** Too many concurrent sessions for the available GPU memory, model loaded at higher precision than the GPU can hold, or another process on the same GPU.

**Fix:**

```bash
# What else is on the GPU?
nvidia-smi

# Restart the offender to release memory
docker-compose restart ai4bharat_tts_server
docker-compose restart ai4bharat_stt_server
```

Concurrency limits go in the AI4Bharat service `.env` — reduce `MAX_BATCH_SIZE` or `MAX_CONCURRENT_REQUESTS`. For sizing per GPU class, see [../services/ai4bharat-stt.md](../services/ai4bharat-stt.md) and [../services/ai4bharat-tts.md](../services/ai4bharat-tts.md).

{% hint style="info" %}
Cloud-only STT/TTS deployments do not need a GPU. You only hit OOM if you opted into running AI4Bharat servers locally.
{% endhint %}

---

## Next steps

- [common-issues.md](common-issues.md) — login, dashboard, MongoDB, MinIO
- [telephony.md](telephony.md) — Vobiz, inbound/outbound, webhooks
- [deployment.md](deployment.md) — Docker, ports, TLS
- [../concepts/voice-pipeline.md](../concepts/voice-pipeline.md) — pipeline stages and budgets
- [../services/voice-server.md](../services/voice-server.md) — voice server reference
- [../services/ai4bharat-stt.md](../services/ai4bharat-stt.md) — local STT server
- [../services/ai4bharat-tts.md](../services/ai4bharat-tts.md) — local TTS server
- [../reference/websocket-api.md](../reference/websocket-api.md) — WebSocket protocol
