---
description: Diagnosing silence, one-way audio, distortion, and interruption problems.
---

# Voice and audio

Audio problems are usually one of three things: the pipeline never started, a sample rate does not match, or a behaviour setting is doing exactly what you told it to.

## No audio at all

### The call connects, then silence

Work outward from the runtime.

```bash
docker compose logs runtime --tail 100
```

| What the logs show | Cause |
| --- | --- |
| No `/answer` request | The provider never reached you. See [Telephony](telephony.md). |
| `/answer` served, no WebSocket | The provider could not open the WSS URL — usually TLS or a proxy that does not upgrade. See [Public voice URLs](../deployment/public-voice-urls.md). |
| WebSocket opened, then an exception | The pipeline failed to build. Read on. |

### The pipeline fails to build

The runtime builds speech-to-text, text-to-speech, and the language model before any audio flows. Any one of them failing ends the call immediately. Common causes:

* **Missing credentials** — the organisation has no `ProviderAuth` for a provider the agent references.
* **An unregistered provider** in the agent's `config.models`.
* **A missing Pipecat extra** — vendor packages are imported inside their creator functions, so a missing dependency surfaces here, not at startup.

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/auth/configured
```

### The agent never greets

The greeting comes from the agent's prompts. An empty greeting means the agent waits for the caller to speak first — which sounds identical to a broken call.

## One-way audio

| You hear the agent, it does not hear you | It hears you, you hear nothing |
| --- | --- |
| Inbound audio is not reaching speech-to-text. Check the STT provider's credentials and that its model supports the agent's language. | Text-to-speech is failing or the audio format is being rejected. Check TTS credentials and, if self-hosting, the format negotiation. |

For a browser session, also check that the page has microphone permission and is served over HTTPS or `localhost` — browsers block `getUserMedia` elsewhere.

## Distorted, robotic, or chipmunk audio

Almost always a **sample-rate mismatch**. Voicera uses two different rates on purpose:

| Path | Variable | Default |
| --- | --- | --- |
| Telephony | `SAMPLE_RATE` | `8000` |
| Browser WebSocket | `WEBSOCKET_SAMPLE_RATE` | `16000` |

Audio played at the wrong rate sounds too fast and high (played faster than recorded) or too slow and deep. If you changed either value, change it back — `8000` is what telephony carriers deliver, and raising it does not add detail that was never there.

Self-hosted TTS adds a second possibility: the model returned a format the runtime did not expect. See [TTS models](../../developer/model-server/tts-models.md).

## Interruption problems

### The agent will not stop talking

Barge-in is gated by `interruption_min_words` — the caller must produce at least that many words before the agent yields. Set too high, the agent talks over people; set to 1, background noise interrupts it.

### The agent interrupts itself, or cuts off its greeting

`ignore_user_speech_before_greeting` exists for exactly this: on noisy lines, the greeting's own audio or line noise can register as speech. Set it `true` to protect the greeting.

See [Agent configuration](../../developer/reference/agent-configuration.md) for the full behaviour table.

## Hold messages never play

`hold_messages` play when a turn takes longer than `hold_message_timeout_seconds`. If they never fire, either the list is empty, the timeout is longer than your slowest turn, or nothing is actually slow — which is the good outcome.

## The agent hangs up unexpectedly

| Setting | Effect |
| --- | --- |
| `user_silence_hangup_seconds` | Ends the call after that much caller silence |
| `automatic_call_ending` | Lets the agent end the call when the conversation is done |
| `user_online_detection_*` | Prompts the caller, then closes after `user_online_detection_repeats` unanswered prompts |

{% hint style="warning" %}
`call_timeout_seconds` is accepted and stored by the API but **no code in the runtime reads it**. Calls are not capped by it. Use `user_silence_hangup_seconds` or `automatic_call_ending` instead.
{% endhint %}

## Latency

Expected budget is under a second per turn. When it is worse:

| Suspect | Check |
| --- | --- |
| The language model | Largest single contributor. Try a smaller or faster model. |
| Provider region | A vendor endpoint far from your server adds a round trip per turn. |
| Self-hosted models | A cold model is slow on the first call. Check GPU utilisation — see [Running on GPUs](../../developer/model-server/gpu-operations.md). |
| Knowledge base | `context` mode retrieves on every turn; `tool` mode only when the model asks. |

## No transcript or recording

Browser sessions are logged too. Check the runtime log for `Registered web call call_id=`; if it is absent, the API call to `POST /api/v1/calls/web` failed and the session ran without a call log.

For a **telephony** call, artifacts are written at call end. If they are missing:

```bash
docker compose logs runtime | grep -i minio
docker compose ps minio
```

Then confirm the call log has the URIs:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/calls/$CALL_ID
```

## Where next

* [Voice pipeline](../concepts/voice-pipeline.md)
* [Agent configuration](../../developer/reference/agent-configuration.md)
* [Telephony](telephony.md)
