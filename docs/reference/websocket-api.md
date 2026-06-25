---
description: WebSocket protocol used by the voice server to stream audio between telephony or browser clients and the Pipecat pipeline.
---

# WebSocket API Reference

The VoicEra voice server accepts a single bidirectional WebSocket per call. Telephony providers (Vobiz, Plivo) and the in-browser test client all speak the same JSON-framed protocol, with a small set of message types in each direction.

{% hint style="info" %}
**Code reference:** `voice_2_voice_server/api/server.py` (`websocket_endpoint`) and `voice_2_voice_server/serializer/vobiz_serializer.py`. For the browser test, see `voice_2_voice_server/docs/talk-on-browser-feature.md`.
{% endhint %}

## Endpoints

| Path | Used by |
|------|---------|
| `/agent/{agent_id}` | Vobiz inbound/outbound and browser test |
| `/plivo/agent/{agent_id}` | Plivo inbound/outbound |

The `agent_id` is the UUID of a VoicEra agent. The voice server looks up the agent's configuration (LLM, STT, TTS, prompt, language) from the backend when the WebSocket opens.

## Transport

- Protocol: WebSocket over TCP
- TLS: required in production (`wss://`)
- Default port: `7860`
- Public URL: configured via `JOHNAIC_WEBSOCKET_URL` (see [environment-variables.md](environment-variables.md))

```javascript
const ws = new WebSocket("wss://voice.yourdomain.com/agent/<agent_id>");
```

## Audio format

All audio frames carry **16-bit linear PCM** (L16) at **16 kHz mono**, base64-encoded inside JSON.

| Property | Value |
|----------|-------|
| Encoding | L16 (PCM signed 16-bit, little-endian) |
| Sample rate | 16000 Hz |
| Channels | 1 (mono) |
| Frame size | ~20 ms chunks typical |
| Transport | Base64 string in JSON |

{% hint style="warning" %}
**μ-law is 8 kHz only.** The Vobiz serializer (`vobiz_serializer.py`) supports L16 at 16 kHz; using μ-law forces the pipeline down to 8 kHz, degrading STT and TTS quality.
{% endhint %}

---

## Message format

Every message is a JSON object with an `event` field at the top level. Some events carry a nested object named after the event (e.g. `media` carries a `media` object).

```json
{
  "event": "<event_name>",
  "<event_name>": { ... }
}
```

## Client to server messages

### `start`

**Required first message.** Signals the start of a call and binds the WebSocket to a call/stream id.

```json
{
  "event": "start",
  "start": {
    "callSid": "vobiz-call-id",
    "streamSid": "vobiz-stream-id"
  }
}
```

`callId` / `streamId` aliases are also accepted.

### `media`

Uplink audio frames from the caller.

```json
{
  "event": "media",
  "media": {
    "contentType": "audio/x-l16",
    "sampleRate": 16000,
    "payload": "<base64 PCM>"
  }
}
```

### `stop`

Sent by the provider on hangup. The pipeline drains, the recording is finalised in MinIO, and the meeting record is updated.

```json
{ "event": "stop", "stop": { "callSid": "vobiz-call-id" } }
```

### `mark` (optional)

Used by some providers for synchronisation marks.

```json
{ "event": "mark", "mark": { "name": "user-checkpoint" } }
```

## Server to client messages

### `media`

Downlink audio frames synthesised by TTS.

```json
{
  "event": "media",
  "media": {
    "contentType": "audio/x-l16",
    "sampleRate": 16000,
    "payload": "<base64 PCM>"
  }
}
```

### `clear`

Tells the client to flush any buffered downlink audio (used when the user barges in over the agent).

```json
{ "event": "clear" }
```

### `mark`

Echo of a previously-issued `mark` from the client, used to confirm playback boundaries.

---

## Call lifecycle

```
+--------+                              +--------------+              +---------+
| Caller |                              | Voice server |              | Backend |
+--------+                              +--------------+              +---------+
    |                                          |                          |
    | dials number bound to Vobiz application  |                          |
    |----------------------------------------->|                          |
    |                                          |                          |
    |   Vobiz HTTP GET /answer?agent_id=...    |                          |
    |   <XML pointing to wss://.../agent/{id}> |                          |
    |                                          |                          |
    |       WebSocket open                     |                          |
    |<---------------------------------------->|                          |
    |                                          | fetch_agent_config       |
    |                                          |------------------------->|
    |                                          |<------- 200 OK ----------|
    |                                          |                          |
    |        { "event": "start", ... }         |                          |
    |----------------------------------------->|                          |
    |                                          |                          |
    |  { "event": "media", payload: <pcm> }    |                          |
    |----------------------------------------->| STT -> LLM -> TTS        |
    |  { "event": "media", payload: <pcm> }    |                          |
    |<-----------------------------------------|                          |
    |              ... (loop)                  |                          |
    |                                          |                          |
    |        { "event": "stop" }               |                          |
    |----------------------------------------->| log_meeting              |
    |                                          |------------------------->|
    |        WebSocket close                   |                          |
    |<---------------------------------------->|                          |
```

### Stages

1. **Connect** — Provider opens the WebSocket; voice server accepts and loads the agent config from the backend.
2. **Start** — First JSON frame must be `event: "start"` carrying `callSid` and `streamSid`.
3. **Stream** — Bidirectional `media` frames. The Pipecat pipeline runs STT, the LLM, and TTS continuously.
4. **Hangup** — `event: "stop"` (or socket close) ends the session; meeting metadata is persisted.

---

## Example: browser test client

The browser test ships with the voice server (`voice_2_voice_server/docs/talk-on-browser-feature.md`) and demonstrates the full protocol.

{% tabs %}
{% tab title="javascript" %}
```javascript
const ws = new WebSocket(`wss://voice.example.com/agent/${agentId}`);

ws.onopen = () => {
  ws.send(JSON.stringify({
    event: "start",
    start: { callSid: crypto.randomUUID(), streamSid: crypto.randomUUID() },
  }));
};

ws.onmessage = (evt) => {
  const msg = JSON.parse(evt.data);
  if (msg.event === "media") {
    playPcm16k(atob(msg.media.payload));
  } else if (msg.event === "clear") {
    flushPlaybackBuffer();
  }
};

function sendMic(pcmBytes) {
  ws.send(JSON.stringify({
    event: "media",
    media: {
      contentType: "audio/x-l16",
      sampleRate: 16000,
      payload: btoa(String.fromCharCode(...new Uint8Array(pcmBytes))),
    },
  }));
}
```
{% endtab %}

{% tab title="python" %}
```python
import asyncio, base64, json, uuid
import aiohttp

async def run(agent_id: str, pcm_chunks):
    url = f"wss://voice.example.com/agent/{agent_id}"
    async with aiohttp.ClientSession() as session:
        async with session.ws_connect(url) as ws:
            await ws.send_json({
                "event": "start",
                "start": {
                    "callSid": str(uuid.uuid4()),
                    "streamSid": str(uuid.uuid4()),
                },
            })

            async def send_audio():
                for chunk in pcm_chunks:
                    await ws.send_json({
                        "event": "media",
                        "media": {
                            "contentType": "audio/x-l16",
                            "sampleRate": 16000,
                            "payload": base64.b64encode(chunk).decode(),
                        },
                    })

            async def recv_audio():
                async for msg in ws:
                    data = json.loads(msg.data)
                    if data["event"] == "media":
                        yield base64.b64decode(data["media"]["payload"])

            await asyncio.gather(send_audio(), anext(recv_audio()))

asyncio.run(run("agent-uuid", []))
```
{% endtab %}
{% endtabs %}

---

## Authentication

Telephony providers reach the voice server over public URLs; the trust boundary is the provider account. The voice server authenticates *to the backend* using `INTERNAL_API_KEY` to fetch agent config and persist meeting records.

| Traffic | Auth mechanism |
|---------|----------------|
| Telephony provider to voice server | Public URLs; provider account |
| Browser test client to voice server | None (gated by deployment) |
| Voice server to backend | `INTERNAL_API_KEY` header |

See [../concepts/telephony-model.md](../concepts/telephony-model.md) for the full trust model.

## Close codes

| Code | Meaning | Action |
|------|---------|--------|
| 1000 | Normal closure | Call ended normally |
| 1002 | Protocol error | Malformed message — check JSON shape |
| 1003 | Unsupported data | Wrong `event` type or audio format |
| 1008 | Policy violation | Agent lookup failed or rejected |
| 1011 | Server error | Pipeline crash — check voice server logs |

## Best practices

1. Send the `start` frame within 5 seconds of opening the socket or providers will time out.
2. Keep `media` frames small (~20 ms PCM, roughly 640 bytes pre-base64) to minimise latency.
3. Handle the `clear` event — without it, barge-in feels broken.
4. Use `wss://` in any non-local environment; some providers refuse plaintext.
5. Always close the socket on hangup so the recording is finalised.

## Next steps

- [rest-api.md](rest-api.md) — HTTP API for agents, calls, and recordings
- [../concepts/voice-pipeline.md](../concepts/voice-pipeline.md) — How STT, LLM, and TTS chain together
- [../concepts/telephony-model.md](../concepts/telephony-model.md) — Inbound and outbound call flows
- [../services/voice-server.md](../services/voice-server.md) — Voice server deep dive
