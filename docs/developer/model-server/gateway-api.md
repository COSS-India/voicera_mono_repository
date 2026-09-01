---
description: Gateway endpoints and their OpenAI compatibility.
---

# Gateway API

The gateway is the model server's single published port. It routes on modality, streams everything, and holds no model-specific knowledge — every model server speaks OpenAI spec natively, so adding a model never touches gateway code. This page is what it exposes and why the two model-listing routes differ.

The container listens on `8000` inside its namespace and is published on `GATEWAY_PORT`, default `8100`. All examples below use `http://localhost:8100`.

## Endpoint table

| Endpoint | Slot | Purpose |
| --- | --- | --- |
| `POST /v1/audio/transcriptions` | stt | Speech to text, one segment per request |
| `WS /v1/asr/ws` | stt | Speech to text, incremental — when the deployed model serves it |
| `WS /v1/realtime` | stt | OpenAI Realtime transcription relay |
| `GET /v1/languages` | stt | What the STT checkpoint in the slot actually speaks |
| `GET /demo` | stt | Live STT demo page from whichever model fills the slot |
| `POST /v1/audio/speech` | tts | Text to speech |
| `POST /v1/chat/completions` | llm | LLM, when a model is deployed |
| `GET /models` | — | Every model in the catalogue, and which are running |
| `GET /v1/models` | — | OpenAI-compatible: only what can be called right now |
| `GET /health` | — | Gateway and upstreams |

Routes are defined in `model-server/gateway/app/main.py`; that file is the current source.

{% hint style="info" %}
`/v1/asr/ws` and `/v1/realtime` are relays, not protocols the gateway understands. Bytes and text frames pass through untouched in both directions, and the protocol on top is entirely between the client and the model. That is why a model that does not stream has nothing listening on its end.
{% endhint %}

Anything a model serves beyond these routes — `/metrics`, `/admin/*`, Orpheus's `/v1/tts` family, a model's own demo page at `/` — is not part of the slot contract and is not forwarded. Reach those with `docker compose exec` for debugging.

## /models vs /v1/models

The two differ on purpose. **OpenAI clients read `/v1/models` as "what can I call", so it must never list something that would answer 503.**

`GET /v1/models` lists only the slots that are actually deployed:

```json
{
  "object": "list",
  "data": [
    {"id": "indic-conformer", "object": "model", "owned_by": "voicera", "kind": "stt"},
    {"id": "indic-parler", "object": "model", "owned_by": "voicera", "kind": "tts"}
  ]
}
```

A slot counts as deployed when its `<SLOT>_MODEL` is set — the same variable that picks the folder for Compose — so the gateway and Compose cannot disagree about what is running. An undeployed LLM slot does not appear.

`GET /models` is the whole catalogue, read from `models.yaml`, with each entry tagged `deployed`:

```json
{
  "object": "list",
  "data": [
    {"kind": "stt", "id": "indic-conformer", "status": "ready", "deployed": true, "...": "..."},
    {"kind": "llm", "id": "gemma", "status": "planned", "deployed": false, "...": "..."}
  ],
  "deployed": {"stt": "indic-conformer", "tts": "indic-parler", "llm": null}
}
```

Use `/models` to see what the server *can* host, including `planned` entries that have no folder yet. Use `/v1/models` from an OpenAI client. `tests/test_catalogue.py` guards the drift specifically — that `/v1/models` never starts advertising something a caller would get a `503` from.

Editing `models.yaml` takes effect on restart, no rebuild: the file is bind-mounted into the container. A missing or unreadable catalogue is not fatal — the gateway still routes traffic, it cannot describe what it is routing to.

## WS /v1/asr/ws

This is the one route that is not OpenAI-shaped, because OpenAI has no equivalent. Claiming OpenAI's realtime path while speaking something else would be worse than an honest name.

It exists because live transcription is genuinely two-directional: audio flows in for as long as someone is talking while partial transcripts flow back, and neither side knows when the other will speak next. TTS is not like that — it is one-directional, so it moved *off* WebSockets to plain HTTP, which gave cancellation for free. Direction of travel decides the transport.

Both directions run concurrently in the relay, and whichever ends first tears the other down — so a caller hanging up mid-sentence closes the upstream session and frees the decoder rather than leaving it transcribing an empty room. `Authorization` and `openai-beta` headers are forwarded to the upstream; the query string is passed through verbatim, which is how `?language=hi` and `&endpoint=1` reach the model.

**This route is not what makes transcription live.** Every STT model here returns partial transcripts while the caller is still speaking, and always has. What differs is where the partials come from. That distinction is set out in full on [STT models](stt-models.md), and it is the thing to read before concluding that a model without this route waits for you to finish a sentence.

`WS /v1/realtime` is a second relay, to the same STT upstream, for models that serve OpenAI Realtime transcription — which is what `indic-conformer` serves. `tests/test_gateway_streaming.py` and `tests/test_stt_streaming.py` cover the relay: binary and text frames both ways, unbuffered, and a hang-up carried through to the model.

## Health

`GET /health` probes every slot concurrently and reports each one:

```json
{
  "status": "healthy",
  "upstreams": {
    "stt": {"deployed": true, "model": "indic-conformer", "reachable": true},
    "tts": {"deployed": true, "model": "indic-parler", "reachable": true},
    "llm": {"deployed": false}
  }
}
```

The status code is `200` when healthy and `503` when degraded. **A slot nobody deployed is not a fault** — reporting it as degraded would make every monitor cry wolf on a stack running exactly as configured. Degraded means a slot that *is* deployed is not reachable.

An upstream probe is a `GET {upstream}/health` with a 2 s timeout, and counts as reachable when the status is below 500. Several models return `503` while loading and `200` once warm, which is exactly what this probe wants: "healthy" means "will answer fast", not "the process is alive".

The gateway has no `depends_on` on the slots. With profiles, a slot may legitimately not be running, so the gateway starts regardless and explains what is missing rather than failing to start.

## The demo page

`GET /demo` forwards to the demo page of whichever model fills the STT slot. With the stack running:

1. Open `http://localhost:8100/demo` in Chrome or Firefox.
2. Pick a language, click **Start**, allow the microphone, and speak.
3. Words stream in as the server emits deltas. Click **Stop** to commit the segment.

`GET /v1/languages` exists for the same page. The picker builds itself from that route, and it fetches relatively — so a page served through the gateway asks the gateway. Without the route that fetch 404s and the picker keeps its short static fallback, which reads as "this model only supports four languages".

The microphone works on `localhost` over HTTP. On a remote machine you need HTTPS or a tunnel — browsers block `getUserMedia` on plain HTTP except for localhost.

## Error shapes for undeployed slots

Every REST route on an empty slot answers `503` with the same body — never a `404`, and never a hang:

```json
{
  "error": {
    "message": "No LLM model is deployed. Set LLM_MODEL in .env and include llm in COMPOSE_PROFILES.",
    "type": "upstream_not_configured"
  }
}
```

The WebSocket routes cannot use a status code, so they **accept the handshake and then explain**, rather than refusing it. A client whose handshake fails is told only "HTTP 403" and cannot tell "wrong URL" from "model still loading". Instead the gateway accepts, sends a JSON frame, and closes with code `1013`:

```json
{
  "type": "error",
  "reason": "upstream_not_configured",
  "error": "No STT model is deployed. Set STT_MODEL in .env and include stt in COMPOSE_PROFILES."
}
```

The same shape with `"reason": "upstream_unreachable"` is sent when the slot is deployed but its WebSocket upstream cannot be reached — which distinguishes "not configured" from "configured but not up yet".

`tests/test_llm_slot.py` pins the REST behaviour: an empty slot answers `503` rather than `404` or a hang, is not advertised at `/v1/models`, and does not mark health degraded. `tests/test_stt_streaming.py` pins the WebSocket behaviour: an empty slot explains itself instead of failing the handshake.

## Streaming

One hard requirement runs through the proxy: **never buffer**. A proxy that collects a full response before forwarding adds hundreds of milliseconds to TTS time-to-first-byte, which is the difference between a natural phone call and an awkward one. Request bodies and response bodies both stream.

Timeouts are asymmetric for the same reason: `connect` and `pool` are 5 s so an upstream that is down fails fast instead of hanging a call, while `read` and `write` are unbounded because a TTS generation or an SSE completion holds the response open for as long as the model takes.

Cancellation matters as much as throughput. When a caller is interrupted, Pipecat cancels the task reading the response; that closes the client connection, which closes the upstream one, which frees the model's slot. `tests/test_gateway_streaming.py` pins both halves — that the gateway streams rather than buffers, and that a client disconnect evicts upstream.

The gateway runs one uvicorn worker: pure async I/O, no CPU work. Scale with replicas, not workers, so upstream connection pools stay predictable.

## Related

* [Overview](overview.md)
* [STT models](stt-models.md)
* [TTS models](tts-models.md)
* [Ports and defaults](../reference/ports-and-defaults.md)
* [WebSocket API](../../api-reference/websocket-api.md)
