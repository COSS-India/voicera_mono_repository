---
description: How a telephony provider becomes a client of the runtime.
---

# Telephony agents

When a call reaches a `telephony` agent, the provider is the client. It fetches an answer document over HTTP, then opens a WebSocket and streams audio. This page is the wire-level view: what the provider sends, what it gets back, and how to exercise it without a phone.

{% hint style="info" %}
This page does not explain the provider abstraction — how Vobiz and Plivo are registered, how applications are provisioned, how credentials are resolved. That is [Telephony model](../../guides/concepts/telephony-model.md). Read it first if you are adding a provider rather than debugging a call.
{% endhint %}

## The answer webhook

`GET|POST /answer` on the runtime, declared in `apps/runtime/routes/telephony.py`. Two query parameters are mandatory:

| Parameter | Required | Meaning |
| --- | --- | --- |
| `agent_id` | Yes | Agent to run. Missing gives `400 agent_id is required`. |
| `org_id` | Yes | Owning organisation. Missing gives `400 org_id is required`. |
| `call_id` | No | Existing `CallLogs` id. Set by the API on outbound calls so the runtime skips inbound registration. |

The API writes this URL onto the agent document when it provisions the provider application, in the form `{VOICE_SERVER_BASE_URL}/answer?agent_id={agent_id}&org_id={org_id}`.

The runtime accepts either encoding for the body. `decode_webhook_body()` in `apps/telephony/webhooks.py` parses JSON when the body starts with `{` or `[`, and `x-www-form-urlencoded` otherwise, then merges any query parameters that the body did not already supply — Vobiz sends `CallUUID` on the URL in some flows. Field names are read case-insensitively across both providers: `From`/`from`, `To`/`to`, `Event`/`event`, `CallStatus`, `HangupCause`, and the call identifier from any of `CallUUID`, `call_uuid`, `call_id`, `callId`, `callSid`, `CallSid`, or `request_uuid`.

The handler branches on the event:

* **A hangup event** — `hangup`, `hangupcomplete`, `callhangup`, or any event whose name contains `hangup` — patches the `CallLog` with `end_time_utc`, `status: "completed"`, and a mapped `call_response` where the provider gave enough to derive one, notifies the campaign orchestrator, and returns `200` with an empty body. No XML.
* **Anything else** is treated as the answer. The runtime loads the agent, rejects it with `400 Agent is not a telephony agent` when `agent_category` is not `telephony`, registers an inbound `CallLog` if the request carried no `call_id` but did carry a provider call identifier, and returns Stream XML.

A backend failure while loading the agent returns `502` with the error text; a routing error returns `400`.

## Stream XML

The response is `application/xml`, built by `build_answer_stream_xml()` in `apps/telephony/xml.py`, which dispatches to the provider's own builder. Vobiz and Plivo currently emit the same document:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
        wss://voice.example.com/agent/YOUR_ORG_ID/YOUR_AGENT_ID?call_id=YOUR_CALL_ID
    </Stream>
</Response>
```

`contentType` is chosen from the sample rate, and only two values exist:

| `SAMPLE_RATE` | `contentType` |
| --- | --- |
| `16000` | `audio/x-l16;rate=16000` |
| anything else | `audio/x-mulaw;rate={SAMPLE_RATE}` |

μ-law is 8 kHz only per the Vobiz specification, which is why 16 kHz switches to L16. The `call_id` query parameter is appended only when the runtime has one — either passed in by the API on an outbound call, or created during inbound registration. Without it the pipeline still runs, but nothing correlates the session to a `CallLog`.

Plivo configures hangup through the application's `hangup_url`, not through `Stream` attributes, so the XML carries no hangup hint.

## The media WebSocket

The provider then connects to the URL inside `<Stream>`: `WS /agent/{org_id}/{agent_id}`, optionally with `?call_id=`.

The runtime accepts the socket, loads the agent, sees `agent_category` is `telephony`, and **blocks on the first text message**. That message must be JSON with `"event": "start"`. Anything else closes the socket with code `1008` and reason `Expected start event`.

From the `start` object the runtime reads:

| It looks for | Falling back to |
| --- | --- |
| Provider call SID | `CallUUID`, `call_uuid`, `call_id`, `callId`, `callSid`, `CallSid`, `request_uuid`, or the same keys nested under `start` — then `callSid`, `callId`, `call_uuid` directly, then the literal `unknown` |
| Stream SID | `streamSid`, then `streamId`, then the literal `unknown` |

If no `call_id` arrived on the URL and the call SID is something other than `unknown`, the runtime registers an inbound `CallLog` at this point. It then loads that `CallLog` to merge its `custom_variables` over the agent's defaults, and starts the pipeline with a provider-specific frame serializer from `create_frame_serializer(provider, stream_sid=..., call_sid=..., sample_rate=...)`.

After the pipeline ends, the runtime patches the `CallLog` with `end_time_utc`, `status: "completed"` and `call_response: "answered"`, and notifies the campaign orchestrator. That is why a hangup webhook and a socket close can both finalise the same call.

## Sample rate

Telephony audio runs at `SAMPLE_RATE`, which defaults to `8000` in `.env.example`. It is read in `apps/runtime/constants.py` and used in three places that must agree: the `contentType` in the answer XML, the frame serializer, and the Pipecat pipeline's input and output rates. Change it in one place — the root `.env` — and all three follow.

Browser sessions use a separate variable, `WEBSOCKET_SAMPLE_RATE`, default `16000`. See [Browser WebSocket agents](browser-websocket.md).

## Provider dispatch

The runtime never guesses the vendor. It reads `telephony.provider` from the agent document and passes that string to both `build_answer_stream_xml()` and `create_frame_serializer()`, each of which looks the provider up in the registry in `apps/telephony/registry.py`. An unregistered provider raises `ValueError`.

This means a single runtime serves Vobiz and Plivo agents side by side on the same port and the same two routes. Nothing is configured per deployment. `GET /api/v1/configuration/telephony` enumerates the registered providers on a running API.

## Testing without a phone

The answer webhook is plain HTTP, so `curl` is enough to check that an agent is provisioned and that your public URL is right.

```bash
curl -s http://localhost:7860/health
```

```json
{"status": "ok", "service": "voicera-runtime"}
```

Then the answer webhook:

```bash
curl -s -X POST \
  'http://localhost:7860/answer?agent_id=YOUR_AGENT_ID&org_id=YOUR_ORG_ID'
```

Expect the Stream XML shown above, with a `wss://` URL derived from `VOICE_SERVER_BASE_URL`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<Response>
    <Stream bidirectional="true" keepCallAlive="true" contentType="audio/x-mulaw;rate=8000">
        wss://voice.example.com/agent/YOUR_ORG_ID/YOUR_AGENT_ID
    </Stream>
</Response>
```

What this tells you:

| Result | What it means |
| --- | --- |
| Stream XML with the right host | The agent exists, is `telephony`, and `VOICE_SERVER_BASE_URL` is set correctly. |
| A `wss://` URL pointing at `localhost` or an empty host | `VOICE_SERVER_BASE_URL` is unset or wrong. No provider will reach it. |
| `400 Agent is not a telephony agent` | The agent's `agent_category` is `websocket`. |
| `502` | The runtime could not reach the API, or the agent does not exist in that organisation. |

There is no equivalent one-liner for the media WebSocket — driving it means sending a `start` frame and then provider-encoded audio. Use a browser `websocket` agent to exercise the pipeline itself, and use this curl to verify only the answer path.

{% hint style="warning" %}
For a real call the provider must reach both URLs from the public internet, over HTTPS and WSS. `VOICE_SERVER_BASE_URL` must be your public host, not `localhost`, and neither route is authenticated. See [Public voice URLs](../../guides/deployment/public-voice-urls.md) and [Security hardening](../../guides/deployment/security-hardening.md).
{% endhint %}

## Related

* [Connecting a client](README.md)
* [Telephony model](../../guides/concepts/telephony-model.md)
* [Browser WebSocket agents](browser-websocket.md)
* [WebSocket API](../../api-reference/websocket-api.md)
* [Telephony troubleshooting](../../guides/troubleshooting/telephony.md)
