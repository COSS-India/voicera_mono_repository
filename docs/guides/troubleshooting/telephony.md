---
description: Diagnosing webhooks, Stream XML, number linking, and recording retrieval.
---

# Telephony

Telephony failures fall into three buckets: the provider cannot reach you, it reaches you and gets something wrong back, or the call connects but the surrounding bookkeeping fails.

## The provider cannot reach `/answer`

The most common failure, and the one Voicera cannot see — nothing reaches your logs.

```bash
docker compose logs runtime | grep answer
```

Empty means the request never arrived. Work through:

| Check | How |
| --- | --- |
| Is the runtime reachable publicly? | `curl -s https://voice.example.com/health` from **outside** your network |
| Is `VOICE_SERVER_BASE_URL` correct? | Must be the public hostname, not `localhost` |
| Is TLS valid? | Providers reject self-signed and expired certificates |
| Does the provider log the attempt? | Its dashboard usually shows webhook failures with a reason |

## Agents that used to work stopped answering

Almost always because `VOICE_SERVER_BASE_URL` changed after the agents were created.

The answer URL is baked into the provider application at **agent-create time**, so an existing agent keeps pointing at the old address. There is no error in Voicera because nothing arrives.

Fix per agent:

```bash
curl -X PATCH http://localhost:8000/api/v1/agents/$AGENT_ID \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Support Line"}'
```

A PATCH re-provisions the application against the current value. Otherwise delete and recreate the agent. See [Public voice URLs](../deployment/public-voice-urls.md).

## `400` from `/answer`

You called it on a `websocket` agent. `/answer` serves `telephony` agents only — a websocket agent has no provider application and no Stream XML to return.

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/agents/$AGENT_ID | grep agent_category
```

`agent_category` cannot be changed to `telephony` without provisioning; PATCH it and the API creates the application, or create a new agent.

## The Stream URL is wrong

```bash
curl -s -X POST "http://localhost:7860/answer?agent_id=$AGENT_ID&org_id=$ORG_ID"
```

| XML shows | Meaning |
| --- | --- |
| `wss://voice.example.com/agent/...` | Correct |
| `wss://localhost/agent/...` | `VOICE_SERVER_BASE_URL` is unset or still the dev value |
| `ws://` not `wss://` | Base URL is `http://`; providers require TLS |

## `/answer` succeeds but no audio

The provider got valid XML and then failed to open the WebSocket. Nearly always the proxy.

```bash
curl -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "https://voice.example.com/agent/$ORG_ID/$AGENT_ID"
```

`101 Switching Protocols` is correct. Anything else means your proxy is not upgrading — you need `proxy_http_version 1.1` plus the `Upgrade` and `Connection` headers.

Also check the read timeout: nginx defaults to 60 seconds, which silently kills any call longer than a minute mid-conversation.

## Creating a telephony agent fails

| Error | Cause |
| --- | --- |
| `VOICE_SERVER_BASE_URL is not configured` | Unset. The API refuses rather than provisioning a broken application. |
| Provider authentication error | No `ProviderAuth` stored for that telephony provider |
| Unknown provider | `telephony_provider` must be a registered id — check `GET /configuration/telephony` |

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/auth/configured
```

## Number attach and detach

### Attach fails with a duplicate key error

Phone numbers are unique **globally**, not per organisation — the index has no `org_id` component. If another organisation in this deployment already holds the number, the attach fails rather than returning a clear conflict.

### The number is in inventory but calls do not route

Attaching without `agent_id` only imports the number; it does not link it to an agent or to the provider application. Re-attach with the agent:

```bash
curl -X POST http://localhost:8000/api/v1/phone-numbers/attach \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"phone_number": "+15551234567", "provider": "vobiz", "agent_id": "'"$AGENT_ID"'"}'
```

Confirm:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/phone-numbers/agent/$AGENT_ID
```

### Detach leaves the number in inventory

Intended. `DELETE /phone-numbers/detach` unlinks from the agent and the provider application but keeps the inventory row.

## Recording never arrives

Recordings are fetched from the provider after the call, so there is a delay — and providers differ.

| Check | Detail |
| --- | --- |
| Recording enabled on the provider account? | Voicera cannot fetch what was never recorded |
| Has enough time passed? | Providers take seconds to minutes to make a recording available |
| Does the call log have the URI? | `GET /calls/{call_id}` — look for `recording_url` |
| Did the runtime log a MinIO error? | `docker compose logs runtime | grep -i minio` |

Plivo exposes `list_recordings_for_call`; Vobiz does not, so retrieval paths differ per provider.

## Provider call SID mismatch

Voicera reconciles its call log with the provider's id via `PATCH /calls/by-provider-sid/{provider_call_sid}`. If a provider webhook arrives with an unknown SID, either the call was not registered or the SID differs from the one stored at dial time.

Inbound registration is idempotent — the same SID twice returns the existing record rather than duplicating.

{% hint style="info" %}
`call_response: "answered"` is terminal, and `end_time_utc` is write-once. Later patches to status or response are dropped deliberately, so a late hangup webhook cannot rewrite a completed call. If a call looks "stuck" at answered, this is why.
{% endhint %}

## Related

* [Telephony model](../concepts/telephony-model.md)
* [Public voice URLs](../deployment/public-voice-urls.md)
* [Calls and call artifacts](../concepts/calls.md)
* [Voice and audio](voice-and-audio.md)
