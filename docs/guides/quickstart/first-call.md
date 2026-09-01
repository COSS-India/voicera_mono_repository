---
description: Talk to your agent and confirm audio, transcript, and recording.
---

# Your first call

Two ways to reach an agent. Start with the browser — it needs no telephony account and costs nothing.

{% tabs %}
{% tab title="Browser (no telephony)" %}
Requires a `websocket` agent from [Create your first agent](first-agent.md).

The runtime accepts a WebSocket at:

```
ws://localhost:7860/agent/{org_id}/{agent_id}
```

Confirm the runtime is up:

```bash
curl -s localhost:7860/health
```

The transport is Pipecat protobuf frames at 16 kHz, so a plain `wscat` will not hold a conversation — you need a client that speaks the protocol. Two options:

* Write one with `@pipecat-ai/websocket-transport` — see [Browser WebSocket agents](../../developer/clients/browser-websocket.md) for a minimal example.
* Run the [Beta dashboard](../../developer/frontend/overview.md), which has a test-call button.

{% hint style="warning" %}
Browser sessions register a `call_type: web` call log on connect, so they produce a transcript and a recording just like telephony calls.
{% endhint %}
{% endtab %}

{% tab title="Phone (telephony)" %}
Requires a `telephony` agent, an attached number, and a publicly reachable `VOICE_SERVER_BASE_URL`.

Verify the answer webhook before dialling:

```bash
curl -s -X POST \
  "http://localhost:7860/answer?agent_id=$AGENT_ID&org_id=$ORG_ID"
```

You should get XML naming the audio address:

```xml
<Response>
  <Stream bidirectional="true">wss://voice.example.com/agent/ORG_ID/AGENT_ID</Stream>
</Response>
```

Check that host is the public one your provider can reach — if it says `localhost`, `VOICE_SERVER_BASE_URL` is unset or wrong, and no real call will connect.

Then dial the number.
{% endtab %}
{% endtabs %}

## Placing an outbound call

```bash
curl -X POST http://localhost:8000/api/v1/calls/outbound \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "'"$AGENT_ID"'",
    "to_number": "+15559876543"
  }'
```

The API takes a concurrency slot, creates the call log, and asks the provider to dial. The response carries the `call_id`.

## What working looks like

```bash
docker compose logs -f runtime
```

In order: the `/answer` request, the WebSocket opening, the pipeline building its three services, the greeting, then transcripts turn by turn.

| Log shows | Meaning |
| --- | --- |
| No `/answer` | The provider never reached you — [Telephony](../troubleshooting/telephony.md) |
| `/answer` but no WebSocket | The provider could not open the WSS URL — TLS or proxy upgrade |
| WebSocket then an exception | The pipeline failed to build — usually missing credentials |
| Transcripts appearing | Working |

## Find the call

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/calls/org/$ORG_ID"
```

Then one call in detail:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/calls/$CALL_ID
```

Watch `status` (`initiated` → `ringing` → `in_progress` → `completed`) and `call_response` (`answered`, `busy`, `no_answer`, …).

## Transcript and recording

Written when the call ends, and fetched through the API rather than from the bucket:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/calls/$CALL_ID/transcript

curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/calls/$CALL_ID/recording -o recording.wav
```

The raw objects live in MinIO under `voicera-calls/{org_id}/{call_id}/`, browsable at [http://localhost:9001](http://localhost:9001).

## If something went wrong

| Symptom | Page |
| --- | --- |
| Silence, one-way audio, distortion | [Voice and audio](../troubleshooting/voice-and-audio.md) |
| The provider cannot reach `/answer` | [Telephony](../troubleshooting/telephony.md) |
| `400` from `/answer` | You called it on a `websocket` agent — it only serves `telephony` agents |
| No transcript after a browser session | Check the runtime log for `Registered web call call_id=` and confirm MinIO is healthy |

## Next

* [Running a campaign](../operator/running-a-campaign.md) — outbound at volume
* [Voice pipeline](../concepts/voice-pipeline.md) — what happened inside the call
* [Agent configuration](../../developer/reference/agent-configuration.md) — tune the behaviour
