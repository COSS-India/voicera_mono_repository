---
description: Diagnosing stuck campaigns, tripped breakers, and workers that are not picking up jobs.
---

# Campaigns

Campaigns involve four moving parts — the API, Redis, the ARQ worker, and the orchestrator — so start by working out which one stopped.

## First: is everything running?

```bash
docker compose ps redis arq-worker campaign-orchestrator
docker compose logs campaign-orchestrator --tail 50
docker compose logs arq-worker --tail 50
```

| Container down | Symptom |
| --- | --- |
| `redis` | Nothing queues or dispatches at all |
| `arq-worker` | Batches are scheduled but no calls are placed |
| `campaign-orchestrator` | The first batch may run, then nothing follows |

## The campaign will not start

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/campaign/$CAMPAIGN_ID
```

| State | Meaning |
| --- | --- |
| `created` | Never started — call `POST /campaign/{id}/start` |
| `syncing` | Still ingesting the CSV; wait, then check the worker logs |
| `running` | Started; if no calls are going out, read on |
| `paused` | Paused manually, or by the circuit breaker |
| `completed` | Finished, or swept as complete |
| `failed` | Stopped abnormally — see below |

## Stuck in `syncing`

The CSV sync runs as an ARQ job. If it never completes:

```bash
docker compose logs arq-worker | grep -i sync
```

Usual causes: the worker is down, the CSV is malformed, or the upload never landed in MinIO.

## `running` but no calls go out

Work through in order:

**1. Are there queued runs left?**

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/campaign/$CAMPAIGN_ID/progress
```

**2. Is a from-number available?** `PhoneNumberPoolExhaustedError` in the logs means no attachable number was free. The dispatcher retries up to 3 times before giving up.

**3. Are concurrency slots free?** If the organisation is already at its ceiling, the dispatcher waits up to 120 seconds and then raises `ConcurrentSlotAcquisitionError`.

```bash
docker exec voicera_oss_redis redis-cli -a "$REDIS_PASSWORD" \
  ZCARD "concurrent_calls:$ORG_ID"
```

**4. Is the orchestrator reacting?** It listens on the `campaign_events` channel. If its logs are silent while batches complete, it is not receiving events — restart it.

## The campaign paused itself

The circuit breaker tripped. It watches a rolling window and pauses the campaign when the failure rate crosses the threshold, so a broken agent cannot burn the whole list.

Defaults (`apps/api/app/constants/campaign.py`):

| Setting | Default |
| --- | --- |
| `failure_threshold` | `0.5` — half the calls failing |
| `window_seconds` | `300` |
| `min_calls_in_window` | `5` — never trips on a tiny sample |

Find out **why** the calls failed before resuming:

```bash
curl -s -H "Authorization: Bearer $TOKEN" \
  "http://localhost:8000/api/v1/calls/org/$ORG_ID" | head -50
```

Common causes: bad credentials, an unreachable answer URL, a number that cannot dial the destination, or a contact list full of invalid numbers. Fix the cause, then:

```bash
curl -X POST -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/campaign/$CAMPAIGN_ID/resume
```

## The campaign is `failed` and will not resume

{% hint style="warning" %}
`failed` is terminal. Resume is only permitted from `paused`, so a campaign that failed cannot be restarted through the API even when queued runs remain. **Redial is the only recovery path.**

A single batch exception is enough to reach this state — including a `ConcurrentSlotAcquisitionError` from a 120-second slot shortage under load, which fails the whole campaign rather than just the batch.
{% endhint %}

```bash
curl -X POST http://localhost:8000/api/v1/campaign/$CAMPAIGN_ID/redial \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}'
```

## Completed too early

If the orchestrator restarted mid-run, completion detection falls back to document timestamps. A campaign idle for more than an hour (`completion_timeout = 3600`) with no pending work is marked `completed` on the first sweep — even if a long-delay retry would otherwise have continued it.

Check `progress` for unprocessed contacts, and redial if any remain.

## Calls dialled twice

You are running more than one campaign orchestrator.

{% hint style="danger" %}
Never run more than one orchestrator replica. Its state is in-memory and it uses Redis **pub/sub**, which delivers each event to *every* subscriber — so two replicas both schedule the next batch and the campaign dials at twice its configured rate.
{% endhint %}

```bash
docker compose ps | grep orchestrator
```

Exactly one. The ARQ worker, by contrast, scales freely.

## Retries do not happen as expected

Defaults: `max_retries: 2`, `retry_delay_seconds: 120`, retry on `busy` and `no_answer`, **not** on `voicemail`.

Two asymmetries surprise people:

* `cancelled` counts as a failure for the circuit breaker but is **never** retried.
* `failed` **is** retried, but has no per-reason toggle — only `busy`, `no_answer`, and `voicemail` can be individually disabled.

Retries create a **new** `QueuedRuns` document rather than mutating the original; the unique index on `(campaign_id, source_uuid, retry_count)` makes that idempotent.

## Inspecting Redis directly

```bash
docker exec -it voicera_oss_redis redis-cli -a "$REDIS_PASSWORD"
```

| Key | Holds |
| --- | --- |
| `campaign_events` | The pub/sub channel |
| `concurrent_calls:{org_id}` | Active slots for an organisation |
| `concurrent_calls_fleet` | Fleet-wide slot set (observability only — no cap is enforced against it) |
| `cb_failures:` / `cb_successes:` / `cb_recent_failures:` | Circuit-breaker windows |
| `rate_limit:{scope}` | Sliding-window rate limiter |

## Related

* [Campaigns](../concepts/campaigns.md)
* [Call concurrency](../concepts/call-concurrency.md)
* [Workers and orchestrator](../../developer/services/workers.md)
* [Running a campaign](../operator/running-a-campaign.md)
