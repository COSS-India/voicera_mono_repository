---
description: CSV-driven outbound campaigns: upload, schedule, run, report.
---

# Campaigns

`apps/api/app/routers/campaign.py`, prefix `/api/v1/campaign`. See [Campaigns](../guides/concepts/campaigns.md).

{% hint style="warning" %}
`max_concurrency`, `schedule_config`, and `circuit_breaker` are accepted as **top-level fields on the request** but are **not** top-level fields on the stored document. The router nests all three inside `orchestrator_metadata` on both create and update, and `CampaignResponse` returns them there. Read `orchestrator_metadata.max_concurrency`, not `max_concurrency`.
{% endhint %}

## `POST /campaign/upload`

Bearer. `201`. `multipart/form-data` with one `file` field. The path is `/upload`.

```bash
curl -X POST http://localhost:8000/api/v1/campaign/upload \
  -H "Authorization: Bearer YOUR_JWT" \
  -F "file=@contacts.csv"
```

The file must end in `.csv`, must be non-empty, must be under `CAMPAIGN_MAX_CSV_BYTES`, and must validate — which means it must have a `phone_number` header column. Returns `CampaignCsvUploadResponse`:

```json
{
  "source_id": "campaigns/org_abc123/9f8e7d6c_contacts.csv",
  "filename": "contacts.csv",
  "contact_rows": 412
}
```

`contact_rows` counts data rows with a non-blank `phone_number`. Pass `source_id` straight to create. Failure codes: `400` for the wrong extension, an empty file, or a validation error; `413` for oversize; `502` when MinIO is unreachable.

## `POST /campaign/create`

Bearer. `201`.

```json
{
  "name": "March outreach",
  "agent_id": "…",
  "source_type": "csv",
  "source_id": "campaigns/org_abc123/9f8e7d6c_contacts.csv",
  "rate_limit_per_second": 1,
  "max_concurrency": 5,
  "from_number": "+14155559999",
  "retry_config": {
    "enabled": true,
    "max_retries": 2,
    "retry_delay_seconds": 120,
    "retry_on_busy": true,
    "retry_on_no_answer": true,
    "retry_on_voicemail": false
  },
  "schedule_config": {
    "enabled": false,
    "timezone": "UTC",
    "slots": [{ "day_of_week": 0, "start_time": "09:00", "end_time": "17:00" }]
  },
  "circuit_breaker": {
    "enabled": true,
    "failure_threshold": 0.5,
    "window_seconds": 300,
    "min_calls_in_window": 5
  }
}
```

| Field | Type | Default | Bounds |
| --- | --- | --- | --- |
| `name` | string | required | — |
| `agent_id` | string | required | Must be a `telephony` agent with a linked number |
| `source_type` | string | `"csv"` | — |
| `source_id` | string | required | From `POST /campaign/upload` |
| `rate_limit_per_second` | int | `1` | `1`–`20` |
| `max_concurrency` | int | `5` | `1`–`20`, and at or below the organisation's ceiling |
| `from_number` | string or null | `null` | One number may back many concurrent calls |
| `retry_config.max_retries` | int | `2` | `0`–`10` |
| `retry_config.retry_delay_seconds` | int | `120` | `30`–`3600` |
| `schedule_config.slots[].day_of_week` | int | required | `0`–`6`, `0` = Monday |
| `schedule_config.slots[].start_time`, `end_time` | string | required | `HH:MM` |
| `circuit_breaker.failure_threshold` | float | `0.5` | `0.1`–`1.0` |
| `circuit_breaker.window_seconds` | int | `300` | `60`–`3600` |
| `circuit_breaker.min_calls_in_window` | int | `5` | `1`–`100` |

`retry_config`, `schedule_config`, and `circuit_breaker` are all optional. Omitting `retry_config` applies `DEFAULT_CAMPAIGN_RETRY_CONFIG` from `apps/api/app/constants/campaign.py`; omitting the other two leaves them absent from `orchestrator_metadata` entirely.

Returns `CampaignResponse`: `campaign_id`, `org_id`, `name`, `agent_id`, `source_type`, `source_id`, `state`, `total_rows`, `processed_rows`, `failed_rows`, `rate_limit_per_second`, `retry_config`, `orchestrator_metadata`, `from_number`, `created_by`, and the five timestamps.

Failure codes: `404` when the agent does not exist, `422` when it is not a telephony agent or has no number, `400` when `max_concurrency` exceeds the organisation ceiling or the source fails validation.

## `GET /campaign/`

Bearer. Array of `CampaignResponse`. The trailing slash is part of the path.

## `GET /campaign/{campaign_id}`

Bearer. One `CampaignResponse`. `404` outside your organisation.

## `POST /campaign/{campaign_id}/start`, `/pause`, `/resume`

Bearer. No body. Each returns a two-field object:

```json
{ "status": "started", "campaign_id": "…" }
```

with `started`, `paused`, or `resumed` respectively. An illegal transition — resuming a campaign that is not paused — returns `400` with the reason.

## `PATCH /campaign/{campaign_id}`

Bearer. All fields optional: `name`, `rate_limit_per_second`, `max_concurrency`, `retry_config`, `schedule_config`, `circuit_breaker`. The three nested-config fields are merged into the existing `orchestrator_metadata` rather than replacing it. A new `max_concurrency` is re-validated against the organisation ceiling. Returns the updated `CampaignResponse`.

## `GET /campaign/{campaign_id}/runs`

Bearer. Query `limit` (default `50`, `1`–`500`) and `offset`. Returns an array of **call log** documents for the campaign, not `QueuedRuns` documents.

## `GET /campaign/{campaign_id}/progress`

Bearer. Returns `CampaignProgressResponse`: `campaign_id`, `state`, `total_rows`, `processed_rows`, `failed_rows`, `progress_percentage`, `rate_limit`, `started_at`, `completed_at`.

## `POST /campaign/{campaign_id}/redial`

Bearer. `201`. Body `{ "name": "March outreach — redial" }`. Creates a **new** campaign over the parent's failed contacts, inheriting the agent, source, rate limit, retry config, and caller ID, and recording `parent_campaign_id` in `orchestrator_metadata`. Returns the child `CampaignResponse`. No failed contacts returns `400 No failed contacts to redial`.

## `GET /campaign/{campaign_id}/source-download-url`

Bearer. Returns `{"download_url": "https://…"}` — a presigned MinIO URL for the uploaded CSV.

## `GET /campaign/{campaign_id}/report`

Bearer. Streams `text/csv` as an attachment named `campaign_{campaign_id}.csv`, with columns `call_id`, `to_number`, `status`, `call_response`, `duration`, `created_at`. Capped at 500 rows.

## `POST /campaign/internal/call-status`

`X-API-Key`. The runtime's terminal-state callback into the orchestrator.

```json
{ "org_id": "…", "call_id": "…", "call_response": "answered" }
```

`call_response` is optional; when omitted the value already on the `CallLog` is used. Returns `{"status": "ok"}`. An unknown `call_id` returns `404`.

## Related

* [Endpoints cheatsheet](endpoints-cheatsheet.md) — every route on one page
* [Authentication](authentication.md) — tokens, headers, and roles
* [Errors](errors.md) — status codes and error shapes
