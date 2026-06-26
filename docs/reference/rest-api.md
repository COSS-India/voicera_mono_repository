---
description: Complete REST API reference for the VoicEra backend, with request parameters, response shapes, and type definitions.
---

# REST API Reference

VoicEra's backend exposes a REST API on port `8000`. All resource routes are mounted under the `/api/v1` prefix. For the auto-generated, always-current schema, use Swagger UI at `http://<host>:8000/docs` or fetch the OpenAPI spec from `http://<host>:8000/openapi.json`.

{% hint style="info" %}
**Swagger is the authoritative source.** This document is a human-readable companion. For the most current endpoint details, see `/docs`.
{% endhint %}

---

## Overview

### Base URL

```
http://localhost:8000        # local development
https://api.yourdomain.com   # production
```

### Authorization

Two authentication schemes are used:

**JWT (dashboard / user-facing endpoints)**

Obtain a token via `POST /api/v1/users/login`. Tokens expire after **30 minutes**.

```http
Authorization: Bearer <jwt_token>
Content-Type: application/json
```

**X-API-Key (internal / service-to-service endpoints)**

Used by the voice server to call the backend. Set via the `INTERNAL_API_KEY` environment variable on both services.

```http
X-API-Key: <internal_api_key>
Content-Type: application/json
```

### Request format

All endpoints accept and return `application/json`.

### Response format

Successful responses return the resource object directly — no envelope wrapper.

Error responses follow FastAPI's standard shape:

```json
{ "detail": "Agent not found" }
```

Validation errors (422) return field-level detail:

```json
{
  "detail": [
    { "loc": ["body", "email"], "msg": "field required", "type": "value_error.missing" }
  ]
}
```

### HTTP status codes

| Code | Meaning |
|------|---------|
| 200 | OK |
| 201 | Created |
| 204 | No Content |
| 400 | Bad Request — validation or business logic failure |
| 401 | Unauthorized — missing or invalid token |
| 403 | Forbidden — authenticated but not permitted |
| 404 | Not Found |
| 409 | Conflict |
| 422 | Unprocessable Entity — schema validation failed |
| 500 | Internal Server Error |
| 502 | Bad Gateway — upstream service (voice server) unreachable |

---

## Users API

Base path: `/api/v1/users`

Handles signup, login, profile lookup, and password reset.

### Signup

`POST /api/v1/users/signup`

Register a new user account. The first user creates a new organization; subsequent users with the same `org_id` join as members.

**Auth:** None (public)

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | User email address |
| `password` | string | yes | Plain-text password (hashed server-side with bcrypt) |
| `name` | string | yes | Display name |
| `company_name` | string | yes | Organization / company name |
| `org_id` | string | no | If provided (from an invite link), joins an existing org instead of creating one |

**Returns** `201 Created` — `{ "status": "success", "message": "...", "org_id": "..." }`

{% hint style="info" %}
The response confirms account creation and returns the assigned `org_id`. It does not include the full user profile. To retrieve the complete profile, first obtain a token via `POST /api/v1/users/login`, then call `GET /api/v1/users/me`.
{% endhint %}

---

### Login

`POST /api/v1/users/login`

Exchange email and password for a JWT.

**Auth:** None (public)

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | Registered email |
| `password` | string | yes | Account password |

**Returns** `200 OK` — [UserLoginResponse](#userloginresponse)

---

### Get current user

`GET /api/v1/users/me`

Return the profile of the currently authenticated user.

**Auth:** JWT

**Returns** `200 OK` — [User](#user)

---

### Get user by email

`GET /api/v1/users/{email}`

Return a user profile by email address. Users can only fetch their own profile.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `email` | string | Email address of the user to fetch |

**Returns** `200 OK` — [User](#user)

---

### Forgot password

`POST /api/v1/users/forgot-password`

Send a password-reset email to the specified address via the configured mail provider.

**Auth:** None (public)

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | Email address of the account to reset |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Reset password

`POST /api/v1/users/reset-password`

Complete a password reset using the token from the reset email.

**Auth:** None (public)

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `token` | string | yes | Reset token from the email |
| `new_password` | string | yes | The new password to set |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

## Agents API

Base path: `/api/v1/agents`

CRUD over voice agent configurations. Each agent encodes its LLM, STT, TTS settings, system prompt, language, telephony provider, and more.

### Create agent

`POST /api/v1/agents`

Create a new agent configuration.

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_type` | string | yes | Unique slug identifier for this agent (e.g. `"support"`) |
| `agent_id` | string | yes | Client-assigned UUID for this agent |
| `agent_config` | object | yes | Free-form config object containing LLM, STT, TTS, and prompt settings |
| `org_id` | string | yes | Organization this agent belongs to (must match caller's org) |
| `agent_category` | string | no | Optional category label |
| `phone_number` | string | no | Phone number to associate with this agent |
| `app_id` | string | no | External telephony application ID |
| `greeting_message` | string | no | First message spoken when a call is answered. Write-only; not included in read responses. |
| `telephony_provider` | string | no | `"Vobiz"` or `"Plivo"` |
| `vobiz_app_id` | string | no | Linked Vobiz application ID |
| `vobiz_answer_url` | string | no | Vobiz answer webhook URL |
| `plivo_app_id` | string | no | Linked Plivo application ID |
| `plivo_answer_url` | string | no | Plivo answer webhook URL |

**Returns** `201 Created` — `{ "status": "success", "message": "..." }`

{% hint style="info" %}
The response confirms agent creation. To retrieve the full agent configuration, call `GET /api/v1/agents/{agent_type}`.
{% endhint %}

---

### List agents for org

`GET /api/v1/agents/org/{org_id}`

List all agents belonging to an organization.

**Auth:** JWT (caller's `org_id` must match path parameter)

| Path parameter | Type | Description |
|----------------|------|-------------|
| `org_id` | string | Organization ID |

**Returns** `200 OK` — `List<`[Agent](#agent)`>`

---

### Get agent

`GET /api/v1/agents/{agent_type}`

Return a single agent by its slug.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `agent_type` | string | Agent slug |

**Returns** `200 OK` — [Agent](#agent)

---

### Update agent

`PUT /api/v1/agents/{agent_type}`

Update any subset of an agent's fields. If `agent_type` is renamed and the agent has a linked Vobiz application, the application name is updated automatically.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `agent_type` | string | Current agent slug |

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_config` | object | **yes** | LLM, STT, TTS, and prompt settings (must be included even for partial updates) |
| `agent_type` | string | no | Rename the agent slug |
| `agent_category` | string | no | Category label |
| `phone_number` | string | no | Associated phone number |
| `app_id` | string | no | Telephony application ID |
| `greeting_message` | string | no | First message spoken when a call is answered. Write-only; not included in read responses. |
| `telephony_provider` | string | no | `"Vobiz"` or `"Plivo"` |
| `vobiz_app_id` | string | no | Linked Vobiz application ID |
| `vobiz_answer_url` | string | no | Vobiz answer webhook URL |
| `plivo_app_id` | string | no | Linked Plivo application ID |
| `plivo_answer_url` | string | no | Plivo answer webhook URL |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Delete agent (path)

`DELETE /api/v1/agents/{agent_type}`

Delete an agent by slug.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `agent_type` | string | Agent slug |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Delete agent (query param)

`DELETE /api/v1/agents?agent_type={value}`

Alternative delete form for agent types that contain `/` characters (URL-safe).

**Auth:** JWT

| Query parameter | Type | Required | Description |
|-----------------|------|----------|-------------|
| `agent_type` | string | yes | Agent slug |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Get agent config (internal)

`GET /api/v1/agents/config/{agent_type}`

Fetch full runtime config by agent slug. Called by the voice server at call-start time.

**Auth:** X-API-Key

| Path parameter | Type | Description |
|----------------|------|-------------|
| `agent_type` | string | Agent slug |

**Returns** `200 OK` — [Agent](#agent)

---

### Get agent config by ID (internal)

`GET /api/v1/agents/config/id/{agent_id}`

Fetch full runtime config by MongoDB `_id`. Used when only the agent ID is available (e.g. outbound calls).

**Auth:** X-API-Key

| Path parameter | Type | Description |
|----------------|------|-------------|
| `agent_id` | string | Agent UUID |

**Returns** `200 OK` — [Agent](#agent)

---

### Get agent by phone number (internal)

`GET /api/v1/agents/by-phone/{phone_number}`

Resolve an agent from an inbound phone number. URL-encode `+` as `%2B`.

**Auth:** X-API-Key

| Path parameter | Type | Description |
|----------------|------|-------------|
| `phone_number` | string | E.164 phone number, e.g. `%2B918071387434` |

**Returns** `200 OK` — [Agent](#agent)

---

## Meetings API

Base path: `/api/v1/meetings`

`/meetings` is the canonical record of every placed or received call.

### List meetings

`GET /api/v1/meetings`

Return a paginated list of calls for the caller's organization.

**Auth:** JWT

| Query parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `page` | integer | `1` | Page number (1-based) |
| `limit` | integer | `50` | Records per page (default 50; up to 10,000 when `for_export=true`) |
| `for_export` | boolean | `false` | Lift the 50-record cap (up to 10 000) for bulk export |
| `agent_type` | string | — | Filter by agent slug |
| `from_number` | string | — | Filter by caller number |
| `to_number` | string | — | Filter by destination number |
| `inbound` | boolean | — | `true` = inbound only, `false` = outbound only |
| `call_status` | string | — | `Busy` \| `Completed` \| `In Progress` |
| `date_from` | string | — | ISO date start, inclusive (e.g. `2026-01-01`) |
| `date_to` | string | — | ISO date end, inclusive |
| `date_sort_order` | string | `latest` | `latest` \| `oldest` |
| `duration_sort_order` | string | — | `longest` \| `shortest` — overrides `date_sort_order` when set |

**Returns** `200 OK` — [PaginatedMeetings](#paginatedmeetings)

---

### Get filter options

`GET /api/v1/meetings/filter-options`

Return distinct `agent_type` and phone number values used to populate the History filter dropdowns.

**Auth:** JWT

**Returns** `200 OK` — [MeetingFilterOptions](#meetingfilteroptions)

---

### Get meeting

`GET /api/v1/meetings/{meeting_id}`

Return full detail for a single call, including transcript and recording URL. The meeting must belong to the caller's organization.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `meeting_id` | string | Provider call UUID (e.g. Vobiz `CallUUID`) |

**Returns** `200 OK` — [Meeting](#meeting)

---

### Stream recording

`GET /api/v1/meetings/{meeting_id}/recording`

Proxy-stream the audio recording for a call. Returns a `StreamingResponse` (WAV, MP3, or M4A). The meeting must belong to the caller's organization. Recordings are served from MinIO object storage via `minio://` URLs.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `meeting_id` | string | Provider call UUID |

**Returns** `200 OK` — audio stream (`audio/wav`, `audio/mpeg`, or `audio/mp4`)

---

### Create meeting (internal)

`POST /api/v1/meetings`

Create the meeting record when a call starts. Called by the voice server.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `meeting_id` | string | yes | Provider call UUID |
| `agent_type` | string | yes | Agent slug |
| `org_id` | string | no | Organization ID |
| `start_time_utc` | string | no | ISO 8601 UTC timestamp |
| `end_time_utc` | string | no | ISO 8601 UTC timestamp (busy calls set this immediately) |
| `inbound` | boolean | no | `true` if the call was inbound |
| `from_number` | string | no | Caller number |
| `to_number` | string | no | Destination number |
| `call_busy` | boolean | no | `true` if the call was never answered |

**Returns** `201 Created` — [Meeting](#meeting)

---

### Update meeting (internal)

`PATCH /api/v1/meetings/{meeting_id}`

Update the end time when a call ends. Called by the voice server.

**Auth:** X-API-Key

| Path parameter | Type | Description |
|----------------|------|-------------|
| `meeting_id` | string | Provider call UUID |

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `end_time_utc` | string | yes | ISO 8601 UTC timestamp of call end |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

## Call Recordings API

Base path: `/api/v1/call-recordings`

### Save recording (internal)

`POST /api/v1/call-recordings`

Called by the voice server after a call completes to attach the recording URL, transcript, and call metadata to the meeting record.

**Auth:** None (internal service call)

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `call_sid` | string | yes | Provider call UUID (used to look up the meeting) |
| `transcript_url` | string | yes | MinIO or HTTP URL of the transcript file |
| `agent_type` | string | yes | Agent slug |
| `recording_url` | string | no | `minio://` URL of the audio recording |
| `transcript_content` | string | no | Full transcript text |
| `call_duration` | float | no | Duration in seconds |
| `end_time_utc` | string | no | ISO 8601 call end time |
| `org_id` | string | no | Organization ID |
| `latency_metrics` | object | no | Pipeline latency breakdown (STT/LLM/TTS timing) |

**Returns** `200 OK` — updated [Meeting](#meeting)

{% hint style="info" %}
To stream or download a recording in the frontend, use `GET /api/v1/meetings/{meeting_id}/recording` — not this endpoint.
{% endhint %}

---

## Campaigns API

Base path: `/api/v1/campaigns`

### Create campaign

`POST /api/v1/campaigns`

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `campaign_name` | string | yes | Unique name for this campaign |
| `org_id` | string | no | Defaults to caller's org |
| `agent_type` | string | no | Agent assigned to this campaign |
| `status` | string | no | Default: `"active"` |
| `campaign_information` | object | no | Arbitrary metadata |

**Returns** `201 Created` — [Campaign](#campaign)

---

### List campaigns for org

`GET /api/v1/campaigns/org/{org_id}`

**Auth:** JWT (caller's `org_id` must match path parameter)

| Path parameter | Type | Description |
|----------------|------|-------------|
| `org_id` | string | Organization ID |

**Returns** `200 OK` — `List<`[Campaign](#campaign)`>`

---

### Get campaign

`GET /api/v1/campaigns/{campaign_name}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `campaign_name` | string | Campaign name |

**Returns** `200 OK` — [Campaign](#campaign)

---

## Audience API

Base path: `/api/v1/audience`

Individual contact entries. Each audience row is a phone number with optional parameters.

### Create audience entry

`POST /api/v1/audience`

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `audience_name` | string | yes | Name / label for this contact |
| `phone_number` | string | yes | Phone number in E.164 format |
| `parameters` | object | no | Arbitrary key-value metadata passed to the agent at call time |

**Returns** `201 Created` — [AudienceEntry](#audienceentry)

---

### List audience entries

`GET /api/v1/audience`

**Auth:** JWT

| Query parameter | Type | Description |
|-----------------|------|-------------|
| `phone_number` | string | Filter to entries matching this phone number |

**Returns** `200 OK` — `List<`[AudienceEntry](#audienceentry)`>`

---

### Get audience entry

`GET /api/v1/audience/{audience_name}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `audience_name` | string | Audience entry name |

**Returns** `200 OK` — [AudienceEntry](#audienceentry)

---

## Batches API

Base path: `/api/v1/batches`

Batches manage outbound calling campaigns from CSV uploads. The `worker/*` sub-routes are called exclusively by the voice server.

### List batches

`GET /api/v1/batches`

**Auth:** JWT

| Query parameter | Type | Description |
|-----------------|------|-------------|
| `agent_type` | string | Filter by agent slug |

**Returns** `200 OK` — `List<`[Batch](#batch)`>`

---

### Upload batch CSV

`POST /api/v1/batches/upload`

Upload a CSV of phone numbers and parse it into contacts. The CSV must have at least one column named `phone_number` or `contact_number`.

**Auth:** JWT

**Request body** (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | yes | CSV file (`.csv` only) |
| `org_id` | string | yes | Must match caller's org |
| `batch_name` | string | yes | Display name for this batch |
| `agent_type` | string | yes | Agent that will place the calls |

**Returns** `201 Created` — [BatchUpload](#batchupload)

---

### Delete batch

`DELETE /api/v1/batches/{batch_id}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `batch_id` | string | Batch ID |

**Returns** `200 OK` — `{ "deleted": true }`

---

### Run batch

`POST /api/v1/batches/{batch_id}/run`

Start dialing all contacts in a batch. Forwards the request to the voice server, which executes calls with bounded concurrency.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `batch_id` | string | Batch ID |

**Request body** (optional)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_type` | string | no | Override agent for this run |
| `concurrency` | integer | no | Max simultaneous calls (1–20). Omit to use the batch's stored concurrency (defaults to 5 at upload time) |

**Returns** `200 OK` — voice server run response

---

### Stop batch

`POST /api/v1/batches/{batch_id}/stop`

Signal the voice server to stop scheduling new calls for this batch. In-flight calls complete normally.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `batch_id` | string | Batch ID |

**Returns** `200 OK` — voice server stop response

---

### Schedule batch

`POST /api/v1/batches/{batch_id}/schedule`

Schedule a batch to run at a future local time.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `batch_id` | string | Batch ID |

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `scheduled_at_local` | string | yes | Local datetime string (e.g. `"2026-07-01T09:00:00"`) |
| `timezone` | string | yes | IANA timezone (e.g. `"Asia/Kolkata"`) |
| `agent_type` | string | no | Override agent |
| `concurrency` | integer | no | Max simultaneous calls (1–20). Omit to use the batch's stored setting |

**Returns** `200 OK` — updated [Batch](#batch)

---

### Cancel scheduled batch

`POST /api/v1/batches/{batch_id}/schedule/cancel`

**Auth:** JWT

**Returns** `200 OK` — updated [Batch](#batch)

---

### Reschedule batch

`POST /api/v1/batches/{batch_id}/schedule/reschedule`

**Auth:** JWT

**Request body** — same as [Schedule batch](#schedule-batch)

**Returns** `200 OK` — updated [Batch](#batch)

---

### Claim next contact (internal)

`POST /api/v1/batches/worker/claim-next`

Atomically claim the next un-dialed contact in a batch for execution. Called by the voice server batch worker.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `batch_id` | string | yes | Batch ID |

**Returns** `200 OK` — `{ "contact": { "row_number": 1, "contact_number": "+91..." } }` or `{ "contact": null }` when exhausted

---

### Get agent call config (internal)

`POST /api/v1/batches/worker/agent-config`

Fetch the agent ID and caller ID needed to place calls for a batch.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `agent_type` | string | yes | Agent slug |

**Returns** `200 OK` — `{ "agent_id": "...", "caller_id": "...", ... }`

---

### Report contact result (internal)

`POST /api/v1/batches/worker/report`

Record the outcome (success or failure) for a single dialed contact.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `batch_id` | string | yes | Batch ID |
| `row_number` | integer | yes | Contact row index |
| `ok` | boolean | yes | `true` if the call was placed successfully |
| `error` | string | no | Error message if `ok` is `false` |

**Returns** `200 OK` — `{ "updated": true }`

---

### Finalize batch (internal)

`POST /api/v1/batches/worker/finalize`

Mark a batch as completed or stopped.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `batch_id` | string | yes | Batch ID |
| `stopped` | boolean | no | `true` if the batch was explicitly stopped before completion |

**Returns** `200 OK` — updated [Batch](#batch)

---

## Knowledge Base API

Base path: `/api/v1/knowledge`

Manage org-scoped PDF documents. Uploaded PDFs are chunked and embedded into a Chroma vector store for RAG at call time.

### Upload PDF

`POST /api/v1/knowledge/upload`

Upload a PDF and initiate background ingest. Returns immediately with status `processing`. Monitor ingest progress via `GET /api/v1/knowledge`; status transitions to `ready` on completion.

**Auth:** JWT

**Request body** (multipart/form-data)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `file` | file | yes | PDF file (`.pdf` only, max 10 MB) |
| `org_id` | string | yes | Must match caller's org |

**Returns** `201 Created` — [KnowledgeUpload](#knowledgeupload)

---

### List documents

`GET /api/v1/knowledge`

List all knowledge PDFs for the caller's organization, including ingest status.

**Auth:** JWT

**Returns** `200 OK` — `List<`[KnowledgeDocument](#knowledgedocument)`>`

---

### Delete document

`DELETE /api/v1/knowledge/{document_id}`

Delete a knowledge document and remove its vectors from Chroma.

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `document_id` | string | Document ID returned by upload |

**Returns** `200 OK` — `{ "deleted": true }`

---

## RAG API

Base path: `/api/v1/rag`

### Retrieve chunks (internal)

`POST /api/v1/rag/retrieve`

Retrieve the top-k most relevant knowledge chunks for a query. Called by the voice server at inference time to ground the LLM in org-specific context.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `question` | string | yes | Query text to search against |
| `top_k` | integer | no | Number of chunks to return (default: 3) |
| `document_ids` | List\<string\> | no | Restrict retrieval to specific document IDs |

**Returns** `200 OK` — [KnowledgeRetrieve](#knowledgeretrieve)

---

## Integrations API

Base path: `/api/v1/integrations`

Per-organization API keys for LLM providers, telephony, STT/TTS, and other services. Keys are stored encrypted and decrypted on read.

{% hint style="warning" %}
**Vobiz and Plivo credentials live here, not in `.env`.** Storing them in `.env` only works for single-tenant dev setups.
{% endhint %}

### Get key (internal)

`POST /api/v1/integrations/bot/get-api-key`

Fetch a decrypted API key by `model` name. Called by the voice server at runtime.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `model` | string | yes | Integration model name (e.g. `"OpenAI"`, `"VobizAuthId"`) |

**Returns** `200 OK` — [Integration](#integration)

---

### Create / update integration

`POST /api/v1/integrations`

Store or update a credential. If an integration for the same `org_id` and `model` already exists, it is updated (upsert). There is no separate PUT.

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Must match caller's org |
| `model` | string | yes | Unique model/service name |
| `api_key` | string | yes | The API key or secret to store (encrypted at rest) |

**Returns** `201 Created` — [Integration](#integration)

---

### List integrations

`GET /api/v1/integrations`

List all integrations for the caller's organization (keys returned decrypted).

**Auth:** JWT

**Returns** `200 OK` — `List<`[Integration](#integration)`>`

---

### Get integration

`GET /api/v1/integrations/{model}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `model` | string | Integration model name |

**Returns** `200 OK` — [Integration](#integration)

---

### Delete integration

`DELETE /api/v1/integrations/{model}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `model` | string | Integration model name |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

## Custom LLM Integrations API

Base path: `/api/v1/custom-llm-integrations`

Per-organization OpenAI-compatible custom LLM endpoints (e.g. self-hosted vLLM, JohnAI, or other proxies).

### Get config (internal)

`POST /api/v1/custom-llm-integrations/bot/get-config`

Fetch the full config (including decrypted API key) for a custom LLM by ID. Called by the voice server.

**Auth:** X-API-Key

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Organization ID |
| `custom_llm_id` | string | yes | Custom LLM config ID |

**Returns** `200 OK` — [CustomLLM](#customllm)

---

### List custom LLMs

`GET /api/v1/custom-llm-integrations`

**Auth:** JWT

**Returns** `200 OK` — `List<`[CustomLLM](#customllm)`>`

---

### Create custom LLM

`POST /api/v1/custom-llm-integrations`

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `org_id` | string | yes | Must match caller's org |
| `name` | string | yes | Display name (e.g. `"My vLLM Server"`) |
| `base_url` | string | yes | OpenAI-compatible base URL (e.g. `"https://llm.example.com/v1"`) |
| `api_key` | string | yes | API key for the custom LLM server |
| `model` | string | yes | Model name to pass in API requests (e.g. `"llama-3-8b"`) |

**Returns** `201 Created` — [CustomLLM](#customllm)

---

### Update custom LLM

`PUT /api/v1/custom-llm-integrations/{custom_llm_id}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `custom_llm_id` | string | Custom LLM config ID |

**Request body** (all optional)

| Parameter | Type | Description |
|-----------|------|-------------|
| `name` | string | Display name |
| `base_url` | string | Base URL |
| `api_key` | string | API key |
| `model` | string | Model name |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Delete custom LLM

`DELETE /api/v1/custom-llm-integrations/{custom_llm_id}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `custom_llm_id` | string | Custom LLM config ID |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

## Telephony APIs

### Vobiz (`/api/v1/vobiz`)

Manage Vobiz applications and phone number links. Credentials (`VobizAuthId`, `VobizAuthToken`) are read from the Integrations collection for the org.

#### Create application

`POST /api/v1/vobiz/application`

Create a Vobiz application bound to an agent's answer URL.

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_type` | string | yes | Agent slug |
| `answer_url` | string | yes | Public HTTPS URL that Vobiz calls when the call is answered |

**Returns** `201 Created` — `{ "status": "success", "app_id": "...", "message": "..." }`

---

#### Delete application

`DELETE /api/v1/vobiz/application/{application_id}`

**Auth:** JWT

| Path parameter | Type | Description |
|----------------|------|-------------|
| `application_id` | string | Vobiz application ID |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

#### List numbers

`GET /api/v1/vobiz/numbers`

List phone numbers on the Vobiz account for the caller's org.

**Auth:** JWT

**Returns** `200 OK` — `{ "status": "success", "numbers": ["..."] }`

---

#### Link number

`POST /api/v1/vobiz/numbers/link`

Assign a phone number to a Vobiz application.

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `phone_number` | string | yes | E.164 phone number |
| `application_id` | string | yes | Vobiz application ID |

**Returns** `201 Created` — `{ "status": "success", "message": "..." }`

---

#### Unlink number

`DELETE /api/v1/vobiz/numbers/unlink`

**Auth:** JWT

**Request body**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `phone_number` | string | yes | E.164 phone number to unlink |

**Returns** `200 OK` — `{ "status": "success", "message": "..." }`

---

### Plivo (`/api/v1/plivo`)

Same surface as Vobiz. Credentials (`PlivoAuthId`, `PlivoAuthToken`) come from Integrations.

#### Create application

`POST /api/v1/plivo/application`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `agent_type` | string | yes | Agent slug |
| `answer_url` | string | yes | Public HTTPS answer URL |

**Auth:** JWT — **Returns** `201 Created`

#### Delete application

`DELETE /api/v1/plivo/application/{application_id}` — **Auth:** JWT

#### List numbers

`GET /api/v1/plivo/numbers` — **Auth:** JWT

#### Link number

`POST /api/v1/plivo/numbers/link` — body: `phone_number`, `application_id` — **Auth:** JWT

#### Unlink number

`DELETE /api/v1/plivo/numbers/unlink` — body: `phone_number` — **Auth:** JWT

---

### Phone numbers (`/api/v1/phone-numbers`)

Track which phone numbers are attached to which agents.

#### List numbers for org

`GET /api/v1/phone-numbers/org/{org_id}`

**Auth:** JWT

**Returns** `200 OK` — `List<`[PhoneNumber](#phonenumber)`>`

---

#### Get number for agent

`GET /api/v1/phone-numbers/agent/{agent_type}`

**Auth:** JWT

**Returns** `200 OK` — [PhoneNumber](#phonenumber)

---

#### Attach number

`POST /api/v1/phone-numbers/attach`

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `phone_number` | string | yes | E.164 phone number |
| `provider` | string | yes | `"Vobiz"` or `"Plivo"` |
| `agent_type` | string | no | Agent slug to attach to |

**Returns** `201 Created` — [PhoneNumber](#phonenumber)

---

#### Detach number

`DELETE /api/v1/phone-numbers/detach`

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `phone_number` | string | yes | E.164 phone number to detach |

**Returns** `200 OK`

---

## Analytics API

Base path: `/api/v1/analytics`

### Get analytics

`GET /api/v1/analytics`

Return aggregate call metrics for the caller's organization. Calculated on-demand from the `CallLogs` collection.

**Auth:** JWT

| Query parameter | Type | Description |
|-----------------|------|-------------|
| `agent_type` | string | Filter metrics to a specific agent |
| `phone_number` | string | Filter metrics to a specific phone number |
| `start_date` | string | ISO date/datetime (inclusive) |
| `end_date` | string | ISO date/datetime (inclusive) |

**Returns** `200 OK` — [Analytics](#analytics)

---

## Members API

Base path: `/api/v1/members`

### Add member

`POST /api/v1/members/add-member`

Add a new user to an existing organization. Used for invite links where the new user has no token yet.

**Auth:** None (public)

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | New member's email |
| `password` | string | yes | New member's password |
| `name` | string | yes | Display name |
| `company_name` | string | yes | Company name |
| `org_id` | string | yes | Organization to join (from invite link URL param) |

**Returns** `201 Created`

---

### List members

`GET /api/v1/members/{org_id}`

**Auth:** JWT (caller must be member of this org)

| Path parameter | Type | Description |
|----------------|------|-------------|
| `org_id` | string | Organization ID |

**Returns** `200 OK` — `{ "status": "success", "members": [...] }`

---

### Remove member

`POST /api/v1/members/delete-member`

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | Email of the member to remove |
| `org_id` | string | yes | Must match caller's org |

**Returns** `200 OK`

---

### Transfer ownership

`POST /api/v1/members/transfer-ownership`

Transfer organization ownership to another existing member. The caller must be the current owner.

**Auth:** JWT

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `email` | string | yes | Email of the member to promote to owner |
| `org_id` | string | yes | Must match caller's org |

**Returns** `200 OK`

---

## Voice Server HTTP API

The voice server is a dedicated FastAPI service on port `7860`. Real-time audio is handled via WebSocket; the HTTP surface manages call control and routing operations.

Base URL: `http://localhost:7860`

| Method | Path | Auth | Description |
|--------|------|------|-------------|
| GET | `/` | None | Status — returns `{"service": "Telephony Server", "status": "running"}` |
| GET | `/health` | None | Health check |
| GET | `/docs` | None | Swagger UI |
| GET | `/telemetry/gpu` | None | GPU telemetry via nvidia-smi (`status: unavailable` on CPU-only deployments) |
| POST | `/outbound/call/` | None | Initiate an outbound call |
| POST | `/outbound/batch/run/` | None | Start batch calling worker (called by backend) |
| POST | `/outbound/batch/stop/` | None | Stop batch calling worker (called by backend) |
| GET/POST | `/answer` | None | Vobiz answer webhook — returns XML directing the call to `/agent/{agent_id}` |
| GET/POST | `/plivo/answer` | None | Plivo answer webhook |
| GET/POST | `/plivo/hangup` | None | Plivo hangup callback |
| WS | `/agent/{agent_id}` | None | Vobiz inbound/outbound audio stream |
| WS | `/plivo/agent/{agent_id}` | None | Plivo audio stream |
| WS | `/browser/agent/{agent_id}` | None | In-browser test client audio stream |

### Outbound call

`POST /outbound/call/`

Initiate an outbound call. The provider (Vobiz or Plivo) is selected automatically based on the agent's `telephony_provider` field.

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `customer_number` | string | yes | E.164 number to dial |
| `agent_id` | string | yes | Agent UUID |
| `custom_field` | string | no | Arbitrary string passed through to the call |
| `caller_id` | string | no | Caller ID to present; falls back to `VOBIZ_CALLER_ID` / `PLIVO_CALLER_ID` env var |

**Returns** `200 OK` — `{ "status": "success", "result": { ... } }`

---

## Types

### User

| Field | Type | Description |
|-------|------|-------------|
| `email` | string | User email |
| `name` | string | Display name |
| `org_id` | string | Organization ID |
| `company_name` | string | Company name |
| `created_at` | string | ISO 8601 creation timestamp |
| `is_owner` | boolean | `true` if this user is the organization owner |

---

### UserLoginResponse

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"success"` or `"fail"` |
| `message` | string | Human-readable description |
| `access_token` | string | JWT (expires in 30 minutes) |
| `token_type` | string | Always `"bearer"` |
| `org_id` | string | Organization ID of the authenticated user |

---

### Agent

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | string | Unique slug identifier |
| `agent_id` | string | Client-assigned UUID |
| `agent_config` | object | LLM, STT, TTS, and prompt settings |
| `org_id` | string | Organization ID |
| `agent_category` | string | Optional category label |
| `phone_number` | string | Associated phone number |
| `app_id` | string | Telephony application ID |
| `telephony_provider` | string | `"Vobiz"` or `"Plivo"` |
| `vobiz_app_id` | string | Linked Vobiz application ID |
| `vobiz_answer_url` | string | Vobiz answer webhook URL |
| `plivo_app_id` | string | Linked Plivo application ID |
| `plivo_answer_url` | string | Plivo answer webhook URL |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |

---

### Meeting

| Field | Type | Description |
|-------|------|-------------|
| `meeting_id` | string | Provider call UUID (e.g. Vobiz `CallUUID`) |
| `agent_type` | string | Agent slug |
| `org_id` | string | Organization ID |
| `agent_category` | string | Agent category at time of call |
| `agent_config` | object | Agent config snapshot at time of call |
| `inbound` | boolean | `true` if the call was inbound |
| `from_number` | string | Caller number |
| `to_number` | string | Destination number |
| `created_at` | string | ISO 8601 record creation time |
| `start_time_utc` | string | ISO 8601 call start time |
| `end_time_utc` | string | ISO 8601 call end time |
| `duration` | float | Call duration in seconds |
| `recording_url` | string | `minio://` URL of the audio recording |
| `transcript_url` | string | URL of the transcript file |
| `transcript_content` | string | Full transcript text |
| `transcript` | List\<object\> | Parsed transcript as structured turns |
| `call_busy` | boolean | `true` if the call was never answered |
| `latency_metrics` | object | Pipeline latency breakdown (STT/LLM/TTS) |

---

### PaginatedMeetings

| Field | Type | Description |
|-------|------|-------------|
| `items` | List\<[Meeting](#meeting)\> | Records for this page |
| `total` | integer | Total matching records across all pages |
| `page` | integer | Current page (1-based) |
| `limit` | integer | Page size used |

---

### MeetingFilterOptions

| Field | Type | Description |
|-------|------|-------------|
| `agent_types` | List\<string\> | Distinct agent slugs with call history |
| `from_numbers` | List\<string\> | Distinct caller numbers |
| `to_numbers` | List\<string\> | Distinct destination numbers |

---

### Campaign

| Field | Type | Description |
|-------|------|-------------|
| `campaign_name` | string | Campaign name |
| `org_id` | string | Organization ID |
| `agent_type` | string | Assigned agent |
| `status` | string | `"active"` or `"inactive"` |
| `campaign_information` | object | Arbitrary metadata |

---

### AudienceEntry

| Field | Type | Description |
|-------|------|-------------|
| `audience_name` | string | Contact name / label |
| `phone_number` | string | E.164 phone number |
| `parameters` | object | Arbitrary metadata passed to the agent |

---

### Batch

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | string | Batch ID |
| `org_id` | string | Organization ID |
| `batch_name` | string | Display name |
| `agent_type` | string | Agent slug |
| `concurrency` | integer | Max simultaneous calls |
| `original_filename` | string | Uploaded CSV filename |
| `status` | string | `uploaded` \| `ready_for_processing` \| `processing` \| `completed` \| `failed` |
| `execution_status` | string | `not_started` \| `queued` \| `running` \| `paused` \| `completed` \| `failed` |
| `total_contacts` | integer | Total rows in CSV |
| `valid_contacts` | integer | Rows with valid phone numbers |
| `invalid_contacts` | integer | Rows with invalid/missing numbers |
| `attempted_calls` | integer | Calls placed so far |
| `successful_calls` | integer | Calls successfully connected |
| `failed_calls` | integer | Calls that failed to place |
| `error_message` | string | Last error if status is `failed` |
| `schedule_mode` | string | `run_now` \| `scheduled` |
| `scheduled_at_utc` | string | ISO 8601 scheduled run time (UTC) |
| `scheduled_timezone` | string | IANA timezone of the scheduled time |
| `scheduled_status` | string | `none` \| `scheduled` \| `triggered` \| `canceled` |
| `scheduled_by` | string | Email of the user who scheduled the run |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |

---

### BatchUpload

Returned immediately after CSV upload. A subset of [Batch](#batch):

| Field | Type | Description |
|-------|------|-------------|
| `batch_id` | string | Batch ID |
| `org_id` | string | Organization ID |
| `batch_name` | string | Display name |
| `agent_type` | string | Agent slug |
| `concurrency` | integer | Max simultaneous calls (default `5`) |
| `original_filename` | string | Uploaded CSV filename |
| `status` | string | Always `"uploaded"` |
| `total_contacts` | integer | Total rows |
| `valid_contacts` | integer | Valid rows |
| `invalid_contacts` | integer | Invalid rows |
| `schedule_mode` | string | Fixed value `"run_now"` at upload time |
| `scheduled_status` | string | Fixed value `"none"` at upload time |
| `created_at` | string | ISO 8601 creation timestamp |

---

### KnowledgeDocument

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Document ID |
| `org_id` | string | Organization ID |
| `original_filename` | string | Uploaded filename |
| `status` | string | `processing` \| `ready` \| `failed` |
| `chunk_count` | integer | Number of chunks embedded |
| `embedding_model` | string | Embedding model used |
| `error_message` | string | Set if status is `failed` |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |

---

### KnowledgeUpload

Returned immediately after PDF upload:

| Field | Type | Description |
|-------|------|-------------|
| `document_id` | string | Document ID |
| `org_id` | string | Organization ID |
| `original_filename` | string | Uploaded filename |
| `status` | string | Always `"processing"` |

---

### KnowledgeRetrieve

| Field | Type | Description |
|-------|------|-------------|
| `chunks` | List\<[KnowledgeChunk](#knowledgechunk)\> | Retrieved chunks in relevance order |

---

### KnowledgeChunk

| Field | Type | Description |
|-------|------|-------------|
| `chunk_id` | string | Chroma chunk ID |
| `document_id` | string | Source document ID |
| `source_filename` | string | Original PDF filename |
| `text` | string | Chunk text |
| `distance` | float | Vector distance (lower = more relevant) |

---

### Integration

| Field | Type | Description |
|-------|------|-------------|
| `org_id` | string | Organization ID |
| `model` | string | Service name / key (e.g. `"OpenAI"`, `"VobizAuthId"`) |
| `api_key` | string | Decrypted API key |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |

---

### CustomLLM

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Config ID |
| `org_id` | string | Organization ID |
| `name` | string | Display name |
| `base_url` | string | OpenAI-compatible base URL |
| `model` | string | Model name passed in API requests |
| `api_key` | string | API key (decrypted) |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |

---

### PhoneNumber

| Field | Type | Description |
|-------|------|-------------|
| `phone_number` | string | E.164 phone number |
| `provider` | string | `"Vobiz"` or `"Plivo"` |
| `agent_type` | string | Currently attached agent slug |
| `org_id` | string | Organization ID |
| `created_at` | string | ISO 8601 creation timestamp |
| `updated_at` | string | ISO 8601 last-updated timestamp |
| `last_link_action` | string | `"attached"` or `"detached"` |
| `last_link_agent_type` | string | Agent involved in last link action |
| `last_link_by_email` | string | Email of user who performed last action |
| `last_link_at` | string | ISO 8601 timestamp of last action |

---

### Analytics

| Field | Type | Description |
|-------|------|-------------|
| `org_id` | string | Organization ID |
| `calls_attempted` | integer | Total call records |
| `calls_connected` | integer | Calls that were answered |
| `average_call_duration` | float | Average duration of connected calls (minutes) |
| `total_minutes_connected` | float | Sum of all connected call durations (minutes) |
| `most_used_agent` | string | Agent slug with the highest call count |
| `most_used_agent_count` | integer | Call count for the most-used agent |
| `agent_breakdown` | List\<[AgentBreakdown](#agentbreakdown)\> | Per-agent call counts |
| `calculated_at` | string | ISO 8601 timestamp of calculation |
| `start_date` | string | Applied date filter start |
| `end_date` | string | Applied date filter end |

---

### AgentBreakdown

| Field | Type | Description |
|-------|------|-------------|
| `agent_type` | string | Agent slug |
| `call_count` | integer | Number of calls placed by this agent |

---

## See also

- [websocket-api.md](websocket-api.md) — Real-time audio streaming protocol
- [endpoints-cheatsheet.md](endpoints-cheatsheet.md) — One-page quick-reference
- [environment-variables.md](environment-variables.md) — All configuration variables
- [../concepts/architecture.md](../concepts/architecture.md) — How the REST API fits into the system
