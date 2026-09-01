---
description: Running Voicera day to day with curl and the OpenAPI console.
---

# Operating via the API

Voicera's core stack ships no user interface. Everything an operator does — creating agents, attaching numbers, placing calls, reading transcripts, running campaigns — is an HTTP request. This page is the working set of those requests.

{% hint style="info" %}
A dashboard (Beta) exists on the `dev-frontend` branch, but it is not part of the Compose stack and several of its screens are not wired to the API. See [Dashboard (Beta)](../../developer/frontend/overview.md). Nothing on this page depends on it.
{% endhint %}

## `/docs` is your console

`apps/api/app/main.py` mounts FastAPI's interactive documentation at `docs_url="/docs"` and ReDoc at `redoc_url="/redoc"`. Both are enabled unconditionally.

| URL | What it is |
| --- | --- |
| `http://localhost:8000/docs` | Swagger UI. Every route, every schema, and an **Authorize** button that puts a Bearer token on subsequent calls. |
| `http://localhost:8000/redoc` | The same OpenAPI document, read-only, better for reading long schemas. |
| `http://localhost:8000/openapi.json` | The raw document. Feed it to a client generator. |

`/docs` is generated from the running code, so it is always correct for the version you are on. When this documentation and `/docs` disagree, `/docs` wins. Use it as the authority for exact field names; use the [REST API reference](../../api-reference/overview.md) for the narrative.

{% hint style="warning" %}
`/docs`, `/redoc`, and `/openapi.json` are unauthenticated and publish your full API surface. Block them at the reverse proxy on any deployment reachable from outside your network. See [Security hardening](../deployment/security-hardening.md).
{% endhint %}

## Getting a token

Every operator route takes `Authorization: Bearer <jwt>`. There are two ways to get one and they are for different callers.

The first user signs up, which creates the organisation and makes them its `super_admin`:

```bash
export API=http://localhost:8000

curl -X POST "$API/api/v1/users/signup" \
  -H 'Content-Type: application/json' \
  -d '{
    "email": "you@example.com",
    "password": "YOUR_PASSWORD",
    "organisation_name": "Your Org"
  }'
```

Afterwards, log in:

```bash
curl -X POST "$API/api/v1/users/login" \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"YOUR_PASSWORD"}'
```

```json
{
  "status": "success",
  "message": "Login successful",
  "access_token": "eyJhbGciOiJIUzI1NiIs…",
  "token_type": "bearer",
  "org_id": "YOUR_ORG_ID",
  "role": "super_admin",
  "organisations": [{"org_id": "YOUR_ORG_ID", "name": "Your Org", "role": "super_admin"}]
}
```

Capture it into a shell variable:

```bash
export TOKEN=$(curl -s -X POST "$API/api/v1/users/login" \
  -H 'Content-Type: application/json' \
  -d '{"email":"you@example.com","password":"YOUR_PASSWORD"}' \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
```

The token carries `org_id`. If you belong to several organisations, `POST /api/v1/users/switch-organisation` issues a new token scoped to another one and persists it as your default for next login:

```bash
curl -X POST "$API/api/v1/users/switch-organisation" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "OTHER_ORG_ID"}'
```

The second path is for services, not people. `POST /api/v1/users/bot/token` exchanges the shared `INTERNAL_API_KEY` for an org-scoped JWT with the `admin` role. The runtime uses it to write call artifacts back. You will not normally call it by hand.

```bash
curl -X POST "$API/api/v1/users/bot/token" \
  -H "X-API-Key: YOUR_INTERNAL_API_KEY" \
  -H 'Content-Type: application/json' \
  -d '{"org_id": "YOUR_ORG_ID"}'
```

A handful of routes take `X-API-Key` directly instead of a Bearer token — `GET /api/v1/agents/by-phone/{phone_number}`, `POST /api/v1/rag/retrieve`, and `POST /api/v1/campaign/internal/call-status`. They are service-to-service and are not part of an operator's routine.

## Keeping a token fresh

`ACCESS_TOKEN_EXPIRE_MINUTES` in `apps/api/app/config.py` defaults to **30**. There is no refresh-token endpoint: when a token expires, you log in again.

| Behaviour | Consequence |
| --- | --- |
| 30-minute lifetime | A shell that sat idle over lunch has a dead `$TOKEN`. `401 Invalid authentication credentials` is almost always this. |
| No refresh route | Scripts must re-login, not refresh. |
| `SECRET_KEY` signs the token | Restarting the API with a **changed or unset** `SECRET_KEY` invalidates every issued token immediately. |

That last row is worth care. When `SECRET_KEY` is unset, `apps/api/app/auth.py` generates a temporary random key at import and logs a warning, so every restart invalidates every token — and two API replicas would reject each other's. Compose refuses to start without it, but a bare `uvicorn` on your host does not. See [Security hardening](../deployment/security-hardening.md).

To lengthen the lifetime, set `ACCESS_TOKEN_EXPIRE_MINUTES` in the root `.env` and restart the API. It is absent from `.env.example`; add it.

A re-login helper for long sessions:

```bash
login() {
  export TOKEN=$(curl -s -X POST "$API/api/v1/users/login" \
    -H 'Content-Type: application/json' \
    -d "{\"email\":\"$VOICERA_EMAIL\",\"password\":\"$VOICERA_PASSWORD\"}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')
}
```

## The tasks you do most

Each recipe assumes `$API` and `$TOKEN` are set. Placeholders are `YOUR_AGENT_ID`, `YOUR_ORG_ID`, `YOUR_CALL_ID`.

### Add provider credentials

Do this first — agents validate their model configuration against configured providers. Credentials are org-scoped, provider-level (one key set covers that vendor's STT, TTS, and LLM), and Fernet-encrypted at rest.

Find out which fields a provider wants:

```bash
curl "$API/api/v1/auth/catalog/openai" -H "Authorization: Bearer $TOKEN"
```

Store them. Only secret fields belong in `auth`:

```bash
curl -X POST "$API/api/v1/auth" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"provider": "openai", "auth": {"api_key": "sk-…"}}'
```

Check what is configured, and remove one:

```bash
curl "$API/api/v1/auth/configured" -H "Authorization: Bearer $TOKEN"
curl -X DELETE "$API/api/v1/auth/openai" -H "Authorization: Bearer $TOKEN"
```

`POST /api/v1/auth` and `DELETE /api/v1/auth/{provider}` need `admin` or `super_admin`. Reading is open to any member, with secrets masked. See [Provider credentials (ProviderAuth)](../concepts/provider-auth.md).

### Create an agent

Browse the catalogs first — they are generated from the [provider registry](../concepts/provider-registry.md), so they are always current:

```bash
curl "$API/api/v1/configuration/stt" -H "Authorization: Bearer $TOKEN"
curl "$API/api/v1/configuration/tts?languages=hi" -H "Authorization: Bearer $TOKEN"
curl "$API/api/v1/configuration/llm" -H "Authorization: Bearer $TOKEN"
curl "$API/api/v1/configuration/telephony" -H "Authorization: Bearer $TOKEN"
```

Then create. A `telephony` agent requires `telephony_provider` and provisions an application at the provider on create; a `websocket` agent must not send `telephony_provider`.

```bash
curl -X POST "$API/api/v1/agents" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "Support Agent",
    "agent_category": "telephony",
    "telephony_provider": "vobiz",
    "config": {
      "schema_version": 1,
      "prompts": {
        "system_prompt": "You are a helpful phone support agent.",
        "greeting_message": "Hello! How can I help you today?"
      },
      "language": {"primary": "en", "secondary": []},
      "models": {
        "stt_config": {"provider": "deepgram", "model": "nova-3-general", "language": "en"},
        "tts_config": {"provider": "cartesia", "model": "sonic-3.5", "language": "en", "voice": "3faa81ae-d3d8-4ab1-9e44-e50e46d33c30"},
        "llm_config": {"provider": "openai", "model": "gpt-4.1", "base_url": "https://api.openai.com/v1"}
      }
    }
  }'
```

The `agents` path has **no trailing slash**. `422` means config validation failed — the message names the field. Full field reference in [Agent configuration](../../developer/reference/agent-configuration.md).

```bash
curl "$API/api/v1/agents" -H "Authorization: Bearer $TOKEN"
curl "$API/api/v1/agents/YOUR_AGENT_ID" -H "Authorization: Bearer $TOKEN"
```

### Attach a number

See what your telephony account holds, then attach:

```bash
curl "$API/api/v1/phone-numbers/providers/vobiz/inventory" \
  -H "Authorization: Bearer $TOKEN"

curl -X POST "$API/api/v1/phone-numbers/attach" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "phone_number": "+919000000000",
    "provider": "vobiz",
    "agent_id": "YOUR_AGENT_ID"
  }'
```

With `agent_id`, this also links the number to the agent's provider application, so inbound calls route to it. Omit `agent_id` to import into inventory only.

Detach is a `DELETE` **with a body**:

```bash
curl -X DELETE "$API/api/v1/phone-numbers/detach" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"phone_number": "+919000000000"}'
```

Detach unlinks at the provider and clears the agent association, keeping the inventory row.

### Place a test call

```bash
curl -X POST "$API/api/v1/calls/outbound" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": "YOUR_AGENT_ID",
    "to_number": "+919876543210",
    "custom_variables": {"customer_name": "Asha"}
  }'
```

`custom_variables` override the agent's `config.custom_variables` defaults for this call only. Add `from_number` to override the caller ID.

The response carries a `call_id`. Everything afterwards keys off it.

### Check a call's transcript

```bash
curl "$API/api/v1/calls/YOUR_CALL_ID" -H "Authorization: Bearer $TOKEN"
```

Read the artifacts through the API proxy, which streams them out of MinIO with your JWT enforced:

```bash
curl "$API/api/v1/calls/YOUR_CALL_ID/transcript" -H "Authorization: Bearer $TOKEN"
curl "$API/api/v1/calls/YOUR_CALL_ID/recording" -H "Authorization: Bearer $TOKEN" -o recording.wav
```

Both return `404` until the runtime uploads the artifact at the end of the call, and `404` permanently for a call that produced none.

{% hint style="info" %}
Browser websocket sessions create a `call_type: web` CallLog, so they produce transcripts and recordings alongside telephony calls. Pre-register one with `POST /api/v1/calls/web` to know the `call_id` up front.
{% endhint %}

List an organisation's calls:

```bash
curl "$API/api/v1/calls/org/YOUR_ORG_ID?limit=50&offset=0" \
  -H "Authorization: Bearer $TOKEN"
```

### Invite a member

```bash
curl -X POST "$API/api/v1/members/invite" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"email": "colleague@example.com", "password": "THEIR_INITIAL_PASSWORD"}'
```

The invite sets the member's initial password directly; there is no email invitation flow. They then log in normally. Requires `admin` or `super_admin`.

```bash
curl "$API/api/v1/members/YOUR_ORG_ID" -H "Authorization: Bearer $TOKEN"

curl -X POST "$API/api/v1/members/assign-admin" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{"email": "colleague@example.com"}'
```

Promoting and removing members requires `super_admin`. See [Multi-tenancy and roles](../concepts/multi-tenancy.md).

### Start a campaign

Three calls: upload, create, start.

```bash
curl -X POST "$API/api/v1/campaign/upload" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@contacts.csv"

curl -X POST "$API/api/v1/campaign/create" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d '{
    "name": "October outreach",
    "agent_id": "YOUR_AGENT_ID",
    "source_type": "csv",
    "source_id": "SOURCE_ID_FROM_UPLOAD",
    "max_concurrency": 5
  }'

curl -X POST "$API/api/v1/campaign/YOUR_CAMPAIGN_ID/start" \
  -H "Authorization: Bearer $TOKEN"
```

The CSV contract, the retry and circuit-breaker blocks, and the report are all in [Running a campaign](running-a-campaign.md).

### Check health

No authentication needed on any of these.

```bash
curl http://localhost:8000/health
curl http://localhost:7860/health
```

```json
{"status": "ok", "database": "up"}
```

Details, including the model-server gateway, are in [Daily operations](operations.md).

## Scripting

Three habits make API-driven operation bearable.

**Keep credentials out of the command.** Put them in the environment and let the shell interpolate:

```bash
export API=http://localhost:8000
export VOICERA_EMAIL=you@example.com
export VOICERA_PASSWORD='…'
```

**Fail loudly.** `curl` exits `0` on a `4xx` by default, which turns a failed script into a silently wrong one. Use `--fail-with-body`:

```bash
curl --fail-with-body -s -X POST "$API/api/v1/agents" \
  -H "Authorization: Bearer $TOKEN" \
  -H 'Content-Type: application/json' \
  -d @agent.json
```

**Keep bodies in files.** Agent configurations are long and quoting them inline invites mistakes. `-d @agent.json` reads from disk and diffs in version control.

A poll loop for a campaign, using only Python's standard library:

```bash
while true; do
  curl -s "$API/api/v1/campaign/YOUR_CAMPAIGN_ID/progress" \
    -H "Authorization: Bearer $TOKEN" \
  | python3 -c 'import sys,json; d=json.load(sys.stdin); print(d["state"], d["processed_rows"], "/", d["total_rows"])'
  sleep 30
done
```

## Recommended tooling

| Tool | Use it for |
| --- | --- |
| `/docs` | Exploring, and one-off calls with the Authorize button. Fastest way to get a body shape right. |
| `curl` | Everything scripted. Present in every container and CI image. |
| `python3 -c` | Parsing JSON without adding a dependency. It is already installed wherever the stack runs. |
| `jq` | Nicer than `python3 -c` if you have it. Not required by anything here. |
| An OpenAPI client generator | Fed from `/openapi.json`, when you are building a real integration rather than operating by hand. |
| `mongosh` | Reading FerretDB directly when the API cannot tell you something. Port **27018** on the host. |

Voicera ships no CLI. There is no `voicerctl`; `scripts/` contains only `start_docker.sh` and `stop_services.sh`.

## Related

* [REST API](../../api-reference/overview.md)
* [Endpoints cheatsheet](../../api-reference/endpoints-cheatsheet.md)
* [Running a campaign](running-a-campaign.md)
* [Daily operations](operations.md)
* [FAQ](faq.md)
