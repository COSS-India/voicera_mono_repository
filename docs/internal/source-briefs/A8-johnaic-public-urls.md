# Brief: What is JOHNAIC? (A8)

**Review gap:** Unclear what JOHNAIC means in voice server and frontend environment variables. Should be explained or renamed.

---

## What it actually is

**JOHNAIC is not a third-party product.** It is a **legacy internal/project name** for the **public base URL of your deployed voice server** — the address on the internet that:

- Telephony providers (Vobiz, Plivo) call for webhooks
- WebSocket clients connect to for live audio
- The frontend uses to build answer URLs and browser test URLs

It likely comes from an early deployment hostname (`johnaic.com`). New documentation should use plain language: **Public voice server URL**.

---

## Environment variables

| Variable | Where used | Meaning |
|----------|------------|---------|
| `JOHNAIC_SERVER_URL` | voice server | Public **HTTPS** base, e.g. `https://voice.example.gov.in` |
| `JOHNAIC_WEBSOCKET_URL` | voice server | Public **WSS** base, e.g. `wss://voice.example.gov.in` |
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | frontend | Same HTTPS base; builds `/answer?agent_id=...`, `/plivo/answer?agent_id=...` |
| `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL` | frontend (optional) | Explicit WSS for "Test on Browser"; else derived from server URL |

**Code references:**

- `voice_2_voice_server/api/server.py` — webhook XML, WebSocket URL construction
- `voicera_frontend/app/(dashboard)/assistants/page.tsx` — answer URL on agent create
- `voicera_frontend/components/assistants/test-browser-dialog.tsx` — browser WebSocket

---

## What URLs are built

| Use case | Example |
|----------|---------|
| Vobiz answer webhook | `{JOHNAIC_SERVER_URL}/answer?agent_id={uuid}` |
| Plivo answer webhook | `{JOHNAIC_SERVER_URL}/plivo/answer?agent_id={uuid}` |
| Audio WebSocket (Vobiz) | `{JOHNAIC_WEBSOCKET_URL}/agent/{agent_id}` |
| Audio WebSocket (Plivo) | `{JOHNAIC_WEBSOCKET_URL}/plivo/agent/{agent_id}` |
| Browser test | `wss://.../agent/{agent_id}` (see `talk-on-browser-feature.md`) |

---

## Writer guidance

1. Document for operators as **"Public voice server address"** with HTTPS and WSS examples.
2. Footnote: *Current environment variable names use `JOHNAIC_*`; a future release may rename to `VOICERA_PUBLIC_URL` / `VOICERA_PUBLIC_WS_URL`.*
3. **Do not** instruct customers to use `vobiz.johnaic.com` — that was one deployment example, not a universal endpoint.

---

## Requirements

- Must be reachable from the public internet (telephony provider callbacks).
- WebSocket path requires **WSS** when the dashboard is served over HTTPS.
- Must match DNS/TLS certificate on the reverse proxy in front of `voice_server`.
