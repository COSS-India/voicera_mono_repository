---
description: What the JOHNAIC environment variables mean and how to set the public voice server URL for telephony webhooks and WebSockets.
---

# Public voice server URLs

Environment variables named `JOHNAIC_*` are not a third-party product. They hold the **public base URL of your deployed voice server** — the address on the internet that telephony providers call for webhooks and that browsers connect to for live audio.

The name is a legacy from an early deployment hostname (`johnaic.com`). Current documentation uses **public voice server URL**.

## What they are used for

- Telephony provider webhooks (Vobiz, Plivo) that signal incoming calls
- WebSocket live-audio connections from telephony bridges and from the dashboard's **Test on Browser** feature
- Answer URLs that the frontend builds when you create an agent

## Variables

| Variable | Where set | Meaning |
|----------|-----------|---------|
| `JOHNAIC_SERVER_URL` | voice server `.env` | Public HTTPS base, e.g. `https://voice.example.gov.in` |
| `JOHNAIC_WEBSOCKET_URL` | voice server `.env` | Public WSS base, e.g. `wss://voice.example.gov.in` |
| `NEXT_PUBLIC_JOHNAIC_SERVER_URL` | frontend `.env.local` | Same HTTPS base; the dashboard uses it to build `/answer?agent_id=...` |
| `NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL` | frontend `.env.local` (optional) | Explicit WSS for **Test on Browser**; derived from the HTTPS base if absent |

The voice server itself listens on port `7860` inside the container. The public URL points at your reverse proxy, which terminates TLS and forwards to `7860`.

## URLs the system builds

| Use case | Pattern |
|----------|---------|
| Vobiz answer webhook | `{JOHNAIC_SERVER_URL}/answer?agent_id={uuid}` |
| Plivo answer webhook | `{JOHNAIC_SERVER_URL}/plivo/answer?agent_id={uuid}` |
| Audio WebSocket (Vobiz) | `{JOHNAIC_WEBSOCKET_URL}/agent/{agent_id}` |
| Audio WebSocket (Plivo) | `{JOHNAIC_WEBSOCKET_URL}/plivo/agent/{agent_id}` |
| Browser test | `wss://.../agent/{agent_id}` |

Code references:

- `voice_2_voice_server/api/server.py`
- `voicera_frontend/app/(dashboard)/assistants/page.tsx`
- `voicera_frontend/components/assistants/test-browser-dialog.tsx`

## Requirements

- Reachable from the **public internet** so telephony callbacks succeed
- Use **WSS** (not `ws://`) when the dashboard is served over HTTPS, otherwise browsers block the WebSocket as mixed content
- DNS and the TLS certificate on the reverse proxy must match the hostname in these variables

## Example production values

```env
# voice_2_voice_server/.env
JOHNAIC_SERVER_URL=https://voice.example.gov.in
JOHNAIC_WEBSOCKET_URL=wss://voice.example.gov.in

# voicera_frontend/.env.local
NEXT_PUBLIC_JOHNAIC_SERVER_URL=https://voice.example.gov.in
NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL=wss://voice.example.gov.in
```

The nginx snippet in [Production deployment](production.md) terminates TLS for `voice.example.gov.in` and forwards both HTTP and WebSocket traffic to the voice server container.

## Local development

Use ngrok or a similar tunnel to expose port `7860` over HTTPS and WSS, then set both bases to the tunnel hostname. Prefer the `wss://` URL the tunnel provides:

```env
JOHNAIC_SERVER_URL=https://abcd1234.ngrok.app
JOHNAIC_WEBSOCKET_URL=wss://abcd1234.ngrok.app
```

{% hint style="warning" %}
Do not copy placeholder hostnames from example configs or earlier documentation. Use a domain name you control. Telephony providers cannot deliver calls to a hostname you do not own.
{% endhint %}

## Next steps

- [Deployment walkthrough](deployment-walkthrough.md)
- [Production deployment](production.md)
- [Telephony model](../../concepts/telephony-model.md)
- [Environment variables](../../reference/environment-variables.md)
