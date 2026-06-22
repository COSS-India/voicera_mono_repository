---
description: Telephony failures — Vobiz auth, WebSocket disconnects, inbound numbers not ringing, outbound campaigns stalled, public voice URL and webhook signature errors.
---

# Telephony

Use this page when the dashboard works and "Test on Browser" works, but real phone calls don't. If browser test also fails, the problem is the voice path itself — start at [voice-and-audio.md](voice-and-audio.md).

The telephony path adds three pieces on top of the voice pipeline:

1. **Vobiz/Plivo credentials** stored in Dashboard → Integrations.
2. **A public, TLS-terminated voice URL** (HTTPS + WSS) reachable from the telephony provider.
3. **Webhook signatures** that the backend validates on every callback.

If any one of these is wrong, calls won't connect. For background, see [../concepts/telephony-model.md](../concepts/telephony-model.md).

---

## Provider credentials

### Vobiz authentication failures

**Symptom:** Outbound calls never dial, inbound numbers don't ring, and voice server or backend logs show `401`, `Invalid auth`, or `Auth ID/Token rejected` from Vobiz.

**Cause:** Vobiz Auth ID or Auth Token is missing, wrong, or saved on the wrong organization.

**Fix:**

1. Open Dashboard → Integrations and re-enter **Vobiz Auth ID** and **Vobiz Auth Token**. Save.
2. Confirm you are logged into the organization that owns the phone numbers.
3. Trigger a fresh call — credentials are read per-organization from the database, not from `.env`.

{% hint style="warning" %}
Do not put Vobiz credentials in a server-side `.env` file. The voice server reads them from MongoDB per organization. A `.env`-based override will only "work" for a single tenant and will silently break multi-org deployments.
{% endhint %}

### Webhook signature validation failures

**Symptom:** Backend logs show `Invalid webhook signature` or `403 Forbidden` on Vobiz/Plivo callbacks. Calls connect but post-call events (recording ready, hangup, status) never reach the dashboard.

**Cause:** Auth Token used to compute the signature does not match the one configured at the provider, or the public URL the provider hits is going through a proxy that rewrites the body.

**Fix:**

```bash
# Tail webhook attempts on the backend
docker-compose logs -f backend | grep -i webhook
```

- Re-copy the Auth Token from the provider console into Dashboard → Integrations.
- Ensure the reverse proxy (nginx, Cloudflare) forwards the request body unchanged. See [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md).
- For Plivo specifically, check that the X-Plivo-Signature header is forwarded by the proxy.

---

## Public voice URL

### Inbound number doesn't ring the agent

**Symptom:** Calling the number connects to a generic provider tone or hangs up immediately. Voice server logs show no incoming session.

**Cause:** The number is not linked to an agent, or the public voice URL configured at the provider is unreachable or wrong.

**Fix:**

1. Dashboard → Numbers: confirm the number shows the expected agent name.
2. From a host outside your network, confirm the public URL responds:

   ```bash
   curl -I https://voice.example.com/health
   ```

3. In the Vobiz/Plivo console, verify the Answer URL / Webhook URL matches the public voice URL exactly, with HTTPS (not HTTP).

### Public voice URL (legacy "JOHNAIC") misconfiguration

**Symptom:** Browser test works on `localhost` but external phone calls fail. Voice server logs show no inbound WebSocket from the provider.

**Cause:** The public voice URL (called `JOHNAIC` in older configs) is still pointing at an example hostname, an internal IP, or a non-TLS endpoint.

**Fix:** Set the public URL to your own TLS-terminated domain. Both forms must be reachable from the public internet:

- `https://voice.example.com` for control / health
- `wss://voice.example.com/ws` for the media stream

```bash
# Quick external check from a non-corporate network
curl -I https://voice.example.com/health
```

If the value is still set to a placeholder, fix it in the voice server `.env` and restart. See [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md) for the full setup (DNS, certs, nginx).

{% hint style="info" %}
"JOHNAIC" is just a legacy name for your public voice server URL. New deployments should use a hostname you control.
{% endhint %}

### WebSocket disconnects mid-call

**Symptom:** Calls connect and start working, then drop after 30–60 seconds. Voice server logs show `WebSocket closed` shortly after session start.

**Cause:** Reverse proxy idle timeout shorter than call duration, mismatched WebSocket headers (`Upgrade`, `Connection`), or TLS termination dropping long-lived connections.

**Fix:** For nginx, raise read/send timeouts and enable WebSocket upgrade:

```nginx
location /ws {
    proxy_pass http://voice_server:7860;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

For Cloudflare, ensure the proxied (orange-cloud) hostname has WebSockets enabled in the network settings.

---

## Inbound

### Call connects but agent never speaks

**Symptom:** Caller hears the line connect and then silence; no greeting plays.

**Cause:** Missing or invalid AI/speech API keys in Integrations, the agent's STT or TTS provider is not reachable, or the agent has no greeting configured.

**Fix:** This is almost always a voice-pipeline problem, not a telephony one. Confirm the agent works via Dashboard → Assistants → **Test on Browser**. If browser test also fails, jump to [voice-and-audio.md](voice-and-audio.md). If browser test works, recheck:

- Integrations has the required cloud API keys.
- The number-to-agent mapping (Dashboard → Numbers) points at the agent you tested.

### Call drops as soon as it connects

**Symptom:** Provider reports call ended in under 2 seconds; voice server logs show a session that opens and closes immediately.

**Cause:** Codec mismatch between provider and voice server, or the answer URL returns an error response.

**Fix:** Confirm the provider is configured for the codec the voice server speaks (PCM µ-law 8 kHz for PSTN). Check backend logs for a 4xx/5xx returned to the provider's answer URL.

---

## Outbound

### Outbound campaign won't dial

**Symptom:** A campaign starts but no calls are placed; campaign status stays at "queued" or "running" with 0 dialed.

**Cause:** Vobiz credentials missing, outbound caller ID not approved by the provider, or rate limit hit on the provider account.

**Fix:**

```bash
# Look for outbound attempt logs
docker-compose logs -f backend | grep -i -E "outbound|campaign"
docker-compose logs -f voice_server | grep -i outbound
```

- Verify credentials in Dashboard → Integrations (see Vobiz auth above).
- Confirm the source number is provisioned and verified at the provider.
- Check provider dashboard for rate limits or balance issues.

### Manual outbound call via API fails

**Symptom:** `POST /outbound/call/` returns 200 but no call is placed, or returns 4xx.

**Fix:** Send a deliberate test call and read both backend and voice-server logs at that exact timestamp:

```bash
curl -X POST http://localhost:7860/outbound/call/ \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{"to":"+919999999999","agent_id":"<agent-id>"}'
```

For the full request/response schema, see [../reference/rest-api.md](../reference/rest-api.md).

---

## Health-check matrix

| Check | Command / where | Healthy |
|-------|------------------|---------|
| Vobiz creds saved | Dashboard → Integrations | Both fields filled and saved |
| Number → agent mapping | Dashboard → Numbers | Agent name shown for each number |
| Public voice URL | `curl -I https://voice.example.com/health` from outside | HTTP 200 |
| WSS reachable | Browser DevTools on Test on Browser | WebSocket upgrades to 101 |
| Webhook validation | `docker-compose logs backend \| grep webhook` | No "invalid signature" lines |

If everything in the matrix is green and calls still fail, the issue is no longer telephony — go to [voice-and-audio.md](voice-and-audio.md) or [common-issues.md](common-issues.md).

---

## Next steps

- [common-issues.md](common-issues.md) — login, dashboard, MongoDB, MinIO
- [voice-and-audio.md](voice-and-audio.md) — STT, TTS, audio quality
- [deployment.md](deployment.md) — Docker, TLS, nginx, env vars
- [../concepts/telephony-model.md](../concepts/telephony-model.md) — how calls flow through VoicEra
- [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md) — public URL setup
- [../services/integrations.md](../services/integrations.md) — Integrations service reference
- [../reference/rest-api.md](../reference/rest-api.md) — backend API
- [../reference/websocket-api.md](../reference/websocket-api.md) — WebSocket protocol
