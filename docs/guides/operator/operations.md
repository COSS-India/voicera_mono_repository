---
description: What operators can do from the dashboard, what needs a hosting partner, and how to triage a failed call.
---

# Day-to-day operations

This page covers routine operator tasks — what you can do yourself in the dashboard, what to hand to your hosting partner, and a triage checklist for failed calls. Audience: non-technical operators.

## What you can do from the dashboard

No SSH or server access is needed for any of these.

| Task | Where in the dashboard |
|------|------------------------|
| View call history | **Meetings** |
| Confirm new calls are landing | **Meetings** — look for new rows after a test call |
| Create, edit, or disable agents | **Assistants** |
| Update telephony or AI credentials | **Integrations** |
| Link or unlink phone numbers | **Phone numbers** |
| Test an agent without a phone | **Test on Browser** on the agent card |
| Manage team access | **Members** (if your role allows) |
| Upload knowledge PDFs | **Knowledge base** |
| Run or monitor outbound batches | **Campaigns** |

{% hint style="info" %}
Log viewing and service management require Docker and shell access. These tasks are the responsibility of the hosting partner.
{% endhint %}

## What needs a hosting partner

These tasks need shell access to the server and are the responsibility of your hosting partner or DevOps team.

| Task | Command or action |
|------|-------------------|
| Restart the whole stack | `make stop-all-services` then `make start-all-services` |
| Restart one service | `docker compose restart backend` (or `voice_server`, `frontend`) |
| View live logs | `docker compose logs -f <service>` |
| Check disk space | `df -h` plus MinIO bucket growth |
| Database backup | MongoDB backup per organisation policy |
| Renew TLS certificates | Reverse proxy or certbot per policy |
| Adjust firewall | Open port 443 to the frontend and voice proxies |

See [Docker Compose deployment](../deployment/docker-compose.md) and [security hardening](../deployment/security-hardening.md) for details.

## Triage: a failed call

Work down the list in order. Stop as soon as you find the problem.

| # | Check | Action |
|---|-------|--------|
| 1 | **Integrations** | Are Vobiz Auth ID and Auth Token present? |
| 2 | **Phone numbers** | Is the number still linked to the correct agent? |
| 3 | **Test on Browser** | Does the agent respond in the browser test? If not, fix voice/AI before looking at telephony. |
| 4 | Browser works, phone fails | Likely a telephony webhook or public URL issue. Send call time and dialled number to your hosting partner. |
| 5 | Call connects but the agent is silent | AI provider key is missing or invalid, or the agent has the wrong STT/TTS provider. Partner checks `voice_server` logs. |
| 6 | Call drops immediately | Webhook or WSS URL is unreachable. Partner checks voice server logs and [public voice URLs](../deployment/public-voice-urls.md). |

When you escalate, send your hosting partner:

- The exact time the call happened.
- The phone number dialled.
- The agent name.
- Whether **Test on Browser** worked for the same agent.
- Any error you saw in the dashboard.

## Checking recordings

{% tabs %}
{% tab title="From the dashboard" %}
1. Open **Meetings**.
2. Click the call you want.
3. Use the recording playback link if your build shows one.
{% endtab %}

{% tab title="Via API" %}
Integrators can fetch recordings through the backend API:

```bash
GET /api/v1/call-recordings
```

See the [REST API reference](../../reference/rest-api.md).
{% endtab %}

{% tab title="Direct storage (partner)" %}
Recordings live in MinIO. Your hosting partner can use the MinIO console on port `9001` from a secure internal network. Do not expose port 9001 publicly.
{% endtab %}
{% endtabs %}

## Suggested maintenance schedule

| Frequency | Action | Who |
|-----------|--------|-----|
| Daily | Glance at **Meetings** for failed or unusually short calls | Operator |
| Weekly | Check disk space and recording storage growth | Hosting partner |
| Monthly | Confirm Integrations keys are still valid | Operator + partner |
| Quarterly | Review and rotate secrets | Partner — see [security hardening](../deployment/security-hardening.md) |
| As needed | Apply OS and Docker image updates | Hosting partner |

## Next steps

- [Operator FAQ](faq.md)
- [Dashboard tour](dashboard-tour.md)
- [Common issues](../../troubleshooting/common-issues.md)
- [Voice and audio troubleshooting](../../troubleshooting/voice-and-audio.md)
