---
description: Verify a fresh VoicEra install with a browser test and an end-to-end phone call.
---

# First call

Walk-through for operators and programme managers to confirm VoicEra is healthy after install. Run the browser test first — it isolates voice and AI configuration from telephony.

## 1. Confirm services are alive

| Check | Action | Good sign | Bad sign |
|-------|--------|-----------|----------|
| Dashboard loads | Open the frontend URL | Login or home page | Blank page, connection error |
| Can log in | Use the provided account | Dashboard home | Error or login loop |
| Backend alive | `GET /health` on backend port | Healthy response | 502/503 or container stopped |
| Voice server alive | `GET /health` on voice port | Healthy response | Crash loop |

Hosting partners can tail logs to diagnose failures:

```bash
docker compose logs -f backend
docker compose logs -f voice_server
docker compose logs -f frontend
```

## 2. Test on browser (do this first)

| Step | Good sign |
|------|-----------|
| Open **Assistants** | List loads, or empty state with a create button |
| Click **Test on Browser** | Dialog opens |
| Allow microphone access | Mic enabled |
| Wait a few seconds | Connected / active session UI appears |
| Speak | Agent responds with voice |
| End test | Dialog closes cleanly |

{% hint style="info" %}
If browser test fails the cause is almost always public voice URLs, missing AI keys in **Integrations**, or a network issue — not telephony. Check [common issues](../troubleshooting/common-issues.md) before testing a phone call.
{% endhint %}

## 3. Test with a phone call

| Step | Good sign |
|------|-----------|
| Enter Vobiz credentials in **Integrations** | Saved successfully |
| Link a number to the agent | Agent name appears on the Numbers page |
| Call the number | You hear the agent greeting |
| Hang up | New row appears in **Meetings** |
| Recording (if enabled) | Play or download available |

## Healthy system summary

- Dashboard loads without error banners
- **Test on Browser** works
- Real calls appear in **Meetings** with realistic durations
- Agents save without errors

## What to report when something fails

1. Date and time of the failure
2. Agent name or ID
3. Phone number used (if applicable)
4. Any error message shown in the dashboard
5. Whether **Test on Browser** worked (yes/no)

{% hint style="success" %}
Once browser test and a real phone call both succeed, harden the install before exposing it to users.
{% endhint %}

## Next steps

- [default-credentials.md](default-credentials.md)
- [../guides/operator/dashboard-tour.md](../guides/operator/dashboard-tour.md)
- [../guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md)
- [../troubleshooting/common-issues.md](../troubleshooting/common-issues.md)
