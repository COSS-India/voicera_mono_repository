# How to verify VoicEra is working

For operators and programme managers after deployment.

## Before any phone call

| Check | What to do | Good sign | Bad sign |
|-------|------------|-----------|----------|
| Dashboard loads | Open site in browser | Login or home page | Blank page, connection error |
| Can log in | Use provided account | Dashboard home | Error or login loop |
| Backend alive | Partner: `GET /health` or container logs | Healthy response | 502/503, container stopped |
| Voice server alive | Partner: `/health` on voice port | Healthy response | Crash loop |

## Test without a phone (do this first)

| Step | Good sign |
|------|-----------|
| Go to **Assistants** | List loads or empty with create button |
| Click **Test on Browser** | Dialog opens |
| Allow microphone | Mic enabled |
| Wait a few seconds | Connected / active session UI |
| Speak | Agent responds with voice |
| End test | Dialog closes cleanly |

If this fails, the issue is usually [public voice URLs](../deployment/public-voice-urls.md), **Integrations** AI keys, or network — not telephony. See [FAQ](faq.md).

## Test with a phone call

| Step | Good sign |
|------|-----------|
| **Integrations** has Vobiz credentials | Saved successfully |
| Number linked to agent | Shows agent name on Numbers page |
| Call the number | Hear agent greeting |
| After hangup | New row in **Meetings** |
| Recording (if enabled) | Play or download available |

## Healthy system (summary)

- Dashboard loads without error banners
- **Test on Browser** works
- Real calls appear in **Meetings** with reasonable duration
- Agents save without errors

## What to report when something fails

1. Date and time
2. Agent name or ID
3. Phone number used (if applicable)
4. Any error message shown in the dashboard
5. Whether **Test on Browser** worked (yes/no)

Hosting partner log commands:

```bash
docker compose logs -f backend
docker compose logs -f voice_server
docker compose logs -f frontend
```

## Related

- [Operations guide](operations.md)
- [FAQ](faq.md)
