# Brief: How do I know it's working? (B6)

**Review gap:** No section describing what a non-technical operator should see when the system is healthy.

**Audience:** Operators and programme managers.

---

## After deployment (no phone call yet)

| Check | What to do | Good sign | Bad sign |
|-------|------------|-----------|----------|
| Dashboard loads | Open website in browser | Login page or home after login | Blank page, connection error |
| Can log in | Use provided account | Dashboard home visible | Error message, loop back to login |
| Backend alive | Hosting partner: open `/health` or check logs | Healthy response | 502/503, container stopped |
| Voice server alive | Hosting partner: `/health` on voice port | Healthy response | Container crash loop |

---

## Test without a phone (recommended first)

| Step | Good sign |
|------|-----------|
| Go to **Assistants** | List of agents (or empty list with create button) |
| Click **Test on Browser** on an agent | Dialog opens |
| Allow microphone when browser asks | Mic enabled |
| Wait a few seconds | Connected state; orb or UI shows active session |
| Speak | Agent responds with voice |
| End test | Dialog closes cleanly |

**If this fails:** Problem is likely voice server URL, Integrations AI keys, or network — not telephony. See B8 FAQ.

---

## Test with a phone call

| Step | Good sign |
|------|-----------|
| Integrations has Vobiz/Plivo credentials | Saved successfully |
| Number linked to agent on Numbers page | Shows agent name |
| Call the number from mobile | Call connects; hear agent greeting |
| After hangup | New row in **Meetings/calls** with time and status |
| Recording (if enabled) | Can play or download from dashboard or storage |

---

## What "healthy" looks like on screen (summary)

- Dashboard loads quickly, no red error banners
- **Test on Browser** connects and you hear the agent
- After a real call, **Meetings** shows the call with reasonable duration
- Operators can create and save agents without errors

---

## What to report if something fails

Give hosting partner:

1. Date and time of test
2. Agent name or ID
3. Phone number used (if applicable)
4. Any error message shown in the dashboard (if any)
5. Whether Test on Browser worked (yes/no)

Partner checks logs:

```bash
docker compose logs -f backend
docker compose logs -f voice_server
docker compose logs -f frontend
```

---

## Related

- [B7-operations-guide.md](./B7-operations-guide.md)
- [B8-faq.md](./B8-faq.md)
