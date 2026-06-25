# Brief: Operations guide (B7)

**Review gap:** No guide for restarting services, viewing logs, checking recordings, troubleshooting failed calls — especially without SSH.

**Audience:** Operators (dashboard-only) + hosting partner appendix (commands).

---

## What operators can do from the dashboard (no SSH)

| Task | Where in dashboard |
|------|-------------------|
| View call history | Meetings / calls section |
| See if calls are happening | New entries after test calls |
| Create / edit / disable agents | Assistants |
| Update telephony or AI credentials | Integrations |
| Link or unlink phone numbers | Phone numbers |
| Test agent without phone | Test on Browser |
| Manage team access | Members (if enabled for role) |
| Upload knowledge PDFs | Knowledge base (if used) |
| Run or monitor outbound batches | Campaigns / batches (if used) |

**Limitation today:** There is no built-in log viewer in the dashboard. Log access requires hosting partner.

---

## What needs a hosting partner

| Task | Command / action |
|------|------------------|
| Restart entire stack | `make stop-all-services` then `make start-all-services` |
| Restart one service | `docker compose restart backend` (or `voice_server`, `frontend`) |
| View live logs | `docker compose logs -f <service_name>` |
| Check disk space | OS tools (`df -h`); MinIO growth |
| Database backup | MongoDB backup procedure (org policy) |
| Renew TLS certificates | Reverse proxy / certbot per org policy |
| Change firewall | Allow 443 to voice and frontend proxies |

---

## Checking if a call was recorded

1. **Dashboard:** Open Meetings/calls for the time of call; check recording link if UI provides it.
2. **Storage:** Recordings stored in MinIO (technical detail — partner may use MinIO console on port 9001 **only on secure internal network**).
3. **Backend API:** `GET /api/v1/call-recordings` and meeting detail endpoints (integrators — see A2).

**Writer:** Confirm exact UI path for recordings on staging and document the steps in the operator guide.

---

## Troubleshooting a failed call (operator steps)

1. **Integrations** — Vobiz Auth ID and Token present? (Plivo keys if using Plivo.)
2. **Numbers** — Is the number still linked to the correct agent?
3. **Test on Browser** — If this fails, fix voice/AI before debugging telephony.
4. **Test on Browser works, phone fails** — Likely telephony URL or provider configuration (A3). Escalate to partner with call time and number dialed.
5. **Call connects but agent silent** — AI key missing or wrong STT/TTS provider on agent; partner checks `voice_server` logs.
6. **Call drops immediately** — Partner checks voice server and webhook reachability (HTTPS/WSS public URL — A8).

---

## Routine maintenance (suggested schedule — writer adapts)

| Frequency | Action | Who |
|-----------|--------|-----|
| Daily | Glance at Meetings for failed/short calls | Operator |
| Weekly | Confirm disk space / recording growth | Partner |
| Monthly | Review Integrations keys still valid | Operator + partner |
| Quarterly | Password/secret rotation review | Partner (A6) |
| As needed | Apply security updates to OS/Docker images | Partner |

---

## Future product note (optional in docs)

A dashboard-based **status page** and **log snippet viewer** would reduce SSH dependency — not available in current release.

---

## Related

- [B6-how-to-know-its-working.md](./B6-how-to-know-its-working.md)
- [B8-faq.md](./B8-faq.md)
- [A3-vobiz-telephony.md](./A3-vobiz-telephony.md)
