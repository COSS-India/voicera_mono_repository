# Brief: FAQ (B8)

**Review gap:** No FAQs for common failure points.

**Writer task:** Expand each into a short Q&A paragraph for end users.

---

## Telephony and Integrations

**Q: Do I put Vobiz username and password in a .env file on the server?**  
A: No. For normal operation, enter **Vobiz Auth ID** and **Vobiz Auth Token** in **Dashboard → Integrations**. The voice server reads them from the database per organization.

**Q: What is JOHNAIC in the configuration?**  
A: Legacy name for your **public voice server URL** (HTTPS and WSS). See [A8-johnaic-public-urls.md](./A8-johnaic-public-urls.md). Use your own domain, not an example hostname from early deployments.

**Q: Test on Browser works but phone calls do not — why?**  
A: Browser test only needs the voice server and AI keys. Phone calls also need correct telephony webhooks, public URL reachable from Vobiz/Plivo, and the phone number linked to the agent. See A3.

**Q: Call connects but the agent does not speak — why?**  
A: Often missing or invalid AI/speech API keys in Integrations, or wrong STT/TTS provider on the agent. Hosting partner should check voice server logs at the time of the call.

---

## Deployment and environment

**Q: Do we need a GPU?**  
A: Only if you run optional local **AI4Bharat** STT/TTS servers. Cloud-only speech providers can run without a GPU on the voice server (sizing still applies for concurrent calls).

**Q: Does the system automatically fall back from cloud to local speech?**  
A: No. Each agent uses the STT/TTS providers configured on that agent. See A4.

**Q: What are the default MinIO and MongoDB passwords?**  
A: Development defaults are documented in the technical README — they must be changed before production. See A6.

**Q: Port already in use when starting?**  
A: Hosting partner can run `make stop-all-ports` or stop conflicting services on ports 3000, 8000, 7860, 27017, 9000, 9001.

---

## Dashboard and agents

**Q: What is an "agent"?**  
A: A configured virtual voice assistant (language, voice, AI settings, instructions) — not a human team member.

**Q: How do I test without spending phone minutes?**  
A: Use **Test on Browser** on the agent card in Assistants.

**Q: How do I make an outbound call?**  
A: May use campaigns/batches in dashboard or technical API `POST /outbound/call/` on voice server — confirm which features are enabled on your deployment.

---

## Documentation and legal

**Q: Where is the full API list?**  
A: Backend: `http://<your-backend>/docs`. Voice server: `http://<your-voice-host>/docs`. See A2.

**Q: Is the software MIT or proprietary?**  
A: **MIT License** only — Copyright (c) 2026 COSS India. See `LICENSE` in the repository and A7.

---

## Getting help

**Q: What information should I send when reporting a problem?**  
A: Time of incident, agent name, phone number (if call-related), whether Test on Browser worked, any error message shown, and your organization name.

**Q: Who fixes server errors?**  
A: Your **hosting partner** uses Docker logs; operators use the dashboard for configuration checks (B7).

---

## Related briefs

| Topic | File |
|-------|------|
| Telephony flow | A3 |
| Public URLs | A8 |
| Security | A6 |
| Success checks | B6 |
