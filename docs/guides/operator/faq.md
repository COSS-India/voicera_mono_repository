---
description: Answers to common operator questions about telephony, deployment, agents, and getting help.
---

# Operator FAQ

Short answers to the questions operators ask most often. For deeper material, follow the links inside each answer.

## Telephony and integrations

### Do I put Vobiz credentials in a `.env` file on the server?

No. For normal operation, enter **Vobiz Auth ID** and **Vobiz Auth Token** in **Dashboard → Integrations**. The voice server reads them from the database per organization. See [integrations service](../../services/integrations.md).

### What is JOHNAIC in the configuration?

It is a legacy name for your **public voice server URL** (HTTPS and WSS). Use your own domain rather than any example hostname from early deployments. See [public voice URLs](../deployment/public-voice-urls.md).

### Test on Browser works but phone calls do not — why?

Browser test only needs the voice server and AI keys. Phone calls also need:

- Correct telephony webhooks on the provider side.
- A public URL that is reachable from Vobiz or Plivo.
- The phone number linked to the agent in **Phone numbers**.

See [telephony model](../../concepts/telephony-model.md) and [telephony troubleshooting](../../troubleshooting/telephony.md).

### Call connects but the agent does not speak — why?

Usually the AI provider key is missing or invalid, or the agent has the wrong STT/TTS provider selected. Your hosting partner should check `voice_server` logs at the time of the call. See [voice and audio troubleshooting](../../troubleshooting/voice-and-audio.md).

## Deployment and environment

### Do we need a GPU?

Only if you run the optional local **AI4Bharat** STT and TTS servers. Cloud-only speech providers run fine without a GPU on the voice host. You still need to size CPU and memory for the expected number of concurrent calls. See [AI4Bharat STT](../../services/ai4bharat-stt.md).

### Does the system fall back from cloud speech to local speech automatically?

No. Each agent uses exactly the STT and TTS providers configured on it. There is no automatic provider fallback.

### What are the default MongoDB and MinIO passwords?

Development defaults are documented in the repo README and must be changed before production. See [security hardening](../deployment/security-hardening.md).

### Port already in use when starting?

Your hosting partner can run `make stop-all-ports` or stop the conflicting services on ports 3000, 8000, 7860, 27017, 9000, and 9001. See [deployment troubleshooting](../../troubleshooting/deployment.md).

## Dashboard and agents

### What is an "agent"?

A configured virtual voice assistant — language, voice, AI settings, instructions, and a linked phone number. It is not a human team member. See [agents, campaigns, and calls](../../concepts/agents-campaigns-calls.md).

### How do I test without spending phone minutes?

Use **Test on Browser** on the agent card under **Assistants**. See the [dashboard tour](dashboard-tour.md).

### How do I make an outbound call?

Run a **Campaign** from the dashboard or call `POST /outbound/call/` on the voice server directly. See [REST API](../../reference/rest-api.md) for the full outbound API reference.

## Documentation and license

### Where is the full API list?

Backend Swagger: `http://<your-backend>/docs`. Voice server Swagger: `http://<your-voice-host>/docs`. See [REST API](../../reference/rest-api.md) and [WebSocket API](../../reference/websocket-api.md).

### Is the software MIT or proprietary?

MIT License only — Copyright (c) 2026 COSS India. See the `LICENSE` file in the repository.

## Getting help

### What information should I send when reporting a problem?

- Time of the incident (with timezone).
- Agent name.
- Phone number, if the issue is call-related.
- Whether **Test on Browser** worked for the same agent.
- Any error shown in the dashboard.
- Your organization name.

### Who fixes server errors?

Your **hosting partner** uses Docker logs and shell access. As an **operator**, you check dashboard configuration first — Integrations, Phone numbers, and Test on Browser — before escalating. See [day-to-day operations](operations.md).

## Next steps

- [Dashboard tour](dashboard-tour.md)
- [Day-to-day operations](operations.md)
- [Common issues](../../troubleshooting/common-issues.md)
- [Glossary](../../concepts/glossary.md)
