---
description: From an empty machine to a working voice agent, in order.
---

# Quickstart

Five pages, meant to be read in sequence. Together they take you from nothing to an agent that answers and speaks.

{% hint style="info" %}
Total time is roughly 20 minutes, most of it Docker pulling images. You need Docker, one AI provider API key, and — for real phone calls only — a public HTTPS hostname.
{% endhint %}

## The path

| Step | Page | What you end with |
| --- | --- | --- |
| 1 | [Prerequisites](prerequisites.md) | A machine that can run the stack, and the accounts you need. |
| 2 | [Install and run](install-and-run.md) | Nine containers up, API answering on `:8000`. |
| 3 | [Create your first agent](first-agent.md) | An agent configured with real provider credentials. |
| 4 | [Your first call](first-call.md) | A conversation you can hear, with a transcript. |
| 5 | [Generated secrets and defaults](secrets-and-defaults.md) | Knowing what was generated for you and what to change. |

## Before you start

You do **not** need a telephony account to try Voicera. A `websocket` agent runs entirely in the browser and needs only an STT, TTS, and LLM key. Add telephony when you want real phone numbers — [Your first call](first-call.md) covers both paths.

{% hint style="warning" %}
Run `./scripts/start_docker.sh`, not a bare `docker compose up`. The script generates `SECRET_KEY`, `INTERNAL_API_KEY`, and `PROVIDER_AUTH_ENCRYPTION_KEY` into `.env`; without them the stack starts misconfigured.
{% endhint %}

## Where next

Once a call works:

* [Architecture](../concepts/architecture.md) — what you just started
* [Running a campaign](../operator/running-a-campaign.md) — outbound at volume
* [Security hardening](../deployment/security-hardening.md) — before anyone else can reach it
