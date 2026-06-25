---
description: Checklist of accounts, infrastructure, and network items needed before installing VoicEra.
---

# Prerequisites

Run through this checklist before installing VoicEra. It applies to operators, programme owners, and hosting partners preparing a server for development or production.

{% hint style="info" %}
Assign a responsible party (operator, programme owner, or hosting partner) for each item. Anything marked as production-only can be skipped for a local test install.
{% endhint %}

## People and accounts

- [ ] Decision owner who approves go-live
- [ ] Operator(s) who will use the dashboard daily
- [ ] Hosting partner who can run Docker and Linux commands
- [ ] Vobiz account with at least one phone number purchased
- [ ] AI provider accounts if using cloud speech or LLM (OpenAI, Bhashini, etc.), or plan for self-hosted AI4Bharat servers
- [ ] Email delivery for signup and password reset (production SMTP or API — not Mailtrap)

## Server and infrastructure

| Item | Minimum | Recommended |
|------|---------|-------------|
| OS | Linux, macOS, or Windows (WSL2) | Ubuntu 20.04 LTS or newer |
| RAM | 8 GB | 16 GB+ |
| CPU | 2 cores | 4+ cores |
| Disk | 50 GB | 100 GB+ NVMe SSD |
| Docker | 20.10+ | latest stable |
| Docker Compose | 1.29+ | latest stable |
| GPU | not required for cloud AI | NVIDIA CUDA GPU if running local AI4Bharat STT/TTS |

{% hint style="warning" %}
Size CPU, RAM, and disk with your hosting partner based on expected concurrent calls. Load-test on staging before production.
{% endhint %}

## Network and DNS

- [ ] Public domain name for dashboard and voice server (recommended)
- [ ] HTTPS certificate for dashboard
- [ ] HTTPS and WSS for the public voice server URL
- [ ] Firewall allows the telephony provider to reach webhook URLs on port 443
- [ ] MongoDB (27017) and MinIO (9000/9001) are not exposed to the public internet
- [ ] Stable internet with inbound connections allowed to your public voice URL

## Software package

- [ ] Copy of `voicera_mono_repository` on the server
- [ ] Environment files prepared for backend, frontend, voice server, and (optional) AI4Bharat servers
- [ ] Vobiz credentials ready to enter in **Dashboard → Integrations** once services are up

## Optional tools

| Tool | When you need it |
|------|------------------|
| Node.js 18+ | Local frontend development outside Docker |
| Python 3.10+ | Local voice server or backend development |
| Make | Convenience commands from the repo Makefile |
| Ngrok | Exposing a local voice server during testing |

## Next steps

- [install-and-run.md](install-and-run.md)
- [default-credentials.md](default-credentials.md)
- [../reference/environment-variables.md](../reference/environment-variables.md)
- [../reference/ports-and-defaults.md](../reference/ports-and-defaults.md)
