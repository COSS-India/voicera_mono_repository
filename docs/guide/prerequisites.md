# Prerequisites checklist

Use this checklist before deployment. Assign a **responsible party** (operator, programme owner, or hosting partner) for each item.

## People and accounts

- [ ] **Decision owner** — approves go-live
- [ ] **Operator(s)** — daily dashboard users
- [ ] **Hosting partner** — can run Docker/Linux if operators cannot
- [ ] **Vobiz account** with at least one phone number purchased
- [ ] **AI provider accounts** if using cloud speech/LLM (OpenAI, Bhashini, etc.), *or* plan for self-hosted [AI4Bharat](../services/ai4bharat-stt.md) servers
- [ ] **Email delivery** for signup and password reset (production SMTP/API — not Mailtrap)

## Server and infrastructure

- [ ] Linux server or VM with **Docker** and **Docker Compose**
- [ ] Sufficient **CPU/RAM** for expected concurrent calls (size with your hosting partner)
- [ ] **NVIDIA GPU** only if using local `ai4bharat_stt_server` / `ai4bharat_tts_server`
- [ ] Disk space for database and call recordings
- [ ] **Stable internet** with inbound HTTPS/WSS to your public voice URL

## Network and DNS

- [ ] **Public domain name** (recommended) for dashboard and voice server
- [ ] **HTTPS** certificate for dashboard
- [ ] **HTTPS + WSS** for [public voice server URLs](../deployment/public-voice-urls.md)
- [ ] Firewall: telephony provider can reach webhook URLs on port 443
- [ ] MongoDB and MinIO **not** exposed to the public internet

## Software package

- [ ] Copy of [voicera_mono_repository](https://github.com/COSS-India/voicera_mono_repository) on server
- [ ] Environment files configured by hosting partner (see [Configuration](../getting-started/configuration.md))
- [ ] **Vobiz credentials in Dashboard → Integrations** after services are up

## Before go-live

- [ ] All **default passwords changed** (MongoDB, MinIO) — [Security hardening](../deployment/security-hardening.md)
- [ ] Integrations filled in on dashboard
- [ ] Test agent created
- [ ] **Test on Browser** succeeded — [Verify it works](verification.md)
- [ ] At least one **test phone call** succeeded
- [ ] [MIT license](../legal/license.md) notice included in distributions if required by policy

## Sizing

| Item | Guidance |
|------|----------|
| RAM (voice only, cloud AI) | Size with hosting partner for expected concurrent calls |
| RAM + GPU (local AI4Bharat) | GPU strongly recommended; pinned VRAM **deferred** — [STT](../services/ai4bharat-stt.md#gpu-vram), [TTS](../services/ai4bharat-tts.md#gpu-vram) |
| Concurrent calls | Load-test on staging before production |
