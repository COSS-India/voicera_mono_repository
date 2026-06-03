# Brief: What do I need before I start? (B2)

**Review gap:** No checklist of server specs, OS, internet, and accounts (e.g. Vobiz).

**Audience:** Programme manager + district IT + hosting partner.

**Writer task:** Turn into a printable checklist with tick boxes and a "responsible party" column.

---

## People and accounts

- [ ] **Decision owner** — who approves go-live
- [ ] **Operator(s)** — who will use the dashboard daily
- [ ] **Hosting partner** — who can run Docker/Linux commands (if operators cannot)
- [ ] **Vobiz account** (and/or **Plivo**) with at least one phone number purchased
- [ ] **AI provider accounts** if using cloud speech/LLM (OpenAI, Bhashini, etc.) — or plan to use only self-hosted AI4Bharat (see A5)
- [ ] **Email delivery** for user signup and password reset (production SMTP/API — not Mailtrap)

---

## Server and infrastructure

- [ ] Linux server or VM (cloud or on-prem) — confirm OS version with hosting partner
- [ ] **Docker** and **Docker Compose** installed (standard deployment path)
- [ ] Sufficient **CPU/RAM** for expected concurrent calls (engineering to publish sizing guide)
- [ ] **NVIDIA GPU** only if using local `ai4bharat_stt_server` / `ai4bharat_tts_server` (see A5)
- [ ] Disk space for database + call recordings (plan for growth)
- [ ] **Stable internet** with inbound connections allowed to your public voice URL

---

## Network and DNS

- [ ] **Public domain name** (recommended) for dashboard and voice server
- [ ] **HTTPS certificate** (valid TLS) for dashboard
- [ ] **HTTPS + WSS** for voice server public URL (see A8)
- [ ] Firewall: telephony provider can reach webhook URLs on port 443
- [ ] MongoDB and MinIO **not** exposed to public internet

---

## Software package

- [ ] Copy of `voicera_mono_repository` on server
- [ ] Environment files configured by hosting partner (see B5)
- [ ] **Vobiz/Plivo auth entered in dashboard Integrations** — not only in server files (see A3)

---

## Before go-live

- [ ] All **default passwords changed** (MongoDB, MinIO) — see A6
- [ ] Integrations filled in on dashboard
- [ ] Test agent created
- [ ] **Test on Browser** succeeded (B6)
- [ ] At least one **test phone call** succeeded (B6)
- [ ] License notice included in distributions if required by policy (MIT — see A7)

---

## Engineering placeholders

| Item | Status |
|------|--------|
| Minimum RAM (voice only, no local AI) | Size with hosting partner / staging load test |
| Minimum RAM (with local AI4Bharat) | Size with hosting partner / staging load test |
| GPU VRAM (STT) | **Deferred** — [AI4Bharat STT](../services/ai4bharat-stt.md#gpu-vram) |
| GPU VRAM (TTS) | **Deferred** — [AI4Bharat TTS](../services/ai4bharat-tts.md#gpu-vram) |
| Recommended concurrent calls per deployment | Load-test on staging |
