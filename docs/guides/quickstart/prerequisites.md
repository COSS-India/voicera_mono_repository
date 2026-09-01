---
description: Hardware, software, and accounts you need before installing Voicera.
---

# Prerequisites

Work through this page before [Install and run](install-and-run.md). Most of it is quick; the accounts can take a day if you are procuring a phone number.

## Software

| Requirement | Version | Notes |
| --- | --- | --- |
| Docker Engine | 24 or newer | Docker Desktop on macOS and Windows |
| Docker Compose | v2 | Invoked as `docker compose`, not `docker-compose` |
| Git | any recent | To clone the repository |
| Python 3 | 3.11+ | Only for running from source. `start_docker.sh` also uses it to generate the encryption key. |

{% tabs %}
{% tab title="Ubuntu" %}
```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose-v2 git python3 python3-pip

# Run docker without sudo
sudo usermod -aG docker $USER
newgrp docker
```
{% endtab %}

{% tab title="macOS" %}
Install [Docker Desktop](https://www.docker.com/products/docker-desktop), then:

```bash
brew install git python@3.11
```
{% endtab %}

{% tab title="Windows" %}
1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop) with the WSL2 backend.
2. Enable WSL2 in Windows Features and restart.
3. Run all commands from inside a WSL2 shell, not PowerShell.
{% endtab %}
{% endtabs %}

Verify:

```bash
docker --version
docker compose version
docker run --rm hello-world
```

{% hint style="warning" %}
`start_docker.sh` needs `python3` with the `cryptography` package to generate `PROVIDER_AUTH_ENCRYPTION_KEY`. Without it the script stops and tells you to set the key manually.

```bash
pip install cryptography
```
{% endhint %}

## Hardware

For the core stack — API, runtime, FerretDB, Redis, MinIO — with cloud model providers:

| Resource | Minimum | Comfortable |
| --- | --- | --- |
| CPU | 2 cores | 4 cores |
| RAM | 4 GB | 8 GB |
| Disk | 20 GB | 50 GB+ |

Recordings and transcripts accumulate in MinIO, so size disk for your call volume.

A **GPU is not required** unless you self-host models. If you do, see [Running on GPUs](../../developer/model-server/gpu-operations.md) — model images are large and the first build needs substantial disk.

## Accounts

Voicera provides no telephony and no models. Bring your own.

### Model providers — required

At least one provider for each of speech-to-text, text-to-speech, and a language model. One vendor can cover all three: `openai`, `google`, and `sarvam` each register STT, TTS, and LLM.

Browse what is available once running:

```bash
curl -H "Authorization: Bearer $TOKEN" \
  http://localhost:8000/api/v1/configuration/stt
```

The alternative is self-hosting — see [Model server](../../developer/model-server/overview.md).

### Telephony — required for phone calls

A [Vobiz](../concepts/telephony-model.md) or Plivo account with at least one number. You need the account credentials and a number you can attach.

{% hint style="info" %}
You can skip telephony entirely at first. A `websocket` agent runs the same pipeline from a browser, with no telephony account and no call charges. See [Your first call](first-call.md).
{% endhint %}

### Optional

| Service | For | Variables |
| --- | --- | --- |
| Embeddings (OpenAI-compatible) | Knowledge base / RAG | `KB_EMBEDDING_API_KEY`, `KB_EMBEDDING_MODEL` |
| Mailtrap | Password-reset email | `MAILTRAP_API_TOKEN`, `MAILTRAP_FROM_EMAIL` |

Without embedding credentials, everything works except knowledge-base ingestion.

## Network

For local evaluation, none of this matters. For **real phone calls**, your telephony provider must reach your runtime from the public internet:

| Requirement | Why |
| --- | --- |
| A public hostname | The provider fetches `/answer` over HTTPS |
| TLS | Providers require HTTPS, and WSS for audio |
| WebSocket upgrade through your proxy | Audio is a WebSocket stream, not HTTP |
| `VOICE_SERVER_BASE_URL` set to that hostname | Baked into the provider application when an agent is created |

For testing, a tunnel such as `ngrok` or `cloudflared` works. See [Public voice URLs](../deployment/public-voice-urls.md).

### Ports

Published on the host by default:

| Port | Service |
| --- | --- |
| `8000` | API |
| `7860` | Runtime |
| `27018` | FerretDB |
| `9000` / `9001` | MinIO API and console |

PostgreSQL and Redis are **not** published. All ports are overridable — see [Ports and defaults](../../developer/reference/ports-and-defaults.md).

## Checklist

- [ ] Docker Engine and Compose v2 installed and working
- [ ] `python3` with `cryptography` available
- [ ] 20 GB+ free disk
- [ ] Credentials for at least one STT, one TTS, and one LLM provider
- [ ] A telephony account and number, if you want real calls
- [ ] A public HTTPS hostname, if you want real calls
- [ ] Host ports free, or overrides chosen

## Next

[Install and run](install-and-run.md)
