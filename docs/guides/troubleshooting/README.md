---
description: Symptom-first index — find your error message, get the fix.
---

# Troubleshooting

Organised by where the symptom shows up. If you are not sure which page applies, scan the table below for the message you actually saw.

{% hint style="info" %}
Before anything else: `docker compose ps` shows which containers are up, and `docker compose logs -f <service>` shows why one is not. Most issues resolve from those two commands.
{% endhint %}

## Symptom index

| What you see | Page |
| --- | --- |
| Stack will not start, containers restart-looping | [Common issues](common-issues.md) |
| `.env` not found, missing or empty secrets | [Common issues](common-issues.md) |
| Port already allocated | [Common issues](common-issues.md) |
| `ModuleNotFoundError: No module named 'apps'` | [Common issues](common-issues.md) |
| Connection refused to FerretDB | [Common issues](common-issues.md) |
| Call connects but there is no audio | [Voice and audio](voice-and-audio.md) |
| Audio in one direction only | [Voice and audio](voice-and-audio.md) |
| Speech is garbled, sped up, or robotic | [Voice and audio](voice-and-audio.md) |
| Agent will not stop talking when interrupted | [Voice and audio](voice-and-audio.md) |
| Greeting is cut off, or hold messages never play | [Voice and audio](voice-and-audio.md) |
| No transcript or recording after the call | [Voice and audio](voice-and-audio.md) |
| Provider cannot reach `/answer` | [Telephony](telephony.md) |
| `/answer` returns `400` | [Telephony](telephony.md) |
| Number will not attach or detach | [Telephony](telephony.md) |
| Recording never arrives from the provider | [Telephony](telephony.md) |
| Campaign stuck in a state, or paused itself | [Campaigns](campaigns.md) |
| Circuit breaker tripped | [Campaigns](campaigns.md) |
| Slot acquisition timeout, phone pool exhausted | [Campaigns](campaigns.md) |
| Worker not picking up jobs | [Campaigns](campaigns.md) |
| WSS fails behind a reverse proxy | [Deployment](deployment.md) |
| GPU not visible to a container | [Deployment](deployment.md) |
| Disk fills during a model build | [Deployment](deployment.md) |
| Volume permission errors | [Deployment](deployment.md) |

## The pages

* [**Common issues**](common-issues.md) — startup, environment, ports, imports, health checks.
* [**Voice and audio**](voice-and-audio.md) — anything you can hear, or cannot.
* [**Telephony**](telephony.md) — the provider boundary: webhooks, applications, numbers, recordings.
* [**Campaigns**](campaigns.md) — the orchestrator, ARQ worker, circuit breaker, and concurrency slots.
* [**Deployment**](deployment.md) — TLS, proxies, GPUs, disk, and volumes.

## Still stuck

Collect these before opening an issue: the failing command, `docker compose ps`, the last 100 log lines from the affected container, and your `.env` with **every secret redacted**.

## Related

* [Daily operations](../operator/operations.md) — health endpoints and which logs matter
* [Ports and defaults](../../developer/reference/ports-and-defaults.md)
