---
description: Container, proxy, volume, and image problems in a deployed stack.
---

# Deployment

Problems that appear once Voicera leaves your laptop: proxies, volumes, images, and GPUs.

## Containers restart in a loop

```bash
docker compose ps
docker compose logs <service> --tail 100
```

| Cause | Fix |
| --- | --- |
| Missing required secret | Compose aborts on `${SECRET_KEY:?...}`. Run `./scripts/start_docker.sh`. |
| Cannot reach the database | Check `postgres` and `ferretdb` are healthy first |
| Cannot reach Redis | Check `REDIS_PASSWORD` matches `REDIS_URL` |
| Out of memory | Check `docker stats` and the host's OOM killer |

A single connection error from `api` on a cold boot is expected: it waits for FerretDB only to *start*, not to be ready, and `restart: unless-stopped` recovers it within seconds. A persistent loop is a real failure.

## TLS and proxy problems

### The WebSocket never connects

The single most common deployment failure. Your proxy must upgrade the connection:

```nginx
proxy_http_version 1.1;
proxy_set_header Upgrade    $http_upgrade;
proxy_set_header Connection "upgrade";
```

Test it:

```bash
curl -i -N \
  -H "Connection: Upgrade" -H "Upgrade: websocket" \
  -H "Sec-WebSocket-Version: 13" -H "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==" \
  "https://voice.example.com/agent/$ORG_ID/$AGENT_ID"
```

`101 Switching Protocols` is correct.

### Calls drop after about a minute

nginx's default `proxy_read_timeout` is 60 seconds. A voice call is a long-lived connection:

```nginx
proxy_read_timeout  3600s;
proxy_send_timeout  3600s;
```

### Providers reject the webhook

Self-signed and expired certificates are refused. Use a publicly trusted certificate, and confirm the full chain is served:

```bash
openssl s_client -connect voice.example.com:443 -servername voice.example.com < /dev/null
```

## Volume and permission problems

### Data vanished after a restart

Something ran `docker compose down -v`. All four volumes are deleted together and there is no undo.

```bash
docker volume ls | grep voicera_oss
```

Restore from backup. See [Daily operations](../operator/operations.md).

### Permission denied on a volume

Usually a host-mounted path with mismatched ownership. Prefer named volumes — the reference stack already does — and check the host directory's owner if you replaced one with a bind-mount.

### The disk filled up

```bash
docker system df
du -sh /var/lib/docker/volumes/voicera_oss_minio_data
```

Recordings accumulate indefinitely; nothing expires automatically. Set a retention policy. `docker system prune` reclaims space from unused images and build cache — it does not touch named volumes.

## Image and build problems

### `ghcr.io` pull failures

The FerretDB and Postgres images come from GitHub Container Registry. If your network blocks it, mirror the images into your own registry and update the tags.

```bash
docker pull ghcr.io/ferretdb/ferretdb:2.7.0
```

### Builds fail near the end, on disk

Model-server images are large. Build **one at a time** — parallel builds double peak usage at the export stage, which is exactly where they fail.

### `minio/minio:latest` changed under you

That tag is unpinned, unlike the others. Pin a digest for reproducible deployments.

## GPU problems

Only relevant if you self-host models.

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi
```

| Symptom | Cause |
| --- | --- |
| No GPU inside the container | NVIDIA Container Toolkit not installed or not configured |
| Out of memory | Another process holds the GPU. Check `nvidia-smi`. |
| Cannot share the GPU | The GPU is in Exclusive Process mode; MPS needs Default mode |
| Model will not download | `ai4bharat/indic-parler-tts` is gated — supply a token with access |

See [Running on GPUs](../../developer/model-server/gpu-operations.md).

## Cross-replica problems

### Users are logged out at random

Your API replicas have different `SECRET_KEY` values, so each rejects tokens signed by the others. It must be identical everywhere — and if it is blank, each replica generates its own temporary key at startup.

### Calls fail when a runtime restarts

Live calls hold a WebSocket on one instance and do not survive its restart. Drain before restarting, and route with session affinity so `/answer` and the audio reach the same instance.

### A campaign dials twice

More than one campaign orchestrator. Run exactly one — see [Campaigns](campaigns.md).

## Health checks

```bash
curl -s localhost:8000/health
curl -s localhost:7860/health
curl -s localhost:8100/health
```

{% hint style="warning" %}
`GET /health` returns HTTP **200 even when the database is down** — only the body changes to `"status": "degraded"`. A probe that checks the status code alone will report a broken API as healthy. Parse the body.
{% endhint %}

## Related

* [Production deployment](../deployment/production.md)
* [Docker Compose](../deployment/docker-compose.md)
* [Security hardening](../deployment/security-hardening.md)
* [Common issues](common-issues.md)
