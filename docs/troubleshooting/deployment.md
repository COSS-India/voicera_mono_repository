---
description: Docker, ports, image builds, env vars, volumes, TLS, nginx, and container restart loops.
---

# Deployment

Use this page for failures that happen before any user-visible feature works: containers won't start, ports collide, images won't build, env vars aren't loading, volumes are wrong, or TLS/nginx terminates incorrectly. If services start cleanly but the app misbehaves, go to [common-issues.md](common-issues.md), [voice-and-audio.md](voice-and-audio.md), or [telephony.md](telephony.md).

For the supported deployment shapes, see [../guides/deployment/docker-compose.md](../guides/deployment/docker-compose.md) and [../guides/deployment/production.md](../guides/deployment/production.md).

---

## Container lifecycle

### Container exits immediately or restart-loops

**Symptom:** `docker-compose ps` shows a service in `Restarting` or `Exited`. The service never serves traffic.

**Cause:** Missing or malformed env var, port collision, a dependency container that hasn't come up yet, or a corrupted image.

**Fix:**

```bash
# Read the actual error
docker-compose logs <service>

# Common follow-ups:
# - Port already in use → see "Port collision" below
# - Missing env var → see "Env vars" below
# - DB connection failure → see common-issues.md

# Force a clean rebuild
docker-compose build --no-cache <service>
docker-compose up -d <service>
```

### Container is "up" but unhealthy

**Symptom:** `docker-compose ps` shows the service running, but the `/health` endpoint returns 500 or the dashboard says the service is down.

**Cause:** Service started but a downstream (Mongo, MinIO, voice server) is unreachable, or it crashed after the initial healthcheck.

**Fix:**

```bash
docker inspect voicera_backend | grep -A 20 "State"
docker-compose logs --tail=200 <service>
```

Restart only after fixing the downstream — restart loops mask the root cause.

---

## Ports

### Port already in use

**Symptom:**

```
Error response from daemon: Ports are not available: listen tcp 0.0.0.0:8000: bind: address already in use
```

**Cause:** Another process (often a previous VoicEra run, a local dev server, or an unrelated service) holds the port.

**Fix:**

```bash
# Find what holds the port
lsof -i :8000

# Kill it if it's stale
kill -9 <PID>

# Or remap the port in docker-compose.yml
# ports:
#   - "8001:8000"

# A repo-provided shortcut frees all VoicEra ports
make stop-all-ports
```

The default port map (3000 frontend, 8000 backend, 7860 voice server, 27017 mongo, 9000/9001 MinIO) is in [../reference/ports-and-defaults.md](../reference/ports-and-defaults.md).

---

## Image builds

### `docker-compose build` fails partway through

**Symptom:** Build aborts with a pip/npm error, a missing system package, or a network timeout.

**Cause:** Stale build cache, the host can't reach the package registry, or a base image was updated and now needs a different system dependency.

**Fix:**

```bash
# Clean rebuild, no cache
docker-compose build --no-cache <service>

# If the failure is network-related, retry with verbose logs
docker-compose build --progress=plain <service>

# Prune dangling layers to reclaim space
docker system prune -f
```

For frontend builds specifically, `npm install --legacy-peer-deps` is required if the lockfile has peer-dep conflicts.

### "No space left on device" during build

**Symptom:** Build fails with `no space left on device` even though `df -h` shows space free.

**Cause:** Docker's overlay filesystem is full even when the host disk isn't.

**Fix:**

```bash
docker system df
docker system prune -af --volumes
```

{% hint style="warning" %}
`docker system prune -af --volumes` deletes all stopped containers, unused images, networks, and unnamed volumes. Run it deliberately, not as a reflex.
{% endhint %}

---

## Environment variables

### Service starts but behaves as if `.env` is empty

**Symptom:** Backend defaults to `localhost` for Mongo, frontend points at the wrong API host, voice server uses placeholder JOHNAIC URL.

**Cause:** Wrong `.env` path, env file not mounted into the container, or values not exported to the process.

**Fix:**

```bash
# What does the container actually see?
docker-compose exec backend env | grep -E "MONGO|JWT|API"
docker-compose exec voice_server env | grep -E "STT|TTS|JOHNAIC|PUBLIC"

# Confirm the file on disk
cat voicera_backend/.env
```

For Next.js, remember `NEXT_PUBLIC_*` vars are baked at **build time**, not runtime — changing them requires a rebuild of the frontend image. See [../reference/environment-variables.md](../reference/environment-variables.md) for the full list and which are build-time vs runtime.

### Changes to `.env` aren't picked up

**Symptom:** You edited `.env`, restarted, but the new value isn't visible.

**Cause:** `docker-compose restart` doesn't reload env vars from the file — it reuses the existing container with its baked env.

**Fix:**

```bash
docker-compose up -d --force-recreate <service>
```

---

## Volumes

### Data disappears after restart

**Symptom:** Users, agents, recordings, or uploaded files vanish after `docker-compose down && up`.

**Cause:** A bind-mount points at an ephemeral host path, or the named volume was removed by a `docker-compose down -v`.

**Fix:**

```bash
# Inspect the volumes the service is using
docker inspect voicera_mongodb | grep -A 10 Mounts
docker volume ls | grep voicera
```

Never run `docker-compose down -v` against a deployment with data you want to keep. For production, mount named volumes (or external block storage) for Mongo and MinIO — see [../guides/deployment/production.md](../guides/deployment/production.md).

### "Permission denied" on a mounted volume

**Symptom:** Container logs show `permission denied` writing to `/data/db`, `/data/minio`, or an uploads directory.

**Cause:** The container's user does not own the bind-mount on the host.

**Fix:**

```bash
# Fix host-side ownership (use the UID/GID the container runs as)
sudo chown -R 1000:1000 /path/to/mounted/dir
```

For Mongo and MinIO, prefer named volumes over bind mounts — Docker manages the permissions.

---

## Networking

### Backend can't resolve `mongodb` or `minio`

**Symptom:** Backend logs show `Name or service not known` or `Temporary failure in name resolution`.

**Cause:** Services are on different Docker networks, or the network was rebuilt but a container still attached to the old one.

**Fix:**

```bash
docker network ls | grep voicera
docker network inspect voicera_mono_repository_voicera_network

# Verify DNS from inside the container
docker-compose exec backend nslookup mongodb
docker-compose exec backend ping -c 2 mongodb

# Last resort: rebuild the network
docker-compose down
docker network prune -f
docker-compose up -d
```

---

## TLS and reverse proxy

### TLS certificate errors

**Symptom:** Browsers warn about an invalid certificate, or `curl https://...` fails with `SSL certificate problem`.

**Cause:** Self-signed cert in production, expired cert, wrong hostname in the cert, or the cert chain is incomplete (missing intermediates).

**Fix:** Issue a real cert via Let's Encrypt (certbot, acme.sh) and configure your reverse proxy to serve the full chain. Verify:

```bash
echo | openssl s_client -showcerts -connect voice.example.com:443 -servername voice.example.com
curl -vI https://voice.example.com/health
```

For full guidance, see [../guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md).

### nginx returns 502 / Bad Gateway

**Symptom:** Reverse proxy returns `502 Bad Gateway` for the dashboard or voice URL.

**Cause:** Upstream container is down, the proxy is pointing at the wrong port, or the proxy can't resolve the Docker service name.

**Fix:**

```bash
# Confirm upstream is healthy from the host
curl http://localhost:8000/health
curl http://localhost:7860/health

# Tail nginx errors
tail -f /var/log/nginx/error.log
```

If nginx itself runs in Docker, make sure it shares a network with the services it proxies, and use service names (not `localhost`) in `proxy_pass`.

### WebSocket connections fail through nginx

**Symptom:** Browser test or telephony WebSocket gets a 400 or upgrades silently fail.

**Cause:** nginx is not configured to upgrade WebSocket connections, or `proxy_read_timeout` is shorter than a normal call.

**Fix:**

```nginx
location /ws {
    proxy_pass http://voice_server:7860;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

See [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md) for a complete nginx example.

---

## Development-mode quirks

### Code changes don't reflect

**Symptom:** You edit a file, the container doesn't pick it up.

**Cause:** Hot-reload is off, the source is not bind-mounted, or you're hitting a cached image.

**Fix:**

```bash
# Backend hot reload
RELOAD=true   # in voicera_backend/.env

# Frontend: usually fine with `npm run dev`, otherwise:
docker-compose build --no-cache frontend
docker-compose up -d frontend
```

See [../guides/developer/local-setup.md](../guides/developer/local-setup.md) for the recommended dev workflow.

### `ModuleNotFoundError` or `ImportError`

**Symptom:** Backend or voice server crashes on import after a dependency change.

**Cause:** Dependencies installed locally but not rebuilt into the image, or `requirements.txt` / `package.json` is out of sync with the lockfile.

**Fix:**

```bash
# Reinstall inside the container (temporary)
docker-compose exec backend pip install --no-cache-dir -r requirements.txt
docker-compose exec frontend npm install --legacy-peer-deps

# Permanent fix: rebuild the image
docker-compose build --no-cache <service>
docker-compose up -d <service>
```

For test-related dependency setup, see [../guides/developer/testing.md](../guides/developer/testing.md).

---

## Monitoring and quick triage

```bash
# Snapshot of all services
docker-compose ps

# Live resource usage
docker stats

# Bump a memory-starved container
docker update --memory 4g voicera_backend

# Service-by-service health
curl http://localhost:8000/health
curl http://localhost:7860/health
curl http://localhost:9000/minio/health/live
```

{% hint style="info" %}
When opening a ticket with your hosting partner, attach: output of `docker-compose ps`, last 200 lines of logs for the affected service, the relevant `.env` (with secrets redacted), and the reverse-proxy config if external traffic is involved.
{% endhint %}

---

## Next steps

- [common-issues.md](common-issues.md) — login, dashboard, MongoDB, MinIO
- [voice-and-audio.md](voice-and-audio.md) — STT, TTS, audio quality
- [telephony.md](telephony.md) — Vobiz, public URLs, webhooks
- [../guides/deployment/docker-compose.md](../guides/deployment/docker-compose.md) — base compose deployment
- [../guides/deployment/production.md](../guides/deployment/production.md) — production hardening
- [../guides/deployment/public-voice-urls.md](../guides/deployment/public-voice-urls.md) — public voice URL setup
- [../guides/deployment/security-hardening.md](../guides/deployment/security-hardening.md) — TLS and security
- [../reference/environment-variables.md](../reference/environment-variables.md) — env var reference
- [../reference/ports-and-defaults.md](../reference/ports-and-defaults.md) — default ports
