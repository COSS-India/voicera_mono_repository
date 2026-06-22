---
description: General VoicEra failures — login, dashboard, JWT auth, MongoDB, MinIO, and services that refuse to start.
---

# Common Issues

Start here when something is broken but you cannot yet tell whether it is voice, telephony, or deployment. This page covers the first symptoms operators and hosting partners hit: the dashboard does not load, login fails, the backend returns 401 or 500, MongoDB refuses connections, or MinIO rejects uploads.

If you can already narrow the problem down, jump directly to:

- [voice-and-audio.md](voice-and-audio.md) — STT, TTS, no/distorted audio, GPU OOM
- [telephony.md](telephony.md) — Vobiz, inbound/outbound calls, webhooks, public voice URLs
- [deployment.md](deployment.md) — Docker, ports, env vars, volumes, TLS, nginx

For what a healthy system looks like before you start digging, see [../guides/operator/dashboard-tour.md](../guides/operator/dashboard-tour.md) and [../guides/operator/operations.md](../guides/operator/operations.md).

---

## Dashboard and login

### Dashboard shows a blank page or connection error

**Symptom:** Browser shows a blank page, "This site can't be reached", or a 404 when opening the dashboard URL.

**Cause:** The frontend container is not running, or `NEXT_PUBLIC_API_BASE_URL` points to a backend the browser cannot reach.

**Fix:**

```bash
# Is the frontend container up?
docker-compose ps frontend
docker-compose logs frontend

# Is the URL reachable at all?
curl -I http://localhost:3000

# Verify the API base URL the frontend was built with
cat voicera_frontend/.env.local
# Expected:
# NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

If `NEXT_PUBLIC_API_BASE_URL` is wrong, fix the value and rebuild the frontend image — Next.js bakes public env vars at build time.

### Login credentials rejected

**Symptom:** Login form returns "Invalid credentials" even with the right email and password, or the page loops back to the login screen.

**Cause:** The user does not exist in MongoDB, stale tokens in browser storage, or the backend cannot reach MongoDB.

**Fix:**

```bash
# Does the user exist?
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.users.findOne({email:'user@example.com'})"

# Create a test user via the backend
docker-compose exec backend python -c \
  "from app.services.auth_service import create_user; create_user('user@example.com', 'password123')"
```

Then in the browser DevTools console:

```js
localStorage.clear()
sessionStorage.clear()
location.reload()
```

For the default seeded accounts, see [../quickstart/default-credentials.md](../quickstart/default-credentials.md).

{% hint style="warning" %}
If you redeploy with a fresh MongoDB volume, all users are wiped. Re-seed before the first login attempt or you will chase a phantom "auth bug".
{% endhint %}

---

## Backend authentication and API

### 401 Unauthorized on protected endpoints

**Symptom:** Every API call returns `401 Unauthorized`, even immediately after logging in.

**Cause:** Missing or expired JWT, wrong `Authorization` header format, or a JWT secret mismatch between issuing and verifying processes.

**Fix:**

```bash
# Issue a fresh token
curl -X POST http://localhost:8000/api/v1/users/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# Use the token explicitly
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/v1/agents
```

If the backend was restarted with a different `JWT_SECRET_KEY`, all previously issued tokens are invalid — log out and log in again. See [../reference/environment-variables.md](../reference/environment-variables.md) for the JWT-related vars.

### 500 Internal Server Error

**Symptom:** API returns `500 Internal Server Error` with no useful body.

**Cause:** Unhandled exception in the backend — most often MongoDB unreachable, a missing API key, or a code-level dependency error after a partial rebuild.

**Fix:**

```bash
# The real error is in the logs
docker-compose logs backend | grep -i -E "error|traceback"

# Check the most common culprits
docker-compose logs mongodb
cat voicera_backend/.env | grep -i api

# Force a clean rebuild if dependencies look off
docker-compose build --no-cache backend
docker-compose up -d backend
```

### Slow API responses (>1s)

**Symptom:** `/agents`, `/call_logs`, or dashboard lists take seconds to load.

**Cause:** Missing MongoDB indexes on hot collections, an exhausted connection pool, or the backend container hitting its CPU/memory limit.

**Fix:**

```bash
# Inspect indexes on the call_logs collection
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.call_logs.getIndexes()"

# Connection pool stats
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.serverStatus().connections"

# Live backend resource usage
docker stats voicera_backend
```

If indexes are missing, run the index-creation script shipped with the backend service. See [../services/backend.md](../services/backend.md) for the index list.

---

## MongoDB

### Connection refused from the backend

**Symptom:** Backend logs show `ServerSelectionTimeoutError` or `connection refused` against `mongodb:27017`.

**Cause:** Mongo started slower than the backend, the credentials are wrong, or both services are on different Docker networks.

**Fix:**

```bash
# Is mongo even up?
docker-compose ps mongodb
docker-compose logs mongodb

# Ping it from the backend
docker-compose exec backend ping -c 2 mongodb
docker exec voicera_mongodb mongosh --eval "db.adminCommand('ping')"

# Bring backend up after mongo is healthy
sleep 30
docker-compose up -d backend
```

If `ping` fails, both containers are not on the same network — see the network section in [deployment.md](deployment.md).

### MongoDB disk full

**Symptom:** Writes start failing with "no space left on device"; new calls don't appear in the dashboard.

**Cause:** Call logs and transcripts accumulated faster than the volume can hold.

**Fix:**

```bash
df -h /data/db

# Drop logs older than 30 days
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.call_logs.deleteMany({created_at: {'\$lt': new Date(Date.now() - 30*24*60*60*1000)}})"
```

For a permanent fix, expand the underlying volume or move to a managed Mongo. See [../guides/deployment/production.md](../guides/deployment/production.md).

### Replica set stuck in recovery

**Symptom:** `rs.status()` shows members stuck in `STARTUP2` or `RECOVERING`.

**Cause:** A node was restored from an outdated snapshot, or the oplog window is too short.

**Fix:**

```bash
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "rs.status()"
```

{% hint style="danger" %}
`rs.initiate()` on an existing cluster will reset the replica set and may lose data. Only run it on a fresh deployment or after confirming the primary's data is expendable.
{% endhint %}

---

## MinIO

### Bucket access denied / uploads fail

**Symptom:** Recording uploads or post-call assets fail with `AccessDenied` or `NoSuchBucket`.

**Cause:** The bucket was never created, credentials in the backend `.env` don't match what MinIO was started with, or the backend isn't reaching the MinIO container.

**Fix:**

```bash
# Is MinIO up?
docker-compose ps minio
docker-compose logs minio

# Create the bucket if it is missing
docker-compose exec minio mc mb minio/recordings

# Confirm the backend can talk to MinIO
docker-compose exec backend python -c \
  "from app.storage.minio_client import get_minio; print(get_minio().list_buckets())"
```

The MinIO console at `http://localhost:9001` is the fastest way to verify credentials and bucket policy. Default credentials and ports live in [../quickstart/default-credentials.md](../quickstart/default-credentials.md) and [../reference/ports-and-defaults.md](../reference/ports-and-defaults.md).

### Pre-signed download URLs expire or 403

**Symptom:** Operators click a recording link from the dashboard and get a 403 or "Request has expired".

**Cause:** Pre-signed URL TTL elapsed, the system clock drifted, or the bucket policy was changed to block public reads.

**Fix:** Regenerate the link from the dashboard. If the failure is consistent, check NTP on the host and confirm bucket policy:

```bash
docker-compose exec minio mc anonymous get minio/recordings
```

---

## Generic "service won't start"

### A service exits immediately or restart-loops

**Symptom:** `docker-compose ps` shows a service as `Restarting` or `Exited`; the dashboard or API is unreachable.

**Cause:** Missing env var, a port collision, the database it depends on is not ready, or a corrupted image.

**Fix:** Always read the logs first — the answer is almost always there:

```bash
docker-compose logs -f <service>
```

Then triage by category:

- Port collision or build failure → [deployment.md](deployment.md)
- Voice server crashing on startup → [voice-and-audio.md](voice-and-audio.md)
- Telephony container failing webhook validation → [telephony.md](telephony.md)

{% hint style="info" %}
When reporting a failure to your hosting partner, include: time of incident, service name, the last 50 log lines, and whether `docker-compose ps` shows the dependency containers (mongo, minio, backend) as healthy. See [../guides/operator/operations.md](../guides/operator/operations.md) for the full incident-report checklist.
{% endhint %}

---

## Health checks at a glance

```bash
# All services
docker-compose ps

# Individual probes
curl http://localhost:8000/health           # Backend
curl http://localhost:7860/health           # Voice server
curl http://localhost:9000/minio/health/live # MinIO
docker exec voicera_mongodb mongosh --eval "db.adminCommand('ping')"
```

A healthy response on each of these is the floor below which nothing else will work. For the full "is it working?" walkthrough, see [../quickstart/first-call.md](../quickstart/first-call.md).

---

## Next steps

- [voice-and-audio.md](voice-and-audio.md) — STT, TTS, audio quality, GPU
- [telephony.md](telephony.md) — Vobiz, webhooks, public voice URLs
- [deployment.md](deployment.md) — Docker, ports, env, TLS, nginx
- [../concepts/architecture.md](../concepts/architecture.md) — how the services fit together
- [../guides/operator/faq.md](../guides/operator/faq.md) — operator-facing FAQ
- [../reference/rest-api.md](../reference/rest-api.md) — backend API surface
