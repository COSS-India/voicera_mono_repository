# Parallel "live-translation" deployment (alongside prod)

Runs the `feat/live-translation` branch as a **second, isolated stack** on the same ace host,
next to `voicera-prod`. It hosts **no models** — its `voice_server` reuses the existing prod
STT/TTS. Public access is via its **own Cloudflare tunnel** (separate URL).

## What is / isn't shared with prod

| Component | Translate stack |
|---|---|
| frontend, backend, voice_server, mongodb, minio, nginx | **own** containers, image, volume (isolated) |
| STT | **reuses** prod `voicera-prod-stt-1:8001` |
| TTS | **reuses** prod `voicera-prod-nginx-1:8080/ttslb/` (keeps 3-replica round-robin) |
| GPU / model weights | none — not hosted here |
| Public URL | own Cloudflare named tunnel |

**Isolation guarantees (no prod interference):**
- Project name `voicera-translate` → containers `voicera-translate-*`, volumes
  `voicera-translate_*`, network `voicera-translate_internal`. No name collision with `voicera-prod-*`.
- Prod containers are **not restarted or reconfigured**. The only cross-project touch is the
  translate `voice_server` *joining* the prod network as an extra member (additive, read-only use).
- No new host ports. Prod's `127.0.0.1:8080` is untouched; translate nginx is only reached by its
  own cloudflared over the internal network.

**Name-collision note:** `voice_server` is on two networks; `backend`/`minio`/`nginx` exist in
*both* projects, so those short names are ambiguous for it. Its env therefore references every
service by **unique container name** (`voicera-translate-*`, `voicera-prod-*`). Do not "simplify"
those back to short names.

---

## 1. Prereqs (already true on ace)

- Prod stack (`voicera-prod`) up with `stt` + `tts` healthy.
- Docker + compose plugin, a Cloudflare account for the tunnel.

Confirm the prod network name and service container names:

```bash
docker network ls | grep voicera            # expect e.g. voicera-prod_voicera_network
docker ps --format '{{.Names}}' | grep -E 'voicera-prod-(stt|tts|nginx)'
# expect: voicera-prod-stt-1  voicera-prod-tts-1/2/3  voicera-prod-nginx-1
```

Confirm prod nginx actually exposes the TTS load-balancer route:

```bash
docker exec voicera-prod-nginx-1 grep -n ttslb /etc/nginx/conf.d/default.conf
# if this prints a /ttslb block, the TTS URL below works as-is.
# if it prints nothing, use INDIC_TTS_SERVER_URL=http://voicera-prod-tts-1:8002 instead
# (single replica, no round-robin) — see §3.
```

## 2. Get the branch onto the host

Use a **separate checkout** so prod's working tree is untouched:

```bash
cd ~
git clone https://github.com/COSS-India/voicera_mono_repository.git voicera_translate
cd voicera_translate
git checkout feat/live-translation
```

> If `deploy/compose/docker-compose.translate.yml`, `deploy/compose/.env.translate.example`,
> and `deploy/nginx/nginx.translate.conf` aren't on the branch yet, copy them from this repo.
> All paths below assume `~/voicera_translate`.

Set a convenience alias (add to `~/.bashrc`):

```bash
alias dcpt='docker compose -p voicera-translate \
  --env-file ~/voicera_translate/deploy/compose/.env.translate \
  -f ~/voicera_translate/deploy/compose/docker-compose.translate.yml'
```

## 3. Env files

### 3a. Compose-level (`deploy/compose/.env.translate`)

```bash
cd ~/voicera_translate/deploy/compose
cp -n .env.translate.example .env.translate
nano .env.translate
```

Fill: `MONGO_ROOT_*`, `MINIO_ROOT_*` (fresh creds, own volume), `PROD_NETWORK_NAME` (from §1),
`TUNNEL_TOKEN` (from §5).

### 3b. Service env files (derived from prod, with deltas)

These are **separate** from prod's `.env` so they carry the translate DB/MinIO/public URL.
Set `PUB` to your translate tunnel hostname first:

```bash
cd ~/voicera_translate
PUB=translate-sandbox.voicera.world          # <-- your tunnel hostname
MROOT=voicera_admin                          # match .env.translate MONGO_ROOT_USER
MPASS='<same-as-.env.translate>'             # match MONGO_ROOT_PASSWORD
SROOT=voicera_admin                          # match MINIO_ROOT_USER
SPASS='<same-as-.env.translate>'             # match MINIO_ROOT_PASSWORD
```

**Backend** — start from a filled prod-style backend env, override the deltas:

```bash
cp voicera_backend/.env voicera_backend/.env.translate      # or from env.example if none
sed -i -e '$a\' voicera_backend/.env.translate
tee -a voicera_backend/.env.translate >/dev/null <<EOF
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USER=$MROOT
MONGODB_PASSWORD=$MPASS
MONGODB_DATABASE=voicera
MONGODB_AUTH_SOURCE=admin
MINIO_ENDPOINT=minio:9000
MINIO_ACCESS_KEY=$SROOT
MINIO_SECRET_KEY=$SPASS
FRONTEND_URL=https://$PUB
EOF
```

**Voice server** — MUST use unique container names (dual-homed, see top note):

```bash
cp voice_2_voice_server/.env voice_2_voice_server/.env.translate
sed -i -e '$a\' voice_2_voice_server/.env.translate
tee -a voice_2_voice_server/.env.translate >/dev/null <<EOF
VOICERA_BACKEND_URL=http://voicera-translate-backend-1:8000
MINIO_ENDPOINT=voicera-translate-minio-1:9000
MINIO_ACCESS_KEY=$SROOT
MINIO_SECRET_KEY=$SPASS
MINIO_SECURE=false
INDIC_STT_SERVER_URL=http://voicera-prod-stt-1:8001
INDIC_TTS_SERVER_URL=http://voicera-prod-nginx-1:8080/ttslb/
JOHNAIC_SERVER_URL=https://$PUB/server
JOHNAIC_WEBSOCKET_URL=wss://$PUB/server
EOF
```

> If §1 showed prod nginx has **no** `/ttslb`, use
> `INDIC_TTS_SERVER_URL=http://voicera-prod-tts-1:8002` instead (one replica, no LB).

**Frontend** — public URL is baked at build time:

```bash
tee voicera_frontend/.env.local.translate >/dev/null <<EOF
NEXT_PUBLIC_JOHNAIC_SERVER_URL=https://$PUB/server
EOF
```

Check for duplicate keys (last-wins at runtime, but keep it clean):

```bash
for f in voicera_backend/.env.translate voice_2_voice_server/.env.translate; do
  echo "== $f =="; grep -v '^#' "$f" | grep '=' | cut -d= -f1 | sort | uniq -d
done
```

## 4. Cloudflare named tunnel (stable URL)

A **named** tunnel gives a stable hostname the frontend can bake in. Quick tunnels
(`--url`) produce a new random URL each restart — avoid for this.

1. Cloudflare Zero Trust → **Networks → Tunnels → Create tunnel** (Cloudflared).
2. Copy the **tunnel token** → `TUNNEL_TOKEN` in `.env.translate`.
3. Add a **Public hostname**: `translate-sandbox.voicera.world` → service **`http://nginx:8080`**
   (cloudflared resolves `nginx` on the internal network). Enable WebSockets (default on).
4. DNS: the dashboard auto-creates the CNAME if `voicera.world` is on this Cloudflare account.

## 5. Build & start

```bash
cd ~/voicera_translate/deploy/compose
docker compose -p voicera-translate --env-file .env.translate \
  -f docker-compose.translate.yml config >/dev/null && echo COMPOSE-OK

dcpt up -d --build
dcpt ps                      # all Up; mongodb/minio (healthy)
```

Frontend bakes `NEXT_PUBLIC_*` at build — if you change `.env.local.translate` later:
`dcpt build --no-cache frontend && dcpt up -d frontend`.

## 6. Verify

```bash
# translate voice_server can reach the reused prod STT/TTS:
dcpt exec voice_server curl -s http://voicera-prod-stt-1:8001/health
dcpt exec voice_server curl -s -o /dev/null -w '%{http_code}\n' \
  http://voicera-prod-nginx-1:8080/ttslb/

# tunnel serving the app:
curl -s -o /dev/null -w 'app:     %{http_code}\n' https://translate-sandbox.voicera.world/
curl -s -o /dev/null -w 'backend: %{http_code}\n' https://translate-sandbox.voicera.world/backend/docs
curl -s -o /dev/null -w 'v2v:     %{http_code}\n' https://translate-sandbox.voicera.world/server/health

# prod untouched:
docker ps --format '{{.Names}}\t{{.Status}}' | grep voicera-prod
```

Then open `https://translate-sandbox.voicera.world`, sign up/login, run a live-translation call.

## 7. Lifecycle

```bash
dcpt logs -f voice_server        # tail
dcpt restart backend             # restart one service
dcpt down                        # stop translate stack (prod unaffected)
dcpt down -v                     # + delete translate mongo/minio volumes
```

> `dcpt down` never touches prod — different project name. Never run a bare
> `docker compose down` from this dir without `-p voicera-translate`.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `network voicera-prod_voicera_network declared as external, but could not be found` | Wrong `PROD_NETWORK_NAME`. | Set it from `docker network ls \| grep voicera`. |
| voice_server: `Cannot connect to host voicera-prod-stt-1:8001` | STT name/network wrong, or voice_server not on prodnet. | Check `dcpt exec voice_server getent hosts voicera-prod-stt-1`; confirm `networks: [internal, prodnet]`. |
| TTS calls all hit one replica | Pointed straight at a `tts` container, not `/ttslb`. | Use `http://voicera-prod-nginx-1:8080/ttslb/` (§3b). |
| Browser test call fails | `NEXT_PUBLIC_JOHNAIC_SERVER_URL` stale / frontend not rebuilt. | Fix `.env.local.translate`; `dcpt build --no-cache frontend && dcpt up -d frontend`. |
| Translate audio landing in prod MinIO | voice_server resolved ambiguous `minio` to prod. | Ensure `MINIO_ENDPOINT=voicera-translate-minio-1:9000` (unique name), not `minio:9000`. |
| Tunnel 502 | cloudflared public hostname not mapped to `http://nginx:8080`. | Fix the public hostname mapping (§4). |
