#!/usr/bin/env bash
# Re-sync the translate stack to the CURRENT cloudflared quick-tunnel URL.
# Quick tunnels get a new random https://<random>.trycloudflare.com URL every
# time cloudflared restarts (reboot/crash/recreate). This script:
#   1. reads the live URL from the cloudflared container logs
#   2. rewrites the 4 public-URL lines in the *.translate env files
#   3. rebuilds the frontend (it bakes NEXT_PUBLIC_* at build time)
#   4. recreates backend + voice_server (env change) — NOT cloudflared
#
# Usage:
#   ./refresh-translate-url.sh                # use current live URL
#   FORCE_NEW=1 ./refresh-translate-url.sh    # restart cloudflared first (new URL)
set -euo pipefail

REPO=~/voicera_translate
COMPOSE_DIR="$REPO/deploy/compose"
dcpt() {
  docker compose -p voicera-translate \
    --env-file "$COMPOSE_DIR/.env.translate" \
    -f "$COMPOSE_DIR/docker-compose.translate.yml" "$@"
}

if [ "${FORCE_NEW:-0}" = "1" ]; then
  echo ">> restarting cloudflared to get a new URL..."
  dcpt restart cloudflared
  sleep 8
fi

URL=$(dcpt logs cloudflared 2>&1 \
  | grep -Eo 'https://[a-z0-9-]+\.trycloudflare\.com' | tail -1)
HOST=${URL#https://}

if [ -z "$URL" ]; then
  echo "!! no trycloudflare URL found in cloudflared logs" >&2
  echo "   check: dcpt logs cloudflared" >&2
  exit 1
fi
echo ">> current tunnel URL: $URL"

cd "$REPO"
sed -i -E "s#^FRONTEND_URL=.*#FRONTEND_URL=$URL#"                                              voicera_backend/.env.translate
sed -i -E "s#^JOHNAIC_SERVER_URL=.*#JOHNAIC_SERVER_URL=$URL/server#"                           voice_2_voice_server/.env.translate
sed -i -E "s#^JOHNAIC_WEBSOCKET_URL=.*#JOHNAIC_WEBSOCKET_URL=wss://$HOST/server#"              voice_2_voice_server/.env.translate
sed -i -E "s#^NEXT_PUBLIC_JOHNAIC_SERVER_URL=.*#NEXT_PUBLIC_JOHNAIC_SERVER_URL=$URL/server#"   voicera_frontend/.env.local.translate

echo ">> updated env lines:"
grep -hE 'trycloudflare' \
  voicera_backend/.env.translate \
  voice_2_voice_server/.env.translate \
  voicera_frontend/.env.local.translate

echo ">> rebuilding frontend + recreating backend/voice_server (cloudflared untouched)..."
cd "$COMPOSE_DIR"
dcpt build --no-cache frontend
dcpt up -d frontend backend voice_server

echo ">> done. App is live at: $URL"
