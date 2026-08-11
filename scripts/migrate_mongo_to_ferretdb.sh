#!/usr/bin/env bash
# Backup MongoDB (voicera) and restore into FerretDB.
#
# Usage:
#   ./scripts/migrate_mongo_to_ferretdb.sh dump
#   ./scripts/migrate_mongo_to_ferretdb.sh restore [path/to/archive.gz]
#   ./scripts/migrate_mongo_to_ferretdb.sh verify
#   ./scripts/migrate_mongo_to_ferretdb.sh cutover   # dump → stop mongo → start ferretdb → restore
#
# Env overrides:
#   MONGO_CONTAINER, FERRET_CONTAINER, MONGODB_DATABASE, BACKUP_DIR, COMPOSE_FILE
#   MONGO_NETWORK, MONGO_TOOLS_IMAGE
#
# Safety: does NOT delete the old Docker volume (e.g. *_mongodb_data).
# Only remove it after you have verified FerretDB and no longer need rollback:
#   docker volume ls | grep mongodb_data
#   docker volume rm <volume_name>

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-${ROOT_DIR}/docker-compose.yml}"
BACKUP_DIR="${BACKUP_DIR:-${ROOT_DIR}/backups/mongodb}"
DB_NAME="${MONGODB_DATABASE:-voicera}"
USER="${MONGODB_USER:-admin}"
PASS="${MONGODB_PASSWORD:-admin123}"
MONGO_CONTAINER="${MONGO_CONTAINER:-voicera_mongodb}"
FERRET_CONTAINER="${FERRET_CONTAINER:-voicera_ferretdb}"
MONGO_TOOLS_IMAGE="${MONGO_TOOLS_IMAGE:-mongo:latest}"
MONGO_NETWORK="${MONGO_NETWORK:-voicera_mono_repository_voicera_network}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
DEFAULT_ARCHIVE="${BACKUP_DIR}/voicera_${TIMESTAMP}.archive.gz"

die() { echo "ERROR: $*" >&2; exit 1; }
info() { echo "==> $*" >&2; }

need_docker() {
  command -v docker >/dev/null 2>&1 || die "docker is required"
}

latest_archive() {
  ls -1t "${BACKUP_DIR}"/voicera_*.archive.gz 2>/dev/null | head -n1 || true
}

container_running() {
  docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null | grep -qx true
}

dump_mongo() {
  need_docker
  mkdir -p "${BACKUP_DIR}"
  local archive="${1:-${DEFAULT_ARCHIVE}}"
  container_running "${MONGO_CONTAINER}" || die "MongoDB container '${MONGO_CONTAINER}' is not running. Start it before dump."

  info "Dumping database '${DB_NAME}' from ${MONGO_CONTAINER} → ${archive}"
  docker exec "${MONGO_CONTAINER}" mongodump \
    -u "${USER}" -p "${PASS}" --authenticationDatabase admin \
    --db "${DB_NAME}" --archive --gzip > "${archive}"

  local size
  size="$(wc -c <"${archive}" | tr -d ' ')"
  [[ "${size}" -gt 32 ]] || die "Dump archive looks empty (${size} bytes): ${archive}"
  info "Dump complete (${size} bytes)."
  info "Keep this file and the old mongodb_data volume until FerretDB is verified."
  echo "${archive}"
}

restore_ferret() {
  need_docker
  local archive="${1:-$(latest_archive)}"
  [[ -n "${archive}" && -f "${archive}" ]] || die "No dump archive found. Run: $0 dump"
  container_running "${FERRET_CONTAINER}" || die "FerretDB container '${FERRET_CONTAINER}' is not running."

  # Resolve absolute path for bind mount
  local abs_archive abs_dir base
  abs_archive="$(cd "$(dirname "${archive}")" && pwd)/$(basename "${archive}")"
  abs_dir="$(dirname "${abs_archive}")"
  base="$(basename "${abs_archive}")"

  local uri="mongodb://${USER}:${PASS}@${FERRET_CONTAINER}:27017/"
  info "Restoring ${abs_archive} into FerretDB via ${MONGO_TOOLS_IMAGE}"
  docker run --rm \
    --network "${MONGO_NETWORK}" \
    -v "${abs_dir}:/backup:ro" \
    "${MONGO_TOOLS_IMAGE}" \
    mongorestore --uri="${uri}" --archive="/backup/${base}" --gzip --nsInclude="${DB_NAME}.*"
  info "Restore complete."
}

verify_counts() {
  need_docker
  container_running "${FERRET_CONTAINER}" || die "FerretDB container '${FERRET_CONTAINER}' is not running."
  info "Collection document counts on FerretDB (db=${DB_NAME}):"
  docker run --rm --network "${MONGO_NETWORK}" "${MONGO_TOOLS_IMAGE}" \
    mongosh "mongodb://${USER}:${PASS}@${FERRET_CONTAINER}:27017/${DB_NAME}" --quiet --eval "
      const cols = db.getCollectionNames().sort();
      cols.forEach(c => print(c + ': ' + db.getCollection(c).countDocuments({})));
      print('---');
      print('collections: ' + cols.length);
    "
  info "Also confirm backend /health after starting the API."
  info "Do NOT docker volume rm *_mongodb_data until you are confident in the cutover."
}

compose() {
  docker compose -f "${COMPOSE_FILE}" "$@"
}

cutover() {
  need_docker

  info "Step 1/5: Dump current MongoDB data (backup)"
  local archive
  archive="$(dump_mongo)"

  info "Step 2/5: Stop backend and MongoDB (volume retained on disk)"
  compose stop backend 2>/dev/null || docker stop voicera_backend 2>/dev/null || true
  # Prefer stopping by container name so we keep the volume even if service was removed from compose
  docker stop "${MONGO_CONTAINER}" 2>/dev/null || true

  info "Step 3/5: Start PostgreSQL + FerretDB"
  # Compose requires frontend secret file to exist even when only starting DB services
  mkdir -p "${ROOT_DIR}/voicera_frontend"
  touch "${ROOT_DIR}/voicera_frontend/.env.local"
  # Remove stale exited containers that may conflict on name
  docker rm -f voicera_postgres voicera_ferretdb 2>/dev/null || true
  compose up -d postgres ferretdb

  info "Waiting for FerretDB to accept connections..."
  local i
  for i in $(seq 1 90); do
    if docker run --rm --network "${MONGO_NETWORK}" "${MONGO_TOOLS_IMAGE}" \
      mongosh "mongodb://${USER}:${PASS}@${FERRET_CONTAINER}:27017/" --quiet --eval 'db.runCommand({ping:1})' >/dev/null 2>&1; then
      break
    fi
    sleep 2
    [[ "${i}" -eq 90 ]] && die "FerretDB did not become ready in time"
  done

  info "Step 4/5: Restore dump into FerretDB"
  restore_ferret "${archive}"

  info "Step 5/5: Start backend and verify"
  compose up -d backend
  sleep 3
  verify_counts
  info "Cutover finished. Smoke-check: login, agents, call logs, batch upload (GridFS)."
  info "When satisfied, you may remove the unused Mongo volume:"
  info "  docker volume ls | grep mongodb_data"
  info "  docker volume rm <name>   # only after verification"
}

usage() {
  cat <<EOF
Usage: $0 <dump|restore|verify|cutover> [archive.gz]

  dump      mongodump voicera DB from MongoDB container
  restore   mongorestore archive into FerretDB
  verify    print collection counts on FerretDB
  cutover   dump → stop Mongo → start FerretDB → restore → start backend

Backups land in: ${BACKUP_DIR}
Old Mongo Docker volumes are never deleted by this script.
EOF
}

main() {
  local cmd="${1:-}"
  shift || true
  case "${cmd}" in
    dump) dump_mongo "${1:-}" ;;
    restore) restore_ferret "${1:-}" ;;
    verify) verify_counts ;;
    cutover) cutover ;;
    -h|--help|help|"") usage; [[ -n "${cmd}" ]] || exit 1 ;;
    *) die "Unknown command: ${cmd}. Run: $0 --help" ;;
  esac
}

main "$@"
