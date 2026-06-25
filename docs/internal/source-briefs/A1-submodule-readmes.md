# Brief: Submodule READMEs (A1)

**Review gap:** None of the five modules have adequate standalone README documentation for someone working on a specific component.

**What the final doc should be:** One README per folder (short: purpose, how it runs, links to other docs).

| Module | Folder | What it does | Run (Docker / make) | Existing doc |
|--------|--------|--------------|---------------------|--------------|
| Backend | `voicera_backend/` | REST API, MongoDB, MinIO, auth, agents, telephony APIs, campaigns, knowledge base | `backend` service; `make start-backend-services` | `voicera_backend/README.md` — short module README (MkDocs for depth) |
| Frontend | `voicera_frontend/` | Next.js dashboard — agents, numbers, integrations, calls | `frontend` :3000; `make start-frontend` | `voicera_frontend/README.md` — module README (replaces create-next-app boilerplate) |
| Voice server | `voice_2_voice_server/` | Real-time voice pipeline, telephony webhooks, WebSocket audio | `voice_server` :7860 | `voice_2_voice_server/README.md` exists — good base |
| STT server | `ai4bharat_stt_server/` | Local Indic speech-to-text (NeMo) | Optional :8001 | **None** — create from this brief |
| TTS server | `ai4bharat_tts_server/` | Local Indic text-to-speech (Parler, WebSocket) | Optional :8002 | **None** — create from this brief |

## Writer task

Turn each row into a 1–2 page README with:

- What is this component?
- When do I need it (required vs optional)?
- How to start/stop (Docker service name or Makefile target)
- What other services must be running first
- Links to `docs/source-briefs/A2`–`A5` and operator docs (`B1`–`B8`)

## Ports (monorepo default)

| Service | Port |
|---------|------|
| frontend | 3000 |
| backend | 8000 |
| voice_server | 7860 |
| mongodb | 27017 |
| minio | 9000 (API), 9001 (console) |
| ai4bharat_stt_server | 8001 |
| ai4bharat_tts_server | 8002 |
