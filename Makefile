.PHONY: build start stop

# Build all services
build-all-services:
	docker compose build backend minio frontend voice_server

# Start all services (postgres+ferretdb pulled/started with up)
start-all-services:
	docker compose up -d postgres ferretdb backend minio frontend voice_server

# Stop all services
stop-all-services:
	docker compose down postgres ferretdb backend minio frontend voice_server

build-backend-services:
	docker compose build backend minio

# Start services except voice_server (detached)
start-backend-services:
	docker compose up -d postgres ferretdb backend minio

# Stop services except voice_server
stop-backend-services:
	docker compose stop postgres ferretdb backend minio

# Models run as containers now, not venvs, and each slot holds a folder per
# model -- so there is no fixed path to a server.py to launch here.
start-voice-only-services:
	docker compose -f model-server/compose.model-server.yml \
	  --project-directory model-server up -d
	bash -c "(cd voice_2_voice_server && source venv/bin/activate && python main.py) & wait"

stop-all-ports:
	-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:27017 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8100 | xargs kill -9 2>/dev/null || true   # model-server gateway
	docker compose -f model-server/compose.model-server.yml \
	  --project-directory model-server down 2>/dev/null || true
	-lsof -ti:7860 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8000 | xargs kill -9 2>/dev/null || true

start-frontend:
	-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	bash -c "(cd voicera_frontend && npm run dev) & wait"

start-dev:
	$(MAKE) start-frontend 
	$(MAKE) start-backend-services
	$(MAKE) start-voice-only-services

stop-dev:
	$(MAKE) stop-backend-services
	$(MAKE) stop-all-ports

# MongoDB → FerretDB migration helpers
dump-mongo:
	./scripts/migrate_mongo_to_ferretdb.sh dump

migrate-to-ferretdb:
	./scripts/migrate_mongo_to_ferretdb.sh cutover

#podman compose up -d postgres ferretdb backend minio
#podman compose stop postgres ferretdb backend minio
