.PHONY: build start stop

# Build all services
build-all-services:
	docker compose build mongodb backend minio frontend voice_server

# Build all services
start-all-services:
	docker compose up -d mongodb backend minio frontend voice_server

# Build all services
stop-all-services:
	docker compose down mongodb backend minio frontend voice_server

build-backend-services:
	docker compose build mongodb backend minio

# Start services except voice_server (detached)
start-backend-services:
	docker compose up -d mongodb backend minio  

# Stop services except voice_server
stop-backend-services:
	docker compose stop mongodb backend minio  

start-voice-only-services:
	bash -c "(cd ai4bharat_stt_server && source venv/bin/activate && python server.py) & (cd ai4bharat_tts_server && source venv/bin/activate && python server.py) & (cd voice_2_voice_server && source venv/bin/activate && python main.py) & wait"

stop-all-ports:
	-lsof -ti:3000 | xargs kill -9 2>/dev/null || true
	-lsof -ti:27017 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8001 | xargs kill -9 2>/dev/null || true
	-lsof -ti:8002 | xargs kill -9 2>/dev/null || true
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