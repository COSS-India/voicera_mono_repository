# VoiceERA Quick Reference

Quick reference for common tasks and commands.

## Getting Started

### Installation (5 minutes)

```bash
# Clone repository
git clone https://github.com/voicera/voicera_mono_repository.git
cd voicera_mono_repository

# Setup environment
cp voicera_backend/.env.example voicera_backend/.env
cp voice_2_voice_server/.env.example voice_2_voice_server/.env

# Start services
docker-compose up -d

# Wait for services to be ready
docker-compose ps
```

### Access Services

| Service | URL | Purpose |
|---------|-----|---------|
| Frontend | http://localhost:3000 | Web dashboard |
| Backend API | http://localhost:8000 | API endpoints |
| API Docs | http://localhost:8000/docs | Swagger UI |
| MinIO | http://localhost:9001 | Object storage |
| Voice Server | http://localhost:7860 | WebSocket server |

## Development Commands

### Backend

```bash
cd voicera_backend

# Install dependencies
pip install -r requirements.txt

# Start dev server
uvicorn app.main:app --reload

# Run tests
pytest

# Run with coverage
pytest --cov=app

# Database migrations
alembic upgrade head
alembic revision --autogenerate -m "Message"
```

### Frontend

```bash
cd voicera_frontend

# Install dependencies
npm install

# Start dev server (http://localhost:3000)
npm run dev

# Build for production
npm run build

# Run tests
npm test

# Run linter
npm run lint
```

### Voice Server

```bash
cd voice_2_voice_server

# Install dependencies
pip install -r requirements.txt

# Start server
python main.py

# Start with debug logging
DEBUG_MODE=true python main.py
```

### Database

```bash
# Start database in Docker
docker-compose up -d mongodb

# Connect to MongoDB
docker exec -it voicera_mono_repository-mongodb-1 mongosh

# Commands in MongoDB
show databases
use voicera_db
show collections
db.agents.find()
db.agents.countDocuments()
```

## API Quick Reference

### Authentication

```bash
# Get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "password"
  }'

# Use token in subsequent requests
curl -H "Authorization: Bearer YOUR_TOKEN" \
  http://localhost:8000/agents
```

### Agents

```bash
# List agents
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/agents

# Create agent
curl -X POST http://localhost:8000/agents \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Support Bot",
    "llm_provider": "openai",
    "system_prompt": "You are a helpful support agent"
  }'

# Get agent details
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/agents/{agent_id}

# Update agent
curl -X PUT http://localhost:8000/agents/{agent_id} \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "New Name"}'

# Delete agent
curl -X DELETE http://localhost:8000/agents/{agent_id} \
  -H "Authorization: Bearer TOKEN"
```

### Campaigns

```bash
# List campaigns
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/campaigns

# Create campaign
curl -X POST http://localhost:8000/campaigns \
  -H "Authorization: Bearer TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Q1 2024 Campaign",
    "agent_id": "agent-uuid",
    "contact_list": ["phone1", "phone2"]
  }'

# Launch campaign
curl -X POST http://localhost:8000/campaigns/{campaign_id}/launch \
  -H "Authorization: Bearer TOKEN"

# Pause campaign
curl -X POST http://localhost:8000/campaigns/{campaign_id}/pause \
  -H "Authorization: Bearer TOKEN"
```

### Call Logs

```bash
# List call logs
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-logs

# Get call details
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-logs/{call_id}

# Get campaign's calls
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-logs/campaign/{campaign_id}
```

### Recordings

```bash
# List recordings
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-recordings

# Download recording
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-recordings/{call_id}/download \
  -o recording.wav

# Get transcript
curl -H "Authorization: Bearer TOKEN" \
  http://localhost:8000/call-recordings/{call_id}/transcript
```

## Docker Commands

### Container Management

```bash
# Start all services
docker-compose up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs

# View logs for specific service
docker-compose logs -f backend

# Restart service
docker-compose restart voice-server

# Run command in container
docker exec -it voicera-backend bash

# Check resource usage
docker stats
```

### Image Management

```bash
# Build all images
docker-compose build

# Build specific image
docker-compose build backend

# List images
docker images

# Remove unused images
docker image prune

# View image layers
docker history image-name
```

### Volumes

```bash
# List volumes
docker volume ls

# Inspect volume
docker volume inspect volume-name

# Clean up unused volumes
docker volume prune

# Backup volume
docker run --rm -v volume-name:/data -v $(pwd):/backup \
  alpine tar czf /backup/volume-backup.tar.gz /data

# Restore volume
docker run --rm -v volume-name:/data -v $(pwd):/backup \
  alpine tar xzf /backup/volume-backup.tar.gz -C /data
```

## Troubleshooting

### Service Won't Start

```bash
# Check logs
docker-compose logs service-name

# Check port conflicts
lsof -i :8000

# Check resource usage
docker stats

# Rebuild images
docker-compose build --no-cache service-name
```

### Database Issues

```bash
# Connect to MongoDB
docker exec -it mongodb mongosh

# Check connections
db.serverStatus().connections

# See slow queries
db.setProfilingLevel(1, { slowms: 100 })
db.system.profile.find().pretty()

# Check indexes
db.agents.getIndexes()

# Create index
db.agents.createIndex({ "user_id": 1, "created_at": -1 })
```

### Performance Issues

```bash
# Check Docker resource limits
docker stats

# Check database performance
docker exec -it mongodb mongosh
db.serverStatus()

# Check API response times
curl -w "@curl_format.txt" http://localhost:8000/agents

# Profile Python code
python -m cProfile -s cumulative script.py
```

### WebSocket Issues

```bash
# Test WebSocket connection
wscat -c ws://localhost:7860

# Check connection logs
docker-compose logs voice-server

# Monitor WebSocket traffic
docker exec -it voice-server tcpdump -i any port 7860
```

## Environment Variables Cheat Sheet

### Backend (.env)

```env
MONGODB_URL=mongodb://localhost:27017
MONGODB_DB=voicera

JWT_SECRET=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION=3600

MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_ACCESS_KEY=minioadmin
MINIO_SECRET_KEY=minioadmin

OPENAI_API_KEY=sk-...
DEEPGRAM_API_KEY=...
CARTESIA_API_KEY=...
```

### Voice Server (.env)

```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=...

TTS_PROVIDER=cartesia
CARTESIA_API_KEY=...

BACKEND_API_URL=http://backend:8000
VOBIZ_WEBHOOK_URL=http://voice-server:7860/webhook/vobiz
```

### Frontend (.env.local)

```env
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:7860
NEXT_PUBLIC_LOG_LEVEL=debug
```

## Testing Quick Reference

```bash
# Backend tests
cd voicera_backend
pytest                          # Run all
pytest tests/test_auth.py      # Single file
pytest -v                       # Verbose
pytest --cov=app               # Coverage
pytest -k "test_name"          # Pattern match

# Frontend tests
cd voicera_frontend
npm test                        # Run tests
npm test -- --watch            # Watch mode
npm test -- --coverage         # Coverage
```

## Git Quick Reference

```bash
# Feature branch
git checkout -b feature/feature-name

# Push branch
git push origin feature/feature-name

# Create pull request
# Go to GitHub

# Update with main
git fetch origin
git rebase origin/main

# Squash commits
git rebase -i HEAD~3

# Undo last commit (keep changes)
git reset --soft HEAD~1

# View history
git log --oneline --graph
```

## Useful Links

- **Docs**: https://voicera-docs.example.com
- **GitHub**: https://github.com/voicera/voicera_mono_repository
- **Issues**: https://github.com/voicera/voicera_mono_repository/issues
- **Discussions**: https://github.com/voicera/voicera_mono_repository/discussions
- **FastAPI Docs**: https://fastapi.tiangolo.com
- **Next.js Docs**: https://nextjs.org/docs
- **MongoDB Docs**: https://docs.mongodb.com

## Need Help?

1. **Check Troubleshooting**: [Troubleshooting Guide](troubleshooting.md)
2. **Search Issues**: [GitHub Issues](https://github.com/voicera/voicera_mono_repository/issues)
3. **Ask Questions**: [GitHub Discussions](https://github.com/voicera/voicera_mono_repository/discussions)
4. **Read Docs**: [Full Documentation](index.md)
5. **Contact**: support@voicera.ai

---

Last updated: January 2024
