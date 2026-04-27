# Quick Start Guide

Get VoiceERA up and running in 5 minutes!

## Prerequisites Checklist

Before starting, ensure you have completed:

- ✅ [Docker & Docker Compose installed](installation.md)
- ✅ Repository cloned
- ✅ Environment files configured (`.env` files)
- ✅ Docker images built (`make build-all-services`)

## 5-Minute Startup

### Step 1: Start All Services

```bash
cd voicera_mono_repository

# Start all services in detached mode
make start-all-services

# Or manually
docker-compose up -d
```

Monitor startup progress:

```bash
# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

Expected output:

```
NAME                COMMAND                  SERVICE             STATUS
voicera_backend     "uvicorn app.main:..."  backend             Up (healthy)
voicera_frontend    "npm run dev"           frontend            Up
voicera_minio       "/usr/bin/minio serve"  minio               Up (healthy)
voicera_mongodb     "mongod --bind_ip_all"  mongodb             Up (healthy)
voicera_voice_server "python main.py"       voice_server        Up
```

### Step 2: Wait for Services to Be Ready

=== "Using Health Checks"
    ```bash
    # Wait for all services to be healthy
    docker-compose ps | grep -i healthy
    ```

=== "Using Docker Logs"
    ```bash
    # Watch for startup messages
    docker-compose logs -f backend
    
    # Look for message like:
    # "Application startup complete"
    ```

=== "Manual Check"
    ```bash
    # Test backend
    curl http://localhost:8000/health
    # Should return: {"status":"ok"}
    
    # Test frontend
    curl -I http://localhost:3000
    # Should return: 200 OK
    ```

### Step 3: Access the Application

Once all services are running, open your browser and navigate to:

| Service | URL | Purpose |
|---------|-----|---------|
| **Frontend** | http://localhost:3000 | Web dashboard |
| **Backend API** | http://localhost:8000/docs | Swagger UI documentation |
| **Backend API** | http://localhost:8000/redoc | ReDoc documentation |
| **MinIO** | http://localhost:9001 | Object storage console |

### Step 4: First Login

1. Open http://localhost:3000
2. Sign up or log in with credentials:
   - **Email:** admin@example.com
   - **Password:** (configured in your `.env` file)

## Common Tasks

### View Service Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs voice_server

# Real-time follow
docker-compose logs -f frontend

# Last 100 lines
docker-compose logs --tail 100 backend
```

### Stop Services

```bash
# Stop all services (keeps data)
make stop-all-services

# Or manually
docker-compose stop

# Remove containers (keeps volumes/data)
docker-compose down

# Remove containers AND volumes (wipes data!)
docker-compose down -v
```

### Restart a Service

```bash
# Restart specific service
docker-compose restart backend

# Restart all services
docker-compose restart
```

### Check Service Status

```bash
# View all containers
docker-compose ps

# Show detailed info
docker inspect voicera_backend

# Check resource usage
docker stats
```

## Troubleshooting Quick Fixes

### "Port already in use"

```bash
# Find and stop the process using the port
lsof -i :8000        # Check port 8000
kill -9 <PID>        # Kill the process

# Or change port in docker-compose.yml
```

### "Connection refused"

```bash
# Services need time to start, wait a moment
sleep 10

# Check if services are running
docker-compose ps

# View startup logs
docker-compose logs backend
```

### "Database connection error"

```bash
# MongoDB may still be initializing
docker-compose logs mongodb | tail -20

# Restart MongoDB
docker-compose restart mongodb

# Wait for health check
docker-compose ps mongodb  # Should show "healthy"
```

### "Out of memory"

```bash
# Check Docker's allocated memory
docker stats

# Increase Docker memory allocation
# Linux: Edit /etc/docker/daemon.json
# macOS/Windows: Docker Desktop Settings > Resources
```

## Next Steps

### Create Your First Agent

1. Go to http://localhost:3000
2. Navigate to **Agents** > **Create Agent**
3. Configure:
   - **Name:** My First Agent
   - **Language:** English
   - **LLM Model:** GPT-3.5 or GPT-4
   - **Voice:** Choose a voice
4. Click **Create**

### Make a Test Call

1. Go to **Campaigns** > **Create Campaign**
2. Configure voice settings
3. Test with a phone number
4. Monitor in **Call Recordings**

### Explore the API

1. Open http://localhost:8000/docs (Swagger UI)
2. Try out API endpoints:
   - `GET /agents` - List all agents
   - `GET /health` - Service health
   - `GET /campaigns` - List campaigns

## Development Mode

For faster iteration during development, run services locally:

```bash
# Start just infrastructure (DB, storage)
make start-backend-services

# In separate terminals, start services locally:

# Terminal 1: Backend
cd voicera_backend
source venv/bin/activate  # or .venv\Scripts\activate on Windows
python -m uvicorn app.main:app --reload --port 8000

# Terminal 2: Frontend
cd voicera_frontend
npm run dev

# Terminal 3: Voice Server
cd voice_2_voice_server
source venv/bin/activate  # or .venv\Scripts\activate on Windows
python main.py
```

## Useful Commands Reference

```bash
# Build all images
make build-all-services

# Start all services
make start-all-services

# Stop all services
make stop-all-services

# View logs
docker-compose logs -f

# Access backend shell
docker-compose exec backend /bin/bash

# Run backend migrations (if needed)
docker-compose exec backend alembic upgrade head

# Seed example data
docker-compose exec backend python -m app.scripts.seed_data
```

## What's Next?

- **[Configuration Guide](configuration.md)** - Customize environment variables
- **[Architecture Overview](../architecture/overview.md)** - Understand system design
- **[API Documentation](../api/rest-api.md)** - Explore REST API
- **[Troubleshooting](../troubleshooting.md)** - Solve common issues

## Getting Help

- 📖 [Full Documentation](../index.md)
- 🐛 [Issue Tracker](https://github.com/voicera/voicera/issues)
- 💬 [Discussions](https://github.com/voicera/voicera/discussions)
- 📧 [Support Email](mailto:support@voicera.ai)
