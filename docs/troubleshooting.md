# Troubleshooting Guide

Common issues and their solutions.

## Service Startup Issues

### Docker Container Won't Start

**Problem:** Container exits immediately or keeps restarting

**Solution:**
```bash
# Check logs
docker-compose logs backend

# Look for common errors:
# - Port already in use
# - Missing environment variables
# - Database connection failure

# Rebuild image
docker-compose build --no-cache backend

# Try again
docker-compose up backend
```

### Port Already in Use

**Problem:** `Error response from daemon: Ports are not available`

**Solution:**
```bash
# Find process using port
lsof -i :8000

# Kill the process
kill -9 <PID>

# Or change port in docker-compose.yml
ports:
  - "8001:8000"  # Map to 8001 instead
```

### Database Connection Failed

**Problem:** Backend can't connect to MongoDB

**Solution:**
```bash
# Ensure MongoDB is running
docker-compose ps mongodb

# Check MongoDB logs
docker-compose logs mongodb

# MongoDB needs time to start
sleep 30
docker-compose up backend

# Verify MongoDB is healthy
docker exec voicera_mongodb mongosh --eval "db.adminCommand('ping')"
```

---

## API & Backend Issues

### 401 Unauthorized Errors

**Problem:** Getting 401 when accessing protected endpoints

**Solution:**
```bash
# Ensure JWT token is in request headers
curl -H "Authorization: Bearer <token>" http://localhost:8000/agents

# Check token validity
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"user@example.com","password":"password"}'

# If login fails, check MongoDB has user data
docker-compose exec mongodb mongosh \
  --username admin --password admin123 \
  --eval "db.users.find()"
```

### 500 Internal Server Error

**Problem:** Backend returns 500 error

**Solution:**
```bash
# Check backend logs for detailed error
docker-compose logs backend | grep -i error

# Common causes:
# 1. Database error - check MongoDB logs
docker-compose logs mongodb

# 2. Missing API keys - check .env file
cat voicera_backend/.env | grep -i api

# 3. Dependency error - reinstall packages
docker-compose build --no-cache backend
```

### Slow API Responses

**Problem:** API endpoints are slow (>1 second)

**Solution:**
```bash
# Check database indexes
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.call_logs.getIndexes()"

# If missing indexes, create them:
# From troubleshooting/create-indexes.js

# Check MongoDB connection pool
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.serverStatus().connections"

# Check backend resource usage
docker stats voicera_backend
```

---

## Voice Server Issues

### WebSocket Connection Failed

**Problem:** Frontend can't establish WebSocket connection

**Solution:**
```bash
# Check voice server is running
docker-compose ps voice_server

# Check WebSocket endpoint
curl http://localhost:7860/health

# Verify backend connection
docker-compose exec voice_server curl http://backend:8000/health

# Check logs
docker-compose logs voice_server

# Firewall issue? Allow WebSocket traffic
sudo ufw allow 7860/tcp
```

### STT Not Working

**Problem:** Voice server can't transcribe audio

**Solution:**
```bash
# Check STT provider configuration
docker-compose exec voice_server cat .env | grep STT

# If using Deepgram, verify API key
export DEEPGRAM_API_KEY=...
docker-compose exec voice_server python -c \
  "from deepgram import DeepgramClient; print('OK')"

# If using AI4Bharat, check service is running
docker-compose ps ai4bharat_stt_server

# Verify connection to STT service
docker-compose exec voice_server curl http://ai4bharat_stt_server:8001/health
```

### TTS Not Working

**Problem:** Voice server can't generate speech

**Solution:**
```bash
# Check TTS provider configuration
docker-compose exec voice_server cat .env | grep TTS

# Test TTS endpoint manually
curl -X POST http://localhost:8002/synthesize \
  -H "Content-Type: application/json" \
  -d '{"text":"Hello world","language":"en"}'

# If using AI4Bharat
docker-compose ps ai4bharat_tts_server
docker-compose logs ai4bharat_tts_server
```

### High CPU Usage

**Problem:** Voice server consuming 100% CPU

**Solution:**
```bash
# Check active sessions
docker-compose exec voice_server curl http://localhost:7860/health | grep sessions

# Kill stuck sessions
docker-compose restart voice_server

# Check audio buffer settings in .env
AUDIO_CHUNK_SIZE=2048  # Reduce if too high
BATCH_SIZE=32          # Reduce for slower machines
```

---

## Frontend Issues

### Page Won't Load

**Problem:** Frontend shows blank page or 404

**Solution:**
```bash
# Check frontend is running
docker-compose ps frontend

# Check logs
docker-compose logs frontend

# Verify API connection
curl http://localhost:3000

# Check .env.local configuration
cat voicera_frontend/.env.local

# Ensure API_BASE_URL is correct
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
```

### Login Not Working

**Problem:** Login credentials rejected

**Solution:**
```bash
# Check if user exists in database
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.users.findOne({email:'user@example.com'})"

# Create test user
docker-compose exec backend python -c \
  "from app.services.auth_service import create_user; create_user('user@example.com', 'password123')"

# Clear browser storage
# Open browser console and run:
# localStorage.clear()
# sessionStorage.clear()
# Then reload page
```

### API Call Errors

**Problem:** Frontend API calls failing with CORS or network errors

**Solution:**
```bash
# Check backend is running and accessible
curl http://localhost:8000/health

# Check CORS configuration in backend
# Should include frontend origin in CORS_ORIGINS

# Frontend .env should have correct API URL
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000

# Check browser console for detailed error
# Open DevTools (F12) and check Network and Console tabs
```

---

## Database Issues

### MongoDB Disk Space Full

**Problem:** MongoDB stops accepting writes

**Solution:**
```bash
# Check disk usage
df -h /data/db

# Clean up old data
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "db.call_logs.deleteMany({created_at: {'\$lt': new Date(Date.now() - 30*24*60*60*1000)}})"

# Or expand volume
# Stop container, expand volume, restart
```

### MongoDB Replication Issues (Replica Set)

**Problem:** Replica set stuck in recovery

**Solution:**
```bash
# Check replica set status
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "rs.status()"

# Reset replica set if needed (DATA LOSS!)
# Only if absolutely necessary
docker-compose exec mongodb mongosh --username admin --password admin123 \
  --eval "rs.initiate()"
```

---

## Storage (MinIO) Issues

### Can't Upload to MinIO

**Problem:** Recording uploads failing

**Solution:**
```bash
# Check MinIO is running
docker-compose ps minio

# Check MinIO console
# Open http://localhost:9001

# Verify credentials
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Create bucket if missing
docker-compose exec minio mc mb minio/recordings

# Check available space
docker stats voicera_minio
```

### Download URLs Not Working

**Problem:** Pre-signed URLs expire or fail

**Solution:**
```bash
# Check S3 bucket policy allows public read (if needed)
# Otherwise use pre-signed URLs with short expiry

# Verify backend can access MinIO
docker-compose exec backend python -c \
  "from app.storage.minio_client import get_minio; print(get_minio())"
```

---

## Monitoring & Performance

### Check Service Health

```bash
# All services
docker-compose ps

# Individual health checks
curl http://localhost:8000/health           # Backend
curl http://localhost:7860/health           # Voice Server
curl http://localhost:9001/minio/health     # MinIO
curl http://localhost:27017                 # MongoDB

# Detailed status
docker inspect voicera_backend | grep -A 20 "State"
```

### Monitor Resource Usage

```bash
# Real-time stats
docker stats

# See which container is using most resources
docker stats --no-stream

# Memory issues?
docker stats voicera_backend
docker update --memory 4g voicera_backend  # Increase limit

# CPU issues?
# Check logs for infinite loops
# Restart container if needed
```

---

## Network Issues

### Services Can't Communicate

**Problem:** Backend can't reach MongoDB

**Solution:**
```bash
# Check network
docker network ls | grep voicera

# Inspect network
docker network inspect voicera_mono_repository_voicera_network

# Test DNS resolution
docker-compose exec backend nslookup mongodb

# Test connectivity
docker-compose exec backend ping mongodb

# If not working, rebuild network
docker-compose down
docker network prune
docker-compose up -d
```

---

## Development Issues

### Changes Not Reflected

**Problem:** Code changes don't appear when running

**Solution:**
```bash
# Ensure hot-reload is enabled for local development
RELOAD=true  # In backend .env

# For frontend, usually works automatically with npm run dev

# Manual rebuild if needed
docker-compose build --no-cache <service>
docker-compose up <service>
```

### Dependency Conflicts

**Problem:** ModuleNotFoundError or ImportError

**Solution:**
```bash
# Backend - reinstall Python dependencies
docker-compose exec backend pip install --no-cache-dir -r requirements.txt

# Frontend - reinstall Node packages
docker-compose exec frontend npm install --legacy-peer-deps

# Or rebuild containers
docker-compose build --no-cache
docker-compose up -d
```

---

## Getting Help

If you're stuck:

1. **Check the logs** - Often has the answer
   ```bash
   docker-compose logs -f <service>
   ```

2. **Google the error** - Exact error message usually helps

3. **Search GitHub issues** - voicera/voicera/issues

4. **Ask in discussions** - voicera/voicera/discussions

5. **Check documentation** - You're reading it!

---

## Next Steps

- **[Quick Start](getting-started/quickstart.md)** - Get started
- **[Configuration](getting-started/configuration.md)** - Setup guide
- **[Docker](deployment/docker.md)** - Container guide
