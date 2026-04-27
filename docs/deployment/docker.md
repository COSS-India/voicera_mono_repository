# Docker & Docker Compose Deployment

Complete guide to deploying VoiceERA using Docker and Docker Compose.

## Overview

Docker containerization ensures consistent behavior across development, staging, and production environments.

**Key Benefits:**
- Reproducible deployments
- Easy scaling
- Service isolation
- Simplified dependency management
- Production-ready setup

## Prerequisites

- Docker 20.10+
- Docker Compose 1.29+
- 8GB+ RAM
- 50GB+ disk space

## Installation

### Docker Desktop (macOS & Windows)

Download and install from [docker.com/products/docker-desktop](https://www.docker.com/products/docker-desktop)

### Linux (Ubuntu/Debian)

```bash
# Install Docker
sudo apt-get update
sudo apt-get install docker.io docker-compose

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

## Building Images

### Build All Services

```bash
# Using Makefile
make build-all-services

# Or manually
docker-compose build
```

### Build Specific Service

```bash
# Build only backend
docker-compose build backend

# Build only frontend
docker-compose build frontend

# Build voice server
docker-compose build voice_server
```

### View Built Images

```bash
# List all images
docker images | grep voicera

# Expected output:
# voicera_mono_repository_backend      latest    abc123...
# voicera_mono_repository_frontend     latest    def456...
# voicera_mono_repository_voice_server latest    ghi789...
```

## Running Services

### Start All Services

```bash
# Using Makefile (recommended)
make start-all-services

# Or manually
docker-compose up -d

# Or with logs
docker-compose up

# Stop gracefully
docker-compose down
```

### Service Startup Sequence

The services start with proper dependencies:

```
1. MongoDB starts first
   (other services depend on it)
         │
2. MinIO starts
   (storage dependency)
         │
3. Backend starts
   (orchestrator)
         │
4. Voice Server starts
   (depends on Backend)
         │
5. Frontend starts
   (depends on Backend)
```

### Health Checks

Docker Compose has health checks configured:

```bash
# View service health status
docker-compose ps

# Example output:
NAME              STATUS
voicera_mongodb   Up (healthy)
voicera_minio     Up (healthy)
voicera_backend   Up (healthy)
voicera_voice_server  Up
voicera_frontend  Up
```

## Service Configuration

### docker-compose.yml Structure

```yaml
version: '3.8'

services:
  # Service definitions
  mongodb:
    image: mongo:latest
    ports:
      - "27017:27017"
    environment:
      MONGO_INITDB_ROOT_USERNAME: admin
      MONGO_INITDB_ROOT_PASSWORD: admin123
    volumes:
      - mongodb_data:/data/db
    networks:
      - voicera_network

  backend:
    build:
      context: ./voicera_backend
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    env_file:
      - ./voicera_backend/.env
    depends_on:
      mongodb:
        condition: service_healthy
    networks:
      - voicera_network

  # ... other services

volumes:
  mongodb_data:
  minio_data:

networks:
  voicera_network:
    driver: bridge
```

### Environment File Management

```bash
# Copy all example files
cp voicera_backend/env.example voicera_backend/.env
cp voice_2_voice_server/.env.example voice_2_voice_server/.env
cp voicera_frontend/.env.example voicera_frontend/.env.local

# Edit each file with your configuration
nano voicera_backend/.env
nano voice_2_voice_server/.env
nano voicera_frontend/.env.local
```

## Managing Services

### View Logs

```bash
# All services
docker-compose logs

# Specific service
docker-compose logs backend
docker-compose logs voice_server

# Follow logs (real-time)
docker-compose logs -f backend

# Last 100 lines
docker-compose logs --tail 100 mongodb

# Since specific time
docker-compose logs --since 2024-01-29T10:00:00 backend
```

### Execute Commands in Container

```bash
# Access backend shell
docker-compose exec backend bash

# Run Python command
docker-compose exec backend python -c "import app; print(app.__version__)"

# Access MongoDB shell
docker-compose exec mongodb mongosh --username admin --password admin123

# Access MinIO console
# Open http://localhost:9001
```

### Restart Services

```bash
# Restart all
docker-compose restart

# Restart specific
docker-compose restart backend

# Hard restart (stop then start)
docker-compose down
docker-compose up -d
```

## Networking

### Service Discovery

Services automatically discover each other via service names:

```
Backend → MongoDB: mongodb:27017
Backend → MinIO: minio:9000
Voice Server → Backend: backend:8000
```

### Exposing Ports

```yaml
# Internal network (not exposed to host)
mongodb:
  ports: ["27017:27017"]  # Only internal

# Exposed to host
backend:
  ports: ["8000:8000"]    # localhost:8000
```

### Custom Network

Services are on `voicera_network`:

```bash
# View network details
docker network inspect voicera_mono_repository_voicera_network

# Test connectivity between containers
docker-compose exec backend ping mongodb
```

## Data Persistence

### Volumes

```yaml
volumes:
  mongodb_data:
    driver: local
  minio_data:
    driver: local
```

### Volume Paths

```bash
# List volumes
docker volume ls | grep voicera

# Inspect volume
docker volume inspect voicera_mono_repository_mongodb_data

# View volume location on host
docker volume inspect voicera_mono_repository_minio_data
```

### Backup & Restore

**Backup MongoDB:**
```bash
# Export database
docker-compose exec mongodb mongodump \
  --username admin \
  --password admin123 \
  --out /backup

# Copy to host
docker cp voicera_mongodb:/backup ./mongodb_backup
```

**Backup MinIO:**
```bash
# Use MinIO client
docker-compose exec minio mc mirror minio/bucket ./backup/minio
```

## Scaling Services

### Horizontal Scaling (Multiple Replicas)

For stateless services (Frontend, Backend):

```bash
# Scale backend to 3 instances
docker-compose up -d --scale backend=3

# Load balancer required (use nginx or traefik)
```

### Vertical Scaling (More Resources)

Increase CPU/Memory allocation:

```yaml
services:
  backend:
    # ... other config
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 4G
        reservations:
          cpus: '1.0'
          memory: 2G
```

## Production Considerations

### Use Production Image Registry

```bash
# Tag images for registry
docker tag voicera_backend:latest registry.example.com/voicera/backend:1.0.0
docker tag voicera_frontend:latest registry.example.com/voicera/frontend:1.0.0
docker tag voicera_voice_server:latest registry.example.com/voicera/voice-server:1.0.0

# Push to registry
docker push registry.example.com/voicera/backend:1.0.0
```

### Update docker-compose.yml for Production

```yaml
services:
  backend:
    image: registry.example.com/voicera/backend:1.0.0
    restart: always
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
  
  mongodb:
    restart: always
    command: mongod --replSet rs0  # Enable replication
    
  minio:
    restart: always
    command: minio server /data --console-address ":9001"
```

### Enable Logging

```yaml
services:
  backend:
    logging:
      driver: "json-file"
      options:
        max-size: "100m"
        max-file: "10"
        labels: "service=backend"
```

## Troubleshooting

### Container won't start

```bash
# Check logs
docker-compose logs backend

# Inspect container
docker inspect voicera_backend

# Rebuild image
docker-compose build --no-cache backend
docker-compose up backend
```

### Port conflicts

```bash
# Find process using port
lsof -i :8000

# Change port in docker-compose.yml
ports:
  - "8001:8000"  # Map to different port
```

### Out of memory

```bash
# Check Docker resource usage
docker stats

# Increase Docker memory
# Docker Desktop: Settings > Resources > Memory

# Linux: Increase swap
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

### Network issues

```bash
# Verify network connectivity
docker network inspect voicera_mono_repository_voicera_network

# Test DNS resolution
docker-compose exec backend nslookup mongodb

# Ping between services
docker-compose exec backend ping minio
```

## Cleanup

```bash
# Stop and remove all containers
docker-compose down

# Also remove volumes (DELETE DATA!)
docker-compose down -v

# Remove unused images
docker image prune -a

# Remove unused volumes
docker volume prune

# Remove everything
docker system prune -a --volumes
```

---

## Next Steps

- **[Production Deployment](production.md)** - Production setup guide
- **[Configuration](../getting-started/configuration.md)** - Environment setup
- **[Quick Start](../getting-started/quickstart.md)** - Get running quickly
