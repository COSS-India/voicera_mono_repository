# Production Deployment

Guidelines and best practices for deploying VoiceERA to production.

## Pre-Deployment Checklist

- [ ] All environment variables configured securely
- [ ] Database backups configured
- [ ] SSL/TLS certificates obtained
- [ ] Load balancer configured
- [ ] Monitoring and logging setup
- [ ] Security scanning completed
- [ ] Performance testing passed
- [ ] Disaster recovery plan documented

## Infrastructure Requirements

### Minimum Production Setup

- **CPU:** 4 cores
- **RAM:** 16GB
- **Disk:** 200GB (SSD preferred)
- **Network:** 100 Mbps bandwidth

### Recommended Production Setup

- **CPU:** 8+ cores
- **RAM:** 32GB+
- **Disk:** 500GB+ SSD (for call recordings)
- **GPU:** NVIDIA GPU (for AI4Bharat services)
- **Network:** 1 Gbps bandwidth

## Deployment Architecture

```
┌─────────────────────────────────────────────────┐
│                Internet                          │
└────────────────────┬────────────────────────────┘
                     │
              ┌──────▼──────┐
              │ SSL/TLS     │
              │ Termination │
              └──────┬──────┘
                     │
              ┌──────▼──────┐
              │ Load         │
              │ Balancer     │
              │ (Nginx/HAProxy)
              └──────┬──────┘
                     │
         ┌───────────┼───────────┐
         ▼           ▼           ▼
    Backend 1  Backend 2  Backend 3
    :8000      :8000      :8000
         │           │           │
         └───────────┼───────────┘
                     │
              ┌──────▼──────┐
              │ MongoDB     │
              │ Replica Set │
              └─────────────┘

Voice Servers (Separate cluster):
    Voice 1 (:7860)
    Voice 2 (:7860)
    Voice 3 (:7860)
         │
         ▼
    Backend (Shared)
```

## SSL/TLS Configuration

### Obtain Certificates

```bash
# Using Let's Encrypt with Certbot
sudo certbot certonly --standalone -d yourdomain.com -d api.yourdomain.com

# Certificates saved to:
# /etc/letsencrypt/live/yourdomain.com/
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-enabled/voicera.conf

# HTTP to HTTPS redirect
server {
    listen 80;
    server_name yourdomain.com api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

# HTTPS - Frontend
server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    # SSL configuration
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    location / {
        proxy_pass http://frontend:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# HTTPS - Backend API
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=api_limit:10m rate=100r/s;
    limit_req zone=api_limit burst=200;

    location / {
        proxy_pass http://backend:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}

# Voice Server WebSocket
upstream voice_servers {
    least_conn;
    server voice-server-1:7860;
    server voice-server-2:7860;
    server voice-server-3:7860;
}

server {
    listen 443 ssl http2;
    server_name voice.yourdomain.com;

    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://voice_servers;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

## Database Configuration

### MongoDB Production Setup

```yaml
# docker-compose.prod.yml
mongodb:
  image: mongo:6.0
  command: mongod --replSet rs0 --bind_ip_all
  environment:
    MONGO_INITDB_ROOT_USERNAME: ${MONGO_ADMIN_USER}
    MONGO_INITDB_ROOT_PASSWORD: ${MONGO_ADMIN_PASSWORD}
  volumes:
    - mongodb_prod_data:/data/db
    - mongodb_prod_config:/data/configdb
  networks:
    - voicera_network
  restart: always
  healthcheck:
    test: ["CMD", "mongosh", "--eval", "db.adminCommand('ping')"]
    interval: 10s
    timeout: 5s
    retries: 5
```

### MongoDB Replica Set Initialization

```bash
# Initialize replica set
docker-compose exec mongodb mongosh --username $MONGO_ADMIN_USER --password $MONGO_ADMIN_PASSWORD

# In mongosh:
rs.initiate({
  _id: "rs0",
  members: [
    { _id: 0, host: "mongodb:27017" }
  ]
})
```

### Backup Strategy

```bash
#!/bin/bash
# Daily MongoDB backup script

BACKUP_DIR=/backups/mongodb
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
docker-compose exec mongodb mongodump \
  --username $MONGO_ADMIN_USER \
  --password $MONGO_ADMIN_PASSWORD \
  --out $BACKUP_DIR/backup_$DATE

# Upload to S3
aws s3 cp $BACKUP_DIR/backup_$DATE s3://voicera-backups/

# Keep only last 30 days
find $BACKUP_DIR -type d -mtime +30 -exec rm -rf {} \;
```

## Monitoring & Observability

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'voicera-backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'

  - job_name: 'voicera-voice-server'
    static_configs:
      - targets: ['voice_server:7860']
    metrics_path: '/metrics'

  - job_name: 'mongodb'
    static_configs:
      - targets: ['mongodb:27017']
```

### Grafana Dashboards

Key dashboards to create:

- **System Health:** CPU, Memory, Disk usage
- **API Metrics:** Request rate, response time, errors
- **Voice Metrics:** Active calls, call duration, success rate
- **Database:** Connection pool, query latency, document count

### Log Aggregation (ELK Stack)

```yaml
# docker-compose.prod.yml additions
elasticsearch:
  image: docker.elastic.co/elasticsearch/elasticsearch:8.0.0
  environment:
    - discovery.type=single-node
    - xpack.security.enabled=false

logstash:
  image: docker.elastic.co/logstash/logstash:8.0.0
  volumes:
    - ./logstash.conf:/usr/share/logstash/pipeline/logstash.conf

kibana:
  image: docker.elastic.co/kibana/kibana:8.0.0
  ports:
    - "5601:5601"
```

## Security Hardening

### Firewall Rules

```bash
# UFW (Ubuntu)
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow SSH
sudo ufw allow 22/tcp

# Allow HTTP/HTTPS
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp

# Allow specific internal services
sudo ufw allow from 10.0.0.0/8 to any port 27017

# Enable firewall
sudo ufw enable
```

### Secrets Management

```bash
# Never commit secrets to git
.env.prod
secrets/
*.key

# Use environment variables or secret management tool
# Example: HashiCorp Vault

# Access secret
curl -H "X-Vault-Token: $VAULT_TOKEN" \
  https://vault.example.com/v1/secret/data/voicera
```

### Database Security

```bash
# Create restricted user
db.createUser({
  user: "backend_user",
  pwd: "strong_password_here",
  roles: [
    { role: "readWrite", db: "voicera" }
  ]
})

# Create read-only user
db.createUser({
  user: "analytics_user",
  pwd: "analytics_password",
  roles: [
    { role: "read", db: "voicera" }
  ]
})
```

## Performance Tuning

### Backend Configuration

```yaml
# voicera_backend environment
MONGODB_MAX_POOL_SIZE=50
MONGODB_TIMEOUT_MS=10000
API_WORKER_COUNT=8
API_WORKER_CLASS=uvicorn.workers.UvicornWorker
```

### Voice Server Optimization

```yaml
# voice_2_voice_server environment
MAX_CONCURRENT_CALLS=100
AUDIO_BUFFER_SIZE=32768
STT_BATCH_SIZE=32
SESSION_TIMEOUT_MINUTES=60
```

### Database Indexes

```javascript
// Ensure critical indexes exist
db.agents.createIndex({ user_id: 1 })
db.campaigns.createIndex({ agent_id: 1, status: 1 })
db.call_logs.createIndex({ campaign_id: 1, created_at: -1 })
db.call_logs.createIndex({ phone_number: 1 })
db.call_logs.createIndex({ status: 1, created_at: -1 })
```

## Disaster Recovery

### Backup Strategy

1. **Daily backups** - Automated MongoDB dumps
2. **Hourly snapshots** - EBS volume snapshots (AWS)
3. **Geographic replication** - Multi-region backup
4. **Point-in-time recovery** - Binary log retention

### Failover Configuration

```yaml
# Use managed services for auto-failover
# AWS RDS for MongoDB Atlas
# Google Cloud MongoDB
# Or Kubernetes with StatefulSets for self-managed
```

## Rolling Deployments

### Zero-Downtime Updates

```bash
#!/bin/bash
# Rolling deployment script

set -e

# Backend rolling update
docker-compose up -d \
  --no-deps \
  --build \
  --remove-orphans \
  backend

# Wait for health check
sleep 30
curl -f http://backend:8000/health || exit 1

# Update frontend
docker-compose up -d \
  --no-deps \
  --build \
  frontend

# Update voice server (requires client reconnection)
# Can be done during low-traffic period
docker-compose up -d \
  --no-deps \
  --build \
  voice_server
```

## Monitoring Checklist

- [ ] Uptime monitoring (e.g., Pingdom, Datadog)
- [ ] Error tracking (e.g., Sentry)
- [ ] Performance monitoring (e.g., New Relic)
- [ ] Security scanning (e.g., Snyk)
- [ ] Log aggregation (e.g., ELK, Splunk)
- [ ] Alerting configured (PagerDuty, OpsGenie)

---

## Next Steps

- **[Environment Variables](environment.md)** - Production configuration
- **[Docker Deployment](docker.md)** - Container deployment
- **[Quick Start](../getting-started/quickstart.md)** - Development setup
