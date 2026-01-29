# Environment Variables Reference

Complete reference for all environment variables used in VoiceERA services.

## Table of Contents

1. [Backend Environment](#backend-environment)
2. [Voice Server Environment](#voice-server-environment)
3. [Frontend Environment](#frontend-environment)
4. [AI4Bharat Services](#ai4bharat-services)
5. [Security & Secrets](#security--secrets)

---

## Backend Environment

### File: `voicera_backend/.env`

#### Database Configuration

```env
# MongoDB Connection
MONGODB_HOST=mongodb
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera

# Connection Pool
MONGODB_MAX_POOL_SIZE=50
MONGODB_TIMEOUT_MS=5000
MONGODB_SERVER_SELECTION_TIMEOUT_MS=5000
```

#### MinIO Configuration

```env
# MinIO S3-Compatible Storage
MINIO_HOST=minio
MINIO_PORT=9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_REGION=us-east-1
MINIO_SECURE=false          # true for HTTPS in production

# Buckets (auto-created if not exist)
MINIO_BUCKET_RECORDINGS=recordings
MINIO_BUCKET_TRANSCRIPTS=transcripts
MINIO_BUCKET_AGENTS=agent-configs
```

#### JWT & Security

```env
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-change-this
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
JWT_REFRESH_EXPIRATION_DAYS=30

# CORS Configuration
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
CORS_ALLOW_CREDENTIALS=true
CORS_MAX_AGE=3600
```

#### API Configuration

```env
# API Settings
API_TITLE=VoiceERA API
API_VERSION=1.0.0
API_DESCRIPTION=Voice AI Platform with Telephony Integration
API_DOCS_ENABLED=true            # /docs endpoint
API_REDOC_ENABLED=true           # /redoc endpoint

# Server
HOST=0.0.0.0
PORT=8000
WORKERS=4
WORKER_CLASS=uvicorn.workers.UvicornWorker
RELOAD=false                     # true in development
```

#### Logging

```env
# Logging Configuration
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_FORMAT=json                  # json or text
LOG_FILE=/var/log/voicera/backend.log
LOG_MAX_BYTES=104857600          # 100MB
LOG_BACKUP_COUNT=10
```

#### Email Configuration (Optional)

```env
# SMTP for email notifications
SMTP_ENABLED=false
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USE_TLS=true
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@voicera.ai
```

---

## Voice Server Environment

### File: `voice_2_voice_server/.env`

#### Vobiz Telephony

```env
# Vobiz API Configuration
VOBIZ_ENABLED=true
VOBIZ_API_BASE=https://api.vobiz.ai/api/v1
VOBIZ_AUTH_ID=your_auth_id
VOBIZ_AUTH_PASSWORD=your_auth_password
VOBIZ_WEBHOOK_URL=https://yourdomain.com/vobiz/webhook
```

#### LLM Provider

=== "OpenAI"
    ```env
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-your-api-key-here
    OPENAI_MODEL=gpt-4                  # gpt-4, gpt-3.5-turbo
    OPENAI_TEMPERATURE=0.7
    OPENAI_MAX_TOKENS=200
    OPENAI_TIMEOUT_SECONDS=30
    ```

=== "Anthropic"
    ```env
    LLM_PROVIDER=anthropic
    ANTHROPIC_API_KEY=sk-ant-...
    ANTHROPIC_MODEL=claude-3-opus      # claude-3-opus, claude-3-sonnet
    ANTHROPIC_TEMPERATURE=0.7
    ```

=== "Local LLM"
    ```env
    LLM_PROVIDER=local
    LOCAL_LLM_API_BASE=http://localhost:8000
    LOCAL_LLM_MODEL=mistral-7b
    LOCAL_LLM_TEMPERATURE=0.7
    ```

#### Speech-to-Text (STT) Provider

=== "Deepgram"
    ```env
    STT_PROVIDER=deepgram
    DEEPGRAM_API_KEY=your-api-key
    DEEPGRAM_MODEL=nova-2              # nova-2, nova, enhanced
    DEEPGRAM_LANGUAGE=en
    DEEPGRAM_ENCODING=linear16
    DEEPGRAM_SAMPLE_RATE=16000
    ```

=== "Google Cloud"
    ```env
    STT_PROVIDER=google
    GOOGLE_CLOUD_STT_CREDENTIALS=/path/to/credentials.json
    GOOGLE_CLOUD_STT_LANGUAGE=en-US
    ```

=== "AI4Bharat"
    ```env
    STT_PROVIDER=ai4bharat
    STT_SERVICE_URL=http://ai4bharat_stt_server:8001
    STT_LANGUAGE=hi                    # Language code
    STT_SAMPLE_RATE=16000
    ```

#### Text-to-Speech (TTS) Provider

=== "Cartesia"
    ```env
    TTS_PROVIDER=cartesia
    CARTESIA_API_KEY=your-api-key
    CARTESIA_VOICE=english_male
    CARTESIA_LANGUAGE=en
    CARTESIA_SAMPLE_RATE=16000
    CARTESIA_SPEED=1.0
    ```

=== "Google Cloud"
    ```env
    TTS_PROVIDER=google
    GOOGLE_CLOUD_TTS_CREDENTIALS=/path/to/credentials.json
    GOOGLE_CLOUD_TTS_VOICE_NAME=en-US-Neural2-C
    GOOGLE_CLOUD_TTS_LANGUAGE=en-US
    ```

=== "AI4Bharat"
    ```env
    TTS_PROVIDER=ai4bharat
    TTS_SERVICE_URL=http://ai4bharat_tts_server:8002
    TTS_LANGUAGE=hi
    TTS_SPEAKER=female
    TTS_SAMPLE_RATE=16000
    TTS_SPEED=1.0
    ```

#### Backend Integration

```env
# Backend API Connection
BACKEND_API_URL=http://backend:8000    # Docker
# BACKEND_API_URL=http://localhost:8000 # Local dev
BACKEND_API_TIMEOUT=30

# Session Management
SESSION_TIMEOUT_MINUTES=30
MAX_CONCURRENT_CALLS=100
ENABLE_CALL_RECORDING=true
ENABLE_TRANSCRIPT_LOGGING=true
```

#### Server Configuration

```env
# Server
HOST=0.0.0.0
PORT=7860
WORKERS=4

# Audio
AUDIO_SAMPLE_RATE=16000
AUDIO_CHANNELS=1
AUDIO_FORMAT=pcm
AUDIO_CHUNK_SIZE=2048
```

#### Logging & Debugging

```env
# Logging
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
LOG_FILE=/var/log/voicera/voice_server.log
DEBUG_MODE=false                 # Enable verbose logging
ENABLE_AUDIO_LOGGING=false       # Log audio streams (CPU intensive!)
```

---

## Frontend Environment

### File: `voicera_frontend/.env.local`

#### API Configuration

```env
# Backend API
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_API_TIMEOUT=30000              # Milliseconds

# Voice Server WebSocket
NEXT_PUBLIC_VOICE_SERVER_URL=http://localhost:7860
NEXT_PUBLIC_WS_URL=ws://localhost:7860
NEXT_PUBLIC_WS_TIMEOUT=30000
```

#### Authentication

```env
# Authentication Settings
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_JWT_STORAGE_KEY=voicera_token
NEXT_PUBLIC_AUTH_REDIRECT_URL=/dashboard
NEXT_PUBLIC_LOGIN_URL=/login
```

#### Application Configuration

```env
# App Info
NEXT_PUBLIC_APP_NAME=VoiceERA
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_APP_ENVIRONMENT=development  # development, staging, production

# Logging
NEXT_PUBLIC_LOG_LEVEL=info                # debug, info, warn, error
NEXT_PUBLIC_SENTRY_DSN=
NEXT_PUBLIC_DEBUG=false
```

#### Analytics & Tracking

```env
# Analytics
NEXT_PUBLIC_ANALYTICS_ENABLED=false
NEXT_PUBLIC_GA_TRACKING_ID=              # Google Analytics ID
NEXT_PUBLIC_MIXPANEL_TOKEN=              # Mixpanel token

# Feature Flags
NEXT_PUBLIC_FEATURE_VOICE_CALLS=true
NEXT_PUBLIC_FEATURE_ANALYTICS=true
NEXT_PUBLIC_FEATURE_RECORDINGS=true
```

#### UI Configuration

```env
# UI Preferences
NEXT_PUBLIC_THEME=light                  # light, dark, auto
NEXT_PUBLIC_ANIMATIONS_ENABLED=true
NEXT_PUBLIC_RECORDS_PER_PAGE=25
```

---

## AI4Bharat Services

### File: `ai4bharat_stt_server/.env`

```env
# Server
HOST=0.0.0.0
PORT=8001
WORKERS=4

# Model Configuration
MODEL_NAME=indic-conformer-hi
MODEL_PATH=/models
DEVICE=cuda                    # cuda or cpu
BATCH_SIZE=32

# Audio
SAMPLE_RATE=16000
AUDIO_FORMAT=wav

# Logging
LOG_LEVEL=INFO
```

### File: `ai4bharat_tts_server/.env`

```env
# Server
HOST=0.0.0.0
PORT=8002
WORKERS=2

# Model Configuration
MODEL_NAME=indic-parler-hi
MODEL_PATH=/models
DEVICE=cuda                    # cuda or cpu

# Audio
SAMPLE_RATE=16000
AUDIO_FORMAT=wav
SPEED=1.0

# Caching
ENABLE_CACHING=true
CACHE_SIZE_MB=500
CACHE_TTL_HOURS=24

# Logging
LOG_LEVEL=INFO
```

---

## Security & Secrets

### Secret Management Best Practices

```bash
# Never commit secrets
echo ".env" >> .gitignore
echo ".env.prod" >> .gitignore
echo "secrets/" >> .gitignore

# Store secrets in environment variables
export MONGODB_PASSWORD=$(aws secretsmanager get-secret-value --secret-id prod/mongodb/password)
export JWT_SECRET_KEY=$(aws secretsmanager get-secret-value --secret-id prod/jwt-secret)

# Or use a secrets management tool
# HashiCorp Vault, AWS Secrets Manager, GCP Secret Manager
```

### Secret Rotation

```bash
# Rotate JWT secret
# 1. Generate new secret
NEW_SECRET=$(openssl rand -hex 32)

# 2. Update in environment
JWT_SECRET_KEY=$NEW_SECRET

# 3. Rotate tokens after grace period
# 4. Monitor for authentication failures
```

### Environment by Stage

**Development:**
```env
DEBUG=true
LOG_LEVEL=DEBUG
MONGODB_SECURE=false
MINIO_SECURE=false
JWT_EXPIRATION_HOURS=24
```

**Staging:**
```env
DEBUG=false
LOG_LEVEL=INFO
MONGODB_SECURE=true
MINIO_SECURE=true
JWT_EXPIRATION_HOURS=12
```

**Production:**
```env
DEBUG=false
LOG_LEVEL=WARNING
MONGODB_SECURE=true
MINIO_SECURE=true
JWT_EXPIRATION_HOURS=1
SESSION_TIMEOUT_MINUTES=15
```

---

## Validation Checklist

- [ ] All required variables set
- [ ] No secrets in .env.example
- [ ] API keys rotated regularly
- [ ] Passwords meet security requirements (16+ chars, mixed case, numbers, symbols)
- [ ] URLs use HTTPS in production
- [ ] Timeout values are reasonable
- [ ] Rate limits are set appropriately
- [ ] Logging doesn't expose sensitive data

---

## Next Steps

- **[Configuration Guide](../getting-started/configuration.md)** - Detailed configuration
- **[Quick Start](../getting-started/quickstart.md)** - Get running
- **[Production Deployment](production.md)** - Production setup
