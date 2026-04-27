# Configuration Guide

Comprehensive guide to configuring VoiceERA services.

## Configuration Overview

VoiceERA uses environment variables for configuration. Each service has its own `.env` file:

- `voicera_backend/.env` - Backend API configuration
- `voice_2_voice_server/.env` - Voice server & AI provider keys
- `voicera_frontend/.env.local` - Frontend configuration
- `ai4bharat_stt_server/.env` - STT service (optional)
- `ai4bharat_tts_server/.env` - TTS service (optional)

## Backend Configuration

### File: `voicera_backend/.env`

#### Database Settings

```env
# MongoDB Connection
MONGODB_HOST=mongodb              # Hostname (docker: 'mongodb', local: 'localhost')
MONGODB_PORT=27017               # Default MongoDB port
MONGODB_USER=admin               # Root username
MONGODB_PASSWORD=admin123        # Root password
MONGODB_DATABASE=voicera         # Database name

# Connection options
MONGODB_MAX_POOL_SIZE=50         # Connection pool size
MONGODB_TIMEOUT_MS=5000          # Connection timeout
```

#### MinIO Storage

```env
# MinIO S3-Compatible Storage
MINIO_HOST=minio                 # Hostname
MINIO_PORT=9000                  # API port
MINIO_ROOT_USER=minioadmin       # Root access key
MINIO_ROOT_PASSWORD=minioadmin   # Root secret key
MINIO_SECURE=false               # Use HTTPS (false for local)
MINIO_REGION=us-east-1           # AWS region

# Buckets
MINIO_BUCKET_RECORDINGS=recordings        # Call recordings
MINIO_BUCKET_TRANSCRIPTS=transcripts      # Transcripts
MINIO_BUCKET_AGENT_CONFIGS=agent-configs # Agent configs
```

#### Security

```env
# JWT Authentication
JWT_SECRET_KEY=your-secret-key-change-this        # Secret for signing tokens
JWT_ALGORITHM=HS256                               # Signing algorithm
JWT_EXPIRATION_HOURS=24                           # Token expiration time

# CORS
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]
```

#### API Keys & Services

```env
# Backend API
API_TITLE=VoiceERA API
API_VERSION=1.0.0
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR

# Email (optional)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
SMTP_FROM=noreply@voicera.ai
```

## Voice Server Configuration

### File: `voice_2_voice_server/.env`

#### Vobiz Telephony

```env
# Vobiz API Configuration
VOBIZ_API_BASE=https://api.vobiz.ai/api/v1
VOBIZ_AUTH_ID=your_auth_id              # From Vobiz dashboard
VOBIZ_AUTH_PASSWORD=your_auth_password  # From Vobiz dashboard
VOBIZ_ENABLED=true
```

#### LLM Providers

=== "OpenAI"
    ```env
    LLM_PROVIDER=openai
    OPENAI_API_KEY=sk-...           # Get from https://platform.openai.com/api-keys
    OPENAI_MODEL=gpt-4              # gpt-4, gpt-3.5-turbo
    OPENAI_TEMPERATURE=0.7
    ```

=== "Local LLM"
    ```env
    LLM_PROVIDER=local
    LOCAL_LLM_API_BASE=http://localhost:8000
    LOCAL_LLM_MODEL=mistral-7b
    ```

=== "Other Providers"
    ```env
    LLM_PROVIDER=anthropic          # or cohere, huggingface
    ANTHROPIC_API_KEY=...
    ```

#### Speech-to-Text (STT) Providers

=== "Deepgram"
    ```env
    STT_PROVIDER=deepgram
    DEEPGRAM_API_KEY=...            # Get from https://console.deepgram.com
    DEEPGRAM_MODEL=nova-2
    DEEPGRAM_LANGUAGE=en
    ```

=== "AI4Bharat (Local)"
    ```env
    STT_PROVIDER=ai4bharat
    STT_SERVICE_URL=http://ai4bharat_stt_server:8001
    STT_LANGUAGE=hi                 # Language code
    ```

=== "Other Providers"
    ```env
    STT_PROVIDER=google             # or azure, ibm, etc.
    GOOGLE_CLOUD_STT_CREDENTIALS=./path/to/credentials.json
    ```

#### Text-to-Speech (TTS) Providers

=== "Cartesia"
    ```env
    TTS_PROVIDER=cartesia
    CARTESIA_API_KEY=...            # Get from https://console.cartesia.ai
    CARTESIA_VOICE=english_male     # Voice selection
    ```

=== "AI4Bharat (Local)"
    ```env
    TTS_PROVIDER=ai4bharat
    TTS_SERVICE_URL=http://ai4bharat_tts_server:8002
    TTS_LANGUAGE=hi                 # Language code
    ```

=== "Other Providers"
    ```env
    TTS_PROVIDER=google             # or azure, elevenlabs, etc.
    GOOGLE_CLOUD_TTS_CREDENTIALS=./path/to/credentials.json
    ```

    #### Johnaic / External Webhook (Optional)

    If you expose your local `voice_2_voice_server` via Ngrok (or another tunnel), set the public URLs here so external systems can call your local server.

    ```env
    # Example (replace with your Ngrok URL)
    JOHNAIC_SERVER_URL="https://abcd-12-34-56-78.ngrok-free.app"
    JOHNAIC_WEBSOCKET_URL="ws://abcd-12-34-56-78.ngrok-free.app"
    ```

    Use `wss://` in `JOHNAIC_WEBSOCKET_URL` when Ngrok provides a secure websocket endpoint.

    See the [Installation](installation.md#optional-ngrok-expose-local-voice-server) section for Ngrok setup steps and examples.

#### Backend Integration

```env
# Backend API Connection
BACKEND_API_URL=http://backend:8000      # For Docker
# BACKEND_API_URL=http://localhost:8000   # For local dev

# Session Management
SESSION_TIMEOUT_MINUTES=30
MAX_CONCURRENT_CALLS=10
```

#### Logging & Debugging

```env
LOG_LEVEL=INFO                   # DEBUG, INFO, WARNING, ERROR
ENABLE_AUDIO_LOGGING=false       # Log audio streams (CPU intensive!)
DEBUG_MODE=false
```

## Frontend Configuration

### File: `voicera_frontend/.env.local`

```env
# API Configuration
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_API_TIMEOUT=30000                # Timeout in ms

# Voice Server (WebSocket)
NEXT_PUBLIC_VOICE_SERVER_URL=http://localhost:7860
NEXT_PUBLIC_WS_URL=ws://localhost:7860

# Authentication
NEXT_PUBLIC_AUTH_ENABLED=true
NEXT_PUBLIC_JWT_STORAGE_KEY=voicera_token

# Application
NEXT_PUBLIC_APP_NAME=VoiceERA
NEXT_PUBLIC_APP_VERSION=1.0.0
NEXT_PUBLIC_LOG_LEVEL=info                  # debug, info, warn, error

# Analytics (optional)
NEXT_PUBLIC_ANALYTICS_ENABLED=false
NEXT_PUBLIC_GA_TRACKING_ID=
```

## AI4Bharat Services (Optional)

### File: `ai4bharat_stt_server/.env`

```env
# Model Configuration
STT_MODEL=indic-conformer-hi    # Language-specific model
MODEL_PATH=/models              # Path to downloaded model
DEVICE=cuda                     # cuda or cpu
BATCH_SIZE=32

# Server
HOST=0.0.0.0
PORT=8001
WORKERS=4
```

### File: `ai4bharat_tts_server/.env`

```env
# Model Configuration
TTS_MODEL=indic-parler-hi       # Language-specific model
MODEL_PATH=/models              # Path to downloaded model
DEVICE=cuda                     # cuda or cpu

# Server
HOST=0.0.0.0
PORT=8002
WORKERS=2
```

## Environment Variables by Use Case

### Development Setup

```env
# backend/.env
LOG_LEVEL=DEBUG
JWT_SECRET_KEY=dev-secret-key-not-secure
CORS_ORIGINS=["http://localhost:3000","http://localhost:8000"]

# voice_server/.env
LLM_PROVIDER=openai
STT_PROVIDER=deepgram
TTS_PROVIDER=cartesia
LOG_LEVEL=DEBUG
DEBUG_MODE=true

# frontend/.env.local
NEXT_PUBLIC_LOG_LEVEL=debug
```

### Production Setup

```env
# backend/.env
LOG_LEVEL=WARNING
JWT_SECRET_KEY=<generated-secure-key>
CORS_ORIGINS=["https://yourdomain.com"]
MONGODB_SECURE=true
MINIO_SECURE=true

# voice_server/.env
LLM_PROVIDER=openai
STT_PROVIDER=deepgram
TTS_PROVIDER=cartesia
LOG_LEVEL=INFO
DEBUG_MODE=false
BACKEND_API_URL=https://api.yourdomain.com

# frontend/.env.local
NEXT_PUBLIC_API_BASE_URL=https://api.yourdomain.com
NEXT_PUBLIC_VOICE_SERVER_URL=https://voice.yourdomain.com
NEXT_PUBLIC_LOG_LEVEL=warn
```

### Testing Setup

```env
# backend/.env
MONGODB_DATABASE=voicera_test
LOG_LEVEL=DEBUG

# voice_server/.env
MAX_CONCURRENT_CALLS=5
SESSION_TIMEOUT_MINUTES=5
```

## Configuration Validation

### Check Backend Config

```bash
# Access backend container
docker-compose exec backend bash

# Test MongoDB connection
python -c "from app.database import get_db; print('✓ MongoDB connected')"

# Test MinIO connection
python -c "from app.storage import get_minio; print('✓ MinIO connected')"
```

### Check Voice Server Config

```bash
# Access voice server container
docker-compose exec voice_server bash

# Test LLM provider
python -c "from api.services import get_llm_service; print('✓ LLM provider ready')"

# Test Backend connection
curl http://backend:8000/health
```

### Check Frontend Config

```bash
# Frontend configuration is embedded during build
# Check build output for configuration summary
docker-compose logs frontend | grep -i config
```

## Regenerating Secrets

### Generate a new JWT secret

```bash
# Using OpenSSL
openssl rand -hex 32

# Using Python
python -c "import secrets; print(secrets.token_hex(32))"

# Output: a1b2c3d4e5f6... (copy this to JWT_SECRET_KEY)
```

### Generate MinIO credentials

```bash
# Using OpenSSL
openssl rand -base64 32   # For username
openssl rand -base64 32   # For password
```

## Reload Configuration

After modifying `.env` files:

```bash
# Restart affected services
docker-compose restart backend voice_server frontend

# View updated configuration in logs
docker-compose logs backend | head -50
```

## Troubleshooting Configuration

### "Connection refused" errors

1. Check host/port in `.env` matches docker-compose.yml
2. For Docker containers, use service names (not localhost)
3. For local development, use localhost/127.0.0.1

### "Invalid API key" errors

1. Verify API key is set correctly (no extra spaces)
2. Check API key is still valid on provider's dashboard
3. Ensure key has correct permissions

### "Port already in use"

1. Change port in docker-compose.yml AND `.env`
2. Example: Change `MINIO_PORT=9001` to `MINIO_PORT=9002`

## Next Steps

- [Quick Start](quickstart.md) - Start services with your configuration
- [Architecture Guide](../architecture/overview.md) - Understand config impact
- [Troubleshooting](../troubleshooting.md) - Solve common issues
