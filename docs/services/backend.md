# Backend API Service

Comprehensive documentation for the VoiceERA Backend API service.

## Overview

The Backend API is the central orchestrator for the VoiceERA platform, built with **FastAPI** and **Python 3.10+**.

**Key Responsibilities:**
- User authentication & authorization
- Agent and campaign management
- Call log storage and retrieval
- Analytics aggregation
- MinIO storage integration
- Integration with Voice Server

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- MongoDB
- MinIO (optional for local development)

### Installation

```bash
cd voicera_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp env.example .env
# Edit .env with your settings
```

### Running Locally

```bash
# Development mode (auto-reload)
uvicorn app.main:app --reload --port 8000

# Production mode
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Via Docker

```bash
# Build image
docker build -t voicera-backend .

# Run container
docker run -p 8000:8000 \
  -e MONGODB_HOST=localhost \
  -e MONGODB_PORT=27017 \
  voicera-backend
```

---

## Project Structure

```
voicera_backend/
├── app/
│   ├── __init__.py
│   ├── main.py                    # Application entry point
│   ├── auth.py                    # Authentication utilities
│   ├── config.py                  # Configuration
│   ├── database.py                # Database connection
│   ├── models/
│   │   ├── __init__.py
│   │   ├── schemas.py             # Pydantic models
│   │   └── database_models.py     # MongoDB models
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── users.py               # User endpoints
│   │   ├── agents.py              # Agent endpoints
│   │   ├── campaigns.py           # Campaign endpoints
│   │   ├── call_logs.py           # Call log endpoints
│   │   ├── call_recordings.py     # Recording endpoints
│   │   ├── analytics.py           # Analytics endpoints
│   │   ├── integrations.py        # Third-party integrations
│   │   └── health.py              # Health check
│   ├── services/
│   │   ├── __init__.py
│   │   ├── agent_service.py       # Agent business logic
│   │   ├── campaign_service.py    # Campaign business logic
│   │   ├── call_recording_service.py
│   │   ├── analytics_service.py   # Analytics queries
│   │   ├── auth_service.py        # JWT & permissions
│   │   └── storage_service.py     # MinIO integration
│   ├── storage/
│   │   └── minio_client.py        # MinIO wrapper
│   ├── utils/
│   │   ├── security.py            # Encryption, hashing
│   │   ├── validators.py          # Data validation
│   │   └── constants.py           # Constant values
│   └── scripts/
│       └── seed_data.py           # Sample data for development
├── tests/
│   ├── test_auth.py
│   ├── test_agents.py
│   ├── test_campaigns.py
│   └── conftest.py                # Test fixtures
├── requirements.txt
├── env.example
├── docker-compose.yml
└── Dockerfile
```

---

## Key Models & Schemas

### User Model

```python
# MongoDB document
{
  "_id": ObjectId,
  "id": UUID,
  "email": str,
  "password_hash": str,
  "role": "admin" | "user",
  "first_name": str,
  "last_name": str,
  "is_active": bool,
  "created_at": datetime,
  "updated_at": datetime
}
```

### Agent Model

```python
{
  "_id": ObjectId,
  "id": UUID,
  "user_id": UUID,
  "name": str,
  "description": str,
  "llm_provider": str,  # "openai", "anthropic", "local"
  "llm_model": str,     # "gpt-4", "claude-3", etc.
  "stt_provider": str,  # "deepgram", "google", "ai4bharat"
  "tts_provider": str,  # "cartesia", "google", "ai4bharat"
  "system_prompt": str,
  "language": str,      # "en", "hi", etc.
  "voice_parameters": {
    "voice_id": str,
    "speed": float,
    "tone": str
  },
  "status": "active" | "inactive" | "archived",
  "created_at": datetime,
  "updated_at": datetime
}
```

### Campaign Model

```python
{
  "_id": ObjectId,
  "id": UUID,
  "user_id": UUID,
  "agent_id": UUID,
  "name": str,
  "description": str,
  "phone_numbers": [str],
  "status": "draft" | "scheduled" | "active" | "completed" | "paused",
  "start_time": datetime,
  "end_time": datetime,
  "max_concurrent_calls": int,
  "retry_config": {
    "max_retries": int,
    "retry_delay": int
  },
  "created_at": datetime,
  "updated_at": datetime
}
```

### CallLog Model

```python
{
  "_id": ObjectId,
  "id": UUID,
  "campaign_id": UUID,
  "agent_id": UUID,
  "phone_number": str,
  "caller_id": str,
  "status": "initiated" | "ringing" | "connected" | "completed" | "failed",
  "duration_seconds": int,
  "transcript": str,
  "summary": str,
  "sentiment": "positive" | "neutral" | "negative",
  "emotions": [str],
  "key_phrases": [str],
  "recording_path": str,       # Path in MinIO
  "error_message": str,        # If failed
  "cost": float,               # API call costs
  "created_at": datetime,
  "updated_at": datetime
}
```

---

## Core Endpoints

### Authentication

```
POST   /auth/register          # User registration
POST   /auth/login             # User login
POST   /auth/refresh-token     # Refresh JWT
GET    /auth/me                # Get current user
POST   /auth/logout            # Logout
POST   /auth/forgot-password   # Password reset
POST   /auth/reset-password    # Confirm password reset
```

### Agents

```
GET    /agents                 # List agents
POST   /agents                 # Create agent
GET    /agents/{agent_id}      # Get agent details
PUT    /agents/{agent_id}      # Update agent
DELETE /agents/{agent_id}      # Delete agent
GET    /agents/{agent_id}/config  # Get agent config
```

### Campaigns

```
GET    /campaigns              # List campaigns
POST   /campaigns              # Create campaign
GET    /campaigns/{campaign_id}    # Get campaign details
PUT    /campaigns/{campaign_id}    # Update campaign
DELETE /campaigns/{campaign_id}    # Delete campaign
POST   /campaigns/{campaign_id}/launch  # Start campaign
POST   /campaigns/{campaign_id}/pause   # Pause campaign
```

### Call Logs

```
GET    /call-logs              # List call logs
GET    /call-logs/{call_id}    # Get call details
GET    /call-logs/campaign/{campaign_id}  # Campaign call logs
DELETE /call-logs/{call_id}    # Delete call log
```

### Call Recordings

```
GET    /call-recordings/{call_id}  # Get recording metadata
GET    /call-recordings/{call_id}/download  # Download recording
GET    /call-recordings/{call_id}/transcript  # Get transcript
```

### Analytics

```
GET    /analytics/calls         # Call statistics
GET    /analytics/sentiment     # Sentiment analysis
GET    /analytics/top-phrases   # Most common phrases
GET    /analytics/agent-performance  # Agent metrics
GET    /analytics/campaign-stats    # Campaign statistics
```

### Health

```
GET    /health                 # Service health check
GET    /readiness              # Readiness probe
GET    /liveness               # Liveness probe
```

---

## Authentication & Authorization

### JWT Token Structure

```json
{
  "sub": "user-uuid",
  "email": "user@example.com",
  "role": "admin",
  "permissions": ["read", "write", "delete"],
  "iat": 1674003600,
  "exp": 1674007200,
  "aud": "voicera-api"
}
```

### Request Header

```http
Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
```

### Permission Scopes

- `read:agents` - List agents
- `write:agents` - Create/edit agents
- `delete:agents` - Delete agents
- `read:campaigns` - List campaigns
- `write:campaigns` - Create/edit campaigns
- `admin` - Full access

---

## Request/Response Examples

### Create Agent

**Request:**
```bash
curl -X POST http://localhost:8000/agents \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Sales Agent",
    "description": "Handles sales inquiries",
    "llm_provider": "openai",
    "llm_model": "gpt-4",
    "stt_provider": "deepgram",
    "tts_provider": "cartesia",
    "system_prompt": "You are a helpful sales agent...",
    "language": "en",
    "voice_parameters": {
      "voice_id": "english_male",
      "speed": 1.0,
      "tone": "professional"
    }
  }'
```

**Response:**
```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "name": "Sales Agent",
  "status": "active",
  "created_at": "2024-01-29T10:30:00Z",
  "updated_at": "2024-01-29T10:30:00Z"
}
```

### Get Call Log with Transcript

**Request:**
```bash
curl http://localhost:8000/call-logs/call-uuid \
  -H "Authorization: Bearer <token>"
```

**Response:**
```json
{
  "id": "call-uuid",
  "campaign_id": "campaign-uuid",
  "phone_number": "+1234567890",
  "status": "completed",
  "duration_seconds": 120,
  "transcript": "User: Hello... Agent: Hi there...",
  "sentiment": "positive",
  "emotions": ["satisfied", "engaged"],
  "recording_path": "s3://voicera-recordings/call-uuid.wav",
  "created_at": "2024-01-29T10:30:00Z"
}
```

---

## Environment Configuration

```env
# Database
MONGODB_HOST=localhost
MONGODB_PORT=27017
MONGODB_USER=admin
MONGODB_PASSWORD=admin123
MONGODB_DATABASE=voicera

# Storage
MINIO_HOST=localhost
MINIO_PORT=9000
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin

# Security
JWT_SECRET_KEY=your-secret-key
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# CORS
CORS_ORIGINS=["http://localhost:3000","https://yourdomain.com"]

# API
API_TITLE=VoiceERA API
API_VERSION=1.0.0
LOG_LEVEL=INFO
DEBUG=False
```

---

## Error Handling

### Standard Error Response

```json
{
  "detail": "Authentication failed",
  "error_code": "AUTH_001",
  "status_code": 401,
  "timestamp": "2024-01-29T10:30:00Z"
}
```

### Common Status Codes

| Code | Meaning |
|------|---------|
| 200 | Success |
| 201 | Created |
| 400 | Bad Request |
| 401 | Unauthorized |
| 403 | Forbidden |
| 404 | Not Found |
| 409 | Conflict |
| 429 | Rate Limited |
| 500 | Server Error |

---

## Next Steps

- **[REST API Details](../api/rest-api.md)** - Complete API documentation
- **[Quick Start](../getting-started/quickstart.md)** - Get started quickly
- **[Configuration](../getting-started/configuration.md)** - Configuration options
