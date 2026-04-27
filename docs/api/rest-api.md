# REST API Documentation

Complete REST API reference for VoiceERA Backend.

## Base URL

```
http://localhost:8000        # Development
https://api.yourdomain.com   # Production
```

## Authentication

All protected endpoints require JWT token in the `Authorization` header:

```http
Authorization: Bearer <jwt_token>
```

## Response Format

All responses are JSON:

```json
{
  "success": true,
  "data": {...},
  "error": null,
  "timestamp": "2024-01-29T10:30:00Z"
}
```

Or for errors:

```json
{
  "success": false,
  "data": null,
  "error": {
    "code": "AUTH_001",
    "message": "Invalid credentials",
    "details": "Email not found"
  },
  "timestamp": "2024-01-29T10:30:00Z"
}
```

---

## Authentication Endpoints

### Register User

```http
POST /auth/register
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!",
  "first_name": "John",
  "last_name": "Doe"
}
```

**Response:** `201 Created`
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "created_at": "2024-01-29T10:30:00Z"
}
```

### Login

```http
POST /auth/login
Content-Type: application/json

{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Response:** `200 OK`
```json
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 86400,
  "user": {
    "id": "user-uuid",
    "email": "user@example.com",
    "role": "user"
  }
}
```

### Get Current User

```http
GET /auth/me
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "id": "user-uuid",
  "email": "user@example.com",
  "first_name": "John",
  "last_name": "Doe",
  "role": "user",
  "created_at": "2024-01-29T10:30:00Z"
}
```

### Refresh Token

```http
POST /auth/refresh-token
Authorization: Bearer <refresh_token>
```

**Response:** `200 OK`
```json
{
  "token": "new-jwt-token",
  "expires_in": 86400
}
```

---

## Agent Endpoints

### List Agents

```http
GET /agents
Authorization: Bearer <token>

# Optional query parameters:
# ?skip=0&limit=10
# ?search=sales
# ?status=active
```

**Response:** `200 OK`
```json
{
  "total": 15,
  "skip": 0,
  "limit": 10,
  "items": [
    {
      "id": "agent-uuid",
      "name": "Sales Agent",
      "description": "Handles sales inquiries",
      "llm_provider": "openai",
      "llm_model": "gpt-4",
      "status": "active",
      "created_at": "2024-01-29T10:30:00Z"
    }
    // ... more agents
  ]
}
```

### Create Agent

```http
POST /agents
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Support Agent",
  "description": "Handles customer support",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "stt_provider": "deepgram",
  "tts_provider": "cartesia",
  "system_prompt": "You are a helpful support agent. Help customers with their issues.",
  "language": "en",
  "voice_parameters": {
    "voice_id": "english_male",
    "speed": 1.0,
    "tone": "professional"
  }
}
```

**Response:** `201 Created`
```json
{
  "id": "new-agent-uuid",
  "name": "Support Agent",
  "status": "active",
  "created_at": "2024-01-29T10:30:00Z"
}
```

### Get Agent

```http
GET /agents/{agent_id}
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "id": "agent-uuid",
  "user_id": "user-uuid",
  "name": "Support Agent",
  "description": "Handles customer support",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "stt_provider": "deepgram",
  "tts_provider": "cartesia",
  "system_prompt": "You are a helpful support agent...",
  "language": "en",
  "status": "active",
  "created_at": "2024-01-29T10:30:00Z"
}
```

### Update Agent

```http
PUT /agents/{agent_id}
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Updated Name",
  "system_prompt": "New system prompt here"
}
```

**Response:** `200 OK`
```json
{
  "id": "agent-uuid",
  "name": "Updated Name",
  "updated_at": "2024-01-29T10:35:00Z"
}
```

### Delete Agent

```http
DELETE /agents/{agent_id}
Authorization: Bearer <token>
```

**Response:** `204 No Content`

---

## Campaign Endpoints

### List Campaigns

```http
GET /campaigns
Authorization: Bearer <token>

# Optional query parameters:
# ?agent_id=uuid
# ?status=active
# ?skip=0&limit=20
```

**Response:** `200 OK`
```json
{
  "total": 5,
  "items": [
    {
      "id": "campaign-uuid",
      "name": "Q1 Sales Campaign",
      "agent_id": "agent-uuid",
      "status": "active",
      "phone_numbers": ["+1234567890"],
      "start_time": "2024-01-29T00:00:00Z",
      "end_time": "2024-03-31T23:59:59Z",
      "created_at": "2024-01-29T10:30:00Z"
    }
  ]
}
```

### Create Campaign

```http
POST /campaigns
Authorization: Bearer <token>
Content-Type: application/json

{
  "name": "Q2 Marketing Campaign",
  "agent_id": "agent-uuid",
  "phone_numbers": ["+1234567890", "+1234567891"],
  "start_time": "2024-04-01T00:00:00Z",
  "end_time": "2024-06-30T23:59:59Z",
  "max_concurrent_calls": 50
}
```

**Response:** `201 Created`
```json
{
  "id": "new-campaign-uuid",
  "status": "draft",
  "created_at": "2024-01-29T10:30:00Z"
}
```

### Launch Campaign

```http
POST /campaigns/{campaign_id}/launch
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "id": "campaign-uuid",
  "status": "active",
  "launched_at": "2024-01-29T10:35:00Z"
}
```

### Pause Campaign

```http
POST /campaigns/{campaign_id}/pause
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "id": "campaign-uuid",
  "status": "paused",
  "paused_at": "2024-01-29T10:36:00Z"
}
```

---

## Call Log Endpoints

### List Call Logs

```http
GET /call-logs
Authorization: Bearer <token>

# Optional filters:
# ?campaign_id=uuid
# ?status=completed
# ?phone_number=%2B1234567890
# ?start_date=2024-01-01
# ?end_date=2024-01-31
# ?skip=0&limit=50
```

**Response:** `200 OK`
```json
{
  "total": 1250,
  "items": [
    {
      "id": "call-uuid",
      "campaign_id": "campaign-uuid",
      "phone_number": "+1234567890",
      "status": "completed",
      "duration_seconds": 120,
      "sentiment": "positive",
      "emotions": ["satisfied"],
      "transcript": "User: Hello... Agent: Hi...",
      "created_at": "2024-01-29T10:30:00Z"
    }
  ]
}
```

### Get Call Details

```http
GET /call-logs/{call_id}
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "id": "call-uuid",
  "campaign_id": "campaign-uuid",
  "phone_number": "+1234567890",
  "status": "completed",
  "duration_seconds": 120,
  "transcript": "Full transcript here...",
  "summary": "Customer called about billing issue...",
  "sentiment": "positive",
  "emotions": ["satisfied", "relieved"],
  "key_phrases": ["billing", "resolution"],
  "recording_url": "https://minio:9001/recordings/call-uuid.wav",
  "cost": 0.50,
  "created_at": "2024-01-29T10:30:00Z"
}
```

---

## Recording Endpoints

### Get Recording

```http
GET /call-recordings/{call_id}
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "call_id": "call-uuid",
  "duration_seconds": 120,
  "format": "wav",
  "size_bytes": 960000,
  "download_url": "https://signed-url-with-expiry",
  "transcript": "Full transcript...",
  "metadata": {
    "agent_id": "agent-uuid",
    "campaign_id": "campaign-uuid",
    "phone_number": "+1234567890"
  }
}
```

### Download Recording

```http
GET /call-recordings/{call_id}/download
Authorization: Bearer <token>
```

**Response:** `200 OK` with audio file (WAV format)

---

## Analytics Endpoints

### Call Statistics

```http
GET /analytics/calls
Authorization: Bearer <token>

# Optional filters:
# ?agent_id=uuid
# ?campaign_id=uuid
# ?days=30
```

**Response:** `200 OK`
```json
{
  "total_calls": 1250,
  "completed_calls": 1200,
  "failed_calls": 50,
  "average_duration": 180,
  "success_rate": 96.0,
  "cost_per_call": 0.45
}
```

### Sentiment Analysis

```http
GET /analytics/sentiment
Authorization: Bearer <token>

# Optional:
# ?agent_id=uuid&days=7
```

**Response:** `200 OK`
```json
{
  "positive": 850,
  "neutral": 300,
  "negative": 100,
  "positive_percentage": 68.0,
  "neutral_percentage": 24.0,
  "negative_percentage": 8.0
}
```

### Top Phrases

```http
GET /analytics/top-phrases
Authorization: Bearer <token>

# Optional:
# ?agent_id=uuid&limit=20
```

**Response:** `200 OK`
```json
[
  {
    "phrase": "billing issue",
    "count": 234,
    "frequency": 18.7
  },
  {
    "phrase": "account access",
    "count": 156,
    "frequency": 12.5
  },
  {
    "phrase": "payment problem",
    "count": 145,
    "frequency": 11.6
  }
]
```

---

## Health & Status

### Health Check

```http
GET /health
```

**Response:** `200 OK`
```json
{
  "status": "ok",
  "timestamp": "2024-01-29T10:30:00Z"
}
```

### Readiness

```http
GET /readiness
```

**Response:** `200 OK`
```json
{
  "database": "ok",
  "storage": "ok",
  "cache": "ok"
}
```

---

## Error Codes

| Code | HTTP | Description |
|------|------|-------------|
| AUTH_001 | 401 | Invalid credentials |
| AUTH_002 | 401 | Token expired |
| AUTH_003 | 401 | Token invalid |
| AUTH_004 | 403 | Permission denied |
| AGENT_001 | 404 | Agent not found |
| AGENT_002 | 409 | Agent already exists |
| CAMPAIGN_001 | 404 | Campaign not found |
| CAMPAIGN_002 | 409 | Cannot launch campaign |
| CALL_001 | 404 | Call log not found |
| SERVER_001 | 500 | Internal server error |

---

## Rate Limiting

API has rate limiting to prevent abuse:

```
General endpoints: 100 requests/minute
Auth endpoints: 10 requests/minute
Upload endpoints: 10 MB/minute
```

Rate limit headers:

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1674007200
```

---

## Next Steps

- **[WebSocket API](websocket-api.md)** - Real-time communication
- **[Quick Start](../getting-started/quickstart.md)** - Get started
- **[Voice Server API](../services/voice-server.md)** - Voice processing
