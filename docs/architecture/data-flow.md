# Data Flow Through the System

This document explains how data moves through VoiceERA during various operations.

## Voice Call Data Flow

The most complex and important flow in VoiceERA.

### Complete Voice Call Sequence

```
┌─────────────────────────────────────────────────────────────────────┐
│ Step 1: Call Initiation                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Caller (Phone) ──[Ring & Answer]──► Vobiz Platform                │
│                                           │                         │
│                                           ▼                         │
│                      Voice Server [WebSocket Ready]                │
│                                           │                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Step 2: Audio Input (Caller Speaks)                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Caller Audio Frames ──[Vobiz]──► Voice Server                     │
│                                        │                            │
│                        ┌───────────────┼───────────────┐            │
│                        │               │               │            │
│                        ▼               ▼               ▼            │
│                    [Buffer]        [Resample]   [Validate Audio]   │
│                        │               │               │            │
│                        └───────────────┼───────────────┘            │
│                                        │                            │
│                                        ▼                            │
│                                   [STT Service]                    │
│                                        │                            │
│                                        ▼                            │
│                                   "Hello, I'd like..."              │
│                                   (Transcript)                      │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Step 3: LLM Processing (Generate Response)                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Transcript ──► [System Prompt + Context] ──► LLM API              │
│                                                   │                 │
│                                                   ▼                 │
│                                              OpenAI, Claude, etc.   │
│                                                   │                 │
│                                                   ▼                 │
│                                      "Sure, I'd be happy to help..."│
│                                      (LLM Response)                 │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Step 4: TTS Synthesis (Voice Generation)                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Response Text ──► TTS API ──► Voice Generation                    │
│                                   │                                │
│                                   ▼                                │
│                            [Audio Synthesis]                       │
│                                   │                                │
│                                   ▼                                │
│                              Audio Frames                          │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Step 5: Audio Output (Caller Hears Response)                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Audio Frames ──► [Vobiz] ──► Caller (Phone Speaker)              │
│                                                                      │
│                        [Loop continues]                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ Step 6: Call Termination & Storage                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Caller hangs up ──► Voice Server                                  │
│                          │                                          │
│          ┌───────────────┼───────────────┐                         │
│          │               │               │                         │
│          ▼               ▼               ▼                         │
│       [Save Audio]   [Save Transcript] [Log Metadata]              │
│          │               │               │                         │
│          ▼               ▼               ▼                         │
│        MinIO          MongoDB         MongoDB                      │
│      (.wav file)      (transcript)    (call_log)                   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Detailed Data Structures

**Audio Frame:**
```json
{
  "type": "audio",
  "session_id": "uuid",
  "timestamp": 1234567890,
  "sequence": 1,
  "data": "<base64-encoded-audio>",
  "format": "pcm_16k"
}
```

**STT Request:**
```json
{
  "audio": "<audio-data>",
  "language": "en",
  "model": "nova-2"
}
```

**STT Response:**
```json
{
  "transcript": "Hello, I'd like to know about your services",
  "confidence": 0.98,
  "alternatives": [
    { "transcript": "Hello, I'd like to know about the services" }
  ]
}
```

**LLM Request:**
```json
{
  "model": "gpt-4",
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful customer service agent..."
    },
    {
      "role": "user",
      "content": "Hello, I'd like to know about your services"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 150
}
```

**LLM Response:**
```json
{
  "choices": [
    {
      "message": {
        "role": "assistant",
        "content": "Sure, I'd be happy to help! We offer..."
      }
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 78,
    "total_tokens": 123
  }
}
```

**Call Log (MongoDB):**
```json
{
  "_id": ObjectId(...),
  "campaign_id": "uuid",
  "phone_number": "+1234567890",
  "caller_id": "uuid",
  "duration_seconds": 120,
  "status": "completed",
  "transcript": "...",
  "emotions": ["satisfied", "engaged"],
  "sentiment": "positive",
  "created_at": ISODate("2024-01-29T10:30:00Z"),
  "updated_at": ISODate("2024-01-29T10:32:00Z")
}
```

**Recording (MinIO):**
```
Path: recordings/{campaign_id}/{call_id}.wav
Size: ~960 KB (2-minute call at 16kHz)
Format: WAV (PCM, 16-bit, 16kHz mono)
```

---

## Agent Creation Data Flow

```
┌─────────────────────────────┐
│ Frontend: Create Agent Form │
└─────────────────────────────┘
          │
          ▼
   ┌──────────────┐
   │ Validate Form│
   └──────────────┘
          │
          ▼
  POST /agents (JSON)
```

**Request Payload:**
```json
{
  "name": "Sales Agent",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "stt_provider": "deepgram",
  "tts_provider": "cartesia",
  "system_prompt": "You are a sales agent...",
  "language": "en"
}
```

**Processing in Backend:**
```
1. Validate request data
2. Check user permissions
3. Create Agent document in MongoDB
4. Generate config file
5. Save config to MinIO
6. Return Agent ID to frontend
```

**Response:**
```json
{
  "id": "agent-uuid",
  "name": "Sales Agent",
  "status": "active",
  "created_at": "2024-01-29T10:30:00Z"
}
```

**Stored in MongoDB (agents collection):**
```json
{
  "_id": ObjectId(...),
  "id": "agent-uuid",
  "user_id": "user-uuid",
  "name": "Sales Agent",
  "llm_provider": "openai",
  "llm_model": "gpt-4",
  "stt_provider": "deepgram",
  "tts_provider": "cartesia",
  "system_prompt": "You are a sales agent...",
  "language": "en",
  "status": "active",
  "created_at": ISODate(...),
  "updated_at": ISODate(...)
}
```

---

## Campaign Launch Data Flow

```
┌──────────────────────────┐
│ Backend: Create Campaign │
└──────────────────────────┘
          │
          ▼
  POST /campaigns (JSON)
  
  {
    "name": "New Year Sale",
    "agent_id": "agent-uuid",
    "phone_numbers": ["+1234567890", "+1234567891"],
    "start_time": "2024-02-01T00:00:00Z",
    "end_time": "2024-02-28T23:59:59Z"
  }

┌──────────────────────────────┐
│ Backend: Validation & Setup  │
└──────────────────────────────┘
          │
  ┌───────┼────────┐
  │       │        │
  ▼       ▼        ▼
1. Fetch 2. Check 3. Create
   Agent  Limits   Campaign
   Config         Document

┌──────────────────────┐
│ Save to MongoDB      │
│ campaigns collection │
└──────────────────────┘
          │
          ▼
  ┌───────────────────┐
  │ Schedule calls    │
  │ via Vobiz API     │
  └───────────────────┘
          │
          ▼
  Response: campaign_id
```

---

## Analytics Data Flow

```
Real-time call data
       │
       ├─► MongoDB (Raw logs)
       │
       ├─► Aggregation Pipeline
       │
       └─► Cached metrics
           (Redis optional)
           
Frontend Dashboard
       │
       ▼
GET /analytics/calls?agent_id=...
GET /analytics/sentiment?campaign_id=...
GET /analytics/top-phrases?agent_id=...
       │
       ▼
  Backend aggregates
       │
       ▼
  Return JSON metrics
       │
       ▼
  Display in charts
```

**Analytics Data in MongoDB:**
```json
{
  "_id": ObjectId(...),
  "type": "call_analytics",
  "agent_id": "agent-uuid",
  "campaign_id": "campaign-uuid",
  "period": "2024-01-29",
  "metrics": {
    "total_calls": 150,
    "completed_calls": 145,
    "average_duration": 180,
    "sentiment": {
      "positive": 95,
      "neutral": 40,
      "negative": 10
    },
    "top_phrases": [
      "pricing",
      "features",
      "support"
    ]
  }
}
```

---

## Recording Retrieval Data Flow

**User Action:** Download call recording

```
Frontend
    │
    ├─► User clicks "Download"
    │
    ├─► GET /call-logs/{call_id}
    │
    ▼
Backend
    │
    ├─► Fetch metadata from MongoDB
    │
    ├─► Generate pre-signed URL for MinIO
    │
    ▼
Response
    │
    ├─► {
         "call_id": "...",
         "download_url": "https://minio:9001/...",
         "transcript": "...",
         "duration": 120
       }
    │
    ▼
Frontend
    │
    ├─► User clicks download link
    │
    └─► Browser downloads audio from MinIO
```

---

## Authentication Data Flow

```
┌──────────────────────────────┐
│ Frontend: Login Form         │
│ Email: user@example.com      │
│ Password: ••••••••           │
└──────────────────────────────┘
          │
          ▼
POST /auth/login
{
  "email": "user@example.com",
  "password": "••••••••"
}

┌──────────────────────────────┐
│ Backend: Authenticate        │
└──────────────────────────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
1. Fetch   2. Hash &
   User       Compare
   from       password
   MongoDB
    │           │
    └─────┬─────┘
          │
          ▼
    ┌──────────────┐
    │ Credentials  │
    │ Valid?       │
    └──────────────┘
          │
          ├─── YES ──► Generate JWT
          │                │
          │                ▼
          │           {
          │             "user_id": "uuid",
          │             "email": "...",
          │             "iat": 1234567890,
          │             "exp": 1234571490,
          │             "permissions": ["read", "write"]
          │           }
          │                │
          │                ▼
          │           Response:
          │           {
          │             "token": "eyJ0eXAi...",
          │             "expires_in": 3600
          │           }
          │
          ├─── NO ──► Response 401 Unauthorized
```

---

## Caching & Performance

### Backend Caching

```
Request 1: GET /agents/123
  │
  ├─► Cache MISS
  │      │
  │      ▼
  │   Query MongoDB
  │      │
  │      ▼
  │   Store in cache (TTL: 10min)
  │      │
  └─────► Return to client

Request 2: GET /agents/123 (within 10min)
  │
  ├─► Cache HIT
  │      │
  └─────► Return from cache (no DB query)
```

### Database Query Optimization

**Without index (slow):**
```
SCAN all documents in call_logs collection
Filter by campaign_id
O(n) - millions of documents
```

**With index (fast):**
```
LOOKUP in index for campaign_id
O(log n) - direct access
```

---

## Error Handling & Recovery

### STT Failure Scenario

```
Voice Server receives audio
    │
    ▼
Send to STT Service
    │
    X Error: Service unavailable
    │
    ▼
Retry with backoff (1s, 2s, 4s)
    │
    ├─► Success on retry 2
    │      │
    │      ▼
    │   Continue call flow
    │
    └─► All retries failed
         │
         ▼
      Fallback: Ask user to repeat
      │
      ▼
      If persistent: End call gracefully
```

---

## Summary

| Operation | Data Path | Storage | Latency |
|-----------|-----------|---------|---------|
| **Voice Call** | Audio frames → STT → LLM → TTS | MinIO + MongoDB | <500ms |
| **Create Agent** | Form → Backend → MongoDB | MongoDB | <200ms |
| **Launch Campaign** | Form → Backend → Vobiz | MongoDB | <500ms |
| **Get Analytics** | Query MongoDB → Aggregate | MongoDB | <1000ms |
| **Download Recording** | Pre-signed URL → MinIO | MinIO | Depends on file size |

---

## Next Steps

- **[System Design Details](system-design.md)** - Component implementation details
- **[REST API](../api/rest-api.md)** - Endpoint specifications
- **[WebSocket API](../api/websocket-api.md)** - Real-time communication protocol
