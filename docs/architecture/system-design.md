# System Design

Detailed technical design and component documentation.

## Component Deep Dives

### 1. Frontend Service

**Stack:** Next.js 16+, React 18+, TailwindCSS 4+

**Port:** 3000

**Key Features:**
- Server-side rendering (SSR) for performance
- Client-side state management
- Real-time WebSocket connections
- Responsive design for mobile/desktop

**Key Modules:**
```
app/
├── (auth)              # Authentication pages
│   ├── login.tsx
│   └── signup.tsx
├── (dashboard)         # Main application
│   ├── agents/
│   ├── campaigns/
│   ├── call-logs/
│   └── analytics/
├── api/                # API routes
└── components/         # Shared components
```

**Dependencies:**
- `axios` or `fetch` - HTTP client
- `socket.io-client` - WebSocket client
- `zustand` or `redux` - State management
- `next-auth` - Authentication

### 2. Backend API Service

**Stack:** FastAPI, Python 3.10+, SQLAlchemy/Motor

**Port:** 8000

**Key Endpoints:**
- `/auth/*` - Authentication
- `/agents/*` - Agent CRUD operations
- `/campaigns/*` - Campaign management
- `/call-logs/*` - Call history
- `/transcripts/*` - Transcription data
- `/health` - Service health

**Database Schema:**
```
Users
├── id (UUID)
├── email
├── password (hashed)
├── role (admin/user)
└── created_at

Agents
├── id (UUID)
├── name
├── llm_provider
├── stt_provider
├── tts_provider
├── system_prompt
├── user_id (FK)
└── created_at

Campaigns
├── id (UUID)
├── name
├── agent_id (FK)
├── phone_numbers
├── status
└── created_at

CallLogs
├── id (UUID)
├── campaign_id (FK)
├── phone_number
├── duration
├── transcript
├── status
└── created_at
```

**Key Services:**
```python
# app/services/
├── agent_service.py      # Agent logic
├── campaign_service.py   # Campaign logic
├── call_recording_service.py
├── auth_service.py       # JWT tokens
└── analytics_service.py  # Call analytics
```

### 3. Voice Server

**Stack:** Pipecat, Python 3.10+

**Port:** 7860

**Responsibilities:**
- Accept WebSocket connections from frontend
- Receive raw audio frames
- Process audio through STT pipeline
- Generate LLM responses
- Convert responses to speech via TTS
- Stream audio back to user

**Pipeline Architecture:**
```
Audio Input
    │
    ▼
[Resampler]
    │
    ▼
[STT Service]
    │
    ▼ (Transcript)
[LLM Service]
    │
    ▼ (Response)
[TTS Service]
    │
    ▼
[Audio Output]
```

**Configuration:**
```yaml
# config/config.example.yaml
stt:
  provider: deepgram  # or ai4bharat, google, etc.
  language: en
  
llm:
  provider: openai
  model: gpt-4
  temperature: 0.7
  
tts:
  provider: cartesia
  voice: english_male
  speed: 1.0
```

### 4. MongoDB

**Purpose:** Primary data store

**Collections:**
- `users` - User accounts
- `agents` - Agent configurations
- `campaigns` - Campaign definitions
- `call_logs` - Call history and metadata
- `transcripts` - Call transcriptions
- `analytics` - Aggregated metrics

**Indices:**
```javascript
// Performance-critical indices
db.users.createIndex({ email: 1 }, { unique: true })
db.agents.createIndex({ user_id: 1 })
db.campaigns.createIndex({ agent_id: 1 })
db.call_logs.createIndex({ campaign_id: 1, created_at: -1 })
db.call_logs.createIndex({ phone_number: 1 })
```

### 5. MinIO (Object Storage)

**Purpose:** Store binary files - recordings, transcripts, etc.

**Bucket Structure:**
```
minio/
├── recordings/
│   └── {campaign_id}/{call_id}.wav
├── transcripts/
│   └── {campaign_id}/{call_id}.json
└── agent-configs/
    └── {agent_id}/config.json
```

**Access:** S3-compatible API (AWS SDK compatible)

### 6. External AI Services

#### LLM Services
- **OpenAI:** GPT-4, GPT-3.5-turbo
- **Anthropic:** Claude
- **Local:** LLaMA, Mistral (self-hosted)

#### STT Services
- **Deepgram:** High-accuracy transcription
- **Google Cloud:** Speech-to-Text
- **AI4Bharat:** Indic language support

#### TTS Services
- **Cartesia:** High-quality voice synthesis
- **Google Cloud:** Text-to-Speech
- **AI4Bharat:** Indic language synthesis

---

## Request-Response Flows

### Authentication Flow

```
1. Frontend sends login credentials
   POST /auth/login { email, password }
   
2. Backend validates credentials
   - Hash password
   - Compare with stored hash
   
3. Backend generates JWT token
   token = sign({ user_id, exp, permissions })
   
4. Frontend stores token
   localStorage.setItem('token', jwt)
   
5. Subsequent requests include token
   headers.Authorization = "Bearer " + jwt
```

### Voice Call Flow

```
1. User initiates call in frontend
   websocket.connect('ws://voice-server:7860')
   
2. Frontend sends auth token
   message: { type: 'auth', token: jwt }
   
3. Voice Server validates token
   - Decodes JWT
   - Checks permissions
   
4. Voice Server sends ready signal
   message: { type: 'ready', session_id: uuid }
   
5. Frontend streams audio chunks
   message: { type: 'audio', data: ArrayBuffer }
   
6. Voice Server processes
   - STT: Audio → Text
   - LLM: Text → Response
   - TTS: Response → Audio
   
7. Voice Server sends response audio
   message: { type: 'audio', data: ArrayBuffer }
   
8. Frontend plays audio
   audioContext.playback(data)
```

### Recording & Storage

```
1. During call, Voice Server buffers audio

2. After call ends
   - Save raw audio to MinIO
   - Save transcript to MongoDB & MinIO
   - Log call metadata to MongoDB
   
3. Frontend fetches call data
   GET /call-logs/{call_id}
   - Returns metadata from MongoDB
   - Returns pre-signed URL to audio in MinIO
```

---

## Error Handling & Resilience

### Circuit Breaker Pattern

```python
class CircuitBreaker:
    def __init__(self, failure_threshold=5, timeout=60):
        self.failure_count = 0
        self.threshold = failure_threshold
        self.timeout = timeout
    
    def call(self, func, *args):
        if self.is_open():
            raise CircuitBreakerOpen()
        
        try:
            result = func(*args)
            self.reset()
            return result
        except Exception as e:
            self.record_failure()
            raise
```

### Retry Logic

```python
@retry(max_attempts=3, backoff=exponential)
def call_external_api(endpoint):
    # Automatically retry on failure
    # with exponential backoff
    pass
```

### Health Checks

**Backend:**
```
GET /health
Response: { status: "ok", timestamp: "..." }
```

**Voice Server:**
```
- STT service connectivity
- LLM service connectivity
- TTS service connectivity
- Memory usage
- Active connections
```

---

## Performance Considerations

### Caching Strategy

```python
# Cache agent configurations (10 min TTL)
@cache(ttl=600)
def get_agent(agent_id):
    return db.agents.find_one({"_id": ObjectId(agent_id)})

# Cache call logs (1 hour TTL)
@cache(ttl=3600)
def get_call_logs(campaign_id, limit=100):
    return db.call_logs.find({...})
```

### Database Optimization

- Connection pooling
- Query indexing
- Pagination for list endpoints
- Aggregation pipelines for analytics

### Audio Processing

- Streaming rather than buffering entire files
- Resampling to match provider requirements
- Compression for storage

---

## Monitoring & Observability

### Logging

All services log to stdout in JSON format:

```json
{
  "timestamp": "2024-01-29T10:30:45Z",
  "level": "INFO",
  "service": "backend",
  "message": "Call initiated",
  "call_id": "abc123",
  "agent_id": "xyz789"
}
```

### Metrics

Key metrics to monitor:

- **API Response Time:** Mean, P95, P99
- **Call Duration:** Average, distribution
- **STT Accuracy:** Word error rate
- **LLM Response Time:** Latency
- **System Resources:** CPU, Memory, Disk
- **Error Rate:** By service, by endpoint

### Tracing

Optional: Implement distributed tracing with Jaeger or Zipkin to track request flows across services.

---

## Security Considerations

### Data Validation

```python
from pydantic import BaseModel, validator

class CreateAgentRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    llm_provider: str = Field(..., pattern="^[a-z0-9_]+$")
    
    @validator('name')
    def name_must_be_valid(cls, v):
        if not v.strip():
            raise ValueError('Name cannot be empty')
        return v.strip()
```

### SQL Injection Prevention

- Use parameterized queries (SQLAlchemy ORM)
- Never concatenate user input into queries

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Rate Limiting

```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/auth/login")
@limiter.limit("5/minute")
async def login(request: Request, ...):
    # Max 5 login attempts per minute
    pass
```

---

## Next Steps

- **[Data Flow](data-flow.md)** - Trace how data moves through the system
- **[Docker Deployment](../deployment/docker.md)** - Containerized deployment
- **[API Documentation](../api/rest-api.md)** - Detailed API endpoints
