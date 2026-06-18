# Voice Server Service

Comprehensive documentation for the VoicEra Voice Server service.

## Overview

The Voice Server is the real-time voice processing engine for VoicEra, built with **Pipecat** and **Python 3.10+**.

**Key Responsibilities:**
- Establish WebSocket connections with clients
- Process real-time audio streams
- Orchestrate STT, LLM, and TTS services
- Manage concurrent voice sessions
- Handle audio input/output
- Store call recordings and transcripts

## Pipeline and providers (per agent)

STT, TTS, and LLM are chosen **per agent** in the dashboard (`llm_model`, `stt_model`, `tts_model`). There is **no automatic fallback** between cloud and local AI4Bharat — configure each agent explicitly.

| Topic | Documentation |
|-------|----------------|
| Provider list | `voice_2_voice_server/README.md` in the repo |
| Bhili language (`bhb`) | Local STT `/transcribe/bhili`; Kenpath Voice Bhili LLM on `dev` |
| Browser test | `voice_2_voice_server/docs/talk-on-browser-feature.md` |
| STT/TTS batching | `dev` branch — `voice_2_voice_server/api/batching.py` |

## Getting Started

### Prerequisites

- Python 3.10+
- pip
- Vobiz credentials in **Dashboard → Integrations** (for production telephony)
- AI keys in Integrations or env (OpenAI, Deepgram, Cartesia, Bhashini, etc.)

### Installation

```bash
cd voice_2_voice_server

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

```bash
# Copy example config
cp .env.example .env

# Edit with your settings
nano .env
```

### Running Locally

```bash
# Development mode
python main.py

# With logging
python main.py --log-level DEBUG

# Via Docker
docker build -t voicera-voice-server .
docker run -p 7860:7860 \
  -e OPENAI_API_KEY=sk-... \
  -e DEEPGRAM_API_KEY=... \
  voicera-voice-server
```

---

## Project Structure

```
voice_2_voice_server/
├── api/
│   ├── __init__.py
│   ├── server.py              # FastAPI server & WebSocket
│   ├── bot.py                 # Voice bot pipeline
│   └── services.py            # Service factories
├── config/
│   ├── __init__.py
│   ├── llm_mappings.py        # LLM provider configs
│   ├── stt_mappings.py        # STT language mappings
│   ├── tts_mappings.py        # TTS language mappings
│   ├── config.yaml            # Main config (gitignored)
│   └── config.example.yaml    # Example config
├── services/
│   ├── ai4bharat/
│   │   ├── __init__.py
│   │   ├── stt.py             # IndicConformer
│   │   └── tts.py             # IndicParler
│   ├── audio/
│   │   ├── greeting_interruption_filter.py
│   │   └── __init__.py
│   ├── bhashini/
│   │   ├── __init__.py
│   │   ├── stt.py
│   │   └── tts.py
│   ├── kenpath_llm/
│   │   ├── __init__.py
│   │   └── llm.py
│   └── __init__.py
├── serializer/
│   ├── __init__.py
│   └── vobiz_serializer.py    # Protocol serializer
├── storage/
│   ├── __init__.py
│   └── minio_client.py        # MinIO integration
├── agent_configs/             # Agent config files
│   ├── default_agent.json
│   ├── sales_agent.json
│   └── indic_english.json
├── main.py                    # Application entry
├── requirements.txt
├── env.example
├── Dockerfile
└── README.md
```

---

## Core Components

### Voice Bot Pipeline

The heart of the system - orchestrates STT, LLM, and TTS.

```python
class VoiceBot:
    """
    Real-time voice processing pipeline
    
    Flow: Audio Input → STT → LLM → TTS → Audio Output
    """
    
    def __init__(self, agent_config):
        self.stt_service = create_stt_service(agent_config)
        self.llm_service = create_llm_service(agent_config)
        self.tts_service = create_tts_service(agent_config)
    
    async def process_audio(self, audio_chunk):
        # Step 1: Convert audio to text
        transcript = await self.stt_service.transcribe(audio_chunk)
        
        # Step 2: Get LLM response
        response = await self.llm_service.generate(
            transcript,
            system_prompt=self.agent_config.system_prompt
        )
        
        # Step 3: Convert response to speech
        audio_response = await self.tts_service.synthesize(response)
        
        return audio_response
```

### WebSocket Protocol

**Message Format:**
```json
{
  "type": "audio|control|status",
  "session_id": "uuid",
  "timestamp": 1674003600000,
  "payload": {...}
}
```

**Audio Message:**
```json
{
  "type": "audio",
  "session_id": "session-uuid",
  "sequence": 1,
  "format": "pcm_16k",
  "data": "base64-encoded-audio"
}
```

**Control Message:**
```json
{
  "type": "control",
  "action": "pause|resume|end",
  "session_id": "session-uuid"
}
```

### Session Management

```python
class SessionManager:
    """Manage active voice sessions"""
    
    async def create_session(self, auth_token, agent_id):
        session = Session(
            session_id=uuid4(),
            agent_id=agent_id,
            user_id=extract_user_id(auth_token),
            created_at=datetime.now(),
            status="active"
        )
        self.active_sessions[session.session_id] = session
        return session
    
    async def end_session(self, session_id):
        session = self.active_sessions.get(session_id)
        if session:
            # Save recording
            await self.save_recording(session)
            # Save transcript
            await self.save_transcript(session)
            # Cleanup
            del self.active_sessions[session_id]
```

---

## Service Providers

### LLM Services

#### OpenAI

```python
class OpenAIService:
    def __init__(self, api_key, model="gpt-4"):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    async def generate(self, prompt, system_prompt=""):
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=200
        )
        return response.choices[0].message.content
```

#### Local LLM

```python
class LocalLLMService:
    def __init__(self, api_base, model="mistral-7b"):
        self.api_base = api_base
        self.model = model
    
    async def generate(self, prompt, system_prompt=""):
        # Call local Ollama or similar
        response = await aiohttp.post(
            f"{self.api_base}/api/generate",
            json={"model": self.model, "prompt": prompt}
        )
        return response.json()
```

### STT Services

#### Deepgram

```python
class DeepgramSTT:
    def __init__(self, api_key):
        self.client = DeepgramClient(api_key=api_key)
    
    async def transcribe(self, audio_data):
        # Streaming transcription
        response = await self.client.transcribe_streaming(
            audio_data,
            model="nova-2",
            language="en"
        )
        return response.transcript
```

#### AI4Bharat (Indic Languages)

```python
class AI4BharatSTT:
    def __init__(self, service_url):
        self.service_url = service_url
    
    async def transcribe(self, audio_data, language="hi"):
        # WebSocket to local AI4Bharat server
        response = await aiohttp.post(
            f"{self.service_url}/transcribe",
            json={"audio": base64.b64encode(audio_data).decode(),
                  "language": language}
        )
        return response.json()["transcript"]
```

### TTS Services

#### Cartesia

```python
class CartesiaTTS:
    def __init__(self, api_key):
        self.client = CartesiaClient(api_key=api_key)
    
    async def synthesize(self, text, voice="english_male"):
        audio = await self.client.synthesize(
            text=text,
            voice_id=voice,
            sample_rate=16000
        )
        return audio
```

#### AI4Bharat

```python
class AI4BharatTTS:
    def __init__(self, service_url):
        self.service_url = service_url
    
    async def synthesize(self, text, language="hi", voice="female"):
        response = await aiohttp.post(
            f"{self.service_url}/synthesize",
            json={
                "text": text,
                "language": language,
                "voice": voice
            }
        )
        return response.json()["audio"]
```

---

## WebSocket API

### Connection Flow

```javascript
// Client-side example
const socket = new WebSocket('ws://localhost:7860/voice');

socket.onopen = () => {
  // Send authentication
  socket.send(JSON.stringify({
    type: 'auth',
    token: jwtToken
  }));
};

socket.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  if (message.type === 'ready') {
    // Server is ready, start sending audio
    const audioContext = new AudioContext();
    // ... capture and send audio frames
  } else if (message.type === 'audio') {
    // Play response audio
    playAudio(message.data);
  }
};
```

### Server-side Handlers

```python
# FastAPI WebSocket endpoint
@app.websocket("/voice")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    try:
        # Authenticate
        auth_msg = await websocket.receive_json()
        user_id = validate_token(auth_msg['token'])
        
        # Create session
        session = await session_manager.create_session(
            auth_msg['token'],
            agent_id=auth_msg.get('agent_id')
        )
        
        # Signal ready
        await websocket.send_json({
            'type': 'ready',
            'session_id': str(session.session_id)
        })
        
        # Process messages
        while True:
            message = await websocket.receive_json()
            
            if message['type'] == 'audio':
                audio_chunk = base64.b64decode(message['data'])
                response_audio = await voice_bot.process_audio(
                    audio_chunk
                )
                await websocket.send_json({
                    'type': 'audio',
                    'data': base64.b64encode(response_audio).decode()
                })
            
            elif message['type'] == 'control':
                if message['action'] == 'end':
                    break
    
    finally:
        await session_manager.end_session(session.session_id)
```

---

## Configuration

### config.yaml Structure

```yaml
stt:
  provider: deepgram          # or ai4bharat, google, azure
  deepgram:
    api_key: ${DEEPGRAM_API_KEY}
    model: nova-2
    language: en
  ai4bharat:
    service_url: http://ai4bharat_stt_server:8001
    language: hi

llm:
  provider: openai            # or anthropic, local
  openai:
    api_key: ${OPENAI_API_KEY}
    model: gpt-4
    temperature: 0.7
    max_tokens: 200
  local:
    api_base: http://localhost:8000
    model: mistral-7b

tts:
  provider: cartesia          # or google, ai4bharat
  cartesia:
    api_key: ${CARTESIA_API_KEY}
    voice: english_male
    sample_rate: 16000
  ai4bharat:
    service_url: http://ai4bharat_tts_server:8002
    language: hi
    voice: female

server:
  host: 0.0.0.0
  port: 7860
  max_connections: 100
  session_timeout: 1800       # 30 minutes

backend:
  api_url: http://backend:8000
  upload_recordings: true
  upload_transcripts: true
```

---

## Monitoring & Health

### Health Check Endpoint

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "services": {
            "stt": await stt_service.health_check(),
            "llm": await llm_service.health_check(),
            "tts": await tts_service.health_check(),
            "backend": await backend.health_check()
        },
        "active_sessions": len(session_manager.active_sessions),
        "timestamp": datetime.utcnow().isoformat()
    }
```

### Metrics

Track these key metrics:
- **Session count** - Active concurrent sessions
- **Audio latency** - Time from audio received to response audio sent
- **Error rate** - STT/LLM/TTS failures
- **Uptime** - Service availability
- **Resource usage** - CPU, memory, connections

---

## Error Handling

### Retry Logic

```python
@retry_with_backoff(max_retries=3, base_delay=1)
async def call_external_service(service, *args):
    try:
        return await service.call(*args)
    except ServiceError as e:
        if e.is_retryable():
            raise  # Will trigger retry
        else:
            # Don't retry for non-transient errors
            raise NoRetryError(str(e))
```

### Graceful Degradation

```python
async def process_audio(audio_chunk):
    try:
        transcript = await stt_service.transcribe(audio_chunk)
    except STTError:
        # Fallback: ask user to repeat
        return await tts_service.synthesize(
            "Sorry, I didn't catch that. Could you repeat?"
        )
    
    try:
        response = await llm_service.generate(transcript)
    except LLMError:
        # Fallback response
        return await tts_service.synthesize(
            "I'm having trouble processing your request. Please try again."
        )
    
    try:
        audio = await tts_service.synthesize(response)
    except TTSError:
        # Log error and disconnect
        logger.error("TTS failed, ending session")
        raise SessionEndError()
    
    return audio
```

---

## Performance Optimization

### Audio Buffering

```python
class AudioBuffer:
    def __init__(self, sample_rate=16000, duration_ms=100):
        self.sample_rate = sample_rate
        self.buffer_size = (sample_rate * duration_ms) // 1000
        self.buffer = []
    
    def add(self, chunk):
        self.buffer.extend(chunk)
        if len(self.buffer) >= self.buffer_size:
            return self.get_and_clear()
        return None
```

### Connection Pooling

```python
# Reuse HTTP connections
session = aiohttp.ClientSession(
    connector=aiohttp.TCPConnector(
        limit=100,
        limit_per_host=30,
        ttl_dns_cache=300
    )
)
```

---

## Next Steps

- **[WebSocket API Details](../api/websocket-api.md)** - Protocol specifications
- **[Quick Start](../getting-started/quickstart.md)** - Get running
- **[Configuration](../getting-started/configuration.md)** - Configure services
