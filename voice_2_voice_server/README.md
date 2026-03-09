# Voice 2 Voice Server

A FastAPI-based telephony server for voice-to-voice interactions using Vobiz telephony platform.

## Project Structure

```
voice_2_voice_server/
├── api/                       # API layer
│   ├── __init__.py            # Module exports
│   ├── server.py              # FastAPI endpoints & webhooks
│   ├── bot.py                 # Voice bot pipeline
│   └── services.py            # Service factories (LLM, STT, TTS)
├── config/                    # Configuration & language mappings
│   ├── __init__.py
│   ├── llm_mappings.py        # LLM provider mappings
│   ├── stt_mappings.py        # STT language code mappings
│   ├── tts_mappings.py        # TTS language code mappings
│   ├── config.yaml            # Server config (gitignored)
│   └── config.example.yaml    # Example configuration
├── services/                  # Custom service implementations
│   ├── ai4bharat/             # AI4Bharat Indic STT/TTS
│   │   ├── stt.py             # IndicConformer WebSocket STT
│   │   └── tts.py             # IndicParler WebSocket TTS
│   └── kenpath_llm/           # Kenpath LLM service
│       └── llm.py
├── serializer/                # Audio frame serializers
│   └── vobiz_serializer.py    # Vobiz protocol serializer
├── agent_configs/             # Agent configuration files
│   ├── default_agent.json
│   ├── sales_agent.json
│   └── indic_english.json
├── audio/                     # Audio files
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies
└── .gitignore
```

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**
   ```bash
   # Vobiz credentials
   export VOBIZ_API_BASE="https://api.vobiz.ai/api/v1"
   export VOBIZ_AUTH_ID="your_auth_id"
   export VOBIZ_AUTH_TOKEN="your_auth_token"
   export VOBIZ_CALLER_ID="your_caller_id"
   
   # Server URLs
   export JOHNAIC_SERVER_URL="https://your-server.com"
   export JOHNAIC_WEBSOCKET_URL="wss://your-server.com"
   
   # Provider API keys (as needed)
   export OPENAI_API_KEY="sk-..."
   export SARVAM_LLM_API_KEY="your-sarvam-llm-api-key"
   export SARVAM_LLM_ENDPOINT="https://e75f29cc-857c-4a05-ac2c-70f4291e0bef.invocation.api.nvcf.nvidia.com/v1/chat/completions"
   export DEEPGRAM_API_KEY="..."
   export CARTESIA_API_KEY="..."
   
   # Audio settings
   export SAMPLE_RATE=8000
   
   # MinIO Configuration
   export MINIO_ENDPOINT="localhost:9000"
   export MINIO_ACCESS_KEY="minioadmin"
   export MINIO_SECRET_KEY="minioadmin"
   export MINIO_SECURE="false"
   
   # Backend API (optional)
   export VOICERA_BACKEND_URL="http://localhost:8000"
   export INTERNAL_API_KEY="your-internal-api-key"
   ```

3. **Run the server:**
   ```bash
   python main.py
   ```
   
   Or with uvicorn:
   ```bash
   uvicorn api.server:app --host 0.0.0.0 --port 7860
   ```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Health check |
| `/docs` | GET | Swagger API documentation |
| `/redoc` | GET | ReDoc API documentation |
| `/outbound/call/` | POST | Initiate outbound call |
| `/answer` | GET/POST | Vobiz answer webhook |
| `/agent/{agent_type}` | WebSocket | Audio streaming endpoint |

### Outbound Call

```bash
curl -X POST "http://localhost:7860/outbound/call/" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_number": "+1234567890",
    "agent_type": "sales_agent",
    "caller_id": "+0987654321"
  }'
```

## Agent Configuration

Agent configs are JSON files in `agent_configs/`. Example:

```json
{
  "system_prompt": "You are a helpful voice assistant.",
  "greeting_message": "Hello, how can I help you?",
  "session_timeout_minutes": 10,
  
  "llm_model": {
    "name": "openai",
    "args": {
      "model": "gpt-4o",
      "temperature": 0.7
    }
  },
  
  "stt_model": {
    "name": "deepgram",
    "language": "English",
    "args": {
      "model": "nova-3"
    }
  },
  
  "tts_model": {
    "name": "cartesia",
    "language": "English",
    "args": {
      "model": "sonic-2",
      "voice_id": "bf0a246a-8642-498a-9950-80c35e9276b5"
    }
  }
}
```

## Supported Providers

### LLM
| Provider | Name | Models |
|----------|------|--------|
| OpenAI | `openai` | gpt-4o, gpt-4o-mini, gpt-3.5-turbo |
| Sarvam | `sarvam` | sarvamai/sarvam-30b |
| Kenpath | `kenpath` | Custom LLM |

### STT (Speech-to-Text)
| Provider | Name | Models |
|----------|------|--------|
| Deepgram | `deepgram` | nova-3, nova-2, flux-general-en |
| Google | `google` | chirp_3, chirp_2, telephony |
| OpenAI | `openai` | whisper-1 |
| AI4Bharat | `indic-conformer-stt` | Indic languages |

### TTS (Text-to-Speech)
| Provider | Name | Models |
|----------|------|--------|
| Cartesia | `cartesia` | sonic-3, sonic-2, sonic-multilingual |
| Google | `google` | Various voices |
| OpenAI | `openai` | alloy, echo, fable, onyx, nova, shimmer |
| AI4Bharat | `indic-parler-tts` | Indic languages |

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Vobiz Telephony                        │
└─────────────────────────┬───────────────────────────────────┘
                          │ WebSocket (audio stream)
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    FastAPI Server                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │
│  │   /answer   │  │ /outbound/  │  │ /agent/{type}       │  │
│  │  (webhook)  │  │   call/     │  │   (WebSocket)       │  │
│  └─────────────┘  └─────────────┘  └──────────┬──────────┘  │
└───────────────────────────────────────────────┼─────────────┘
                                                │
                          ┌─────────────────────▼─────────────────────┐
                          │              Voice Bot Pipeline           │
                          │  ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐      │
                          │  │ STT │→ │ LLM │→ │ TTS │→ │ Out │      │
                          │  └─────┘  └─────┘  └─────┘  └─────┘      │
                          └───────────────────────────────────────────┘
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VOBIZ_API_BASE` | Yes | - | Vobiz API base URL |
| `VOBIZ_AUTH_ID` | Yes | - | Vobiz account auth ID |
| `VOBIZ_AUTH_TOKEN` | Yes | - | Vobiz account auth token |
| `VOBIZ_CALLER_ID` | No | - | Default caller ID for outbound calls |
| `JOHNAIC_SERVER_URL` | Yes | - | Public server URL for webhooks |
| `JOHNAIC_WEBSOCKET_URL` | Yes | - | Public WebSocket URL |
| `SAMPLE_RATE` | No | 8000 | Audio sample rate in Hz |
| `MINIO_ENDPOINT` | Yes | - | MinIO server endpoint (e.g., `localhost:9000`) |
| `MINIO_ACCESS_KEY` | Yes | - | MinIO access key |
| `MINIO_SECRET_KEY` | Yes | - | MinIO secret key |
| `MINIO_SECURE` | No | `false` | Use secure connection (HTTPS) for MinIO |
| `VOICERA_BACKEND_URL` | No | `http://localhost:8000` | Backend API URL |
| `INTERNAL_API_KEY` | No | - | Internal API key for backend communication |
| `OPENAI_API_KEY` | * | - | OpenAI API key |
| `SARVAM_LLM_API_KEY` | * | - | Sarvam LLM API key (Bearer token) |
| `SARVAM_LLM_ENDPOINT` | No | NVIDIA NVCF chat completion URL | Sarvam LLM endpoint |
| `SARVAM_LLM_MODEL` | No | `sarvamai/sarvam-30b` | Sarvam LLM model id |
| `SARVAM_LLM_TEMPERATURE` | No | `0.8` | Sarvam LLM temperature |
| `SARVAM_LLM_MAX_TOKENS` | No | `2048` | Sarvam LLM max tokens |
| `SARVAM_LLM_REPETITION_PENALTY` | No | `1.0` | Sarvam repetition penalty |
| `SARVAM_LLM_TIMEOUT_SECONDS` | No | `60` | Sarvam request timeout |
| `DEEPGRAM_API_KEY` | * | - | Deepgram API key |
| `CARTESIA_API_KEY` | * | - | Cartesia API key |
| `GOOGLE_STT_CREDENTIALS_PATH` | * | credentials/google_stt.json | Google STT credentials |
| `GOOGLE_TTS_CREDENTIALS_PATH` | * | credentials/google_tts.json | Google TTS credentials |
| `INDIC_STT_SERVER_URL` | * | - | AI4Bharat STT server WebSocket URL |
| `INDIC_TTS_SERVER_URL` | * | - | AI4Bharat TTS server URL |

\* Required based on configured providers

## Development

### Code Structure

- **`api/services.py`** - Factory functions for creating LLM, STT, TTS services
- **`api/bot.py`** - Pipecat pipeline setup and event handlers
- **`api/server.py`** - FastAPI routes and Vobiz integration

### Adding a New Provider

1. Create service implementation in `services/`
2. Add factory function in `api/services.py`
3. Add language mappings in `config/` if needed
