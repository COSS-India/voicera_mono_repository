# Local Development Setup

Complete guide for setting up VoiceERA for local development.

## Development Environment

### Prerequisites

- Python 3.10+
- Node.js 18+
- Docker & Docker Compose
- Git
- Text editor or IDE (VS Code, PyCharm, etc.)

## Frontend Development

### Setup

```bash
cd voicera_frontend

# Install dependencies
npm install

# Create development environment
cp .env.example .env.local

# Edit configuration
nano .env.local
```

### Environment Configuration

```env
# .env.local
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
NEXT_PUBLIC_VOICE_SERVER_URL=http://localhost:7860
NEXT_PUBLIC_WS_URL=ws://localhost:7860
NEXT_PUBLIC_LOG_LEVEL=debug
NEXT_PUBLIC_DEBUG=true
```

### Running Dev Server

```bash
npm run dev

# Open http://localhost:3000
```

### Hot Reload

Changes to files automatically reload the browser. If not:

```bash
# Hard restart
npm run dev -- --turbo  # With Turbopack for faster builds
```

### Building for Production

```bash
npm run build
npm start  # Test production build
```

---

## Backend Development

### Setup

```bash
cd voicera_backend

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create development environment
cp env.example .env

# Edit configuration
nano .env
```

### Environment Configuration

```env
# .env for development
DEBUG=true
LOG_LEVEL=DEBUG
RELOAD=true

# Database
MONGODB_HOST=localhost
MONGODB_PORT=27017

# JWT
JWT_SECRET_KEY=dev-secret-key-change-in-production

# API
CORS_ORIGINS=["http://localhost:3000"]
```

### Start Database (Docker)

```bash
# From repo root
docker-compose up -d mongodb minio

# Wait for healthy status
docker-compose ps
```

### Running Dev Server

```bash
# With auto-reload
uvicorn app.main:app --reload --port 8000

# Open http://localhost:8000/docs (Swagger UI)
```

### Database Migrations (if applicable)

```bash
# Run migrations
alembic upgrade head

# Create new migration
alembic revision --autogenerate -m "Add new column"
```

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest

# Run with coverage
pytest --cov=app

# Run specific test file
pytest tests/test_auth.py
```

---

## Voice Server Development

### Setup

```bash
cd voice_2_voice_server

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create configuration
cp .env.example .env

# Edit configuration
nano .env
```

### Environment Configuration

```env
# .env for development
LOG_LEVEL=DEBUG
DEBUG_MODE=true
ENABLE_AUDIO_LOGGING=false

# LLM
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-...

# STT
STT_PROVIDER=deepgram
DEEPGRAM_API_KEY=...

# TTS
TTS_PROVIDER=cartesia
CARTESIA_API_KEY=...

# Backend
BACKEND_API_URL=http://localhost:8000
```

### Running Dev Server

```bash
# Start voice server
python main.py

# Should output: 
# INFO: Started server process [12345]
# Open http://localhost:7860
```

### Testing Audio Processing

```bash
# Create test script
cat > test_audio.py << 'EOF'
import asyncio
from services.ai4bharat.stt import AI4BharatSTT

async def test_stt():
    stt = AI4BharatSTT("http://localhost:8001")
    
    # Test with audio file
    with open("test_audio.wav", "rb") as f:
        audio_data = f.read()
    
    transcript = await stt.transcribe(audio_data)
    print(f"Transcript: {transcript}")

asyncio.run(test_stt())
EOF

# Run test
python test_audio.py
```

---

## IDE Setup

### VS Code Extensions

Recommended extensions:

```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.vscode-pylance",
    "ms-python.debugpy",
    "esbenp.prettier-vscode",
    "dbaeumer.vscode-eslint",
    "GitHub.copilot",
    "ms-vscode.makefile-tools",
    "ms-vscode.docker"
  ]
}
```

Install with:
```bash
code --install-extension <extension-id>
```

### Python Debugging (VS Code)

Create `.vscode/launch.json`:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Backend",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/voicera_backend/app/main.py",
      "console": "integratedTerminal",
      "env": {
        "PYTHONPATH": "${workspaceFolder}/voicera_backend",
        "RELOAD": "true"
      }
    },
    {
      "name": "Voice Server",
      "type": "python",
      "request": "launch",
      "program": "${workspaceFolder}/voice_2_voice_server/main.py",
      "console": "integratedTerminal"
    }
  ]
}
```

Press `F5` to start debugging.

### JavaScript Debugging (VS Code)

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "node",
      "request": "launch",
      "name": "Next.js",
      "skipFiles": ["<node_internals>/**"],
      "program": "${workspaceFolder}/voicera_frontend/node_modules/.bin/next",
      "args": ["dev"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## Development Workflow

### 1. Create Feature Branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

Edit files in your editor with hot-reload:

```bash
# Terminal 1: Frontend
cd voicera_frontend && npm run dev

# Terminal 2: Backend
cd voicera_backend && uvicorn app.main:app --reload

# Terminal 3: Voice Server
cd voice_2_voice_server && python main.py
```

### 3. Test Changes

```bash
# Backend tests
cd voicera_backend && pytest

# Frontend tests (if added)
cd voicera_frontend && npm test

# Manual testing via Swagger
curl http://localhost:8000/docs
```

### 4. Commit and Push

```bash
git add .
git commit -m "feat: Add new feature"
git push origin feature/your-feature-name
```

### 5. Create Pull Request

Submit PR with:
- Clear description
- Tests for new features
- Updated documentation

---

## Debugging Tips

### Backend Debugging

```python
# Add breakpoint
import pdb; pdb.set_trace()

# Or use VS Code debugger - set breakpoint and press F5
```

### Frontend Debugging

```javascript
// Open DevTools (F12)
// Set breakpoints in Sources tab
// Use console for debugging
console.log("Debug info", variable);
```

### Log Analysis

```bash
# View logs in real-time
docker-compose logs -f <service>

# Filter logs
docker-compose logs | grep "error"

# Save logs to file
docker-compose logs > logs.txt
```

---

## Common Development Tasks

### Add New Database Model

1. Create model in `voicera_backend/app/models/`
2. Add Pydantic schema
3. Create MongoDB collection with indexes
4. Add API endpoints in `routers/`

### Add New API Endpoint

```python
# voicera_backend/app/routers/new_feature.py
from fastapi import APIRouter, Depends
from app.models.schemas import MySchema

router = APIRouter(prefix="/feature", tags=["feature"])

@router.get("/")
async def list_items():
    """List all items"""
    return {"items": []}

# In app/main.py
from app.routers import new_feature
app.include_router(new_feature.router)
```

### Add Frontend Component

```bash
# Create component file
touch voicera_frontend/components/MyComponent.tsx

# Create story for Storybook (if using)
touch voicera_frontend/components/MyComponent.stories.tsx

# Use component
import MyComponent from '@/components/MyComponent';

export default function Page() {
  return <MyComponent />;
}
```

### Add Test

```bash
# Backend
touch voicera_backend/tests/test_new_feature.py

# Frontend (if using Jest)
touch voicera_frontend/__tests__/MyComponent.test.tsx
```

---

## Performance Optimization

### Backend Profiling

```python
# Add to code
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# ... your code ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.print_stats()
```

### Frontend Performance

```bash
# Build analysis
npm run build -- --analyze

# Lighthouse audit
npm run build && npm start
# Then run Chrome Lighthouse
```

---

## Dependency Management

### Update Dependencies

```bash
# Backend
pip list --outdated
pip install --upgrade package-name

# Frontend
npm outdated
npm update
```

### Security Checks

```bash
# Backend
safety check

# Frontend
npm audit
```

---

## Next Steps

- **[Contributing](contributing.md)** - Contribution guidelines
- **[Testing](testing.md)** - Testing guide
- **[Quick Start](../getting-started/quickstart.md)** - Get running
