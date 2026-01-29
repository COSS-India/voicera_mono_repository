# Testing Guide

Complete testing guide for VoiceERA development.

## Testing Overview

VoiceERA uses automated testing across all services:

- **Backend**: Unit tests, integration tests, API tests
- **Frontend**: Component tests, integration tests, E2E tests
- **Voice Server**: Integration tests, audio processing tests
- **CI/CD**: All tests run automatically on PRs

## Backend Testing

### Setup

```bash
cd voicera_backend
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_auth.py

# Run specific test function
pytest tests/test_auth.py::test_login

# Verbose output
pytest -v

# With coverage report
pytest --cov=app

# Stop on first failure
pytest -x

# Run only failed tests
pytest --lf

# Run tests matching pattern
pytest -k "auth"

# Parallel testing
pytest -n auto
```

### Test Structure

```
voicera_backend/
├── tests/
│   ├── conftest.py              # Shared fixtures
│   ├── test_auth.py             # Auth tests
│   ├── test_agents.py           # Agent tests
│   ├── test_campaigns.py        # Campaign tests
│   ├── test_api/
│   │   ├── test_auth_routes.py
│   │   ├── test_agent_routes.py
│   │   └── test_campaign_routes.py
│   ├── test_services/
│   │   ├── test_agent_service.py
│   │   └── test_campaign_service.py
│   └── test_models/
│       ├── test_schemas.py
│       └── test_database.py
```

### Writing Tests

#### Unit Test Example

```python
# tests/test_agents.py
import pytest
from app.models.schemas import AgentCreate
from app.services.agent_service import AgentService

@pytest.fixture
def agent_service():
    """Fixture for agent service"""
    return AgentService()

def test_create_agent(agent_service):
    """Test agent creation"""
    agent_data = AgentCreate(
        name="Support Bot",
        llm_provider="openai",
        system_prompt="You are a helpful support agent"
    )
    
    agent = agent_service.create_agent(agent_data)
    
    assert agent.name == "Support Bot"
    assert agent.llm_provider == "openai"

def test_get_agent(agent_service):
    """Test get agent"""
    # Create agent first
    agent_data = AgentCreate(
        name="Test Agent",
        llm_provider="openai"
    )
    created_agent = agent_service.create_agent(agent_data)
    
    # Get agent
    retrieved = agent_service.get_agent(created_agent.id)
    
    assert retrieved.id == created_agent.id
    assert retrieved.name == "Test Agent"
```

#### Async Test Example

```python
# tests/test_voice_server.py
import pytest
from voice_2_voice_server.services.llm import LLMService

@pytest.mark.asyncio
async def test_llm_response():
    """Test LLM response generation"""
    service = LLMService("openai")
    
    response = await service.generate(
        messages=[{"role": "user", "content": "Hello"}],
        max_tokens=100
    )
    
    assert response.content
    assert len(response.content) > 0
```

#### Mocking Example

```python
# tests/test_external_services.py
from unittest.mock import patch, AsyncMock
import pytest

@pytest.mark.asyncio
async def test_stt_fallback():
    """Test STT fallback on error"""
    with patch('services.deepgram.transcribe') as mock_stt:
        # Mock Deepgram error
        mock_stt.side_effect = Exception("API Error")
        
        # Should fall back to AI4Bharat
        result = await transcribe_with_fallback(audio_data)
        
        assert result.provider == "ai4bharat"
```

### Test Fixtures

```python
# tests/conftest.py
import pytest
from app.database import get_db
from app.models.user import User

@pytest.fixture
def db():
    """Test database fixture"""
    db = get_test_db()
    yield db
    db.cleanup()

@pytest.fixture
def auth_headers(db):
    """Authentication headers for tests"""
    user = User.create(
        email="test@example.com",
        password="testpass"
    )
    token = create_test_token(user)
    return {"Authorization": f"Bearer {token}"}

@pytest.fixture
def client(db):
    """FastAPI test client"""
    from fastapi.testclient import TestClient
    from app.main import app
    return TestClient(app)
```

### API Testing

```python
# tests/test_api/test_agents.py
from fastapi.testclient import TestClient

def test_list_agents(client, auth_headers):
    """Test list agents endpoint"""
    response = client.get(
        "/agents",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data

def test_create_agent(client, auth_headers):
    """Test create agent endpoint"""
    response = client.post(
        "/agents",
        json={
            "name": "Support Bot",
            "llm_provider": "openai",
            "system_prompt": "Help users"
        },
        headers=auth_headers
    )
    
    assert response.status_code == 201
    data = response.json()
    assert data["name"] == "Support Bot"
    assert data["id"]

def test_unauthorized(client):
    """Test unauthorized access"""
    response = client.get("/agents")
    
    assert response.status_code == 401
```

## Frontend Testing

### Setup

```bash
cd voicera_frontend

# Install dependencies
npm install --save-dev jest @testing-library/react @testing-library/jest-dom
```

### Running Tests

```bash
# Run all tests
npm test

# Watch mode
npm test -- --watch

# Coverage report
npm test -- --coverage

# Run specific test
npm test -- auth.test.tsx

# Update snapshots
npm test -- -u
```

### Writing Tests

#### Component Test Example

```typescript
// __tests__/components/AgentCard.test.tsx
import { render, screen } from '@testing-library/react';
import { AgentCard } from '@/components/AgentCard';

describe('AgentCard', () => {
  it('renders agent information', () => {
    const agent = {
      id: '123',
      name: 'Support Bot',
      status: 'active'
    };
    
    render(<AgentCard agent={agent} />);
    
    expect(screen.getByText('Support Bot')).toBeInTheDocument();
    expect(screen.getByText('active')).toBeInTheDocument();
  });
  
  it('handles delete action', async () => {
    const onDelete = jest.fn();
    const agent = { id: '123', name: 'Bot', status: 'active' };
    
    render(<AgentCard agent={agent} onDelete={onDelete} />);
    
    const deleteButton = screen.getByRole('button', { name: /delete/i });
    fireEvent.click(deleteButton);
    
    expect(onDelete).toHaveBeenCalledWith('123');
  });
});
```

#### Hook Test Example

```typescript
// __tests__/hooks/useAuth.test.ts
import { renderHook, act } from '@testing-library/react';
import { useAuth } from '@/hooks/useAuth';

describe('useAuth', () => {
  it('returns user after login', async () => {
    const { result } = renderHook(() => useAuth());
    
    await act(async () => {
      await result.current.login('user@example.com', 'password');
    });
    
    expect(result.current.user).toBeDefined();
    expect(result.current.user?.email).toBe('user@example.com');
  });
  
  it('clears user on logout', async () => {
    const { result } = renderHook(() => useAuth());
    
    await act(async () => {
      await result.current.logout();
    });
    
    expect(result.current.user).toBeNull();
  });
});
```

## Voice Server Testing

### Audio Processing Tests

```python
# voice_2_voice_server/tests/test_audio_processing.py
import pytest
import numpy as np
from api.bot import VoiceBot

@pytest.mark.asyncio
async def test_audio_frame_processing():
    """Test audio frame processing"""
    bot = VoiceBot()
    
    # Create test audio (16-bit PCM, 16kHz, mono)
    audio_data = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    
    result = await bot.process_audio(audio_data)
    
    assert result is not None
    assert hasattr(result, 'transcript')
```

### STT/TTS Tests

```python
# voice_2_voice_server/tests/test_services.py
import pytest

@pytest.mark.asyncio
async def test_deepgram_stt():
    """Test Deepgram STT service"""
    from services.deepgram.stt import DeepgramSTT
    
    stt = DeepgramSTT(api_key="test-key")
    
    # Mock API response
    with patch.object(stt, '_call_api') as mock:
        mock.return_value = {"transcript": "hello world"}
        
        result = await stt.transcribe(audio_data)
        
        assert result == "hello world"

@pytest.mark.asyncio
async def test_cartesia_tts():
    """Test Cartesia TTS service"""
    from services.cartesia.tts import CartesiaTTS
    
    tts = CartesiaTTS(api_key="test-key")
    
    audio = await tts.synthesize("Hello world")
    
    assert audio is not None
    assert len(audio) > 0
```

## Performance Testing

### Load Testing

```python
# tests/load_test.py
import pytest
from locust import HttpUser, task

class VoiceERAUser(HttpUser):
    """Simulated VoiceERA user"""
    
    @task(3)
    def list_agents(self):
        self.client.get("/agents")
    
    @task(1)
    def create_agent(self):
        self.client.post(
            "/agents",
            json={
                "name": "Load Test Bot",
                "llm_provider": "openai"
            }
        )
```

Run with:
```bash
locust -f tests/load_test.py
```

### Latency Testing

```python
# tests/test_latency.py
import pytest
import time

def test_agent_creation_latency():
    """Test agent creation latency"""
    start = time.time()
    
    for _ in range(100):
        create_agent({"name": "Bot"})
    
    duration = time.time() - start
    avg_latency = duration / 100
    
    # Should create agent in < 100ms
    assert avg_latency < 0.1
```

## Continuous Integration

### GitHub Actions

Tests run automatically on:

```yaml
# .github/workflows/tests.yml
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Backend tests
        run: |
          cd voicera_backend
          pip install -r requirements.txt
          pytest
      
      - name: Frontend tests
        run: |
          cd voicera_frontend
          npm install
          npm test
      
      - name: Upload coverage
        uses: codecov/codecov-action@v3
```

## Coverage Reports

### Generate Coverage

```bash
# Backend
cd voicera_backend
pytest --cov=app --cov-report=html
# Open htmlcov/index.html

# Frontend
cd voicera_frontend
npm test -- --coverage
# Open coverage/lcov-report/index.html
```

### Coverage Goals

- Minimum: 70% code coverage
- Target: 85% code coverage
- Critical paths: 95% coverage

## Common Testing Issues

### Issue: Async Tests Failing

**Solution**: Use `@pytest.mark.asyncio`

```python
@pytest.mark.asyncio
async def test_something():
    result = await async_function()
    assert result
```

### Issue: Database Tests Interfering

**Solution**: Use test fixtures to isolate

```python
@pytest.fixture(autouse=True)
def reset_db():
    """Reset database before each test"""
    clear_test_db()
    yield
    clear_test_db()
```

### Issue: Timeout on WebSocket Tests

**Solution**: Increase timeout

```python
@pytest.mark.asyncio(timeout=30)
async def test_websocket():
    # ...
```

## Best Practices

1. **Write tests alongside code** - Test-driven development
2. **Use descriptive names** - `test_create_agent_with_valid_data`
3. **One assertion per test** - Keep tests focused
4. **Use fixtures** - Reduce code duplication
5. **Mock external services** - Don't hit real APIs
6. **Test edge cases** - Negative cases are important
7. **Keep tests fast** - <100ms per test
8. **Use CI/CD** - Automate test running

## Next Steps

- **[Contributing Guide](contributing.md)** - How to contribute
- **[Local Setup](local-setup.md)** - Development environment
- **[Architecture](../architecture/overview.md)** - System design
