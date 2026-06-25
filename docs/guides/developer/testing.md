---
description: Run and write tests for the VoicEra backend, frontend, and voice server.
---

# Testing

How to run and write tests across the three VoicEra services. Audience: engineers writing new code or fixing bugs.

## What is tested where

| Service | Framework | Common test types |
|---------|-----------|--------------------|
| Backend (`voicera_backend`) | `pytest` | Unit, service, API |
| Frontend (`voicera_frontend`) | Jest + React Testing Library | Component, hook |
| Voice server (`voice_2_voice_server`) | `pytest` + `pytest-asyncio` | Audio, STT/TTS, pipeline |

CI runs the same commands on every PR. See [contributing](contributing.md) for the workflow.

## Backend

### Install test dependencies

```bash
cd voicera_backend
pip install pytest pytest-asyncio pytest-cov pytest-mock
```

### Run

```bash
pytest                          # all tests
pytest tests/test_auth.py       # one file
pytest tests/test_auth.py::test_login   # one function
pytest -v                       # verbose
pytest -x                       # stop on first failure
pytest --lf                     # rerun last failures
pytest -k "auth"                # match by name
pytest -n auto                  # parallel
pytest --cov=app                # coverage
```

### Layout

```
voicera_backend/tests/
├── conftest.py                 # shared fixtures
├── test_auth.py
├── test_agents.py
├── test_api/
│   ├── test_auth_routes.py
│   └── test_agent_routes.py
├── test_services/
│   └── test_agent_service.py
└── test_models/
    └── test_schemas.py
```

### Unit test

```python
# tests/test_agents.py
import pytest
from app.models.schemas import AgentCreate
from app.services.agent_service import AgentService

@pytest.fixture
def agent_service():
    return AgentService()

def test_create_agent(agent_service):
    data = AgentCreate(name="Support Bot", llm_provider="openai")
    agent = agent_service.create_agent(data)
    assert agent.name == "Support Bot"
```

### Async test

```python
import pytest
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_agent_service_creates_agent():
    with patch("app.services.agent_service.get_db") as mock_db:
        mock_db.return_value.agents.insert_one = AsyncMock(return_value=True)
        from app.services.agent_service import AgentService
        service = AgentService()
        result = await service.create({"name": "Test", "llm_provider": "openai"})
        assert result is not None
```

### Mock an external service

```python
from unittest.mock import patch
import pytest

@pytest.mark.asyncio
async def test_stt_returns_empty_on_error():
    with patch("services.deepgram.transcribe") as mock_stt:
        mock_stt.side_effect = Exception("API Error")
        result = await transcribe_audio(audio_data)
        assert result is None or result.text == ""
```

### Shared fixtures

```python
# tests/conftest.py
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.fixture
def client():
    return TestClient(app)

@pytest.fixture
def auth_headers(client):
    token = create_test_token(email="test@example.com")
    return {"Authorization": f"Bearer {token}"}
```

### Test an HTTP endpoint

```python
def test_list_agents(client, auth_headers):
    response = client.get("/agents", headers=auth_headers)
    assert response.status_code == 200
    assert "agents" in response.json()

def test_unauthorized(client):
    assert client.get("/agents").status_code == 401
```

## Frontend

### Install

```bash
cd voicera_frontend
npm install --save-dev jest @testing-library/react @testing-library/jest-dom
```

### Run

```bash
npm test                        # run once
npm test -- --watch             # watch mode
npm test -- --coverage          # coverage
npm test -- auth.test.tsx       # one file
npm test -- -u                  # update snapshots
```

### Component test

```tsx
// __tests__/components/AgentCard.test.tsx
import { render, screen, fireEvent } from '@testing-library/react';
import { AgentCard } from '@/components/AgentCard';

describe('AgentCard', () => {
  it('renders agent information', () => {
    const agent = { id: '123', name: 'Support Bot', status: 'active' };
    render(<AgentCard agent={agent} />);
    expect(screen.getByText('Support Bot')).toBeInTheDocument();
  });

  it('handles delete action', () => {
    const onDelete = jest.fn();
    const agent = { id: '123', name: 'Bot', status: 'active' };
    render(<AgentCard agent={agent} onDelete={onDelete} />);
    fireEvent.click(screen.getByRole('button', { name: /delete/i }));
    expect(onDelete).toHaveBeenCalledWith('123');
  });
});
```

### Hook test

```tsx
// __tests__/hooks/useAuth.test.ts
import { renderHook, act } from '@testing-library/react';
import { useAuth } from '@/hooks/useAuth';

describe('useAuth', () => {
  it('returns user after login', async () => {
    const { result } = renderHook(() => useAuth());
    await act(async () => {
      await result.current.login('user@example.com', 'password');
    });
    expect(result.current.user?.email).toBe('user@example.com');
  });
});
```

## Voice server

### Audio frame processing

```python
import pytest
import numpy as np
from api.bot import VoiceBot

@pytest.mark.asyncio
async def test_audio_frame_processing():
    bot = VoiceBot()
    audio = np.random.randint(-32768, 32767, 16000, dtype=np.int16)
    result = await bot.process_audio(audio)
    assert hasattr(result, 'transcript')
```

### Provider integration

```python
@pytest.mark.asyncio
async def test_deepgram_stt():
    from services.deepgram.stt import DeepgramSTT
    stt = DeepgramSTT(api_key="test-key")
    with patch.object(stt, "_call_api") as mock:
        mock.return_value = {"transcript": "hello world"}
        assert await stt.transcribe(audio_data) == "hello world"
```

## Coverage

```bash
# Backend
cd voicera_backend
pytest --cov=app --cov-report=html
# open htmlcov/index.html

# Frontend
cd voicera_frontend
npm test -- --coverage
# open coverage/lcov-report/index.html
```

Coverage targets:

- Minimum: 70%
- Target: 85%
- Critical paths: 95%

## Continuous integration

GitHub Actions runs tests on push to `main` and on every PR. The workflow lives at `.github/workflows/tests.yml`:

```yaml
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
```

## Common pitfalls

{% hint style="warning" %}
**Async tests silently pass.** Decorate every async test with `@pytest.mark.asyncio` or pytest will skip the body.
{% endhint %}

{% hint style="warning" %}
**Database state leaks between tests.** Add an `autouse` fixture that resets the test database before each test:

```python
@pytest.fixture(autouse=True)
def reset_db():
    clear_test_db()
    yield
    clear_test_db()
```
{% endhint %}

{% hint style="warning" %}
**WebSocket tests time out.** Increase the per-test timeout for long-running socket scenarios.
{% endhint %}

## Best practices

1. Write tests alongside the code change.
2. Use descriptive names — `test_create_agent_with_valid_data`.
3. Keep tests focused — one behaviour per test.
4. Mock external services. Do not call real APIs from unit tests.
5. Cover negative paths, not just the happy path.
6. Keep each test under ~100 ms to keep the suite fast.

## Next steps

- [Contributing](contributing.md)
- [Local setup](local-setup.md)
- [Architecture](../../concepts/architecture.md)
