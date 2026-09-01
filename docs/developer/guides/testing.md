---
description: Running the Voicera test suites.
---

# Testing

Voicera has five test suites, one per package. Each lives inside the code it covers — there is no top-level `tests/` directory and no aggregate runner. This page tells you how to run each, what it needs, and what breaks if you skip it.

{% hint style="warning" %}
There is **no CI**. The repository has no `.github/` directory, no workflows, and no pre-commit hooks. Nothing runs on a push. Every check on this page is one you run yourself, before opening a pull request.
{% endhint %}

## The five suites

| Suite | Modules | Covers | Needs Docker |
| --- | --- | --- | --- |
| `apps/api/tests` | 18 | Auth, agents, telephony provisioning, campaigns, knowledge base, call artifacts, secret encryption. | No |
| `apps/runtime/tests` | 5 | Route dispatch, prompt substitution, hold messages, call ending, knowledge injection. | No |
| `apps/telephony/tests` | 6 | Registry, clients, answer XML, serializers, webhooks, catalog schema. | No |
| `apps/providers/tests` | 1 | The provider registry and the catalog dump, across every registered vendor. | No |
| `model-server/tests` | 20 | Catalogue, gateway streaming, slot behaviour, audio parity, KV page allocation. | Only `test_model_switching` |

None of them needs a database, a GPU, or a network. One model-server module shells out to `docker compose config`.

## Running them

Every command below assumes you are at the repository root and the root is on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD"
```

`apps/providers`, `apps/telephony`, and `apps/runtime` import each other as `apps.*`, so this is not optional. See [Local setup](local-setup.md#the-pythonpath-and-the-apps-namespace).

### Providers and telephony

These two need only `pytest` on top of the API requirements:

```bash
pip install -r apps/api/requirements.txt pytest
pytest apps/providers/tests apps/telephony/tests -v
```

Both packages are pure Pydantic and `httpx`. Neither imports Pipecat at module level, which is exactly what makes this possible.

### API

```bash
pip install -r apps/api/requirements.txt pytest
pytest apps/api/tests -v
```

`apps/api/tests/conftest.py` puts both `apps/api` (for `app.*`) and the repository root (for `apps.*`) on `sys.path`, so this works from any directory. Routes are exercised through FastAPI's `TestClient` against in-memory stores and `unittest.mock.patch`, not a live FerretDB.

One module is opt-in. `test_agent_telephony_integration.py` places real calls against real provider credentials and is skipped unless you ask for it:

```bash
RUN_TELEPHONY_INTEGRATION=1 \
TEST_ORG_ID=<your-org-id> \
VOICE_SERVER_BASE_URL=https://voice.example.com \
INTERNAL_API_KEY=<key> \
pytest apps/api/tests/test_agent_telephony_integration.py -q
```

That needs the API stack up, FerretDB reachable, and provider credentials stored. It costs money at the telephony vendor. Leave it off unless you are changing provisioning.

### Runtime

```bash
python -m venv .venv-runtime
source .venv-runtime/bin/activate
pip install -r apps/runtime/requirements.txt pytest
export PYTHONPATH="$PWD"
pytest apps/runtime/tests -v
```

A separate virtualenv is worth it — `apps/runtime/requirements.txt` pulls in `pipecat-ai[deepgram,cartesia,openai,silero,websocket]==1.8.1`, which the other suites neither need nor want.

### Model server

```bash
cd model-server
pip install -r tests/requirements-dev.txt
pytest tests/ -v
ruff check .
```

`tests/requirements-dev.txt` is deliberately small: `pytest`, `pytest-asyncio`, `httpx`, `numpy`, `fastapi`, `uvicorn`, `ruff`, `websockets`. No torch, no NeMo, no CUDA. `tests/pytest.ini` sets `asyncio_mode = auto`, so async tests need no decorator.

## What needs Docker

Exactly one module: `model-server/tests/test_model_switching.py`. It shells out to `docker compose config`, which interpolates the compose files without contacting a daemon, and it **skips** if the `docker` CLI is absent. So you can run the whole model-server suite on a machine with no Docker at all — you just lose that one module's coverage.

No other suite touches Docker. The database-only Compose stack described in [Local setup](local-setup.md#database-only-compose) is for running the services by hand, not for testing.

## Why the model-server suite needs no GPU

The GPU stack is stubbed. `model-server/tests/stubs/` holds stand-ins for the three heavy imports:

```text
model-server/tests/stubs/
├── torch.py
├── nemo/collections/asr/models.py
└── inference/runner.py
```

`tests/conftest.py` puts that directory first on `sys.path`, ahead of any real package:

```python
sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
sys.path.insert(0, str(ROOT / "gateway"))
```

Everything *else* is the real code — routing, batching, protocol handling, transport. As the suite's README puts it: the NeMo, torch, and Parler-runner layers are stubbed, so everything else is real. That is what makes it worth running on a laptop: it cannot tell you the model transcribes correctly, but it can tell you the gateway streams instead of buffering, that a client disconnect evicts the upstream request, and that the KV page allocator never hands one page to two calls.

Two scripts in that directory are **not** part of the suite because they need real models on a GPU:

| Script | Use |
| --- | --- |
| `smoke_gpu.py` | End-to-end round trip on the box: TTS speaks, STT transcribes it back. |
| `bench_tts.py` | Latency and real-time factor, sequential or at a chosen concurrency. |

## Fixtures and stubs

There are only two `conftest.py` files under `apps/`, and both do one narrow job.

`apps/api/tests/conftest.py` fixes imports:

```python
API_ROOT = Path(__file__).resolve().parents[1]
VOICERA_ROOT = Path(__file__).resolve().parents[3]

for path in (str(API_ROOT), str(VOICERA_ROOT)):
    if path not in sys.path:
        sys.path.insert(0, path)
```

`apps/runtime/tests/conftest.py` stubs the Pipecat runners so route tests do not build a pipeline:

```python
_mock_runners = MagicMock()
_mock_runners.run_telephony_bot = AsyncMock()
_mock_runners.run_websocket_bot = AsyncMock()
sys.modules["apps.runtime.services.pipecat.runners"] = _mock_runners

from apps.runtime.app import app  # noqa: E402
```

The `sys.modules` assignment has to happen before `apps.runtime.app` is imported, which is why the import sits below it with a `noqa`. It also provides the shared `client` fixture wrapping `TestClient(app)`.

`apps/providers/tests` and `apps/telephony/tests` have no `conftest.py` at all. They rely on `PYTHONPATH` and on `load_providers()` doing its own discovery.

`model-server/tests/conftest.py` does more: the stub path insertion above, a `free_port()` helper, a `serve()` helper that runs an ASGI app on a background thread and waits for it to accept, and a `find_setup()` helper that locates `setup.sh` in either of the two places it has lived. Its comment on that last one is worth reading — hardcoding either path "turns a move into six silent skips: the suite stays green while the checks are simply not running."

## What each suite protects

### apps/providers

One module, `test_provider_schemas.py`, testing the registry as a whole rather than each vendor. It pins the union member counts (11 STT, 14 TTS, 9 LLM), asserts every registered config has a creator and the reverse, and walks every provider asserting the catalog dump contains no `$defs`, `$ref`, or `anyOf`, that every secret field is marked and carries no `input_mode`, and that every language id emitted exists in the canonical `LANGUAGES` map. See [Adding an AI provider](adding-a-provider.md#testing).

### apps/telephony

`test_registry.py` asserts the registered set is exactly `{"vobiz", "plivo"}` and that the lazy serializer load works. `test_xml.py` pins the answer XML per sample rate — the string most likely to be silently wrong, because malformed XML produces a call that connects and then goes quiet.

### apps/api

The broadest suite. `test_secret_crypto.py` covers Fernet encryption of `ProviderAuth`. `test_agent_telephony_service.py` and `test_agent_telephony.py` cover application provisioning and teardown on agent create, update, and delete. The campaign modules cover the dispatcher, the repository, CSV sync, and the status processor.

### apps/runtime

`test_routing.py` covers `/answer` and the `/agent` WebSocket handshake. `test_hold.py`, `test_call_ending.py`, and `test_prompt_substitution.py` cover pipeline behaviour that only shows up mid-call.

### model-server

`model-server/tests/README.md` has a full table. The shape of it: the early entries guard the audio itself (`test_stt_audio_parity`, `test_tts_request_parity`, `test_pcm_chunk_boundaries`), the middle ones guard the wiring between files (`test_catalogue`, `test_model_switching`, `test_setup_selection`), and the last ones guard the seam between the model server and the voice pipeline.

## Four model-server modules silently skip

{% hint style="danger" %}
`test_llm_wiring.py`, `test_client_selection.py`, `test_tts_format_negotiation.py`, and `test_partial_transcripts.py` all locate the voice pipeline at `ROOT.parent / "voice_2_voice_server"` — a directory that **no longer exists**. Every test in those four modules therefore skips, and a skip does not fail a run.
{% endhint %}

The paths are hardcoded at the top of each module:

```python
ROOT = Path(__file__).resolve().parent.parent          # → model-server/
V2V = ROOT.parent / "voice_2_voice_server"             # → voicera/voice_2_voice_server

pytestmark = pytest.mark.skipif(
    not V2V.is_dir(), reason="voice_2_voice_server not present in this checkout"
)
```

`voice_2_voice_server` was renamed to `apps/runtime` in the revamp. The directory those tests point at was never recreated, so `V2V.is_dir()` is `False` and 29 tests across the four modules never run.

The suite is honest about it. `conftest.py` installs a `pytest_terminal_summary` hook that prints a red **NOT VERIFIED** block after the summary line, naming what is unverified while that is true:

* every model marked `ready` can actually be named by an agent config
* the client decodes the audio format each TTS model declares
* partial transcripts still reach the caller mid-utterance

The comment above the hook explains why it exists: "A skip is invisible in a green summary line. That is the failure mode this hook exists for: the suite says '165 passed' while a quarter of what it claims to cover is not running."

These four modules should be repointed at `apps/runtime`. The specific paths they look for are `voice_2_voice_server/api/services.py`, `voice_2_voice_server/services/ai4bharat/stt.py`, and `voice_2_voice_server/services/ai4bharat/tts.py`, none of which map one-to-one onto the current runtime layout — the client selection logic now lives in `apps/providers` and the pipeline in `apps/runtime/services/pipecat/`. Repointing them is a real piece of work, not a path substitution.

{% hint style="warning" %}
Until that is done, treat the model-server suite's pass count as covering the server side only. The seam between the model server and the voice pipeline is unverified, and that seam is where a mistake stays invisible until a live call drops.
{% endhint %}

## Related

* [Local setup](local-setup.md)
* [Repository layout](repository-layout.md)
* [Contributing](contributing.md)
* [Adding an AI provider](adding-a-provider.md)
* [Adding a telephony provider](adding-a-telephony-provider.md)
