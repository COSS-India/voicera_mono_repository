---
description: A map of the Voicera monorepo.
---

# Repository layout

Where everything lives, what owns what, and which files are placeholders you should not read anything into.

{% hint style="info" %}
For the runtime relationship between these directories — which process talks to which — read [Architecture](../../guides/concepts/architecture.md) first. This page is about the tree on disk.
{% endhint %}

## Top level

```text
voicera/
├── apps/                  Application code — four Python packages
├── model-server/          Optional self-hosted STT, TTS, and LLM stack
├── scripts/               start_docker.sh, stop_services.sh
├── docs/                  This documentation
├── docker-compose.yaml    The whole local stack
├── .env.example           The single environment template
├── .gitbook.yaml          GitBook root and redirects
├── README.md              Short pointer to the Docker quick start
├── CHANGELOG.md           Keep a Changelog history
├── CONTRIBUTING.md        Contributor entry point
├── SECURITY.md            Vulnerability reporting policy
├── CODE_OF_CONDUCT.md     Placeholder — empty
├── LICENSE                Placeholder — empty
├── Makefile               Placeholder — empty
├── pyproject.toml         Placeholder — empty
└── __init__.py            Empty; makes the checkout importable as a package
```

| Path | Role |
| --- | --- |
| `docker-compose.yaml` | Nine services: `postgres`, `ferretdb`, `api`, `arq-worker`, `campaign-orchestrator`, `runtime`, `redis`, `minio`, `minio-init`. Container and volume names are prefixed `voicera_oss_`; the network is `app-network`. |
| `.env.example` | The one environment template. Copy it to `.env` at the root. See [Environment variables](../reference/environment-variables.md). |
| `scripts/start_docker.sh` | Ensures a root `.env`, generating `SECRET_KEY`, `INTERNAL_API_KEY`, and `PROVIDER_AUTH_ENCRYPTION_KEY` if missing, then starts the stack. Prefer it over a bare `docker compose up`. |
| `scripts/stop_services.sh` | Stops the stack. |

## apps

Four packages under one namespace. `apps/__init__.py` is empty and exists so `apps.providers` and `apps.telephony` are importable from the repository root.

```text
apps/
├── __init__.py
├── api/          FastAPI REST surface (also the ARQ worker and orchestrator)
├── runtime/      Answer webhook and the Pipecat audio pipeline
├── providers/    STT, TTS, and LLM vendor registry
├── telephony/    Vobiz and Plivo clients, answer XML, frame serializers
└── schemes/      Empty — see below
```

### apps/api

```text
apps/api/
├── Dockerfile          Builds api, arq-worker, and campaign-orchestrator
├── README.md           Route tables and auth model
├── requirements.txt
├── app/
│   ├── main.py         FastAPI application
│   ├── config.py       Pydantic BaseSettings; pins the root .env
│   ├── auth.py         JWT and X-API-Key dependencies
│   ├── database.py     FerretDB client
│   ├── database_init.py
│   ├── routers/        Thin HTTP layer under /api/v1
│   ├── services/       Business rules — agents, campaigns, concurrency, RAG
│   ├── models/         Pydantic documents and request/response schemas
│   ├── constants/      Campaign and call constants
│   ├── rag/            Chroma ingest and retrieval
│   ├── storage/        MinIO helpers
│   ├── tasks/          ARQ job definitions, including WorkerSettings
│   └── utils/
└── tests/              18 test modules plus conftest.py
```

One package, **three containers**. `api`, `arq-worker`, and `campaign-orchestrator` all build from `apps/api/Dockerfile` and differ only in their `command:`. See [Workers and orchestrator](../services/workers.md).

### apps/runtime

```text
apps/runtime/
├── Dockerfile          Sets PYTHONPATH=/app
├── requirements.txt    FastAPI, Pipecat 1.8.1, MinIO
├── app.py              FastAPI app plus a main() entry point
├── constants.py        os.getenv at call time, not at import
├── routes/             health, telephony (/answer), agent (WS)
├── services/
│   ├── agent_routing.py
│   ├── backend.py            Fetches agent config and credentials from the API
│   ├── ai_service_factory.py Builds STT, TTS, and LLM from apps.providers
│   ├── pipecat/              The pipeline, split into ten modules
│   ├── knowledge/            RAG context injection
│   └── storage/              Call artifacts to MinIO
└── tests/              5 test modules plus conftest.py
```

### apps/providers

```text
apps/providers/
├── base.py         Kind and ProviderType enums, Auth/Settings/Config bases
├── registry.py     @register_stt / @register_tts / @register_llm, load_providers()
├── factory.py      Discriminated unions from the registry, create_*_service dispatch
├── schema.py       provider_schemas() and configuration_defaults() catalog dump
├── languages.py    Canonical language ids and language_schema_extra()
├── readme.md
├── cloud/          21 vendor folders → provider_type=cloud
├── adapters/       bhashini → provider_type=adapter
├── local/          Reserved for self-hosted providers; currently only __init__.py
└── tests/          1 test module
```

Every vendor folder is four files: `catalog.py`, `config.py`, `languages.py`, `service.py`. See [Adding an AI provider](adding-a-provider.md).

### apps/telephony

```text
apps/telephony/
├── base.py             Kind, Credentials, ApiResult, shared httpx helpers, config bases
├── registry.py         @register_client / @register_answer_xml / @register_frame_serializer
├── xml.py              build_answer_stream_xml dispatcher
├── serializers.py      create_frame_serializer dispatcher (needs pipecat)
├── calls.py            initiate_outbound dispatcher
├── webhooks.py         Provider webhook parsing
├── schema.py           provider_schemas / configuration_telephony
├── readme.md
├── scripts/print_schemas.py
├── providers/
│   ├── vobiz/          9 files
│   └── plivo/          9 files
└── tests/              6 test modules
```

The package-root `xml.py`, `calls.py`, and `serializers.py` are dispatch facades. They contain no provider `if`/`elif` chains and must not grow any. See [Adding a telephony provider](adding-a-telephony-provider.md).

{% hint style="info" %}
`apps/telephony/readme.md` documents the package's public API and is current.
{% endhint %}

## model-server

A separate Docker Compose stack with its own `.env`, run only if you want to host models yourself.

```text
model-server/
├── README.md
├── setup.sh, stop.sh, compose-files.sh
├── compose.model-server.yml       The stack
├── compose.mps.yml                NVIDIA MPS overlay
├── compose.shared-hf-cache.yml    Shared HuggingFace cache overlay
├── models.yaml                    The catalogue
├── ruff.toml                      The only lint config in the repository
├── gateway/                       One published port; routes to the slots
├── stt/                           indic-conformer, indic-transcribe
├── tts/                           indic-mio, indic-parler, orpheus
├── llm/                           qwen3.5-4b
├── tests/                         20 test modules plus stubs
└── hindi.wav                      Fixture for the GPU smoke script
```

Three **slots** — STT, TTS, LLM — each filled by naming a folder in `STT_MODEL`, `TTS_MODEL`, or `LLM_MODEL`. See [Model server overview](../model-server/overview.md).

## scripts

Two shell scripts, both meant to be run from the repository root:

| Script | What it does |
| --- | --- |
| `start_docker.sh` | Creates or updates the root `.env`, generating the three secrets if they are missing, then brings the stack up detached. |
| `stop_services.sh` | Brings the stack down. |

## Where tests live

Tests sit inside the package they cover. There is no top-level `tests/` directory.

| Suite | Modules | Covers |
| --- | --- | --- |
| `apps/api/tests` | 18 | Auth, agents, telephony provisioning, campaigns, knowledge base, call artifacts. |
| `apps/runtime/tests` | 5 | Routing, prompt substitution, hold, call ending, knowledge. |
| `apps/telephony/tests` | 6 | Registry, clients, XML, serializers, webhooks, schema. |
| `apps/providers/tests` | 1 | The registry and the catalog dump. |
| `model-server/tests` | 20 | Catalogue, gateway streaming, slot behaviour, audio parity. |

Full instructions in [Testing](testing.md).

## Files that are intentionally empty

Several files exist so tooling and GitHub find them, but hold no content yet. Do not infer a workflow from their presence.

| File | State | What this means |
| --- | --- | --- |
| `Makefile` | Empty | There are **no** `make` targets. Never document one. Use the explicit `pip` and `uvicorn` commands in [Local setup](local-setup.md). |
| `pyproject.toml` | Empty | The project is **not** pip-installable. There is no `pip install -e .`, no build backend, and no tool configuration. Dependencies come from `apps/<app>/requirements.txt`. |
| `CODE_OF_CONDUCT.md` | Empty | A placeholder. No code of conduct has been adopted yet. |
| `LICENSE` | Empty | A placeholder. The licence has not been declared in the repository. |
| `apps/schemes/__init__.py` | Empty | `apps/schemes/` contains nothing but this empty file. It is not imported anywhere. |
| `__init__.py` (root) | Empty | Makes the checkout itself importable as a package, which lets pytest import the tree as `voicera.apps.providers` when the parent directory is on `sys.path`. `apps/providers/registry.py` resolves vendor roots from `__package__` rather than a hardcoded `apps.providers` prefix for exactly this reason. |

There is also **no `.github/` directory**, and therefore no CI, no workflows, no issue templates, and no pull-request template. Nothing runs automatically on a push. Every check is something you run locally before opening a pull request.

{% hint style="warning" %}
Because there is no CI, a green local run is the only signal a change has. Run the relevant [test suites](testing.md) and `ruff check .` in `model-server/` yourself.
{% endhint %}

## Related

* [Local setup](local-setup.md)
* [Testing](testing.md)
* [Architecture](../../guides/concepts/architecture.md)
* [Services overview](../services/README.md)
