---
description: Every self-hosted model in one place, behind one gateway.
---

# Overview

`model-server` is the self-hosted half of Voicera's AI providers. It runs speech-to-text, text-to-speech and a language model on your own GPUs, and puts all three behind a single OpenAI-compatible gateway on port `8100`. This page is the orientation: what the layout is, how to start it, and what state each piece is actually in.

{% hint style="info" %}
If you only want to point an agent at self-hosted models, read [Self-hosted models](../../guides/deployment/self-hosted-models.md) first. This section is about the model server itself.
{% endhint %}

## Why one gateway

Three **slots**, each holding as many **models** as you have folders for. A slot is one container on a fixed port; which model fills it is a folder name in `.env`.

Only the gateway publishes a port. The model containers listen inside their own network namespace and are reachable only by Compose service name, so this stack sits beside other stacks on the same host without competing for ports. Everything is routed on modality, and the gateway holds no model-specific knowledge — each model server speaks the OpenAI shape natively, so adding a model never touches gateway code.

```mermaid
flowchart LR
  RT["Runtime<br/>apps/runtime"]
  GW["Gateway<br/>:8100 published"]
  STT["stt slot<br/>:8001 internal"]
  TTS["tts slot<br/>:8002 internal"]
  LLM["llm slot<br/>:8003 internal"]

  RT --> GW
  GW --> STT
  GW --> TTS
  GW --> LLM
```

The gateway container listens on `8000` inside its network namespace; `GATEWAY_PORT` (default `8100`) is what it is published as on the host. Defined in `model-server/compose.model-server.yml`.

## Layout

```
model-server/
├── gateway/               the only published port    :8100
├── stt/                   speech-to-text slot        :8001 (internal)
│   ├── indic-conformer/
│   └── indic-transcribe/
├── tts/                   text-to-speech slot        :8002 (internal)
│   ├── indic-parler/
│   ├── indic-mio/
│   └── orpheus/
├── llm/                   language-model slot        :8003 (internal)
│   └── qwen3.5-4b/
├── models.yaml            catalogue of every model, served at /models
└── tests/                 run without a GPU
```

Adding a model is adding a folder — no service, no port, no gateway change, which is what makes swapping one a one-line edit rather than a project. See [Adding a model](adding-a-model.md).

## Running it

```bash
cd model-server
STT_MODEL=indic-conformer ./setup.sh
```

`setup.sh` asks which model should fill each slot, fetches weights, builds the images and starts the stack. When it finishes it prints the demo URL:

```text
  Demo:     http://localhost:8100/demo
  Gateway:  http://localhost:8100
  Health:   curl http://localhost:8100/health
  Models:   curl http://localhost:8100/models
```

Stop it with `./stop.sh`. Or drive Compose directly:

```bash
cp .env.example .env
docker compose -f compose.model-server.yml up -d --build
```

Driving Compose by hand with only the base file skips the overlays a model or host may need — a `compose.extra.yml` a model brings, the shared HuggingFace cache, and the MPS attachment. `compose-files.sh` produces the correct `-f` list:

```bash
docker compose $(sh model-server/compose-files.sh) --project-directory model-server up -d
```

First build takes 20-40 minutes. See [Running on GPUs](gpu-operations.md) for GPU selection and disk requirements.

## What it replaces

The earlier layout had three separate services — one for STT, one for TTS, one for the LLM — each with its own port, its own repository conventions and its own way of being started. `model-server` is one gateway with three slots instead. The consequences:

| Then | Now |
| --- | --- |
| Three services to start, configure and monitor | One `setup.sh`, one `/health`, one published port |
| Each service exposed its own host port | Only `GATEWAY_PORT` binds on the host |
| Adding a model meant a new service | Adding a model means a folder and a catalogue entry |
| Each service had its own environment file | One `model-server/.env` |
| Model choice was baked into which service ran | Model choice is `STT_MODEL` / `TTS_MODEL` / `LLM_MODEL` |

## Current state

Verified on the `ace-h200` box on 26 August, running beside the production and translate stacks:

| | |
| --- | --- |
| Both models on GPU 1 | via the shared MPS daemon, ~12.3 GB |
| TTS time-to-first-audio | 1.5 s cold, ~250 ms warm |
| Realtime factor | 0.69x |
| Round trip | TTS speaks a sentence, STT transcribes it back word for word |
| Effect on production | none — `voicera-prod` stayed `running(11)` throughout |

{% hint style="warning" %}
Two things are **not** verified on hardware, and `model-server/README.md` is explicit about both.

**A real call through the runtime has not been tested.** That needs a second runtime pointed at the gateway via `MODEL_SERVER_URL`, plus an agent configured for `indic-conformer-stt` and `indic-parler-tts`.

**The LLM slot has not been run on hardware at all.** `llm/qwen3.5-4b/` is written but has never been built or started, so the vLLM flags in it are unverified against a live model. The numbers above cover STT and TTS only.
{% endhint %}

Per-model status is recorded in `model-server/models.yaml` and summarised on [STT models](stt-models.md), [TTS models](tts-models.md) and [LLM models](llm-models.md).

## Relationship to apps/providers

The model server is one provider among the others in the [provider registry](../../guides/concepts/provider-registry.md). An agent names a self-hosted model the same way it names a cloud one; the registry routes to the gateway instead of to a vendor API.

The naming convention between the two is `<catalogue id>-<slot>`: the folder `orpheus` under `tts/` is `orpheus-tts` in an agent config. That convention is a convention, not a lookup — the catalogue does not record the client-facing name — so `tests/test_client_selection.py` pins it in both directions: every model marked `ready` must be nameable by an agent config, and every name the client accepts must have a model behind it.

That test exists because the failure has no symptom until a call drops. A model can be catalogued, built, healthy and listed at `/models` while the runtime has never heard of its name. Deploying it looks like success right up to the first agent that asks for it.

## Related

* [Slots and models](slots-and-models.md)
* [Gateway API](gateway-api.md)
* [Self-hosted models](../../guides/deployment/self-hosted-models.md)
* [Ports and defaults](../reference/ports-and-defaults.md)
