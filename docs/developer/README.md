---
description: Build on Voicera, extend it, or run it on your own infrastructure.
---

# Developer

How Voicera is put together, and how to change it. If you want to *use* the API rather than modify the system, the [API Reference](../api-reference/overview.md) tab is the shorter path.

## Where to start

| If you want to… | Start here |
| --- | --- |
| Run the stack from source | [Local setup](guides/local-setup.md) |
| Understand the moving parts | [Services overview](services/README.md) |
| Add a new STT, TTS, or LLM vendor | [Adding an AI provider](guides/adding-a-provider.md) |
| Add a new telephony carrier | [Adding a telephony provider](guides/adding-a-telephony-provider.md) |
| Run models on your own GPUs | [Model server](model-server/README.md) |
| Connect something to Voicera | [Connecting a client](clients/README.md) |
| Look up a variable or a port | [Configuration reference](reference/environment-variables.md) |

## Services

Nine containers, five Python packages. `apps/api` alone runs as three containers off one image.

* [**Overview**](services/README.md) — containers, ports, start-up order, who talks to whom
* [**API**](services/api.md) — routers, service layer, persistence, lifecycle
* [**Runtime**](services/runtime.md) — the answer webhook and the Pipecat pipeline
* [**Providers**](services/providers.md) — the registry that makes vendors pluggable
* [**Telephony**](services/telephony.md) — carrier clients, answer XML, frame serializers
* [**Workers and orchestrator**](services/workers.md) — ARQ jobs and campaign scheduling

## Model server

Optional. Run STT, TTS, and LLM on your own hardware behind one gateway on `:8100`.

* [**Model server**](model-server/README.md) — the section index
* [**Slots and models**](model-server/slots-and-models.md) — swap a model with one line
* [**Running on GPUs**](model-server/gpu-operations.md) — device selection, MPS, caching

## Contributing

* [**Local setup**](guides/local-setup.md) · [**Repository layout**](guides/repository-layout.md)
* [**Adding an AI provider**](guides/adding-a-provider.md) · [**Adding a telephony provider**](guides/adding-a-telephony-provider.md)
* [**Testing**](guides/testing.md) — the five suites and what each protects
* [**Contributing**](guides/contributing.md) — branches, commits, pull requests

{% hint style="info" %}
There is no CI. Run the test suites yourself before opening a pull request — [Testing](guides/testing.md) lists all five and what each needs.
{% endhint %}

## Configuration reference

* [**Environment variables**](reference/environment-variables.md) — the single root `.env`, annotated
* [**Ports and defaults**](reference/ports-and-defaults.md) — published versus internal
* [**Data model**](reference/data-model.md) — collections, fields, enums, indexes
* [**Agent configuration**](reference/agent-configuration.md) — the full `config` contract

## Dashboard (Beta)

{% hint style="warning" %}
The Next.js dashboard lives on the **`dev-frontend`** branch, is not merged into `dev`, and is not part of `docker-compose.yaml`. See [Dashboard (Beta)](frontend/README.md).
{% endhint %}
