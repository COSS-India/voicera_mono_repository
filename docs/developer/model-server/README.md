---
description: Run STT, TTS, and LLM on your own GPUs behind one gateway.
---

# Model server

An optional, self-contained subsystem for running speech and language models on your own hardware. One gateway on `:8100` fronts three slots — STT, TTS, LLM — and each slot runs whichever model you selected, without the gateway knowing which one.

{% hint style="info" %}
The model server is **optional**. Voicera runs fine against cloud providers alone. Deploy this when data residency, per-minute cost, or Indic-language coverage makes self-hosting worth the GPUs.
{% endhint %}

## Start here

| Page | What it answers |
| --- | --- |
| [Overview](overview.md) | What it is, what it replaces, and its current state. |
| [Slots and models](slots-and-models.md) | The slot/model split, and switching models with one line. |
| [Gateway API](gateway-api.md) | Every endpoint, including `WS /v1/asr/ws`. |

## The models

| Page | Covers |
| --- | --- |
| [STT models](stt-models.md) | `indic-conformer`, `indic-transcribe`, and partial transcripts versus streaming. |
| [TTS models](tts-models.md) | `indic-parler`, `orpheus`, `indic-mio`, and audio format negotiation. |
| [LLM models](llm-models.md) | `qwen3.5-4b` and the model-id agreement across files. |

## Operating it

| Page | Covers |
| --- | --- |
| [Adding a model](adding-a-model.md) | The container contract, and the two steps to add one. |
| [Running on GPUs](gpu-operations.md) | Device selection, MPS, shared HuggingFace cache, disk. |

## Honest status

{% hint style="warning" %}
Two gaps are documented rather than hidden. The LLM slot has not yet been run on real hardware, and a full call routed from the voice runtime through a self-hosted model has not been verified end to end. Both are called out on the pages concerned.
{% endhint %}

## Related

* [Self-hosted models](../../guides/deployment/self-hosted-models.md) — wiring the model server to the runtime
* [Provider registry](../../guides/concepts/provider-registry.md) — how agents select a provider
