---
description: The LLM slot and the model that fills it.
---

# LLM models

The LLM slot is the third of the model server's three slots, on internal port 8003, answering `POST /v1/chat/completions`. It is the only slot that ships turned off — `LLM_MODEL` is empty in `.env.example` — and it is the one part of the model server that has never been run on hardware.

{% hint style="warning" %}
**The LLM slot has not been run on hardware at all.** `model-server/README.md` states it plainly: `llm/qwen3.5-4b/` is written but has never been built or started, so the vLLM flags in it are unverified against a live model. `models.yaml` repeats it against the catalogue entry. Everything on this page describes intended behaviour that no live model has confirmed.
{% endhint %}

## qwen3.5-4b

| | |
| --- | --- |
| Catalogue id | `qwen3.5-4b` |
| Name | Qwen3.5-4B |
| Status | `ready` — folder exists; **not run on hardware** |
| Runtime | vLLM `v0.27.1` |
| HuggingFace model | `Qwen/Qwen3.5-4B` |
| Quantization | none — bf16, ~8 GB of weights |
| Context | 8192, capped for telephony; the model natively does 262,144 |
| Thinking | on by default in the model; the runtime sends `enable_thinking=false` per request |
| Languages | 201 languages and dialects |

Two other LLM ids are catalogued as `planned`: `gemma` and `llama`, both vLLM.

Every flag in the Dockerfile is there for a reason:

| Flag | Why |
| --- | --- |
| `--served-model-name qwen3.5-4b` | vLLM rejects any request whose `model` field is not the name it was started with, so this is pinned to the catalogue id. Without it, clients would have to send `Qwen/Qwen3.5-4B`. |
| `--language-model-only` | The checkpoint registers as `Qwen3_5ForConditionalGeneration` and ships a 24-layer vision encoder. This stack is voice only, so skipping it hands that memory back to the KV cache. |
| `--max-model-len 8192` | Native context is 262,144. Telephony turns are short, and reserving KV cache for context nobody uses is the fastest way to run out of GPU. |
| `--reasoning-parser qwen3` | Thinking is on by default in Qwen3.5. The parser splits `<think>` blocks into `reasoning_content` so they never reach TTS as spoken text. |
| `--enable-chunked-prefill` | Interleaves prefill with decode, which is what holds time-to-first-token down when several calls are in flight. |
| no `--quantization` | At 4B the weights are ~8 GB in bf16. FP8 saves about 4 GB this box can spare, and small-model quantisation degrades hardest in exactly the Indic languages this serves. Set `VLLM_QUANTIZATION=fp8` in `.env` if memory ever becomes the constraint. |

Runtime limits come from `.env` and apply to whichever model fills the slot: `VLLM_MAX_MODEL_LEN` (8192), `VLLM_MAX_NUM_SEQS` (20), `VLLM_GPU_MEMORY_UTILIZATION` (0.10), `VLLM_QUANTIZATION` (empty means bf16). Which model and which model-specific flags live in `llm/<model>/Dockerfile`, not in `.env`.

`GPU_MEMORY_UTILIZATION` defaults to `0.10`, and note what that means: it is a fraction of the card's **total** memory, not of what is free. On a 143 GB H200 that is a hard ~14 GB reservation taken at startup. MPS does not partition memory, so oversizing it takes memory away from the production workers on the same GPU rather than failing cleanly. Never use vLLM's own 0.9 default here.

There is no `fetch.sh`: vLLM downloads its own weights from HuggingFace into the `hf_cache` volume on first start, and that volume survives restarts. First start therefore takes several minutes with no output on `/health` — watch `docker compose logs -f llm` rather than assuming it has hung.

### Two vLLM bugs that affect this

Both produce a call with dead air rather than an error, so they are worth knowing before debugging a silent bot.

* [vllm#35574](https://github.com/vllm-project/vllm/issues/35574) — `chat_template_kwargs: {enable_thinking: false}` did not always disable thinking on Qwen3.5. Closed February 2026, so fixed well before the `v0.27.1` pinned here, but it is why the runtime also appends `/no_think` to the system prompt as a second signal.
* [vllm#38894](https://github.com/vllm-project/vllm/issues/38894) — with the `qwen3` reasoning parser, generated text can arrive in `delta.reasoning` while `delta.content` stays empty. Pipecat only forwards `content` to TTS, so the caller hears nothing. The runtime's `VllmQwenVoiceLLMService._normalize_qwen_chunk` copies one into the other, but only while thinking is disabled — with thinking on, that mapping would speak the chain of thought aloud.

`tests/test_llm_wiring.py` pins both behaviours.

## Why the folder is small

`llm/qwen3.5-4b/` contains a `Dockerfile` and a `README.md`. There is no code, because there is nothing to write: vLLM already serves `/v1/chat/completions`, `/v1/models` and `/health` in the shape the gateway forwards.

That is the clearest demonstration of what the [container contract](adding-a-model.md) actually asks for. Some model folders are a full server — `tts/indic-parler/` carries a paged-KV-cache engine. This one is about 30 lines of Dockerfile. Both satisfy the same interface.

Adding another vLLM model is copying the folder, changing `MODEL_ID`, `SERVED_NAME` and `EXTRA_ARGS`, adding the id to `models.yaml`, and setting `LLM_MODEL` to the new folder name. It appears in `setup.sh`'s menu on its own. Nothing in `compose.model-server.yml` or `gateway/` changes.

## The model-id agreement

vLLM rejects any request whose `model` field is not the name it was started with, and it does so at call time with a `400` — during a live phone call, not at deploy. Four files have to agree on the same string:

| File | What holds the string |
| --- | --- |
| `model-server/llm/<model>/Dockerfile` | `SERVED_NAME=...` and the folder name |
| `model-server/models.yaml` | the catalogue id the gateway reports at `/models` |
| `apps/runtime` — `services/vllm_qwen/llm.py` | `VLLM_MODEL`, what the client puts in the request |
| `apps/runtime` — `config/llm_mappings.py` | the default for provider `qwen` |

Nothing else checks that, so `tests/test_llm_wiring.py` does. It reads the real files — by AST where they are Python — so the test fails when one of them drifts rather than passing against a copy. It skips when the runtime is not present in the checkout.

Note that the LLM slot does not select by model name the way STT and TTS do. `tests/test_client_selection.py` enforces the `<catalogue id>-<slot>` convention for STT and TTS only; the LLM is asked for its served name instead, which is what `test_llm_wiring.py` covers.

## Status

The slot mechanics are tested; the model is not.

`tests/test_llm_slot.py` covers the slot itself, over a real socket rather than an ASGI test client — an ASGI client would not show whether tokens arrive as they are produced, which for a voice agent is the entire point. It checks that an empty slot answers `503` rather than a `404` or a hang, is not advertised at `/v1/models`, and does not mark `/health` degraded; and that a filled slot routes and streams token by token. The upstream in that test is a stand-in for vLLM, not vLLM.

What that leaves unverified is everything about the model itself: whether the pinned `vllm/vllm-openai:v0.27.1` image builds and starts here, whether `--language-model-only` behaves as expected against this checkpoint, whether the `qwen3` reasoning parser splits `<think>` blocks the way the flag table assumes, and what the real memory footprint is at `GPU_MEMORY_UTILIZATION=0.10`.

Treat the deployment as untried. Start it on its own, watch `docker compose logs -f llm` through the first weight download, and confirm `/health` and a plain `POST /v1/chat/completions` before pointing an agent at it.

## Related

* [Gateway API](gateway-api.md)
* [Adding a model](adding-a-model.md)
* [Running on GPUs](gpu-operations.md)
* [Provider registry](../../guides/concepts/provider-registry.md)
