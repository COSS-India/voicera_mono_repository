---
description: Self-hosted Qwen3 LLM server for VoicEra voice agents, powered by vLLM.
---

# LLM Server

The LLM server (`llm_server/`) is an **optional** component that lets you run a local large language model instead of (or alongside) cloud providers such as OpenAI or Grok. It wraps [vLLM](https://docs.vllm.ai/) with a pre-tuned launcher for **Qwen3-8B** and exposes an OpenAI-compatible HTTP API on port **8003**.

The voice server connects to it via `VLLM_BASE_URL` / `VLLM_API_KEY` environment variables — no code changes needed.

## When to use it

| Scenario | Recommendation |
|----------|----------------|
| Production telephony, lowest latency | Cloud providers (OpenAI, Grok) via Dashboard → Integrations |
| On-premise / data-sovereignty requirement | Local LLM server |
| Cost optimisation at scale | Local LLM server with GPU hardware |
| Development / offline testing | Local LLM server |

## Prerequisites

- NVIDIA GPU with CUDA support (Ampere or newer recommended)
- CUDA 12.1+
- Python 3.10+
- `vllm` installed: `pip install vllm`

VRAM requirements depend on the model variant chosen (see [Model options](#model-options) below).

## Quick start

```bash
cd llm_server
python server.py
```

The server prints its full vLLM command, then starts. When ready:

```
Server will be available at: http://localhost:8003/v1
Health check: http://localhost:8003/health
```

## Configuration

All settings live at the top of `llm_server/server.py` as Python constants. Edit them before starting the server — no CLI flags needed.

### GPU selection

```python
CUDA_VISIBLE_DEVICES = "2"        # single GPU
# CUDA_VISIBLE_DEVICES = "0,1"    # two GPUs (set TENSOR_PARALLEL_SIZE = 2)
```

### Model options

| Constant value | VRAM | Notes |
|---------------|------|-------|
| `"Qwen/Qwen3-8B"` | ~16 GB | Full BF16 — default |
| `"Qwen/Qwen3-8B-FP8"` | ~9 GB | Pre-quantized FP8; requires Ada Lovelace / Hopper GPU |
| `"Qwen/Qwen3-8B-AWQ"` | ~6 GB | Pre-quantized AWQ; works on any CUDA GPU |
| `"/path/to/local/model"` | — | Local HuggingFace checkpoint directory |

```python
MODEL = "Qwen/Qwen3-8B"
```

### Quantization

Runtime quantization (applied by vLLM on the fly, independent of the model variant):

```python
QUANTIZATION = "fp8"    # None | "fp8" | "awq" | "gptq" | "bitsandbytes"
```

`None` disables runtime quantization (uses full BF16/FP16). Use `"fp8"` for Ampere+.

### Context length

```python
MAX_MODEL_LEN = 8192    # 8192 tokens is recommended for telephony agents
```

For context longer than 32 768 tokens, set `ROPE_SCALING` — see the comments in `server.py`.

### Thinking mode (Qwen3)

Qwen3 supports chain-of-thought reasoning inside `<think>…</think>` blocks. For telephony the recommended default is **off** (latency matters more than reasoning depth).

```python
REASONING_PARSER      = "qwen3"   # "qwen3" | "deepseek_r1" | None
ENABLE_THINKING_DEFAULT = False   # False = thinking OFF by default; can be toggled per request
```

### Host and port

```python
HOST = "0.0.0.0"    # bind address
PORT = 8003         # listen port
```

### Multi-GPU (tensor parallelism)

```python
CUDA_VISIBLE_DEVICES  = "0,1"
TENSOR_PARALLEL_SIZE  = 2         # must equal number of GPUs listed above
```

### Other performance knobs

| Setting | Default | Effect |
|---------|---------|--------|
| `GPU_MEMORY_UTILIZATION` | `0.9` | Fraction of VRAM pre-allocated for the KV cache |
| `MAX_NUM_SEQS` | `32` | Maximum concurrent sequences; 32 suits telephony workloads |
| `ENABLE_CHUNKED_PREFILL` | `True` | Reduces time-to-first-token; recommended for voice agents |
| `ENFORCE_EAGER` | `False` | Disables CUDA graphs; use `True` if you hit OOM |

## Connecting to VoicEra

Point the voice server at your LLM server by setting these in `voice_2_voice_server/.env`:

```env
VLLM_BASE_URL=http://localhost:8003/v1
VLLM_API_KEY=                          # leave blank if you haven't configured vLLM auth
```

Then, when creating an agent in the dashboard, select **vLLM** as the LLM provider and set the model name to match `MODEL` in `server.py` (e.g. `Qwen/Qwen3-8B`).

## API surface

The server exposes an OpenAI-compatible API. Key endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Health check |
| GET | `/v1/models` | List loaded models |
| POST | `/v1/chat/completions` | Chat inference |
| POST | `/v1/completions` | Raw completion inference |

Interactive docs are available at `http://localhost:8003/docs`.

## Troubleshooting

**CUDA out of memory** — lower `GPU_MEMORY_UTILIZATION` (e.g. `0.8`) or switch to a smaller model variant (`FP8` or `AWQ`).

**Slow first response** — normal on first request; vLLM compiles CUDA kernels on startup. Subsequent calls are fast.

**Server exits immediately** — check that `vllm` is installed (`pip install vllm`) and that `CUDA_VISIBLE_DEVICES` points to a valid GPU index.

**Model not found** — on first run vLLM downloads the model from HuggingFace (~16 GB). Set `USE_MODELSCOPE = True` in `server.py` for faster downloads in India.

## Related

- [Voice server](voice-server.md) — connects to the LLM server at call time
- [Environment variables](../reference/environment-variables.md) — `VLLM_BASE_URL` and `VLLM_API_KEY` reference
- [Integrations](integrations.md) — per-org API key storage for cloud LLM providers
