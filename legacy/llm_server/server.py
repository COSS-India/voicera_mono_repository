import os
import subprocess
import sys

# ============================================================
# QWEN3-8B vLLM SERVER LAUNCHER
# Edit the variables below to configure your server
# ============================================================

# ── GPU SELECTION ────────────────────────────────────────────
# Which GPU(s) to use. Comma-separated indices. e.g. "0", "2", "0,1,2,3"
CUDA_VISIBLE_DEVICES = "2"

# ── MODEL ────────────────────────────────────────────────────
# HuggingFace model ID or local path to model directory
# Options:
#   "Qwen/Qwen3-8B"          → full BF16 model (~16GB VRAM)
#   "Qwen/Qwen3-8B-FP8"      → pre-quantized FP8 (~9GB VRAM), needs Ada Lovelace / Hopper GPU
#   "Qwen/Qwen3-8B-AWQ"      → pre-quantized AWQ (~6GB VRAM), works on any CUDA GPU
#   "/path/to/local/model"   → local directory
MODEL = "Qwen/Qwen3-8B"

# ── QUANTIZATION ─────────────────────────────────────────────
# Runtime quantization applied by vLLM on the fly (different from pre-quantized models above)
# Options:
#   None          → no quantization, full precision (BF16/FP16)
#   "fp8"         → FP8 weight quantization. W8A8 on Ada/Hopper, W8A16 on Ampere
#   "awq"         → AWQ 4-bit quantization (use only if model is AWQ)
#   "gptq"        → GPTQ 4-bit quantization (use only if model is GPTQ)
#   "squeezellm"  → SqueezeLLM quantization
#   "bitsandbytes"→ bitsandbytes quantization
QUANTIZATION = "fp8"

# ── SERVER HOST & PORT ───────────────────────────────────────
# HOST:
#   "0.0.0.0"    → accessible from other machines on your network
#   "127.0.0.1"  → localhost only (more secure, only your machine)
HOST = "0.0.0.0"
PORT = 8003

# ── CONTEXT / SEQUENCE LENGTH ────────────────────────────────
# Maximum number of tokens (prompt + response) per request
# Lower = less VRAM for KV cache, faster, good for telephony
# Higher = can handle longer conversations but needs more VRAM
# Options:
#   None         → uses model default (40960 for Qwen3-8B)
#   8192         → good for telephony agents (short turns)
#   16384        → medium length conversations
#   32768        → full pretraining context
#   131072       → extended context (requires YaRN rope scaling, see below)
MAX_MODEL_LEN = 8192

# ── THINKING MODE / REASONING PARSER ────────────────────────
# Controls how Qwen3's <think>...</think> blocks are handled
# Options:
#   None         → thinking tokens appear raw inside "content" field
#   "qwen3"      → recommended for vLLM >= 0.9.0, splits thinking into
#                  separate "reasoning_content" field. Works with enable_thinking=False
#   "deepseek_r1"→ older parser for vLLM 0.8.5, does NOT support enable_thinking=False
REASONING_PARSER = "qwen3"

# ── THINKING MODE DEFAULT ────────────────────────────────────
# Controls whether thinking is ON or OFF by default for all requests
# This sets the DEFAULT — individual requests can still override it
# Options:
#   True   → thinking ON by default (best for math/coding tasks)
#   False  → thinking OFF by default (best for telephony/voice agents)
ENABLE_THINKING_DEFAULT = False

# ── CHAT TEMPLATE CONTENT FORMAT ────────────────────────────
# How vLLM handles the chat template output format
# Options:
#   "string"      → plain string output (recommended for Qwen3)
#   "openai"      → OpenAI-style structured output
CHAT_TEMPLATE_CONTENT_FORMAT = "string"

# ── GPU MEMORY UTILIZATION ───────────────────────────────────
# Fraction of GPU VRAM vLLM pre-allocates (rest goes to KV cache)
# Options: 0.0 to 1.0
#   0.9  → default, maximizes KV cache
#   0.85 → safer, leaves headroom for other processes
#   0.7  → conservative, reduces KV cache significantly
GPU_MEMORY_UTILIZATION = 0.9

# ── TENSOR PARALLELISM ───────────────────────────────────────
# Number of GPUs to split the model across (multi-GPU inference)
# Options: 1, 2, 4, 8 (must equal number of GPUs in CUDA_VISIBLE_DEVICES)
#   1 → single GPU (default)
#   2 → split across 2 GPUs (set CUDA_VISIBLE_DEVICES="0,1")
#   4 → split across 4 GPUs
TENSOR_PARALLEL_SIZE = 1

# ── DTYPE ────────────────────────────────────────────────────
# Data type for model weights (when not quantized)
# Options:
#   "auto"     → auto-detect from model config (recommended)
#   "bfloat16" → BF16, best for Ampere+ (A100, 3090, 4090)
#   "float16"  → FP16, better for older GPUs (V100, T4)
#   "float32"  → FP32, very slow, only for debugging
DTYPE = "auto"

# ── TOOL CALLING ─────────────────────────────────────────────
# Enable automatic tool/function calling support
# Options:
#   False              → disabled
#   True               → enable with hermes parser (for Qwen3)
ENABLE_TOOL_CALLING = False
TOOL_CALL_PARSER = "hermes"  # used only if ENABLE_TOOL_CALLING=True

# ── CHUNKED PREFILL ──────────────────────────────────────────
# Interleaves prefill and decode steps to reduce time-to-first-token
# Options:
#   True   → enabled (recommended for telephony, reduces TTFT)
#   False  → disabled
ENABLE_CHUNKED_PREFILL = True

# ── MAX CONCURRENT SEQUENCES ─────────────────────────────────
# Maximum number of requests being processed simultaneously
# Options: any integer
#   None  → vLLM decides automatically based on available memory
#   32    → good for telephony (many short concurrent calls)
#   64    → higher throughput server
MAX_NUM_SEQS = 32

# ── ROPE SCALING (for context > 32768 tokens) ────────────────
# Only needed if MAX_MODEL_LEN > 32768
# If you set MAX_MODEL_LEN <= 32768, leave this as None
# Example for 131072 context:
#   ROPE_SCALING = '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":32768}'
# For 65536 context:
#   ROPE_SCALING = '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}'
ROPE_SCALING = None

# ── ENFORCE EAGER MODE ───────────────────────────────────────
# Disables CUDA Graphs (torch.compile), slower but uses less VRAM
# Options:
#   False  → CUDA Graphs ON (default, faster inference)
#   True   → CUDA Graphs OFF (slower, use if you hit OOM)
ENFORCE_EAGER = False

# ── MODEL SOURCE ─────────────────────────────────────────────
# Where to download the model from
# Options:
#   False  → HuggingFace Hub (default)
#   True   → ModelScope (faster in China/India sometimes)
USE_MODELSCOPE = False

# ============================================================
# DO NOT EDIT BELOW THIS LINE
# ============================================================

def build_command():
    cmd = ["vllm", "serve", MODEL]

    cmd += ["--host", HOST]
    cmd += ["--port", str(PORT)]

    if QUANTIZATION:
        cmd += ["--quantization", QUANTIZATION]

    if MAX_MODEL_LEN:
        cmd += ["--max-model-len", str(MAX_MODEL_LEN)]

    if REASONING_PARSER:
        cmd += ["--reasoning-parser", REASONING_PARSER]

    if CHAT_TEMPLATE_CONTENT_FORMAT:
        cmd += ["--chat-template-content-format", CHAT_TEMPLATE_CONTENT_FORMAT]

    cmd += ["--gpu-memory-utilization", str(GPU_MEMORY_UTILIZATION)]
    cmd += ["--tensor-parallel-size", str(TENSOR_PARALLEL_SIZE)]
    cmd += ["--dtype", DTYPE]

    if ENABLE_CHUNKED_PREFILL:
        cmd += ["--enable-chunked-prefill"]

    if MAX_NUM_SEQS:
        cmd += ["--max-num-seqs", str(MAX_NUM_SEQS)]

    if ROPE_SCALING:
        cmd += ["--rope-scaling", ROPE_SCALING]

    if ENFORCE_EAGER:
        cmd += ["--enforce-eager"]

    if ENABLE_TOOL_CALLING:
        cmd += ["--enable-auto-tool-choice", "--tool-call-parser", TOOL_CALL_PARSER]

    if not ENABLE_THINKING_DEFAULT:
        # Appends /no_think behavior note — actual disable is per-request via chat_template_kwargs
        # This default is handled at request time in your client code
        pass

    return cmd


def main():
    env = os.environ.copy()

    env["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES

    if USE_MODELSCOPE:
        env["VLLM_USE_MODELSCOPE"] = "true"

    command = build_command()

    print("=" * 60)
    print("  Qwen3-8B vLLM Server Launcher")
    print("=" * 60)
    print(f"\n  GPU(s)         : {CUDA_VISIBLE_DEVICES}")
    print(f"  Model          : {MODEL}")
    print(f"  Quantization   : {QUANTIZATION or 'None (BF16/FP16)'}")
    print(f"  Max Ctx Length : {MAX_MODEL_LEN or 'Default (40960)'}")
    print(f"  Reasoning Mode : {REASONING_PARSER or 'Raw (no parser)'}")
    print(f"  Thinking Mode  : {'ON' if ENABLE_THINKING_DEFAULT else 'OFF (use /no_think in system prompt)'}")
    print(f"  Host:Port      : {HOST}:{PORT}")
    print(f"  Chunked Prefill: {ENABLE_CHUNKED_PREFILL}")
    print(f"  Max Sequences  : {MAX_NUM_SEQS}")
    print(f"  GPU Mem Util   : {GPU_MEMORY_UTILIZATION}")
    print(f"  Enforce Eager  : {ENFORCE_EAGER}")
    print(f"  Tool Calling   : {ENABLE_TOOL_CALLING}")
    print(f"\n  Full command:\n  {' '.join(command)}")
    print("\n" + "=" * 60)
    print(f"  Server will be available at: http://localhost:{PORT}/v1")
    print(f"  Health check: http://localhost:{PORT}/health")
    print("=" * 60 + "\n")

    try:
        subprocess.run(command, env=env, check=True)
    except KeyboardInterrupt:
        print("\n\nServer stopped.")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"\nServer exited with error code {e.returncode}")
        sys.exit(e.returncode)


if __name__ == "__main__":
    main()