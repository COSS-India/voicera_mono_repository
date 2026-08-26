# vLLM. The official image already serves the OpenAI spec, so this is a thin
# wrapper that reads the same knobs server.py exposes as constants.
# server.py is kept for the bare-metal path and as documentation of the flags.
FROM vllm/vllm-openai:v0.11.0

ENV MODEL_ID=Qwen/Qwen3-8B \
    PORT=8003 \
    QUANTIZATION=fp8 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=20 \
    GPU_MEMORY_UTILIZATION=0.30 \
    REASONING_PARSER=qwen3

EXPOSE 8003
ENTRYPOINT ["sh", "-c", "vllm serve $MODEL_ID \
  --host 0.0.0.0 --port $PORT \
  --quantization $QUANTIZATION \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  --reasoning-parser $REASONING_PARSER \
  --chat-template-content-format string \
  --enable-chunked-prefill"]
