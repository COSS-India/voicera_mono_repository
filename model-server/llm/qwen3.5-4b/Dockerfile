# Qwen3.5-4B on vLLM.
#
# There is no code in this folder because there is nothing to write: vLLM
# already serves /v1/chat/completions, /v1/models and /health in the shape the
# gateway forwards. A model folder here is a Dockerfile and nothing else.
#
# To add another vLLM model, copy this folder, change MODEL_ID, SERVED_NAME and
# EXTRA_ARGS, and set LLM_MODEL to the new folder name. Nothing else moves.
FROM vllm/vllm-openai:v0.27.1

# SERVED_NAME is what clients must put in the request's "model" field -- vLLM
# rejects anything else. Pinning it to the catalogue id means the voice server
# never has to know the HuggingFace path.
ENV MODEL_ID=Qwen/Qwen3.5-4B \
    SERVED_NAME=qwen3.5-4b \
    PORT=8003 \
    MAX_MODEL_LEN=8192 \
    MAX_NUM_SEQS=20 \
    GPU_MEMORY_UTILIZATION=0.10

# --language-model-only: Qwen3.5 registers as a multimodal class, so without
#   this vLLM loads a vision tower this stack will never call and charges the
#   KV cache for it.
# --reasoning-parser qwen3: thinking is on by default in Qwen3.5. The parser
#   splits <think> blocks into reasoning_content so they never reach TTS; the
#   voice server also sends enable_thinking=false per request.
# --enable-chunked-prefill: interleaves prefill with decode, which is what keeps
#   time-to-first-token low when several calls are in flight.
ENV EXTRA_ARGS="--language-model-only --reasoning-parser qwen3 --enable-chunked-prefill --chat-template-content-format string"

# No --quantization: at 4B the weights are ~8 GB in bf16 and quantising buys
# about 4 GB we are not short of, at the cost of quality that is hardest to spot
# in exactly the Indic languages this serves. Set VLLM_QUANTIZATION if that
# trade ever changes.
EXPOSE 8003
ENTRYPOINT ["sh", "-c", "exec vllm serve $MODEL_ID \
  --served-model-name $SERVED_NAME \
  --host 0.0.0.0 --port $PORT \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
  ${VLLM_QUANTIZATION:+--quantization $VLLM_QUANTIZATION} \
  $EXTRA_ARGS"]
