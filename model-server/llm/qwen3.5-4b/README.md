# Qwen3.5-4B

Fills the LLM slot. There is no code here — vLLM already serves
`/v1/chat/completions`, `/v1/models` and `/health` in the shape the gateway
forwards, so the Dockerfile is the entire implementation.

No `fetch.sh` either, unlike the STT and TTS folders: vLLM downloads its own
weights from HuggingFace into the `hf_cache` volume on first start, and that
volume survives restarts. First start therefore takes several minutes with no
output on `/health` — watch `docker compose logs -f llm` rather than assuming it
has hung.

## Why each flag

| flag | why |
|---|---|
| `--served-model-name qwen3.5-4b` | vLLM rejects any request whose `model` field is not the name it was started with, so this is pinned to the catalogue id. Without it, clients would have to send `Qwen/Qwen3.5-4B`. |
| `--language-model-only` | The checkpoint registers as `Qwen3_5ForConditionalGeneration` and ships a 24-layer vision encoder. This stack is voice only, so skipping it hands that memory back to the KV cache. |
| `--max-model-len 8192` | Native context is 262,144. Telephony turns are short, and reserving KV cache for context nobody uses is the fastest way to run out of GPU. |
| `--reasoning-parser qwen3` | Thinking is on by default in Qwen3.5. The parser splits `<think>` blocks into `reasoning_content` so they never reach TTS as spoken text. |
| `--enable-chunked-prefill` | Interleaves prefill with decode, which is what holds time-to-first-token down when several calls are in flight. |
| no `--quantization` | At 4B the weights are ~8 GB in bf16. FP8 saves about 4 GB this box can spare, and small-model quantisation degrades hardest in exactly the Indic languages this serves. Set `VLLM_QUANTIZATION=fp8` in `.env` if memory ever becomes the constraint. |

`GPU_MEMORY_UTILIZATION` defaults to `0.10`, and note what that means: it is a
fraction of the card's **total** memory, not of what is free. On a 143 GB H200
that is a hard ~14 GB reservation taken at startup. MPS does not partition
memory, so oversizing it takes memory away from the production workers on the
same GPU rather than failing cleanly.

## Two vLLM bugs that affect this

Both produce a call with dead air rather than an error, so they are worth
knowing before debugging a silent bot.

- [vllm#35574](https://github.com/vllm-project/vllm/issues/35574) —
  `chat_template_kwargs: {enable_thinking: false}` did not always disable
  thinking on Qwen3.5. Closed February 2026, so fixed well before the v0.27.1
  pinned here, but it is why the voice server also appends `/no_think` to the
  system prompt as a second signal.
- [vllm#38894](https://github.com/vllm-project/vllm/issues/38894) — with the
  `qwen3` reasoning parser, generated text can arrive in `delta.reasoning` while
  `delta.content` stays empty. Pipecat only forwards `content` to TTS, so the
  caller hears nothing. `VllmQwenVoiceLLMService._normalize_qwen_chunk` in the
  voice server copies one into the other, but only while thinking is disabled —
  with thinking on, that mapping would speak the chain of thought aloud.

`tests/test_llm_wiring.py` pins both of those behaviours.

## Adding another vLLM model

Copy this folder, change `MODEL_ID`, `SERVED_NAME` and `EXTRA_ARGS`, add the id
to `models.yaml`, and set `LLM_MODEL` to the new folder name. It appears in
`setup.sh`'s menu on its own. Nothing in `compose.model-server.yml` or
`gateway/` changes.
