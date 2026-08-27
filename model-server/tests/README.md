# model-server tests

Run without a GPU. The NeMo / torch / Parler-runner layers are stubbed in
`stubs/`, so everything else -- routing, batching, protocol, transport -- is the
real code.

```bash
pip install -r tests/requirements-dev.txt
pytest tests/ -v
ruff check .
```

`test_model_switching` shells out to `docker compose config`, which interpolates
without needing a running daemon; it skips if the `docker` CLI is absent.

These guard the things the revamp could silently break:

| Test | Guards |
|---|---|
| `test_stt_audio_parity` | multipart upload decodes to the same samples as the old base64 path |
| `test_tts_request_parity` | `voice` + `instructions` recompose into the exact prompt the model used to get |
| `test_pcm_chunk_boundaries` | a chunk ending mid-float does not desynchronise the audio |
| `test_gateway_streaming` | the gateway streams rather than buffers, and a client disconnect evicts upstream (barge-in) |
| `test_page_table` | the KV page allocator never hands one page to two calls |
| `test_catalogue` | catalogue and folders agree in both directions, and profiles stay slot names |
| `test_model_switching` | naming a different model really builds a different folder, and the slot's service name and port do not move |
| `test_llm_wiring` | the LLM's model id means the same string in the Dockerfile, the catalogue, the voice server and the provider mapping |
| `test_setup_selection` | setup.sh offers a menu per slot instead of assuming a model, and runs the chosen model's `fetch.sh` |
| `test_llm_slot` | an empty slot answers 503 rather than 404 or a hang, is not advertised at `/v1/models`, and does not mark health degraded; a filled one routes and streams token by token |
| `test_tts_format_negotiation` | the client decodes by declared format, so 44.1 kHz float32 and 24 kHz int16 both work; gain never wraps; a sample split across chunks survives at either width |

Two scripts are not part of the suite because they need real models on a GPU:

| Script | Use |
|---|---|
| `smoke_gpu.py` | end-to-end round trip on the box: TTS speaks, STT transcribes it back |
| `bench_tts.py` | latency and real-time factor, sequential or at a chosen concurrency |
