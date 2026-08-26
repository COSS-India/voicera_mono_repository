# model-server tests

Run without a GPU. The NeMo / torch / Parler-runner layers are stubbed in
`stubs/`, so everything else -- routing, batching, protocol, transport -- is the
real code.

```bash
pip install -r tests/requirements-dev.txt
pytest tests/ -v
ruff check .
```

These guard the three things the revamp could silently break:

| Test | Guards |
|---|---|
| `test_stt_audio_parity` | multipart upload decodes to the same samples as the old base64 path |
| `test_tts_dialect_parity` | legacy and `speech.create` requests reach the runner identically |
| `test_gateway_streaming` | the gateway streams rather than buffers, and a client disconnect evicts upstream (barge-in) |
