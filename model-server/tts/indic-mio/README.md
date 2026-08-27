# Indic-Mio (TTS)

SPRINGLab's Indic-Mio — a Qwen3-0.6B backbone with the MioCodec vocoder. Fills
the TTS slot; set `TTS_MODEL=indic-mio` in `model-server/.env`.

Voice is a preset speaker embedding rather than a text description, and the
embedding carries the accent, so `language` is informational here. `GET
/v1/voices` on the slot lists the roster — Aditi, Meera, Ananya, Rahul, Arjun,
built from AI4Bharat Rasa reference clips in `voices/`. The model also supports
cloning from a new reference clip; `scripts/build_voices.py` is how the bundled
ones were made.

## Two containers, and why

This is the first model here that is not one process. Token generation is
delegated to a **vLLM sidecar** serving `SPRINGLab/Indic-Mio`; this container
does only the MioCodec decode and the orchestration. That is how upstream runs
it, and it is not worth undoing: the model's Dockerfile pins torch 2.7.1/cu128
for Blackwell `sm_120` kernels, and vLLM's own image brings a different torch.
Folding them together would mean fighting that pinning for no gain.

So the folder brings the sidecar with it, in **`compose.extra.yml`**. The slot
contract is unchanged — one service named `tts` on 8002 answering the OpenAI
route — and the sidecar is an internal service only `tts` talks to, publishing
nothing on the host.

`setup.sh` adds the overlay automatically when this model is selected, by
looking for the file rather than knowing the model. Driving Compose by hand:

```bash
docker compose -f compose.model-server.yml \
               -f tts/indic-mio/compose.extra.yml \
               --project-directory . up -d --build
```

**Watch the sidecar's memory setting.** Upstream uses
`--gpu-memory-utilization 0.35`, which is a fraction of the card's *total*
memory — on a 143 GB H200 that reserves about 50 GB, on a GPU shared with
production through MPS, which does not partition memory. The overlay defaults to
`0.06` (~8.6 GB) and exposes `MIO_VLLM_GPU_MEMORY_UTILIZATION` to change it.

## Transport

The upstream server spoke a WebSocket protocol — one JSON message in, float32
frames out, `{"type":"done"}` to finish. That is gone. `server.py` now serves:

| Endpoint | Purpose |
|----------|---------|
| `POST /v1/audio/speech` | OpenAI-compatible; streams PCM as it is produced |
| `GET /health` | 503 until the codec is loaded and the engine is up |
| `GET /v1/voices` | speaker roster, read from `voices/manifest.json` |

The engine is untouched. Only the transport changed, and it bought two things:

**Cancellation is free.** `synthesize_stream` is an async generator, so when the
caller hangs up mid-sentence Starlette closes it, `GeneratorExit` unwinds into
the aiohttp response reading from vLLM, and that request is aborted. Barge-in
stops GPU work instead of letting a sentence nobody is listening to run to the
end. The WebSocket version had to notice a `ConnectionClosed` and return.

**Self-description.** Responses carry `X-Audio-Format`, `X-Sample-Rate` and
`X-Channels`. This model emits **44.1 kHz float32** (`pcm_f32le`), the same shape
as Indic Parler, and will encode 16-bit `pcm` on request. Orpheus in the same
slot emits 24 kHz signed 16-bit — the client reads the headers rather than
knowing which model answered.

`UPSTREAM-README.md` is the original documentation, kept verbatim. Note it still
describes the WebSocket protocol.

## No `fetch.sh`

Weights come from HuggingFace on first start, into the `hf_cache` volume: the
codec here, and the backbone in the sidecar. First start is slow with little
output — watch `docker compose logs -f tts vllm-mio` rather than waiting on
`/health`, which stays 503 until the codec has loaded.

The bundled reference clips in `voices/refs/` are in the repository, so no
download is needed for the preset voices.

## Not yet run on hardware

Neither container has been built or started. The transport rewrite is covered by
tests that need no GPU; the vLLM sidecar's flags and the codec's behaviour are
unverified.
