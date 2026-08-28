# indic-transcribe-core — streaming ASR

Live, word-by-word speech recognition for **25 Indic languages**, over a WebSocket, with a
browser demo. Wraps the `indic-transcribe-core` checkpoint (Canary 1.2 B) in NVIDIA NeMo's
AlignAtt streaming decoder behind a FastAPI gateway.


### 1. Prerequisites

| | Requirement | Check |
|---|---|---|
| GPU | NVIDIA, ≥16 GB VRAM | `nvidia-smi` |
| Driver | CUDA 13.x — the image pins `torch 2.12.0+cu132` | `nvidia-smi \| grep CUDA` |
| Docker | with the NVIDIA Container Toolkit | `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi` |
| Disk | ~30 GB (14 GB image, 4.9 GB checkpoint, 4.6 GB converted) | `df -h .` |

### 2. Get the code

```bash
git clone <this-repo> core-serve && cd core-serve
cp .env.example .env
```

Edit `.env` and set `CORE_PUBLIC_HOST`. Use `localhost` for a local run — Caddy then issues an
internal certificate instead of attempting an ACME challenge.

### 3. Download the checkpoint

The HuggingFace repository is private, so you need a token with read access to it
(<https://huggingface.co/settings/tokens>):

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx

hf download <org>/indic-transcribe-core --local-dir models/core
```

`<org>/indic-transcribe-core` is the repository id 
`huggingface_hub` the command is `huggingface-cli download`.

Use the indic transcribe core model.
 language gate are specific to `core`:

```bash
python3 -c "import json;d=json.load(open('models/core/tokenizer_config.json'));print(len(d['prompt_langs']),'languages')"
# -> 25 languages          (27 means you have flex, not core)
```

If the weights already live elsewhere on the host, set `CORE_MODELS_DIR` in `.env` instead of
copying them.

### 4. Build, and convert the checkpoint

```bash
docker compose build
```

The build asserts its own environment and refuses to produce an image otherwise:
`BUILD GATE OK: 2.12.0+cu132 | 2.11.0+cpu`.

The checkpoint ships in HuggingFace format; the streaming decoder is NeMo. Convert it once:

```bash
docker run --rm --gpus all \
  -v "${CORE_MODELS_DIR:-$PWD/models/core}:/models/core:ro" \
  -v "$PWD/artifacts:/artifacts" \
  core-asr:latest python /app/tools/hf_to_nemo.py
```

### 5. Run

```bash
docker compose up -d
docker compose logs -f core-asr      # first start loads 4.9 GB; allow ~3 minutes
curl -s localhost:9002/health        # {"status":"ok","sessions":0}
```

Open `https://<CORE_PUBLIC_HOST>/`, pick a language, and talk.

### Verify it

```bash
./verify.sh
```


---

### WebSocket — the streaming path

```python
import asyncio, json, websockets, soundfile as sf, numpy as np

async def main():
    wav, sr = sf.read("clip.wav", dtype="float32")        # must be 16 kHz mono
    pcm = (np.clip(wav, -1, 1) * 32767).astype(np.int16)
    async with websockets.connect("ws://localhost:9002/v1/asr/ws?language=hi") as ws:
        print(json.loads(await ws.recv()))                # {"type": "ready", ...}
        for i in range(0, len(pcm), 1600):                # 100 ms blocks, paced at 1x
            await ws.send(pcm[i:i + 1600].tobytes())
            await asyncio.sleep(0.1)
        await ws.send(json.dumps({"type": "stop"}))
        while True:
            m = json.loads(await ws.recv())
            if m["type"] == "closed":
                print(m["transcript"]); break
            print(m["type"], m["turn"], m["text"])

asyncio.run(main())
```

Three things to know:

* **The stream does not end on its own.** It runs until the client sends `{"type":"stop"}`.
  Add `&endpoint=1` to opt into pause-based turn commits.
* **`turn_final` ends a turn, not the stream.** A client that closes on it drops the rest.
* **You always state the language.** There is no auto-detection — see below.

### Endpoints

| | |
|---|---|
| `WS /v1/asr/ws?language=hi` | Raw PCM16 in; JSON `partial` / `turn_final` / `closed` out |
| `POST /v1/audio/transcriptions` | OpenAI-shaped. 16 kHz WAV/FLAC/OGG only — no ffmpeg in the image |
| `GET /v1/languages` | The 25 languages, and what the service refuses |
| `GET /health` | 503 loading, 200 ready, 503 after an unrecoverable fault |
| `GET /metrics` | Engine and batcher counters, including capacity |
| `POST /admin/batcher` | Retune `batch_window_ms` / `max_batch` live |

### Languages

`as` `bhb` `bho` `bn` `brx` `doi` `en` `gu` `hi` `kn` `kok` `ks` `mai` `ml` `mni` `mr` `ne`
`or` `pa` `sa` `sat` `sd` `ta` `te` `ur`

### Two things the API refuses on purpose

**No language auto-detection.** Measured top-1 accuracy is 0.047 for `bho`, 0.258 for `hi`,
0.490 for `ur` — each absorbed by a close neighbour. A wrong language yields a confidently
wrong *script*, not an error.

---

## Capacity


Measured to 60 concurrent streams in **[LOADTEST.md](LOADTEST.md)**.

---

## Things you should know before deploying

**Transcription pauses for ~1.9 s every ~22 s of continuous speech.** This is deliberate and
not optional: the decoder's state grows against a 1024-position limit, and with the reset
disabled a 43 s stream stalls near 30 s and ends with 76% of its words. [REPORT.md §3](REPORT.md)
proves the cause and documents the fix that was tried and rejected.

**Do not remove `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`** from `docker-compose.yml`.
Without it the service crashes reproducibly at ~16 concurrent streams. [REPORT.md §7](REPORT.md)
has the bisection.

**The defaults set for current running setup ** Geometry `0.24/0.16` rather than NeMo's `1.0/0.5`,
`alignatt_thr=8`, rotation at 12–20 s. Each is reversible by environment variable and each
costs something when reversed — [REPORT.md](REPORT.md) and [docs/BENCHMARKS.md](docs/BENCHMARKS.md)
say what.

---

## Documentation

| | |
|---|---|
| **[SETUP.md](SETUP.md)** | The full install, with a check after every step, plus troubleshooting |
| **[REPORT.md](REPORT.md)** | What one stream experiences — latency, the pause, accuracy, known defects |
| **[LOADTEST.md](LOADTEST.md)** | Behaviour up to 60 concurrent streams |
| **[bench/README.md](bench/README.md)** | Running and reading the benchmarks |
| **[docs/BENCHMARKS.md](docs/BENCHMARKS.md)** | How the shipped defaults were chosen |

## Layout

```
core_engine.py        AlignAtt streaming engine — sessions, buffer, the tick
batcher.py            GPU worker and batch formation
app.py                FastAPI gateway — WebSocket protocol, turns, admission
vad.py                Silero VAD via onnxruntime, on CPU
nemo_patch/           the tokenizer module NeMo's release is missing
tools/                checkpoint conversion and verification
bench/                load-test harness; report.py generates the documents
static/demo.html      the 25-language live demo
verify.sh             every gate, in order
```

## License

MIT, covering the serving code in this repository. The checkpoint carries its own separate
license and access terms and is not redistributed here.
