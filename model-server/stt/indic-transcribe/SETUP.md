# Setup

From a bare GPU host to a live streaming-ASR service. Every step below has a check you can run
before moving to the next one — this stack fails in slow, expensive ways (a bad wheel surfaces
as an import error inside a five-gigabyte model load), so the gates are the point.

Budget about 40 minutes, most of it the image build.

---

## 1. What you need first

| | Requirement | Check |
|---|---|---|
| GPU | NVIDIA, ≥16 GB VRAM. Developed on an RTX PRO 6000 Blackwell (sm_120). | `nvidia-smi` |
| Driver | CUDA 13.x. The image installs `torch 2.12.0+cu132`. | `nvidia-smi \| grep CUDA` |
| Docker | With the NVIDIA Container Toolkit. | `docker run --rm --gpus all nvidia/cuda:13.0.1-base-ubuntu24.04 nvidia-smi` |
| Disk | ~30 GB: ~14 GB image, 4.9 GB HF checkpoint, 4.6 GB converted checkpoint. | `df -h .` |
| Ports | 9002, plus 80/443 if you want HTTPS. | `ss -ltnp \| grep -E ':(80\|443\|9002)'` |

**On an older GPU:** the pinned `torch 2.12.0+cu132` is what carries sm_120 (Blackwell) kernels.
Ampere and Ada work with an earlier torch, but `constraints.txt` pins a validated 171-package
set — change one pin and you are re-validating the whole graph. Start here, then move.

### The checkpoint

`indic-transcribe-core` is a Canary 1.2 B model covering **25 Indic languages**. It is **not
part of this repository and not redistributed by it** — you obtain it yourself, under whatever
terms it carries, from the HuggingFace repository you were given access to.

The repository is private, so you need a token with read access to it
(<https://huggingface.co/settings/tokens>):

```bash
pip install -U "huggingface_hub[cli]"
export HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxxxxxx        # your token

hf download <org>/indic-transcribe-core --local-dir models/core
```

On older `huggingface_hub` the command is `huggingface-cli download` with the same arguments.
`<org>/indic-transcribe-core` is the repository id from the link you were sent.

That downloads ~4.9 GB and should leave you with:

```
models/core/
├── config.json
├── tokenizer_config.json          # prompt_ids_by_lang lives here — the authoritative roster
├── model.safetensors              # ~4.89 GB
├── tokenizer_spl_tokens.model
├── feature_extraction_indic_canary.py
├── tokenization_indic_canary.py
├── modeling_indic_canary.py
└── indic_transcribe.py
```

**Check you have the right checkpoint before going further** — `flex` and the ONNX exports are
different models, and the language roster and prompt table here are specific to `core`:

```bash
test -f models/core/model.safetensors && python3 -c \
"import json;d=json.load(open('models/core/tokenizer_config.json'));print(len(d['prompt_langs']),'languages')"
# -> 25 languages
```

If it prints **27** you have `flex`, not `core`. The shipped wrapper code advertises 27
including `bgc` and `hne`, but those two tokens are absent from core's vocabulary and are
skipped at tokenizer init — `tokenizer_config.json` is the materialised truth.

Nothing in this stack ever writes into `models/`; it is bind-mounted read-only throughout. If
the weights already live somewhere else on the host, set `CORE_MODELS_DIR` in `.env` to point
at them instead of copying.

---

## 2. Configure

```bash
cp .env.example .env
```

Edit `CORE_PUBLIC_HOST`. For a local run, `localhost` — Caddy then issues an internal
certificate and attempts no ACME challenge. For a public host, any name that resolves to it;
`sslip.io` gives you one for free, since `a-b-c-d.sslip.io` resolves to `a.b.c.d` with nothing
to register.

**HTTPS is not decoration.** Browsers expose `getUserMedia` only in a secure context, so the
demo's microphone is unavailable over plain HTTP on a public IP — the page loads and the mic
button does nothing.

---

## 3. Build the image

```bash
docker compose build
```

The build asserts its own environment before it ships, and refuses to produce an image otherwise:

```
BUILD GATE OK: 2.12.0+cu132 | 2.11.0+cpu
```

That gate exists because of one specific trap. **There is no cu132 build of torchaudio.** torch
publishes 2.12.0+cu132; torchaudio's newest release is 2.11.0 and ships only
cpu/cu126/cu128/cu129/cu130. The checkpoint's own feature extractor imports torchaudio at module
top level, so it cannot simply be omitted — and a CUDA-flavoured torchaudio beside a CUDA-13.2
torch raises *"PyTorch and TorchAudio were compiled with different CUDA versions"*, which breaks
NeMo itself rather than only the front end.

The resolution: `torchaudio==2.11.0+cpu` installed `--no-deps`. A CPU-only build has no CUDA
extension, so there is no version check to fail. Nothing is lost — `Resample` is pure ATen, and
all audio here is already 16 kHz so it never runs.

---

## 4. Convert the checkpoint, and prove the conversion

The service runs on NeMo's AlignAtt streaming implementation; the checkpoint ships in
HuggingFace format. Converting is a 1:1 tensor rename — encoder 1292 ↔ 1292, decoder 630 ↔ 630,
four prefix rules, zero reshapes — but "the keys all matched" is not evidence the weights
landed correctly, so each step is checked by value.

```bash
RUN="docker run --rm --gpus all \
  -v ${CORE_MODELS_DIR:-$PWD/models/core}:/models/core:ro \
  -v $PWD/artifacts:/artifacts \
  -v $PWD/results:/results \
  core-asr:latest"
```

**a. Offline oracle** — run the HF implementation and keep what it produces. This is the
reference every later step is compared against.

```bash
$RUN python /app/tools/transcribe_hf.py --verify-only --model-dir /models/core --lang hi
```

Expect `1.2214 B` parameters and a coherent transcript in the right script.

**b. Convert.**

```bash
$RUN python /app/tools/hf_to_nemo.py --hf-dir /models/core \
    --out /artifacts/indic_transcribe_core.nemo
```

Produces ~4.6 GB and reports 1926 tensors, vocab 7152.

**c. The gate** — the converted model must produce **byte-identical token ids**, not merely
similar text.

```bash
$RUN python /app/tools/verify_nemo.py \
    --ckpt /artifacts/indic_transcribe_core.nemo \
    --audio /corpus/hi/medium/hi_medium_000.wav --lang hi \
    --expect-ids '[...]'          # the ids printed by step (a)
```

```
[gate] prompt table OK: 25 languages, all 10-token
[gate] PASS — token ids byte-identical to the HF oracle
```

> **Compare ids, not strings.** The HF port's `decode()` ends in `.strip()`; NeMo's
> `Hypothesis.text` keeps the leading SentencePiece space. Two identical models differ by one
> U+0020, and a text comparison reports a failure that is not there.

---

## 5. Run

```bash
docker compose up -d
docker compose logs -f core-asr        # first start loads 4.9 GB; allow ~3 minutes
```

```bash
curl -s localhost:9002/health          # {"status":"ok","sessions":0}
curl -s localhost:9002/v1/languages    # the 25
```

`/health` returns **503 while loading and 200 only once the model is restored *and* warmed**, so
"healthy" means "will answer fast", not "the process is alive". The compose healthcheck allows a
240 s start period for exactly this reason.

Then open `https://<CORE_PUBLIC_HOST>/` for the live demo: pick a language, press start, talk.

### Using it from code

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

Three things to know before you build on this:

* **The stream does not end on its own.** It runs until the client sends `{"type": "stop"}`.
  Add `&endpoint=1` to opt into pause-based turn commits — off by default, because guessing a
  turn boundary from pause length cuts people off mid-thought.
* **`turn_final` means the turn ended, not the stream.** A client that closes on it
  reintroduces the bug that flag was built to fix.
* **The language is always yours to state.** There is no auto-detection; see the README for the
  measured reason.

---

## 6. Verify the whole thing

```bash
./verify.sh
```

Runs every gate in order: oracle → conversion → token-id equality → long-form streaming →
turn handling → gap-free continuous speech → the fatal-error path. Anything red means do not
ship it.

---

## 7. Benchmarks (optional)

Reproduces every number in [REPORT.md](REPORT.md). The corpus is FLEURS, downloaded on demand.

```bash
BENCH="docker run --rm --gpus all --network host \
  -v $PWD/bench:/app/bench:ro -v $PWD/results:/results -v $PWD/corpus:/corpus:ro"

# corpus + offline references (needs the model; run without --network)
$RUN python /app/tools/make_corpus.py --langs hi --per-bucket 6
$RUN python /app/tools/offline_reference.py

# Run F — latency, per language
$BENCH python /app/bench/latency_profile.py --repeats 3

# Run G — the periodic pause. Capture the server log alongside it: the per-turn warm-up
# attribution that proves the cause comes from the server's own timings, not the client's.
docker logs -f --since 1s core-asr > results/_runG_server.log 2>&1 &
$BENCH python /app/bench/rotation_profile.py --repeats 3 --seconds 90 \
       --server-log /results/_runG_server.log
kill %1

# Runs H + I — capacity at the shipped geometry, with GPU counters sampled around the same load
$BENCH python /app/bench/concurrency.py --chunk-s 0.24 --levels 1,4,8,16,24 \
       --repeat 3 --gpu-sample --buckets short,medium \
       --out /results/runH_concurrency_t7.json

$BENCH python /app/bench/report.py --report      # -> results/REPORT.md
cp results/REPORT.md REPORT.md
```

See **[REPORT.md](REPORT.md)** for what one stream experiences and **[LOADTEST.md](LOADTEST.md)**
for behaviour up to 60 concurrent streams.

`report.py` reads `results/*.json` and writes the document. No number in it is typed by hand; if
one cannot be traced to a JSON file, that is a bug in the generator. Run it without `--report`
to regenerate `docs/BENCHMARKS.md`, the campaign that chose the defaults.

**These runs occupy the GPU.** The service stays up but the demo will be slow while they
execute, and `concurrency.py` deliberately loads it to saturation.

---

## Troubleshooting

**`/health` is 503 forever.** Watch `docker compose logs core-asr`. A missing
`/artifacts/*.nemo` or an unreadable `models/core` shows up here immediately.

**`/health` returns 503 with an error message.** The engine hit an unrecoverable CUDA fault and
deliberately shut down; compose replaces the container. This is designed behaviour — a poisoned
CUDA context is process-wide, so the process exits rather than serving a GPU it can no longer
use. If it recurs, see the known defects in [REPORT.md](REPORT.md).

**Caddy will not get a certificate.** `CORE_PUBLIC_HOST` must resolve to this host and port 80
must be reachable from the internet — ACME uses the HTTP-01 challenge. For a local run set it
to `localhost`.

**Transcript is fluent but in the wrong script.** You requested the wrong language. A wrong
language produces confidently wrong output rather than an error; there is no auto-detection to
fall back on.

**`bgc` or `hne` is rejected.** Correct. They are advertised by the wrapper but absent from
core's vocabulary.

**Out of memory with other things on the GPU.** The service reserves ~15 GB. One worker only —
a second uvicorn worker loads a second copy of the model onto the same device.
