---
description: Running the model server on GPUs.
---

# Running on GPUs

The model server assumes NVIDIA GPUs and the NVIDIA Container Toolkit. This page covers picking a device, sharing one with other workloads through MPS, reusing a HuggingFace cache, and the gotchas that have cost time on real hardware.

{% hint style="info" %}
`setup.sh` checks the prerequisites and prints what it finds. It errors if GPU models are selected and `nvidia-smi` is unavailable, and warns if the NVIDIA runtime is not visible to Docker.
{% endhint %}

## Picking a GPU

Pick the device with `GPU_DEVICE_IDS` in `model-server/.env`. Every GPU-bearing service — `stt`, `tts`, `llm`, and any sidecar a model brings — reserves the same device. The gateway takes none: it is pure async I/O with no CUDA context.

```bash
GPU_DEVICE_IDS=1
```

Check `nvidia-smi` first:

```bash
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
nvidia-smi -q | grep -i "compute mode"
```

Two things to avoid: any GPU in `Exclusive Process` mode without a running MPS daemon (see below), and splitting a tensor-parallel pair.

On Voicera's `ace-h200` box GPU 1 is the team's allocation. STT and TTS together need roughly 12 GB.

Memory reservations are worth planning before starting. The vLLM-backed slots take a fraction of the card's **total** memory at startup, not of what is free:

| Setting | Default | What it reserves |
| --- | --- | --- |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.10` | the LLM slot — ~14 GB of a 143 GB H200 |
| `MIO_VLLM_GPU_MEMORY_UTILIZATION` | `0.06` | `tts/indic-mio`'s vLLM sidecar — ~8.6 GB |

Those are two separate reservations on the same card when both are running, not alternatives. Never use vLLM's own `0.9` default here: MPS does not partition memory, so a greedy pre-allocation starves whatever else shares the card rather than failing cleanly.

Every GPU service also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.

## MPS and sharing

NVIDIA's Multi-Process Service lets several processes share one GPU. A GPU in `Exclusive Process` mode can *only* be shared through an MPS daemon; a GPU in the ordinary `Default` mode needs none of it.

**MPS is a property of the host, not of this stack**, so it is detected rather than configured. The daemon publishes a `control` pipe, and that file existing is the fact:

```bash
[ -e "$MPS_PIPE_DIR/control" ] || pgrep -x nvidia-cuda-mps-control
```

When a daemon is found, `compose-files.sh` layers in `compose.mps.yml`; `setup.sh` does the detection and writes the real paths into `.env`. By hand:

```bash
docker compose -f compose.model-server.yml -f compose.mps.yml \
               --project-directory . up -d
```

Both mistakes here are silent. Attach with no daemon behind the pipe directory and the client finds nothing; skip it on an `Exclusive Process` GPU and the container never gets a CUDA context. That is why the detection exists and why `setup.sh` prints a warning naming the check when it finds no daemon.

The pipe directory is per-GPU by convention: `/tmp/nvidia-mps-gpu<N>`, following `GPU_DEVICE_IDS`. Override it outright with `MPS_PIPE_DIR` and `MPS_LOG_DIR` when the daemon lives elsewhere — which is also the escape hatch when `GPU_DEVICE_IDS` names more than one device and `gpu<N>` stops meaning anything.

{% hint style="warning" %}
This used to be wired into the base Compose file with the pipe directory hardcoded to `/tmp/nvidia-mps-gpu1`. That was wrong twice: it ignored `GPU_DEVICE_IDS`, so selecting GPU 3 gave you GPU 3 with GPU 1's pipe — a mismatch nothing reports, because the client fails to find its daemon — and it applied on hosts with no daemon at all. `tests/test_mps.py` pins both, and renders with the real `docker compose config` rather than reading YAML.
{% endhint %}

An MPS client needs the host IPC namespace — the daemon's shared memory lives there and a container in its own namespace cannot see it — plus unlimited `memlock` and a 64 MB stack. `test_mps.py` also checks that the overlay *adds to* each service rather than replacing it: an overlay carrying `volumes:` can shadow the base list instead of extending it, which would unmount the model's own source and leave a container that starts and finds no code.

A model that brings GPU sidecars brings their MPS wiring too, as `<slot>/<model>/compose.mps.yml` — the shared file cannot mention `vllm-mio`, because naming a service that only exists when another overlay is layered in is a Compose error rather than a no-op. `compose-files.sh` adds a model's own MPS file only when both conditions hold: the model is selected *and* a daemon is detected.

## Shared HuggingFace cache

`ai4bharat/indic-parler-tts` is a **gated** HuggingFace repository. The Parler checkpoint comes from elsewhere, but the tokenizer and T5 encoder do not. Two ways round it:

{% tabs %}
{% tab title="Reuse a cache" %}
On a box where another stack has already downloaded them, mount that cache read-only:

```bash
USE_SHARED_HF_CACHE=1 ./setup.sh
```

Or by hand:

```bash
docker compose -f compose.model-server.yml -f compose.shared-hf-cache.yml \
               --project-directory . up -d
```

The overlay sets `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` on the TTS slot, which stops HuggingFace making the gated check that would `401`, and mounts the external volume named by `HF_CACHE_VOLUME` (default `voicera-prod_hf_cache`) read-only — the cache belongs to the other stack and must not be written.

`setup.sh` records the choice in `.env`, so a later restart keeps it. It used to be a setup-time variable only, and the overlay quietly vanished on the next start.
{% endtab %}

{% tab title="Supply a token" %}
Set `HF_TOKEN` in `.env` to an account with access to `ai4bharat/indic-parler-tts`. Compose passes it to every slot as `HUGGING_FACE_HUB_TOKEN`, and containers download into the stack's own `hf_cache` volume.

`setup.sh` prompts for it when a TTS model is selected and neither a token nor the shared-cache option is given, and warns if you skip both.
{% endtab %}
{% endtabs %}

`HF_HUB_OFFLINE` defaults to `0`, which lets a container fetch from HuggingFace on first start — what the vLLM-backed models rely on. Set it to `1` only when pointing at a cache that already holds everything; offline mode turns a gated `401` into a clean miss. A model whose image cannot fetch at all pins this itself: `stt/indic-transcribe` bakes `HF_HUB_OFFLINE=1` into its image, because a container that silently tried to reach a private repository would fail slowly and obscurely instead of quickly.

## Disk and build order

First build takes 20-40 minutes. Budget generously:

| | |
| --- | --- |
| `stt/indic-conformer` checkpoint | ~2.4 GB, downloaded by `fetch.sh` |
| `stt/indic-transcribe` | ~30 GB total: ~14 GB image, 4.9 GB HF checkpoint, 4.6 GB converted checkpoint |
| `llm/qwen3.5-4b` | ~8 GB of bf16 weights, into the `hf_cache` volume on first start |
| `tts/orpheus` | ~7 GB for a 3B backbone in bf16 plus cache |

**Build one image at a time on a tight disk.** Building both in parallel doubles peak usage at the export stage, which is where it fails.

Weights are not in the repository. `stt/indic-conformer/models/IndicConformer.nemo` and `tts/indic-parler/checkpoints/` are gitignored — fetch or copy them before building. They live inside the model's own folder, which is what the slot bind-mounts, so editing a model's `server.py` also does not need a rebuild.

## Gotchas

These are the ones that have cost time on real hardware, from `model-server/README.md`.

**The gated repository.** Covered above. Either supply a token with access, or read a cache that already has the files.

**Weights are not in the repository.** Covered above.

**`additional_contexts` paths resolve against the compose file, not the build context.** The NeMo fork path stays `../../ai4bharat_nemo` even though the model folders moved a level deeper. A test pins this. Note that `compose.extra.yml` overlays follow a *different* rule — their paths resolve against the project directory — so the two cannot be reasoned about interchangeably.

**Build one image at a time on a tight disk.** Covered above.

**Nothing binds on the host except the gateway.** The model containers publish nothing, so this stack can run beside others without competing for ports. To reach a model directly for debugging, use `docker compose exec` or add a temporary `ports:` mapping.

**`/health` may be 503 for minutes.** Several models return 503 while loading — that is by design, so "healthy" means "will answer fast". For the vLLM-backed models the first start also downloads weights with little output. Watch `docker compose logs -f <slot>` rather than assuming it has hung.

### Checking a live box

Two scripts are not part of the test suite because they need real models on a GPU:

```bash
# End-to-end round trip: TTS speaks a sentence, STT transcribes it back
docker compose -f compose.model-server.yml exec -T gateway python - < tests/smoke_gpu.py
docker compose -f compose.model-server.yml cp gateway:/tmp/tts_out.wav .

# Latency and real-time factor, sequential or at a chosen concurrency
python tests/bench_tts.py -n 20                       # sequential
python tests/bench_tts.py -n 16 --concurrency 16      # a busy box
python tests/bench_tts.py --url http://localhost:8002 # skip the gateway
```

`smoke_gpu.py` covers both models plus the full gateway path without needing a sample audio file; listen to `tts_out.wav` afterwards. In `bench_tts.py`, `ttft` is time to first audio byte — what a caller hears as the pause before the bot speaks. Under about 300 ms feels natural.

Everything else in `tests/` runs without a GPU: the NeMo, torch and Parler-runner layers are stubbed, so routing, batching, protocol and transport are all real code.

## Apple MPS

There is no Apple Metal path. **MPS throughout this repository means NVIDIA's Multi-Process Service, not Apple's Metal Performance Shaders** — `compose.mps.yml`, `MPS_PIPE_DIR` and `tests/test_mps.py` are all about sharing an NVIDIA GPU with other processes.

Nothing in `model-server` targets Apple silicon. The GPU reservations name `driver: nvidia`, the images are built on CUDA bases, and `setup.sh` requires `nvidia-smi` when any slot is selected. The `indic-conformer` folder notes that CPU inference works but is far too slow for a live call, which is the closest thing to a non-NVIDIA path here.

To develop on a Mac, run the test suite — it needs no GPU — and point an agent at cloud providers via the [provider registry](../../guides/concepts/provider-registry.md) instead of self-hosting.

## Related

* [Overview](overview.md)
* [Slots and models](slots-and-models.md)
* [Self-hosted models](../../guides/deployment/self-hosted-models.md)
* [Environment variables](../reference/environment-variables.md)
