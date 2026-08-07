---
description: Scalability and operational-quality issue log for ai4bharat_stt_server (+ its deploy config and v2v client).
---

# STT Scalability & Operations Issue Log

Scope: `ai4bharat_stt_server/`, `deploy/nginx/nginx.conf`, `deploy/compose/docker-compose.prod.yml`,
and the calling client `voice_2_voice_server/services/ai4bharat/stt.py`.

Audit date: 2026-08-07. Status values: `OPEN`, `FIXED`, `PARTIAL`, `WONTFIX-NOW`.

Line references are to the pre-fix revision (`a72d370`) unless a fix note says otherwise.

---

## A. Scalability

| # | Status | Issue | Location | Effect |
|---|--------|-------|----------|--------|
| S1 | **PARTIAL** | No `/sttlb` load-balancer route existed and `INDIC_STT_SERVER_URL` points straight at `stt:8001`; aiohttp keep-alive + DNS caching pins each v2v process to one replica. Route now added; **the env change is still pending** | `deploy/nginx/nginx.conf`, `deploy/OPERATIONS.md:177` | `--scale stt=N` has no effect — STT cannot be scaled horizontally until the voice server is repointed at `http://nginx:8080/sttlb`. Same failure mode `OPERATIONS.md:411-415` documents for TTS |
| S2 | **FIXED** | Interim re-transcription is O(n²): the entire growing segment is re-sent every 600 ms | `voice_2_voice_server/services/ai4bharat/stt.py:211-230` | ~11x GPU work amplification on a 12 s utterance (~126 audio-seconds of inference for 12 s of speech). Dominant capacity limit |
| S3 | **WONTFIX-NOW** | `STT_NUM_WORKERS=4` → 4 model copies, 4 CUDA contexts, 4 independent batchers | `deploy/compose/docker-compose.prod.yml:73` | Batching efficiency ÷4: 16 concurrent requests become 4 batches of 4, not 1 of 16. Wastes VRAM and defeats the server's only GPU optimisation |
| S4 | **PARTIAL** | fp32 everywhere; no TF32, no bf16/fp16 | `server.py:80-101` | ~2x throughput and ~half the VRAM left unused |
| S5 | **FIXED** | `torch.cuda.empty_cache()` on every idle second | `server.py:198-209` | Returns allocator segments to the driver during every quiet gap → `cudaMalloc` + fragmentation latency spike on the first request after idle. Redundant with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| S6 | **OPEN** | Batches are capped by **count** (`MAX_BATCH_SIZE=16`), not by padded audio-seconds; NeMo pads to the longest item | `server.py:145-156` | One 20 s clip batched with fifteen 0.8 s clips costs 16x20 s of compute for 32 s of audio |
| S7 | **FIXED** | `asyncio.to_thread(response_queue.get)` — one blocked threadpool thread per in-flight request | `server.py:260, 273` | Hard concurrency ceiling of `min(32, cpu+4)` per worker, unrelated to GPU capacity |
| S8 | **FIXED** | No stale-request drop: queue holds 256 items, client gives up after 10 s | `server.py:45, 128`, `stt.py:200` | Under overload the GPU burns cycles on requests nobody is waiting for → collapse instead of load-shed |
| S9 | **FIXED** | base64 decode + numpy conversion ran on the event loop; `request_queue.put(..., timeout=1.0)` also blocked the event loop for up to 1 s when full | `server.py:114-116, 128, 258` | One large payload or a full queue stalls every other request in that worker process |
| S10 | **WONTFIX-NOW** | `BATCH_TIMEOUT=0.100` is paid per request across 4 near-empty batchers | `server.py:47` | +100 ms latency for no throughput gain at the current worker count. Now env-tunable (`STT_BATCH_TIMEOUT_MS`); retune together with S3 |
| S11 | **OPEN** | `set $var` + `proxy_pass http://$var` on every route → no `upstream {}` block → no keepalive connection pool | `nginx.conf:33, 39, 45, 52, 74` | Fresh TCP connection per proxied request. Deliberate for DNS re-resolution of scaled services, wrong for `backend`/`frontend` |
| S12 | **FIXED** | `proxy_set_header Connection "upgrade"` set unconditionally at server level | `nginx.conf:27-28` | Plain HTTP requests carry `Connection: upgrade` with an empty `Upgrade` header, breaking upstream keep-alive |
| S13 | **FIXED** (STT route) | `proxy_request_buffering` left at the default `on` | `nginx.conf` | nginx fully buffers each base64 body before forwarding — pure added latency on the STT path |
| S14 | **OPEN** | CUDA-graph RNNT decoding unused; RNNT greedy decode is launch-bound | `server.py:88, 100` | Same class of win as the CUDA-graph TTS fix noted in `OPERATIONS.md:426` |
| S15 | **OPEN** | No autoscaling; `--scale` reverts on any plain `up` | `OPERATIONS.md:396-399` | Static capacity, manual toil |

## B. Operational quality

| # | Status | Issue | Location | Effect |
|---|--------|-------|----------|--------|
| O1 | **FIXED** | **Worker thread death = permanent brick.** No `try/except` in `batch_worker`; `/health` still reported `"healthy"`; `restart: unless-stopped` never fired | `server.py:176-219, 277-289` | One transient CUDA OOM or a bad tensor killed the only inference thread. Every subsequent request hung on an untimed `response_queue.get`, the queue filled, and the service returned `503 STT queue is full` forever with no restart and no alert |
| O2 | **FIXED** | **Zero logging.** `loguru` declared in `requirements.txt`, never imported | `server.py` (whole file) | No request count, latency, batch size, or error rate. Nothing to triage or capacity-plan with |
| O3 | **FIXED** | No `/metrics`, no RTFx measurement | — | Could not answer "are we GPU-saturated". `nvidia-smi dmon` by hand does not scale |
| O4 | **FIXED** | No `healthcheck:` on the `stt` service, while model load takes 2-3 min | `docker-compose.prod.yml:56-85` | 3 minutes of connection-refused after every deploy, swallowed by the client as `""` → silent dead air, no signal in `dcp ps` |
| O5 | **FIXED** | `/health` returned HTTP 200 with `main_loaded: false` | `server.py:277-289` | Unusable as a readiness probe |
| O6 | **FIXED** | Client collapsed 503 / timeout / connection-refused / genuine silence into one `""` | `stt.py:205-209` | "STT overloaded" indistinguishable from "user said nothing" |
| O7 | **FIXED** | No graceful shutdown: daemon threads, no shutdown handler | `server.py:222-233` | Every deploy dropped in-flight transcriptions mid-call; callers waited out their full 10 s timeout |
| O8 | **OPEN** | No VRAM cap on `stt` / `tts` / `mio-tts`; all request `count: all` (only `vllm-mio` is capped, at 0.35) | `docker-compose.prod.yml:79-85, 106-112, 171-177` | Cross-service OOM, which used to trigger O1 |
| O9 | **FIXED** | nginx `log_format` carried no `$upstream_response_time` / `$upstream_addr` | `nginx.conf` | Blind to which upstream or replica is slow |
| O10 | **PARTIAL** | 600 s proxy timeouts applied to every route | `nginx.conf:13-16` | A hung upstream holds an nginx worker connection for 10 minutes |
| O11 | **OPEN** | No `max_fails` / `proxy_next_upstream` (consequence of S11) | `nginx.conf` | A dead replica keeps receiving traffic |
| O12 | **PARTIAL** | `limit_req` configured only for `/minio/` | `nginx.conf:3, 61` | No overload protection on the routes that carry load |
| O13 | **FIXED** | `.env.example` omitted every **required** variable (`INDIC_NEMO_PATH`, `BHILI_NEMO_PATH`, `BHILI_ENABLE`) | `.env.example` | A copy of the example cannot boot — `_required_model_path` hard-fails |
| O14 | **PARTIAL** | Unpinned deps plus wrong packages: `dotenv` (PyPI stub of `python-dotenv`), `pyarr` (Sonarr/Radarr client, typo for `pyarrow`, unused), unused `onnx`/`onnxruntime`/`onnxruntime-gpu` | `requirements.txt` | Non-reproducible builds, bloated image |
| O15 | **PARTIAL** | No `HEALTHCHECK`, runs as root, nemo fork revision unpinned | `Dockerfile` | No container-level liveness; unreproducible image |
| O16 | **PARTIAL** | No STT load test and no CI; `test_bhili_endpoint.py` was the only test | — | Replica counts and batch parameters set by guess, not measurement |
| O17 | **FIXED** | Docs instruct `python server.py --port 8001`; `server.py` parses no argv | `docs/services/ai4bharat-stt.md:104` | Runbook does not work as written |
| O18 | **FIXED** | Dead `model.py`: different model (`ai4bharat/indic-conformer-600m-multilingual`), different API (`UploadFile`), imported by nothing | `ai4bharat_stt_server/model.py` | Misleads operators into thinking file upload is supported |

## C. Out of scope for this log, still open

Correctness/security findings from the same audit, recorded here so they are not lost. **Not fixed** —
they are not scalability or operational-quality items.

| # | Status | Issue | Location |
|---|--------|-------|----------|
| C1 | **OPEN** | Mixed-language batch corruption: `main_infer` applies `language_ids[0]` to the whole batch, so a Tamil caller batched behind a Hindi caller is decoded as Hindi. Silent, load-dependent quality destruction | `server.py:164-165` |
| C2 | **OPEN** | No authentication on `/transcribe*`. Contained under Compose (internal network, no published port), but `setup.sh:390` exposes it on the host's private IP with no auth, no TLS, and no rate limit for bare-metal installs | `server.py`, `setup.sh:390` |
| C3 | **OPEN** | No audio format validation — a WAV file's 44-byte header is interpreted as PCM samples | `server.py:114-116` |

---

## Fix log

Applied 2026-08-07. Each entry lists what changed and how it was verified.

### server.py — rewritten around a supervised worker (O1, O2, O3, O5, O7, S5, S7, S8, S9, S4-partial)

- **O1** `batch_worker` now wraps batch collection and inference in `try/except`. On failure it
  increments `WorkerStats.failures`, logs with a traceback, replies `""` to every item in the failed
  batch, and calls `torch.cuda.empty_cache()` when the message looks like a CUDA OOM. The loop
  cannot exit on an exception. Each worker publishes a monotonic heartbeat; `/health` reports
  `degraded` + HTTP 503 when a heartbeat is older than `STT_WORKER_STALL_SECONDS` (default 60),
  so `restart: unless-stopped` plus the new Compose healthcheck now actually recycle the container.
- **S7** The response path no longer uses `queue.Queue` + `asyncio.to_thread`. Each request creates
  an `asyncio.Future`; the worker completes it with `loop.call_soon_threadsafe`. Zero threadpool
  threads are held per in-flight request, removing the ~32/worker ceiling. Replies are idempotent,
  so a late worker reply after a client timeout is dropped instead of raising `InvalidStateError`.
- **S8** Every queued item carries a monotonic deadline (`STT_REQUEST_DEADLINE_S`, default 10 s, matched
  to the client's `ClientTimeout(total=10)`). The worker drops expired items before inference and
  answers them `""`, so overload sheds instead of amplifying. The handler also bounds its own wait
  with `asyncio.wait_for` and returns **504** rather than hanging forever.
- **S9** base64 decode and the numpy conversion run in `asyncio.to_thread`. The blocking
  `request_queue.put(..., timeout=1.0)` became `put_nowait`, so a full queue sheds immediately
  (503) instead of stalling the event loop for a second.
- **O2** Structured `loguru` logging: startup/model-load timing, per-batch debug lines (size, audio
  seconds, inference ms, RTFx), a rolled-up info summary every `STT_LOG_SUMMARY_EVERY` batches
  (default 100), and warnings for shed/stale/failed batches.
- **O3** New `/metrics` endpoint emitting Prometheus text format with **no new dependency**:
  `stt_requests_total`, `stt_batches_total`, `stt_batch_failures_total`, `stt_stale_dropped_total`,
  `stt_rejected_total{reason=...}`, `stt_audio_seconds_total`, `stt_infer_seconds_total`,
  `stt_queue_depth`, `stt_worker_up`, `stt_model_loaded`, `stt_ready`. RTFx is
  `stt_audio_seconds_total / stt_infer_seconds_total`.
- **O5** `/health` returns 503 while `status` is `loading` or `degraded` and 200 only when ready. All
  pre-existing keys (`status`, `main_loaded`, `bhili_enabled`, `main_queue_size`, …) are retained so
  `setup.sh` and the runbooks keep working.
- **O7** A `lifespan` shutdown phase signals the workers to stop and drains both queues, replying
  `""` to everything still pending. In-flight callers get an immediate empty result instead of
  waiting out a 10 s timeout. This also replaces the deprecated `@app.on_event` hooks.
- **S5** `torch.cuda.empty_cache()` moved behind `STT_IDLE_EMPTY_CACHE_SECONDS` (default 60 s of
  continuous idle) instead of firing every second.
- **S4 (partial)** `torch.set_float32_matmul_precision(STT_MATMUL_PRECISION)`, default `high`, enables
  TF32 matmuls on Ampere+; set to `highest` to revert. bf16/fp16 weights are **not** applied — that
  needs a WER check against the production checkpoint on real GPU hardware.
- Batch parameters are now env-tunable without a rebuild: `STT_MAX_BATCH_SIZE`, `STT_BATCH_TIMEOUT_MS`,
  `STT_QUEUE_MAXSIZE`, `STT_MAX_AUDIO_SECONDS`.
- Input guards added while restructuring the decode path: oversized payloads → 413, undecodable or
  odd-length payloads → 400 (previously both produced a 500 with a traceback).

### voice_2_voice_server/services/ai4bharat/stt.py (S2, O6)

- **S2** The interim interval is now adaptive: an interim fires only after the new audio exceeds
  `min(max(AI4BHARAT_INTERIM_MS, len(segment_buffer) * AI4BHARAT_INTERIM_GROWTH), AI4BHARAT_INTERIM_MAX_S)`.
  Defaults: growth `0.5` (50 % more audio required before the next interim), ceiling `2.0` s.
  Measured (simulation of the real chunk/interim arithmetic, 200 ms chunks, `AI4BHARAT_INTERIM_MS=600`):

  | utterance | before | after | GPU work |
  |-----------|--------|-------|----------|
  | 5 s | 8 interims, 26.6 audio-s inferred | 4 interims, 13.6 audio-s | **1.96x less** |
  | 12 s | 20 interims, 138.0 audio-s | 7 interims, 45.8 audio-s | **3.01x less** |
  | 30 s | 50 interims, 795.0 audio-s | 16 interims, 247.4 audio-s | **3.21x less** |

  The `AI4BHARAT_INTERIM_MAX_S=2.0` ceiling exists because uncapped growth reached 4.5x on a 12 s
  utterance but let the interim gap drift to ~4 s, and left the last interim covering only 9.6 s of
  12 s. With the ceiling the gap never exceeds 2 s and the last interim covers to within 1.6 s of the
  end, so the "promote latest interim when the final comes back empty" fallback at `stt.py:249-263`
  keeps its quality. Final transcripts are untouched either way — interims are display-only and the
  window is still the whole segment, never a trailing slice. Set `AI4BHARAT_INTERIM_GROWTH=0` to
  restore the old fixed-interval behaviour.
- **O6** `_transcribe_buffer` now distinguishes 503 (server shed: queue full or not ready), 429, 504,
  other HTTP statuses, `asyncio.TimeoutError`, and `aiohttp.ClientError` in its log lines, each with
  its own message. "STT overloaded" is no longer indistinguishable from "user said nothing".

### deploy/nginx/nginx.conf (S1, S12, S13, O9, O10-partial, O12-partial)

- **S1** New `/sttlb/` location: `set $stt_up stt:8001;` + `rewrite` + variable `proxy_pass`, so nginx
  re-resolves `stt` every 10 s (existing `resolver 127.0.0.11 valid=10s`) and round-robins over
  replicas. Mirrors the `/ttslb2` pattern.
  **Deployment still required:** set `INDIC_STT_SERVER_URL=http://nginx:8080/sttlb` in
  `voice_2_voice_server/.env` and recreate `voice_server`. Until that is done the route exists but is
  unused and S1 stays OPEN — pointing the client at `stt:8001` keeps pinning to one replica.
- **S12** Added `map $http_upgrade $connection_upgrade { default upgrade; '' close; }` and switched the
  server-level and `/minio/` headers to `Connection $connection_upgrade`. WebSocket routes are
  unaffected (they send `Upgrade`, so the map yields `upgrade`).
- **S13 / O10 / O12 (STT route only)** Inside `/sttlb/`: `proxy_request_buffering off`,
  `client_max_body_size 4m`, `proxy_connect_timeout 2s`, `proxy_send_timeout 20s`,
  `proxy_read_timeout 20s` (instead of the global 600 s), and
  `limit_req zone=stt_limit burst=200 nodelay` with `limit_req_status 429`. The global 600 s timeouts
  are unchanged for the WebSocket routes that need them, so O10 stays PARTIAL.
- **O9** Added a `stt_upstream` log format carrying `rt`, `uct`, `urt`, and `ua`, applied via
  `access_log`, plus `server_tokens off`.

### deploy/compose/docker-compose.prod.yml (O4)

- **O4** `stt` gained a `healthcheck` that calls `/health` with stdlib `urllib` (no `curl` in the
  image), `start_period: 300s` to cover model load, and `interval: 20s`. Combined with the O1/O5
  changes, a wedged worker now surfaces as `unhealthy` in `dcp ps`.
- Deliberately **not** done: gating `voice_server` on `stt: {condition: service_healthy}`. STT is an
  optional provider, so a failed STT load must not prevent the voice server (and its cloud-STT
  agents) from starting. Boot ordering stays as-is.

### Dockerfile, requirements.txt, .env.example, docs (O13, O14, O15, O17, O18)

- **O15 (partial)** Added a `HEALTHCHECK` matching the Compose one. Non-root user and a pinned nemo
  fork revision are still open — both need volume-permission and build-context changes to verify.
- **O14 (partial)** `dotenv` → `python-dotenv` (the former is a PyPI stub), removed `pyarr` (a
  Sonarr/Radarr API client, unused — almost certainly a typo for `pyarrow`) and the unused
  `onnx` / `onnxruntime` / `onnxruntime-gpu` trio, and dropped the trailing whitespace on
  `torchaudio`. Version pins are still open: they need a resolve-and-test pass against the NeMo fork.
- **O13** `.env.example` now documents `INDIC_NEMO_PATH`, `BHILI_NEMO_PATH`, `BHILI_ENABLE` and every
  new tunable, so a copy of it can boot.
- **O17** `docs/services/ai4bharat-stt.md` run instructions corrected to `PORT=8001 python3 server.py`.
- **O18** Deleted the dead `model.py` (no importer anywhere in the repo; recoverable from git history).
- **O16 (partial)** Added `test_server_contracts.py` — 11 pytest cases that stub NeMo and exercise the
  worker-crash, stale-drop, queue-shed, oversize, malformed-input, readiness, metrics, and
  shutdown-drain paths. A GPU load test is still open.

### Verification

**Tests.** `python3 -m pytest ai4bharat_stt_server/test_server_contracts.py -q` → **11 passed**
(NeMo import surface stubbed, fake model injected, runs on CPU).

**O1 confirmed against the pre-fix code.** Ran the same flaky-inference scenario against
`git show a72d370:ai4bharat_stt_server/server.py`:

```
pre-fix behaviour:
  request 1 (triggers the exception): HUNG (no response after 6s)
  worker thread alive after exception: False
  request 2 (after the exception): HUNG (no response after 6s)
```

The repro process could not even exit — both HTTP calls were parked on the untimed `queue.get`
forever. Post-fix, `test_worker_survives_inference_exception` shows request 1 returning
`200 {"text": ""}`, request 2 returning `200 {"text": "ok"}`, `failures == 1`, worker still alive.

**S8** `test_stale_requests_dropped_before_inference` asserts the model is never invoked for an
expired item (`batches == 0`, `stale_dropped == 1`) and the worker stays alive.

**S2** Adaptive-interim arithmetic simulated directly (table above). `python3 -m py_compile` on the
edited `stt.py` passes; the full v2v import graph needs pipecat, which is not installed here.

**nginx** `docker run --rm -v .../nginx.conf:/etc/nginx/conf.d/default.conf:ro nginx:alpine nginx -t`
→ `syntax is ok` / `test is successful`.

**O4** Compose YAML parses and the healthcheck command was exercised against a stub server:
exit `0` on HTTP 200, exit `1` on 503, exit `1` when nothing is listening — i.e. `loading` and
`degraded` both mark the container unhealthy, as intended. (`docker compose config` on the whole file
stops earlier on a pre-existing unrelated issue: `voicera_frontend/.env.local` is not in the repo.)

**Caveat on the deploy docs.** `deploy/OPERATIONS.md` and `deploy/DEPLOYMENT_v2.md` were updated on
disk (STT now documented as `http://nginx:8080/sttlb`, plus a balancing-verification note), but both
files are listed in `.git/info/exclude`, so those edits are untracked and will **not** be committed.
Remove them from the exclude file if the changes should ship.

**Not verified here — needs the real GPU host and checkpoint:** actual RTFx gain from
`STT_MATMUL_PRECISION=high` (S4), the `/sttlb` round-robin across `--scale stt=2` (S1 completion),
and end-to-end call quality with the new interim cadence.

### Recommended next, in order

1. **S1 completion** — set `INDIC_STT_SERVER_URL=http://nginx:8080/sttlb` and load-test `--scale stt=2`.
2. **C1** — group batches by `language_id` (~8 lines; silent quality bug).
3. **S3 + S10** — `STT_NUM_WORKERS=1`, retune `STT_BATCH_TIMEOUT_MS`, measure RTFx from `/metrics`
   before and after.
4. **S4 completion** — bf16 weights with a WER check on the production checkpoint.
5. **O8** — per-service VRAM caps so a TTS burst can no longer OOM STT.
6. **S6 / S14** — length-bucketed batching and CUDA-graph RNNT decoding.
