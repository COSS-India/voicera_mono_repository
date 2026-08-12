# Test and benchmark suite

`orpheus_test.py` verifies every endpoint of a running server and measures its
latency and throughput. It requires only `requests`.

```bash
pip install requests
python3 tests/orpheus_test.py --suite all
```

The process exits non-zero if any functional check fails, so it can be used as a
deployment gate.

## Options

| Option | Default | Description |
|---|---|---|
| `--base-url` | `http://localhost:9000` | Server to test |
| `--suite` | `all` | `api`, `batch`, `live`, `concurrency`, `latency`, or `all` |
| `--concurrency` | `1,2,4,8,16,32,64` | Levels for the concurrency and latency suites |
| `--duration` | `25` | Seconds per sustained-load level |
| `--json` | — | Write all results, including captured response schemas, to this file |

Set `--concurrency` to span your configured `ORPHEUS_MAX_NUM_SEQS`. Include a level
above it to observe admission queueing.

```bash
# Fast functional check only
python3 tests/orpheus_test.py --suite api

# Full run against a server configured for max_num_seqs=256
python3 tests/orpheus_test.py --suite all \
    --concurrency 1,4,16,32,64,128,256 --duration 20 --json results.json
```

## Measurement basis

Performance suites request `response_format=pcm`, where every byte maps to a known
frame: **4096 bytes = one 85.333 ms frame at 24 kHz**. The first and last chunks of
a stream carry two frames each, because the SNAC decoder widens its emit window at
both ends. Frame arrival timestamps drive every latency statistic.

| Metric | Definition |
|---|---|
| `ttfa_ms` | Time from request send to the first audio byte |
| `rtf` | Real-time factor: wall time to produce the stream ÷ duration of the audio. Below 1.0 is faster than real time. |
| `jitter_p99_ms` | 99th percentile of the gap between consecutive frame arrivals |
| `late frame` | A frame arriving after `first_audio + 100 ms + frame_index × 85.333 ms` — the moment a real-time player needs it |
| `clean_pct` | Percentage of streams in which **no** frame was late |
| `agg_audio_s_per_s` | Total audio produced ÷ wall clock. The server's delivered throughput. |

`clean_pct` and `ttfa` measure different failures. A queued request may start late
(high `ttfa_p95`) yet play back perfectly once it starts (`clean_pct` 100%). Read
them together: `ttfa` is how long the caller waits for audio, `clean_pct` is whether
that audio stutters.

## Suites

### `api`

Functional verification of every endpoint, plus the request/response schemas.

- Catalog endpoints return the expected roster (22 languages, 40 speakers, 12 styles)
- `POST /v1/tts`, `GET /v1/tts/stream`, and the WebSocket produce valid audio
- WebSocket head and tail frames are double-width, confirming the end-of-stream flush
- Every `response_format` × `stream_format` combination behaves as documented
- Chunked formats deliver the first byte early; buffered formats deliver it at the end
- Eleven validation cases return the correct status and an OpenAI error envelope

With `--json`, live response bodies for each endpoint are captured under `schemas`.

### `batch` — non-live inference

One complete file per request, across three text lengths and all five formats.
Reports wall time, server-reported RTF, and encoded size. Also verifies the WAV
container declares a truthful duration.

This is the mode to measure when clients download audio before playing it.

### `live` — streaming inference

Single-stream latency across three text lengths: TTFA percentiles, RTF, inter-frame
jitter, worst gap, and how many streams played without a late frame.

TTFA should be roughly constant regardless of output length. If it scales with text
length, streaming is broken somewhere between the encoder and the client.

### `concurrency` — continuous batching

Two measurements.

**Burst sweep.** N streams start simultaneously at each level. Shows aggregate
throughput rising with concurrency until the GPU saturates, and TTFA degrading as
the batch widens. The level with peak `agg_audio_s_per_s` is the server's capacity;
past it, throughput falls while latency climbs.

**Sequential versus concurrent.** The same eight requests are run one at a time,
then all at once. The ratio is the continuous-batching speedup. Per-request latency
increases slightly while total wall time drops sharply — that is the trade batching
makes.

### `latency` — delayed latency under load

Two arrival patterns, because they stress different things.

**Sustained (closed loop).** N workers loop for `--duration` seconds, each starting
a new request as soon as its previous one finishes. Demand never stops. Admission
queueing appears as `ttfa_p95` rising far above `ttfa_p50` and
`started_within_500ms_pct` falling.

**Fixed arrival rate (open loop).** Requests are issued on a clock at 2, 5, 10 and
20 requests per second regardless of whether earlier ones have finished. This is how
production traffic actually arrives. If the rate exceeds capacity the queue grows
without bound and `ttfa_p95` climbs steeply — the clearest signal that a deployment
is under-provisioned for its offered load.

## Interpreting the results

**Pick `ORPHEUS_MAX_NUM_SEQS` from the burst sweep.** The highest level where
`clean_pct` is 100 is the most concurrent live streams the GPU serves smoothly. The
level with peak `agg_audio_s_per_s` is the throughput ceiling. For live delivery
choose the former; for batch work choose the latter.

**Watch for throughput falling at high concurrency.** Once admitted work exceeds
what the GPU can retire, queued requests idle while the wall clock runs, so
`agg_audio_s_per_s` drops even as more requests are in flight.

**Queueing is silent.** The server returns no signal when a request waits for
admission — the caller only sees latency. If `ttfa_p95` is high while `clean_pct`
stays at 100, requests are queueing rather than degrading. Set client-side timeouts
accordingly.

**`rtf_worst` above 1.0 under load is expected** when it includes queue wait; the
stream itself may still be smooth. Compare against `clean_pct` before concluding the
server is too slow.

## Notes

- Run against an idle server. Concurrent traffic from other clients will distort
  every throughput number.
- The suite generates real audio and will occupy the GPU for the duration of a full
  run — several minutes depending on `--concurrency` and `--duration`.
- All timings are client-observed and therefore include network transit. On
  localhost this is negligible; across a network, compare against the server's own
  `X-TTFA-Ms` and `X-RTF` response headers.
