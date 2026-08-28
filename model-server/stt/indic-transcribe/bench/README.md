# The measurement harness

Two campaigns live here, and they answer different questions.

| | Question | Runs | Document |
|---|---|---|---|
| **First** | Which configuration should ship? | A geometry, B concurrency, C batching window, D soak, E offline throughput | `docs/BENCHMARKS.md` |
| **Second** | What does the shipped configuration actually do? | F latency, G the periodic pause, H concurrency at the shipped geometry, I GPU | `REPORT.md` |

The second exists because the first measured concurrency at `chunk 0.96 / right 0.48`, and the
service ships `0.24 / 0.16`. Those numbers do not transfer — the shipped geometry spends roughly
twice the GPU per stream — so quoting Run B for the live service would overstate capacity by
about a factor of two. Run B is kept, and labelled.

## Running them

Everything targets the live server over the host network. **Not** `--network
container:core-asr`: sharing the server's network namespace looks tidier, but when the server
restarts — which it does, see the CUDA fault in REPORT.md §7 — the namespace is destroyed and
the benchmark container loses networking permanently, turning every remaining cell into an
error. That cost a full sweep here before it was noticed.

```bash
BENCH="docker run --rm --gpus all --network host \
  -v $PWD/bench:/app/bench:ro -v $PWD/results:/results -v $PWD/corpus:/corpus:ro"

$BENCH python /app/bench/latency_profile.py  --repeats 3
$BENCH python /app/bench/rotation_profile.py --repeats 3 --seconds 90 \
       --server-log /results/_runG_server.log
$BENCH python /app/bench/concurrency.py --chunk-s 0.24 --levels 1,4,8,12,16,24 \
       --gpu-sample --repeat 3 --out /results/runH_concurrency_t7.json

$BENCH python /app/bench/report.py --report      # -> REPORT.md
$BENCH python /app/bench/report.py               # -> BENCHMARKS.md
```

`rotation_profile.py` wants the server's own log to attribute per-turn warm-ups. Capture it
alongside the run:

```bash
docker logs -f --since 1s core-asr > results/_runG_server.log 2>&1 &
```

## How to run a concurrency test, and how to read it

```bash
# The admission limit defaults to the real-time capacity (8). To measure BEYOND it you must
# raise it deliberately -- otherwise the extra streams are refused, which is the point.
CORE_MAX_SESSIONS=32 docker compose up -d core-asr

BENCH="docker run --rm --gpus all --network host \
  -v $PWD/bench:/app/bench:ro -v $PWD/results:/results -v $PWD/corpus:/corpus:ro"

$BENCH python /app/bench/concurrency.py \
  --url ws://localhost:9002/v1/asr/ws --base http://localhost:9002 \
  --chunk-s 0.24 --levels 8,10,12,14,16 --repeat 3 --gpu-sample \
  --out /results/runH_ceiling.json
```

`--chunk-s` must match the server's geometry: it paces arrivals, and a mismatch measures a
different experiment. Put the levels around the capacity you expect, not far above it — the
interesting behaviour is at the crossing, and everything past it looks the same.

### The full load test

`LOADTEST.md` is generated from three runs. Reproduce it with:

```bash
CORE_MAX_SESSIONS=96 docker compose up -d core-asr     # admit more than capacity, on purpose

# the curve: 1 -> 60 streams, mixed audio sizes, 3 repeats
$BENCH python /app/bench/concurrency.py --url ws://localhost:9002/v1/asr/ws \
  --base http://localhost:9002 --chunk-s 0.24 --levels 1,4,8,12,16,24,32,48,60 \
  --repeat 3 --gpu-sample --buckets short,medium,long \
  --out /results/runL1_load_sweep.json

# arrival shape, at a fixed level (merge the three into runL2_arrival.json)
for a in stagger poisson sync; do
  $BENCH python /app/bench/concurrency.py --url ws://localhost:9002/v1/asr/ws \
    --base http://localhost:9002 --chunk-s 0.24 --levels 24 --repeat 3 --arrival $a \
    --buckets short,medium,long --out /results/_arr_$a.json
done

# sustained, well past capacity
$BENCH python /app/bench/soak.py --url ws://localhost:9002/v1/asr/ws \
  --base http://localhost:9002 --streams 32 --seconds 420 \
  --out /results/runL3_soak_n32.json

$BENCH python /app/bench/report.py --loadtest      # -> results/LOADTEST.md
```

`--buckets short,medium,long` matters: the buckets are interleaved so every level carries all
three sizes, and short streams are the ones that degrade first. Testing on medium clips alone
hides the effect that binds capacity.

### Reading the result

**`normalized_latency_by_bucket` is the capacity metric.** It is `e2e / audio_duration`: 1.00
means the stream finished in exactly its own audio duration, 1.70 means it took 70% longer.
Read the **worst** bucket, not the average — short streams degrade first and an average across
lengths hides them.

**Do not size capacity from TTFP.** It is the most tempting column and the most misleading one.
Measured on this service:

| N | worst-bucket normalized latency | TTFP p50 | verdict |
|---|---|---|---|
| 8 | 1.04 | 1927 ms | keeping up |
| 10 | **1.12** | **1849 ms** | **already slipping — and TTFP got BETTER** |
| 16 | 1.50 | 2070 ms | far behind |

TTFP at 10 streams is *lower* than at 8 while streams are finishing 12% late. Anyone watching
latency would have concluded it was fine. The backlog shows up as drift, not as a slow first
word.

**The two columns that do reveal it:**

| Column | Meaning |
|---|---|
| `normalized_latency_by_bucket` | 1.00 = real time. Past ~1.10 you are over capacity |
| `delta_lag_p95_ms` | the drift itself. 174 ms at N=8, 3149 ms at N=16, 15325 ms at N=60 |

**`hol_suspected`** firing means short streams scored *worse* than long ones — they are stuck
behind long ones. It is the first thing to break: short clips degrade at N=10 while medium
clips are still at 1.00.

**`client_bound` rows measure the harness, not the server.** This host runs the load generator
and the server on the same 8 vCPUs.

**`/metrics` says it directly too.** `over_realtime_capacity` and `capacity_warning` appear once
active sessions exceed `realtime_capacity`, so an operator does not have to infer it.

## The rules these scripts follow

**Real-time pacing is mandatory.** Feeding faster than 1× measures batch throughput and calls it
streaming.

**Timing is open-loop.** Arrival times are fixed before the run and every latency is measured
against that schedule, not against when the client managed to send. Closing the loop is how
coordinated omission hides a stall: a server that freezes stops receiving sends during exactly
the slow window, so the samples that would have been slow are never taken.

**A client that falls behind is not a server result.** This host runs the load generator and the
server on the same 8 vCPUs. Any stream more than 100 ms behind its own schedule is flagged
`client_bound` and excluded from server-side aggregates.

**Percentiles come from raw samples.** Repeats are pooled before the percentile is taken.
Nothing averages a percentile.

**A failed cell is a result.** Scripts record the error and continue. They used to die instead,
and twice discarded a completed campaign when the server hit its intermittent CUDA fault.

**Nothing is typed into a document by hand.** `report.py` reads `results/*.json`. A run that was
not executed prints `NOT MEASURED`; it is never inferred or interpolated.

**Accuracy here is a sanity check.** CER is computed against the model's own offline transcript
of the same audio, which asks whether streaming degraded the text and removes the dataset's
labelling noise. It is not a WER benchmark.

## Files

| | |
|---|---|
| `ws_client.py` | One streaming client, real-time paced. The primitive everything else is built from. |
| `metrics_lib.py` | Metric definitions, written out rather than imported because most of them are argued about. |
| `latency_profile.py` | Run F — TTFP, gaps, tail, per language, with repeats. |
| `rotation_profile.py` | Run G — the periodic pause: how often, how long, and why. |
| `concurrency.py` | Runs B and H — capacity, with `--gpu-sample` for Run I. |
| `soak.py` | Run D — 300 s at N=8, watching for leaks. |
| `gpu_sampler.py` | 10 Hz `nvidia-smi` sampling, and an honest note on what it cannot show. |
| `report.py` | Generates both documents. |
| `test_longform.py` `test_turns.py` `test_continuous.py` `test_fatal_path.py` | Regression gates, chained by `../verify.sh`. |
