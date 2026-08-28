# Performance report — `indic-transcribe-core` streaming ASR

[README](README.md) · **Performance** · [Load test](LOADTEST.md) · [Setup](SETUP.md)

Generated 2026-08-28T05:47:47+00:00 from the raw run data by `bench/report.py --report` — not written by hand.

What this measures: **the configuration that actually ships**. The campaign that chose that configuration — geometry sweep, batching window, soak — is a separate document, `docs/BENCHMARKS.md`.

## 1. Summary

| | | |
|---|---|---|
| **First word** | 1866 ms p50 | from the first audio byte; 1365 ms from speech onset |
| **Word-to-word gap** | 288 ms p50 | between partials inside a turn — this is what "smooth" means |
| **Periodic pause** | 1919 ms every 22 s | decoder-state rotation; deliberate, and the alternative is a stall (§3) |
| **Tail** | 51 ms | stop talking → transcript settles |
| **Concurrent streams** | 8 | largest level where every stream still finished in about its own audio duration — §4, and `LOADTEST.md` for the curve to 60 |
| **Offline throughput** | 91x real time | whole-utterance batch decoding, for reference (§6) |
| **Languages** | 25 | not the 27 the wrapper advertises |

Everything below was generated from the raw run data by `bench/report.py`, not written by hand. A run that was not executed says `NOT MEASURED`; a cell that failed prints its error. Nothing is inferred, interpolated, or typed by hand.

## Environment and shipped configuration

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120) |
| Host | AWS `g7e.2xlarge`, 8 vCPU — the load generator shares these with the server |
| torch | 2.12.0+cu132 (CUDA 13.2) |
| torchaudio | 2.11.0+cpu — no cu132 build exists; see SETUP.md |
| NeMo | 3.0.0 |
| Model | `indic-transcribe-core`, Canary 1.2 B, 1.2214 B parameters, bf16 |
| Languages | 25 |
| Policy | AlignAtt, `alignatt_thr=8`, `waitk_lagging=1` |
| **Geometry** | **chunk 0.24 s / right 0.16 s** — a 0.40 s theoretical latency floor |
| Rotation | soft 12 s, hard 20 s of speech |

### Why this geometry, and not NeMo's default

NeMo recommends `chunk 1.0 / right 0.5`. This service ships `0.24 / 0.16`, which was chosen by
measurement, not preference. Over 43 s of continuous speech that is 95 commits at a 0.30 s
median gap against the default's 36 commits and a 3.36 s worst gap, for the same transcript
accuracy — the difference between text appearing word by word and appearing in clumps.

**It costs capacity, not accuracy:** roughly twice the GPU per stream, so roughly half the
concurrent sessions. `CORE_CHUNK_SECS=1.0 CORE_RIGHT_SECS=0.5` trades smoothness back for
capacity, and every number in §2–§5 changes if you do.

The full geometry sweep that settled this is in `docs/BENCHMARKS.md` (Run A).


## 2. Latency

One stream, real-time paced, 3 repeats per language, on the shipped configuration. Percentiles are computed over the pooled raw samples; nothing here averages a percentile.

**Time-to-first-token and time-to-first-partial are the same event on this server, so one number is reported rather than two.** AlignAtt emits a partial at the moment it commits a token, so the first token *is* the first visible word. There is no sub-word streaming layer underneath whose latency could differ.

**Two TTFP columns, because a clip's own leading silence is not the server's doing.** TTFP runs from the first audio byte — the honest number for a client, since that is when it started sending. *TTFP from speech* subtracts the silence the clip opens with. Where the two differ, the corpus is the cause, not the engine.

| language | clip | lead silence | **TTFP** | TTFP from speech | spread over 3x | gap p50 | gap p90 | gap p99 | tick p95 | tail | CER vs offline |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `hi` | `hi_medium_000.wav` | 800 ms | **1863 ms** | 1063 ms | 55.4 ms | 281 ms | 510 ms | 1420 ms | 68 ms | 51 ms | 0.1061 |
| `bn` | `bn_medium_000.wav` | 0 ms | **1866 ms** | 1866 ms | 0.8 ms | 300 ms | 709 ms | 719 ms | 64 ms | 51 ms | 0.0000 |
| `ta` | `ta_medium_000.wav` | 2000 ms | **3965 ms** | 1965 ms | 2.1 ms | 289 ms | 691 ms | 790 ms | 64 ms | 51 ms | 0.0000 |
| `te` | `te_medium_000.wav` | 200 ms | **1545 ms** | 1345 ms | 0.2 ms | 289 ms | 719 ms | 891 ms | 64 ms | 101 ms | 0.0163 |
| `mr` | `mr_medium_000.wav` | 700 ms | **2065 ms** | 1365 ms | 0.8 ms | 281 ms | 499 ms | 701 ms | 63 ms | 51 ms | 0.0000 |
| `en` | `en_medium_000.wav` | – | **–** | – | – | – | – | – | – | – | – |

Pooled over all 15 streams: TTFP p50 **1866 ms** / p95 3966 ms; from speech onset p50 **1365 ms** / p95 1966 ms; tail p50 51 ms.

### What each column measures

| | |
|---|---|
| **TTFP** | First audio byte sent → first partial carrying text. What a user reads as "it started working". |
| **gap p50 / p90 / p99** | Time between consecutive partials *within a turn*. This is smoothness. Gaps spanning a decoder-state rotation are excluded here and measured in §3, because pooling the two yields a p99 that describes neither. |
| **tick p95** | Server compute for one chunk. Against the 240 ms chunk period, the fraction used is what one stream costs in real time. |
| **tail** | Last audio sample sent → stream closed. How long after you stop talking the transcript settles. |
| **CER vs offline** | Character error rate against **the model's own offline transcript of the same audio**, not a human label. It asks whether streaming degraded the text, and removes the dataset's labelling noise from the comparison. A sanity check, not an accuracy claim. |

### What this establishes

**The stream is deterministic.** Across 3 repeats of identical audio the worst TTFP spread is 55.4 ms — well inside one chunk period. Repeating does not sample a distribution here so much as confirm there is barely one. That is why few repeats suffice at N=1, and why §4, where arrival timing genuinely varies, is treated differently.

**Per-chunk compute is stable across languages and scripts:** tick p95 spans 63 ms–68 ms against a 240 ms chunk period. Where a language looks slow in the TTFP column, read its lead-silence column first.

**Streaming costs little accuracy.** CER against the model's own offline output runs 0.0000–0.1061, and is exactly 0.0000 for several languages — on those clips the streaming path reproduces the offline transcript character for character. Where it is non-zero the loss is dropped words mid-utterance, not a garbled tail; the tail figures above are uniformly ~50 ms.

## 3. The periodic pause

The most visible behaviour in the live demo: transcription stops for a moment every few seconds, then resumes. It is decoder-state rotation, it is deliberate, and this section is what it costs and why it is worth paying.

### Why it exists

`decoder_mems_list` grows one position per decode step against the decoder's 1024-position limit. A stream that never rotates does not degrade gracefully — it stalls outright and never recovers. Rotation resets the decoder before that point, at 12 s of speech (soft: waits for a ≥250 ms gap so the cut lands between words) or 20 s regardless.

### What it costs

Measured on **gap-free** speech — silence trimmed, so the soft trigger never fires early and every turn runs to the hard cap. That is both the worse case and the realistic one: a person talking steadily. 90 s per run, 3 repeats.

| arm | rotations/min | s between | steady gap p50 | p90 | p99 | **boundary gap p50** | boundary max | TTFP | tail | partials/min |
|---|---|---|---|---|---|---|---|---|---|---|
| `shipped` | 2.67 | 22.5 s | 280 ms | 509 ms | 2278 ms | **1919 ms** | 2626 ms | 1565 ms | 102 ms | 148.0 |

On the shipped configuration a rotation lands roughly every **22 s** and costs **1919 ms** at p50 (worst 2626 ms), against a steady-state gap of 280 ms. Between rotations nothing else changes: 148 partials per minute keep arriving.

### The cause, measured on both sides of the wire

The server records how long each turn took to produce its first word. Turn 0 is a cold start with no prior state; every later turn follows a rotation. If the two are the same size, the pause is time-to-first-partial being re-paid — not decoder reload, not GPU work, not the model deliberating.

| | |
|---|---|
| cold start (turn 0), p50 | 1560 ms over 3 samples |
| rotation warm-up (turn > 0), p50 | 1950 ms over 12 samples |
| rotation warm-up, max | 2660 ms |
| **ratio** | **1.250** |

A ratio near 1.0 settles it: **the pause is a cold start, re-paid.** Rotation builds a new session with an empty audio window, and AlignAtt cannot commit until that window refills — `usable - attended - 1 >= alignatt_thr` is unsatisfiable while `usable` is three frames. Every rotation therefore pays time-to-first-partial again from scratch.

### The fix that was built, measured, and rejected

Rotation exists to reset the decoder. The audio window is bounded by construction and was never the problem — so carry the window across the boundary, reset only the decoder, and seed it with the last K emitted tokens so the model continues rather than re-transcribes.

**The mechanism worked**: a rotated turn committed in 0.37 s against 2.17 s cold. **End to end it was a regression in every configuration tried**, losing 15% of the transcript and producing a *worse* worst-case gap than carrying nothing. K = 24, 64 and 128 gave byte-identical results, which rules out the carried text as the lever — it is the carried audio. A decoder handed a stretch it has already transcribed predicts EOS on a context that reads as a finished utterance, and stops producing.

It ships behind `CORE_SEAMLESS_ROTATION=1`, **off by default**.

### The honest summary

A 1919 ms hiccup every 22 s is the price of a stream that does not stall. It is a real cost, it is visible to users, and no configuration measured so far removes it without costing more than it saves.

## 4. Concurrency and capacity

Real-time paced, staggered arrivals, open-loop timing from a schedule fixed **before** the run. That last point is not pedantry: closing the timing loop is how coordinated omission hides a stall — a server that freezes stops receiving sends during exactly the slow window, so the samples that would have been slow are never taken and p99 looks healthy.

This host runs the load generator and the server on the same 8 vCPUs, so any client that fell more than 100 ms behind its own schedule is counted as `client_bound` and excluded from server-side aggregates.

| N | repeats | streams completed | errors | **finished late by** | **drift p95** | TTFP p50 | TTFP p95 | sess/tick |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 3 | 0 | **0%** | **65 ms** | 1847 ms | 1847 ms | 1.00 |
| 4 | 3 | 12 | 0 | **0%** | **90 ms** | 1859 ms | 2766 ms | 1.12 |
| 8 | 6 | 48 | 0 | **4%** | **178 ms** | 1927 ms | 2908 ms | 2.44 |
| 10 | 3 | 30 | 0 | **12%** | **375 ms** | 1849 ms | 3100 ms | 4.44 |
| 12 | 3 | 36 | 0 | **25%** | **851 ms** | 1942 ms | 3303 ms | 4.68 |
| 14 | 3 | 42 | 0 | **38%** | **1385 ms** | 2010 ms | 3602 ms | 5.29 |
| 16 | 6 | 96 | 0 | **50%** | **2525 ms** | 2070 ms | 3909 ms | 6.23 |
| 24 | 3 | 72 | 0 | **69%** | **4853 ms** | 2442 ms | 4940 ms | 9.41 |

**Every stream at every level completed — no errors anywhere in the sweep.** That is worth stating plainly because it was not true until recently: this sweep previously could not be completed at all, because the service crashed at 16 streams and above on every attempt. See §7.

Head-of-line blocking suspected at: N=8, N=16, N=24.

### Reading the capacity number

**`finished late by` is the capacity metric, not TTFP.** It is how much longer a stream took than the audio it carried: 0% is exactly real time, 50% means a ten-second clip took fifteen seconds. It is taken from the *worst* audio-length class at each level, because short streams degrade first and an average across classes hides the population that binds capacity.

**The honest ceiling is N = 8.** By N = 10 the slowest class is already past it, so the service is no longer real time there however healthy the TTFP column looks.

Note the shape of the TTFP column: it degrades far more gently than the other two. A reader who sizes capacity from latency alone would put the ceiling two to three times too high. TTFP stays tolerable well past the point where the server has stopped keeping up, because the backlog shows up as *drift*, not as a slow first word. `LOADTEST.md` takes this to 60 streams, where the gap between the two is starkest.

### What limits it

**The decoder is a Python loop over sessions.** `_decode_one` runs serially, so decoder work per chunk period scales with N no matter how sessions are grouped. Run C showed this directly: widening the batch-formation window lifts sessions per tick from 1.00 to 2.89, yet the per-tick budget *rises*, because the same serial decoder work is concentrated into fewer, longer ticks. Grouping recovers the encoder term only. §5 shows the same thing from the GPU side: encode time is flat under load, decode time is not.

Batching the decoder is the one change that would lift this ceiling, and it is not shipped — its correctness gate (byte-identical transcripts for eight staggered sessions) was never run, and silent corruption is the failure mode.

**AlignAtt re-encodes the whole buffer every chunk**, roughly a twelvefold multiplier on encoder work per second of audio versus offline decoding. §6 measures that ceiling. Smaller chunks make it worse: that is the real price of word-by-word output, and it is capacity, not accuracy.

### One artefact worth knowing about

The **first stream served after a container restart** measured a tick p95 of 642 ms against the ~63 ms every later stream measured — a tenfold, one-off warm-up on a server whose `/health` was already returning 200. The startup warm-up does not cover this path. It is excluded from the medians above and recorded here rather than averaged away, because it is a real cost a real user pays after a restart.

### Why the earlier concurrency numbers do not apply

The first campaign measured concurrency at `chunk 0.96 / right 0.48`, which is **not what ships**. Capacity does not transfer between geometries — the shipped one spends roughly twice the GPU per stream — so those numbers would overstate this service by about a factor of two. Kept because they are real, labelled because they describe a different configuration.

| N | TTFP p50 | errors |
|---|---|---|
| 1 | 2446 ms | 0 |
| 8 | 2388 ms | 0 |
| 16 | 2491 ms | 0 |
| 24 | 2808 ms | 0 |
| 32 | 2930 ms | 0 |

## 5. GPU usage

Sampled at 10 Hz around the same streams that produced §4 — 2161 samples across the cells that completed cleanly. Sampling GPU counters in a separate pass would describe a differently loaded machine.

| N | util % p50 | util % max | VRAM used MB | SM clock MHz | power W p50 | power W max | temp °C max | encode ms | decode ms | tick p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | 20 | 20252 | 2325 | 98 | 104 | 39 | 64.0 | 25.4 | 642 |
| 1 | 0 | 22 | 20254 | 2325 | 98 | 103 | 38 | 22.8 | 19.6 | 63 |
| 1 | 5 | 18 | 20254 | 2325 | 98 | 103 | 38 | 23.0 | 19.7 | 63 |
| 4 | 18 | 37 | 20258 | 2325 | 118 | 134 | 40 | 23.8 | 20.9 | 63 |
| 4 | 15 | 35 | 20266 | 2325 | 118 | 134 | 40 | 23.2 | 20.6 | 64 |
| 4 | 22 | 36 | 20266 | 2325 | 119 | 137 | 40 | 22.7 | 18.3 | 63 |
| 8 | 31 | 38 | 20272 | 2325 | 135 | 151 | 41 | 27.3 | 50.4 | 161 |
| 8 | 32 | 38 | 20282 | 2325 | 135 | 150 | 42 | 24.7 | 47.0 | 157 |
| 8 | 26 | 39 | 20284 | 2325 | 134 | 149 | 41 | 23.9 | 45.4 | 136 |
| 16 | 34 | 53 | 20284 | 2325 | 145 | 176 | 44 | 27.5 | 110.2 | 421 |
| 16 | 34 | 52 | 20288 | 2325 | 147 | 174 | 45 | 25.3 | 110.8 | 444 |
| 16 | 34 | 52 | 20294 | 2325 | 146 | 176 | 45 | 25.2 | 106.8 | 434 |
| 24 | 33 | 53 | 20294 | 2325 | 143 | 168 | 44 | 28.0 | 165.5 | 591 |
| 24 | 33 | 54 | 20294 | 2325 | 142 | 169 | 44 | 25.7 | 151.1 | 608 |
| 24 | 34 | 52 | 20296 | 2325 | 141 | 171 | 45 | 25.4 | 140.0 | 600 |

The first `N = 1` row is the first stream served after a container restart and carries a one-off warm-up — a 626 ms tick and a 62.8 ms encode against the ~63 ms and ~23 ms every later row shows. It is left in rather than dropped, because it is a real cost a real user pays after a restart (§4).

### What these numbers do and do not show

**`utilization.gpu` is not utilisation in the sense people mean.** `nvidia-smi` reports the fraction of time at least one kernel was resident, not how much of the GPU was doing work. One small kernel occupying a single SM reads as 100%. For a decoder issuing on the order of a couple of thousand tiny launches per session per tick from Python, that figure is actively misleading and must not be read as saturation.

**The counters that would settle it are unavailable on this host.** `SM_ACTIVE`, `SM_OCCUPANCY`, `DRAM_ACTIVE` and `PIPE_TENSOR_ACTIVE` need DCGM, which is not installed; they are `NOT MEASURED` rather than estimated. The standing hypothesis — that the decoder is launch-bound, high kernel residency against low occupancy, for which CUDA graphs would be the right lever — is therefore **neither confirmed nor refuted here**, and the optimisation that depends on it has deliberately not been attempted. Guessing would have been cheaper and worse.

**What the encode/decode split does show**, and it is the useful part: **encode time is flat under load while decode time grows with it.** Per tick, encode moves barely at all from one stream to eight while decode roughly doubles or worse. That is exactly the signature of the serial Python decoder loop identified in §4 — the encoder batches across sessions, the decoder does not — and it is the mechanism behind the capacity limit, independent of the stability limit that binds first.

**Power and clocks say the same thing from the other side.** Draw rises modestly with load and the SM clock never moves off its ceiling: this GPU is nowhere near thermally or electrically limited by this workload. Whatever is constraining the service, it is not the silicon.

VRAM is essentially flat across load — the allocator reserves once and each additional session adds little.

## 6. Offline throughput, for reference

Whole-utterance batch decoding: the ceiling streaming's ~12x re-encode amplification spends against. RTFx higher is better; RTF lower is better. Pooled over total audio and total time, **not** a mean of per-clip ratios.

| batch | mix | clips | audio s | process s | RTFx ↑ | RTF ↓ |
|---|---|---|---|---|---|---|
| 1 | uniform_medium | 6 | 83.4 | 3.691 | 22.6 | 0.04425 |
| 1 | mixed | 11 | 105.06 | 4.813 | 21.83 | 0.04582 |
| 2 | uniform_medium | 6 | 83.4 | 2.176 | 38.33 | 0.02609 |
| 2 | mixed | 11 | 105.06 | 3.515 | 29.89 | 0.03345 |
| 4 | uniform_medium | 6 | 83.4 | 1.721 | 48.45 | 0.02064 |
| 4 | mixed | 11 | 105.06 | 2.004 | 52.44 | 0.01907 |
| 8 | uniform_medium | 6 | 83.4 | 0.972 | 85.83 | 0.01165 |
| 8 | mixed | 11 | 105.06 | 1.813 | 57.94 | 0.01726 |
| 16 | uniform_medium | 6 | 83.4 | 0.948 | 87.96 | 0.01137 |
| 16 | mixed | 11 | 105.06 | 1.149 | 91.4 | 0.01094 |

## 7. Known defects, and one that is now fixed

### Fixed: the CUDA illegal memory access under concurrent load

For two campaigns this was the service's defining defect: a streaming session could fault the
CUDA context with `cudaErrorIllegalAddress`, and because a poisoned context is process-wide, one
bad session took every other session down with it. It is now understood and fixed.

**What it was not.** Two earlier hypotheses were tested and both are wrong, and a third
assumption was wrong as well:

* *Decoder memory outgrowing the 1024-position limit.* Falsified: 75 s of gap-free speech, which
  drives every turn to the hard rotation cap, produced no fault.
* *Language mismatch — speaking one language while the selector declares another.* Falsified:
  faults occurred with correctly declared, supported languages and matching audio.
* *A memory leak.* There is none. Allocated memory held flat at ~2.3 GiB and reserved at
  ~9.5 GiB right up to the moment of the fault, with 77 GB free on the device. The soak run
  agrees. Nothing was leaking.

**How it was found.** The fault was made to reproduce in about thirty seconds in-process,
which turned an intermittent production incident into a bisectable experiment. The bisection is
the argument:

| Condition | Result |
|---|---|
| 1 session | clean |
| 16 sessions, **identical** audio | clean — 4576 decoder steps |
| 8 sessions, **different** audio | **fault**, always at the same tick |
| 16 sessions, different audio | **fault**, always at the same tick |
| Same run, `float32` | clean, 400 ticks |
| Same run, `float16` | **fault, at the identical tick as bfloat16** |
| Same run, caching allocator disabled | clean |
| Same run, `expandable_segments:True` | clean |

Four things fall out of that table. It is **deterministic**, so it is not a race. It needs
**several sessions decoding different audio in one tick**, so it is not any single input.
`float16` and `bfloat16` fail at the *identical* point while `float32` never does — and those
two share an allocation layout that `float32` does not, which points at memory layout rather
than at arithmetic. And it disappears when the caching allocator is taken out of the picture.

Every index the engine computes was checked at the moment of the fault and all were in range:
context lengths of 10–15 against a 512-wide buffer, positional indexes of 0–14 against a
1024-row table. The failing call is a cross-attention GEMM whose shapes are tiny and which had
already succeeded three times in the same tick for other sessions. It is a victim, not the
culprit: the reported `CUBLAS_STATUS_INTERNAL_ERROR` is simply the first call to notice a
context that something else had already corrupted.

**The fix.**

```
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It is set in `docker-compose.yml` and is **not optional**. The allocator grows a segment through
CUDA's virtual-memory API instead of repeatedly splitting and recombining cached blocks, and
that is enough to avoid the fault entirely.

**Verified, on the real service through the real gateway:**

| | before | after |
|---|---|---|
| 8 concurrent streams | ~10 of 24 lost | **0 errors** |
| 16 concurrent streams | 48 of 48 lost, every attempt | **0 errors** |
| 24 concurrent streams | 72 of 72 lost, every attempt | **0 errors** |
| CUDA faults during the sweep | 9 | **0** |
| Long-form output | 95 commits / 98% words / 2.38 s | **unchanged** |

The full concurrency sweep in §4 completed for the first time.

**And then re-verified on the cases that were not concurrency at all.** Three of the original
faults were single streams, so the concurrency result alone would not have settled it:

| Previously faulting scenario | Result with the fix |
|---|---|
| 90 s gap-free single stream × 3 (faulted on repeat 3, twice, in separate campaigns) | **3/3 complete** |
| the `en` latency cell (faulted twice) | **3/3 complete** |
| isolated reproduction, 16 sessions, 1000 ticks | **17,610 decoder steps, no fault** — the unfixed build faults at 156 |

Sustained load is covered below. Across every run in this section: **zero faults, zero
restarts.**

**What this is honestly not:** a root cause in the sense of naming the offending kernel. The
mitigation is verified and the mechanism is narrowed to allocator block reuse under 16-bit
compute, but which kernel writes out of bounds is still unidentified. `compute-sanitizer` would
name it and would not attach in this container. Anyone continuing should start there; the
reproduction is eight concurrent sessions on differing audio, which faults deterministically
within a few seconds at 16-bit and never at float32.

**Defence in depth remains, and should stay.** A fatal CUDA error is still classified by
`is_fatal_cuda()`, still exits the worker with code 70 so the container is replaced with a clean
context, and `/health` still turns 503 rather than reporting a healthy service that refuses every
connection. That machinery was exercised nine times during this campaign and worked every time.
It is what kept a fixed bug from being an outage while it was still unfixed.

### Fixed: the service accepted eight times the load it could serve

Separate from the crash, and easy to miss because nothing about it looks like a failure.
`CORE_MAX_SESSIONS` was 64 against a measured real-time capacity of 8. Past 8 the service
refused nothing — it accepted the work and fell behind, and the only symptom was transcripts
arriving progressively later. §4 shows why that was invisible: TTFP at 10 streams is *lower*
than at 8, while streams are already finishing 12% later than the audio they carry.

Now: the admission limit defaults to the measured capacity, a refused client is told which of
the two limits it hit, and `/metrics` carries `realtime_capacity`, `over_realtime_capacity` and
a `capacity_warning` string so an operator never has to infer it from latency.

Raising `CORE_MAX_SESSIONS` above `CORE_REALTIME_CAPACITY` is still allowed — admitting a short
burst is often better than refusing it — but the trade is now reported rather than silent. Note
that benchmarking above the ceiling requires raising it deliberately, which is the point.

**One defect fixed on the way:** a refused connection was closed *before* the WebSocket
handshake completed, which discards the close code and reason and hands the client a bare
`HTTP 403`. Being over capacity and asking for an unsupported language were indistinguishable,
and 403 describes neither. Refusals now complete the handshake, send a structured error
(`{"reason": "at_capacity", ...}`), and close with the correct code.

### T5 geometry (`0.32 / 0.24`) faults on medium clips

Reproducible, and duration-dependent — it succeeds on short clips. Its neighbours `0.48/0.24`
and `0.32/0.16` both work. Not investigated since the allocator fix; it may well be the same
bug. Avoid that geometry.

### `max_generation_length` cannot simply be raised

`initialize_aed_model_state` allocates `pred_tokens_ids` at the state's default of 256. Assigning
512 afterwards raises the loop bound without resizing the buffer, producing out-of-bounds device
writes that surface later as an illegal memory access from an unrelated kernel. The engine grows
both tensors; without that, every geometry below the baseline crashed.

### Deliberate limitations

* **No language auto-detection.** Measured top-1 accuracy is 0.047 for `bho`, 0.258 for `hi`,
  0.356 for `mai`, 0.490 for `ur` — each absorbed by a close neighbour. A wrong language yields
  a confidently wrong *script*, not an error, so the caller always states it.
* **Native script only.** `<|itn|>` and `<|romanized|>` exist in the vocabulary but are
  untrained on this checkpoint; requesting them returns fluent wrong text rather than failing.
* **`bgc` and `hne` are rejected.** The wrapper advertises them; core's vocabulary lacks them.

### Not attempted, and why

| | |
|---|---|
| CUDA-graphing the inner decode step | Its justification rests on occupancy counters this host cannot provide (§5). Attempting it would have been guessing. |
| Batching the decoder | The only change that lifts the concurrency ceiling, and the one with a real silent-corruption risk. Its gate — byte-identical transcripts for eight staggered sessions — was not run, so it is not shipped. |


### Sustained load

Short cells measure throughput; they do not certify a memory-corruption fix. These do.

| arm | duration | streams completed | errors | VRAM allocated drift | VRAM reserved drift |
|---|---|---|---|---|---|
| at capacity (8 streams) | 600 s | 288 | **0** | 0.000 GB | 0.000 GB |
| deliberately over (16 streams) | 300 s | 288 | **0** | 0.001 GB | 0.000 GB |

Drift is first-half mean against second-half mean, sampled throughout the run. At capacity it is 0.000 GB on both counters; over capacity it is 0.001 GB — one megabyte across five minutes, which is measurement noise on a 2.5 GB working set, not a trend. There is no leak, and there never was one; that hypothesis is dead on measurement rather than on argument.

The over-capacity arm matters as much as the other: 16 streams for five minutes produced **no errors and no restarts**. Past its ceiling this service falls behind, it does not fail. The backlog is visible in the session count — 11 sessions still draining when the 16-stream run ended, against 1 at 8 streams.

## 8. Method, and what this report does not claim

**Every number is read from `results/*.json` by `bench/report.py`.** Nothing is transcribed by
hand. Regenerate the document and it reproduces; a number that cannot be traced to a JSON file
is a bug in the generator, not a rounding difference.

**Percentiles come from the raw samples that produced them.** Nothing here averages a
percentile, which is not a meaningful operation. Where repeats exist, the samples are pooled
before the percentile is taken.

**Latency is measured open-loop.** Arrival times are fixed before each run and every latency is
measured against that schedule, not against when the client managed to send. Closing that loop
is how coordinated omission hides a stall.

**Accuracy here is a sanity check, not an accuracy campaign.** CER is computed against the
model's own offline transcript of the same audio, which asks whether *streaming* degraded the
text and removes the dataset's labelling noise from the comparison. It is not a WER benchmark
and must not be quoted as one.

### What is not settled

* **Six languages of twenty-five are measured** for latency, bounded by the corpus that exists.
  The remaining nineteen are `NOT MEASURED`, not assumed equivalent.
* **No p99 in this document rests on a thousand samples.** Percentiles at the tail are
  indicative.
* **Occupancy counters are unavailable** (§5), so the launch-bound hypothesis stands untested.
* **The CUDA fault is fixed and the fix is verified, but the offending kernel is not named**
  (§7). The mechanism is narrowed to allocator block reuse under 16-bit compute; that is a
  characterisation, not a culprit.
