# Load test — `indic-transcribe-core` streaming ASR

[README](README.md) · [Performance](REPORT.md) · **Load test** · [Setup](SETUP.md)

Generated 2026-08-28T05:47:48+00:00 by `bench/report.py --loadtest` from the raw run data. Every number is read from a file; nothing is typed in by hand.

Companion to `REPORT.md`, which measures what a *single* stream experiences. This measures what happens when many arrive at once — a different question, with different failure modes: queueing, head-of-line blocking, and a periodic pause that behaves nothing like it does at rest.

## 1. Verdict

| | | |
|---|---|---|
| **Real-time capacity** | **8 concurrent streams** | every class of audio finished within 5% of its own duration. This is the number to size on. |
| **Tested to** | 60 concurrent streams | 7.5x the capacity; streams finished 70% later than their own audio |
| **Stability** | 615 streams, 0 errors | across every level from 1 to 60. Nothing crashed, nothing was refused. |
| **Harness** | 0 client-bound streams | the load generator kept up, so these are the server's numbers |

### What over-capacity actually looks like

It does not look like failure. At 60 streams — 7.5x the capacity — every stream still completed and every transcript was still produced. What changed is *when*:

| | at capacity (8) | at 60 streams | change |
|---|---|---|---|
| First word (TTFP p50) | 1926 ms | 3881 ms | only 2.0x — **this is the trap** |
| First word (TTFP p95) | 2899 ms | 19925 ms | 7x — the tail is where it shows |
| Drift behind real time | 174 ms | 15325 ms | **88x** — the honest signal |
| Word-to-word gap | 287 ms | 825 ms | 2.9x — what a reader perceives |

**The median first word roughly doubles while the service carries 7.5x its capacity — and the drift behind real time grows 88x over the same span.** A dashboard watching TTFP p50 would show almost nothing wrong. The damage is in the p95 and in the drift: the transcript keeps arriving, just later and later, and no error is ever raised. That is the single most important thing in this document.

## Method

| | |
|---|---|
| Geometry | the shipped one: chunk 0.24 s / right 0.16 s |
| Audio | short (~4 s), medium (~14 s) and long (~34 s) clips, interleaved so every level carries all three |
| Pacing | real time. Feeding faster measures batch throughput and calls it streaming |
| Timing | open loop — arrival times fixed *before* the run, latency measured against that schedule |
| Repeats | 3 per level; tables show the median across them |
| Admission | `CORE_MAX_SESSIONS` raised to 96 for the test. The shipped default is lower on purpose |

**Open-loop timing is not pedantry.** Closing the loop is how coordinated omission hides a
stall: a server that freezes stops receiving sends during exactly the slow window, so the
samples that would have been slow are never taken, and p99 comes out looking healthy. Every
latency here is measured from a schedule fixed in advance.

**The load generator shares this box with the server** — 8 vCPUs between them. Any stream that
fell more than 100 ms behind its own send schedule is counted `client-bound` and excluded from
server-side aggregates. That count is reported in every table; it was zero throughout, which is
what makes these the server's numbers rather than the harness's.

### The terms

| Term | Definition |
|---|---|
| **TTFP** (time to first partial) | first audio byte sent → first partial carrying text |
| **TTFT** (time to first token) | **the same event.** AlignAtt emits a partial when it commits a token, so the first token *is* the first visible word. There is no sub-word layer beneath it, so one number is reported rather than two invented ones |
| **TTFP from speech** | TTFP minus the clip's own leading silence — the server's contribution alone |
| **Inter-word gap** | time between consecutive commits *within* a turn. This is smoothness |
| **Rotation pause** | the gap that spans a decoder-state reset. Measured separately, because pooling it with the ordinary gaps gives a p99 that describes neither |
| **Drift** (`delta_lag`) | how far behind real time a stream ran |
| **Finished late by** | how much longer a stream took than the audio it carried. 0% = real time, 70% = a 10 s clip took 17 s. Reported for the *worst* audio-length class, since short streams degrade first |
| **sess/tick** | how many sessions the batcher served per tick — the batching efficiency |


## 2. The concurrency curve

| N | runs | streams | err | **finished late by** | **drift p95** | TTFP p50 | TTFP p95 | gap p50 | gap p99 | sess/tick |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | 3 | 3 | 0 | **0%** | **73 ms** | 1847 ms | 1847 ms | 288 ms | 1200 ms | 1.0 |
| **4** | 3 | 12 | 0 | **1%** | **105 ms** | 1864 ms | 2068 ms | 276 ms | 814 ms | 1.1 |
| **8** | 3 | 24 | 0 | **5%** | **174 ms** | 1926 ms | 2899 ms | 287 ms | 1189 ms | 2.7 |
| **12** | 3 | 36 | 0 | **26%** | **884 ms** | 1914 ms | 3271 ms | 288 ms | 1200 ms | 5.0 |
| **16** | 3 | 48 | 0 | **57%** | **3149 ms** | 2100 ms | 3974 ms | 348 ms | 1270 ms | 6.5 |
| **24** | 3 | 72 | 0 | **69%** | **8206 ms** | 2501 ms | 3609 ms | 528 ms | 1831 ms | 9.0 |
| **32** | 3 | 96 | 0 | **70%** | **10301 ms** | 3088 ms | 6769 ms | 642 ms | 2488 ms | 11.3 |
| **48** | 3 | 144 | 0 | **69%** | **14184 ms** | 3508 ms | 14619 ms | 824 ms | 2771 ms | 13.3 |
| **60** | 3 | 180 | 0 | **70%** | **15325 ms** | 3881 ms | 19925 ms | 825 ms | 2783 ms | 12.4 |

**615 streams across 27 cells, 0 errors, 0 client-bound.** Every level from 1 to 60 completed. Nothing crashed, nothing was refused, and the load generator kept up throughout — so every number above is the server's, not the harness's.

### Where it stops being real time

**Two columns decide this, and both are plain outcomes rather than internal counters.**

| Column | What it means |
|---|---|
| **finished late by** | how much longer a stream took than the audio it carried. 0% is exactly real time; 70% means a 10-second clip took 17 seconds to transcribe. Taken from the *worst* audio-length class at each level, because short streams degrade first and an average across classes hides them |
| **drift p95** | how far behind real time the stream actually ran |

**The ceiling is 8 concurrent streams.** Up to there the slowest class of stream finishes within 10% of its own duration and drift stays in the low hundreds of milliseconds. Past it both climb steeply and never recover: the service does not fail, it falls behind, and keeps falling behind for as long as the load lasts.

**TTFP is not one of the deciding columns, and that is the trap in this table.** Read TTFP alone and the service looks healthy far past the point where it has stopped keeping up — the first word still arrives in about two seconds while the *rest* of the transcript falls minutes behind. Time-to-first-partial measures how quickly you were admitted, not whether the service can sustain you. Sizing from it overshoots by several times.

## 3. Latency in detail

| N | TTFP p50 | TTFP p95 | TTFP p99 | p95 ÷ p50 | drift p95 | gap p50 | gap p99 |
|---|---|---|---|---|---|---|---|
| **1** | 1847 ms | 1847 ms | 1847 ms | **1.0x** | 73 ms | 288 ms | 1200 ms |
| **4** | 1864 ms | 2068 ms | 2068 ms | **1.1x** | 105 ms | 276 ms | 814 ms |
| **8** | 1926 ms | 2899 ms | 2899 ms | **1.5x** | 174 ms | 287 ms | 1189 ms |
| **12** | 1914 ms | 3271 ms | 3271 ms | **1.7x** | 884 ms | 288 ms | 1200 ms |
| **16** | 2100 ms | 3974 ms | 3974 ms | **1.9x** | 3149 ms | 348 ms | 1270 ms |
| **24** | 2501 ms | 3609 ms | 5628 ms | **1.4x** | 8206 ms | 528 ms | 1831 ms |
| **32** | 3088 ms | 6769 ms | 7112 ms | **2.2x** | 10301 ms | 642 ms | 2488 ms |
| **48** | 3508 ms | 14619 ms | 16818 ms | **4.2x** | 14184 ms | 824 ms | 2771 ms |
| **60** | 3881 ms | 19925 ms | 20515 ms | **5.1x** | 15325 ms | 825 ms | 2783 ms |

**TTFT and TTFP are one number on this server, not two.** AlignAtt emits a partial at the moment it commits a token, so the first token is the first visible word. Reporting them separately would mean inventing a distinction the architecture does not have.

**Read the `p95 ÷ p50` column.** At low load the two are close: everyone gets the same service. As load rises the ratio opens up — the median stays respectable while the unlucky tail waits many times longer. Averages, and medians, conceal this completely.

**The gap columns are what a person actually perceives.** TTFP is a one-off; the inter-word gap is every word after it. When it triples, the transcript stops reading as live typing and starts reading as intermittent bursts — the same total text, a different experience.

## 4. Audio length: short streams suffer first

Every level carries a mix of short (~4 s), medium (~14 s) and long (~34 s) clips, interleaved deterministically so each level sees all three. `nl` is normalized latency — 1.00 means the stream finished in exactly its own audio duration.

| N | **short** nl / TTFP | **medium** nl / TTFP | **long** nl / TTFP |
|---|---|---|---|
| **1** | – | – | 1.00 / 1847 ms |
| **4** | 1.01 / 2068 ms | 1.00 / 1855 ms | 1.00 / 1848 ms |
| **8** | 1.07 / 2171 ms | 1.00 / 1926 ms | 1.00 / 1848 ms |
| **12** | 1.27 / 1655 ms | 1.01 / 2047 ms | 1.00 / 1914 ms |
| **16** | 1.58 / 1762 ms | 1.21 / 2346 ms | 1.00 / 2100 ms |
| **24** | 1.72 / 2354 ms | 1.21 / 2924 ms | 1.09 / 2483 ms |
| **32** | 1.73 / 2693 ms | 1.21 / 3404 ms | 1.09 / 3090 ms |
| **48** | 1.72 / 2874 ms | 1.21 / 3554 ms | 1.09 / 3564 ms |
| **60** | 1.72 / 3062 ms | 1.21 / 3881 ms | 1.09 / 4763 ms |

Head-of-line blocking detected at: **N=4, N=8, N=12, N=16, N=24, N=32, N=48, N=60**.

**Short streams degrade first and worst, and that is backwards from what matters.** A four-second utterance has no slack: it is scheduled behind whatever long streams are already mid-flight, and a fixed queueing delay is a far larger fraction of four seconds than of thirty-four. So the quick interactions — the ones where a user is waiting on a reply — are the ones that feel the load first, while the long dictations they are queued behind still look fine.

If a deployment mixes short and long audio, size it by the short streams.

## 5. The periodic pause, under load

The visible flush: transcription stops for a moment, then resumes. It is decoder-state rotation, it is deliberate, and `REPORT.md` §3 explains why it exists and proves the cause. The question here is different — **does contention make it worse?**

| N | rotations/stream | pauses seen | **pause p50** | pause max | steady gap p50 | **pause ÷ steady** |
|---|---|---|---|---|---|---|
| **1** | 2.00 | 6 | **2337 ms** | 2337 ms | 288 ms | **8.1x** |
| **4** | 1.25 | 15 | **2004 ms** | 3770 ms | 276 ms | **7.3x** |
| **8** | 1.12 | 29 | **1609 ms** | 3746 ms | 287 ms | **5.6x** |
| **12** | 0.92 | 36 | **1589 ms** | 3718 ms | 288 ms | **5.5x** |
| **16** | 1.00 | 189 | **199 ms** | 2193 ms | 348 ms | **0.6x** |
| **24** | 1.04 | 875 | **248 ms** | 1346 ms | 528 ms | **0.5x** |
| **32** | 0.91 | 1203 | **241 ms** | 2467 ms | 642 ms | **0.4x** |
| **48** | 0.42 | 784 | **230 ms** | 2956 ms | 824 ms | **0.3x** |
| **60** | 0.23 | 539 | **253 ms** | 2529 ms | 825 ms | **0.3x** |

The answer is not the one to expect, and the `pause ÷ steady` column is where to read it. That column is how many ordinary word-gaps one rotation is worth.

**Lightly loaded, the pause dominates.** At one to twelve streams a rotation costs about 1.6–2.3 s where the reader expects a quarter-second — five to eight times a normal gap. It is the single most conspicuous thing about the output, which is why it gets its own section in `REPORT.md`.

**Past capacity it vanishes into the noise.** From sixteen streams up the ratio falls below 1.0: the rotation gap is no longer even the *longest* pause a user sees. That is not an improvement. Nothing about rotation got faster — the ordinary gaps grew to meet it, from 287 ms to 825 ms, and queueing delay took over as the dominant source of silence. The flush stops being the problem because everything else has become one.

Two honest caveats on this table. The `rotations/stream` column falls with load partly because an overloaded server emits fewer partials, so fewer turn transitions are *observed* — it is measuring what reached the client, not what the engine did. And at low N the pause count is small (single digits per level), so those percentiles describe a handful of events; the N=1 row is three rotations, not a distribution.

The practical reading: **rotation is a fixed cost you notice when the service is healthy, and the least of your problems when it is not.**

## 6. GPU under load

Sampled at 10 Hz around the same streams that produced the table above.

| N | util % p50 | util % max | VRAM MB | power W p50 | power W max | temp °C | encode ms | decode ms | decode ÷ encode |
|---|---|---|---|---|---|---|---|---|---|
| **1** | 12 | 35 | 20290 | 105 | 115 | 39 | 23.3 | 16.8 | 0.7x |
| **4** | 16 | 37 | 20290 | 113 | 130 | 39 | 23.8 | 20.1 | 0.8x |
| **8** | 24 | 40 | 20292 | 122 | 156 | 41 | 25.4 | 48.5 | 1.9x |
| **12** | 27 | 43 | 20292 | 123 | 162 | 42 | 25.7 | 94.8 | 3.7x |
| **16** | 33 | 55 | 20294 | 147 | 175 | 45 | 26.8 | 119.3 | 4.4x |
| **24** | 33 | 63 | 20316 | 161 | 179 | 48 | 29.7 | 140.9 | 4.7x |
| **32** | 32 | 69 | 20334 | 157 | 179 | 48 | 26.9 | 171.1 | 6.4x |
| **48** | 32 | 69 | 20346 | 155 | 178 | 48 | 27.6 | 210.4 | 7.6x |
| **60** | 31 | 69 | 20346 | 156 | 180 | 49 | 34.0 | 192.7 | 5.7x |

**Encode time is flat under load; decode time is not.** That single contrast is the whole capacity story. The encoder batches across sessions, so serving ten streams costs it barely more than serving one. The decoder is a serial Python loop over sessions — `_decode_one` — so its cost scales with N no matter how the batcher groups them. Watch the `decode ÷ encode` column climb.

**Do not read `util %` as saturation.** `nvidia-smi` reports the fraction of time at least one kernel was resident, not how much of the GPU was working; one small kernel on one SM reads as 100%. The counters that would settle it (`SM_ACTIVE`, `SM_OCCUPANCY`) need DCGM, which is not installed here — they are `NOT MEASURED` rather than estimated. Power draw and clocks tell the same story from the other side: this GPU is nowhere near limited by this workload. The bottleneck is a Python loop, not silicon.

## 7. How the streams arrive

The same 24 streams, started three different ways. Real traffic is none of them exactly; the spread between them is what matters.

| | |
|---|---|
| `stagger` | starts jittered across one chunk period — the default, and the closest to independent users arriving on their own schedules |
| `poisson` | exponential inter-arrival times — bursty, the standard traffic model |
| `sync` | every stream starts on the same instant — an artificial herd, kept as a worst case because it hands the batcher a full batch it would never see in production |

| arrival | runs | TTFP p50 | **TTFP p95** | finished late by | drift p95 | **sess/tick** | err |
|---|---|---|---|---|---|---|---|
| `stagger` | 3 | 2380 ms | **3425 ms** | 69% | 7464 ms | **10.7** | 0 |
| `poisson` | 3 | 2450 ms | **3614 ms** | 69% | 7415 ms | **10.5** | 0 |
| `sync` | 3 | 2586 ms | **3231 ms** | 70% | 7326 ms | **14.2** | 0 |

### The herd is the *best* case, not the worst

`sync` — the pattern included as a stress test — produced the highest batching efficiency (14.2 sessions per tick against 10.7 for staggered arrivals) and the **lowest** TTFP p95. That is the opposite of the intuition it was added to test.

The mechanism is straightforward once seen. Synchronised streams hit their chunk boundaries together, so every tick finds a full batch waiting and the encoder — which *does* batch across sessions — runs once for all of them. Staggered arrivals spread the same work across more, emptier ticks. The herd wins on throughput for exactly the reason it looks alarming.

It costs a little at the median: `sync` has the *highest* TTFP p50, because every stream queues behind the same instant instead of slotting into a gap. So the trade is a slightly worse typical first word for a materially better tail.

**What this means practically:** arrival shape moves the numbers by a few percent, while stream count moves them by hundreds. Do not spend effort smoothing traffic arrival — spend it on the stream count. And note that none of the three patterns produced a single error.

## 8. Sustained load, well past capacity

32 concurrent streams held for 420 s — roughly four times the real-time capacity, sustained rather than in bursts. Short cells can hide a slow leak or a slow degradation; this is the arm that would surface either.

| | |
|---|---|
| streams completed | 672 |
| errors | **0** |
| VRAM allocated, first half → second half | 2.534 GB → 2.533 GB (**-0.001 GB**) |
| VRAM reserved, drift | 0.000 GB |
| tick p95, first half → second half | 748 ms → 780 ms |
| sessions still draining at the end | 6 |

**No leak, and no collapse.** Memory is flat across the run; the service absorbs four times its real-time capacity for minutes on end by falling behind, not by failing. The residual session count at the end is the backlog made visible — work accepted that had not yet drained when the clock stopped.

## 9. What to do with this

**Size on 8 streams per replica.** Not on latency, and not on the fact that 60 streams
"worked" — they completed, minutes late, with no error to tell you so.

**Watch the right signal.** `normalized_latency` in the benchmark output — over about 1.1 and
you are past capacity — or `over_realtime_capacity` in `/metrics`, which the service reports
itself. Not TTFP: it is the most available metric and the most misleading one here.

**If the traffic is short utterances, size lower.** Short streams degrade first and hardest
(§4) — a fixed queueing delay is a much larger fraction of a four-second clip than of a
thirty-four-second one. The interactions where a human is actively waiting are the ones that
break first.

**To go past 8 per replica**, in ascending order of risk:

| Option | Effect | Cost |
|---|---|---|
| More replicas | linear, and the GPU has room — one replica uses ~10 GB of 96 GB | this box has 8 vCPUs and the decoder is a CPU-launch-bound Python loop, so CPU binds before VRAM does. A larger instance scales further |
| Baseline geometry (1.0 / 0.5) | roughly 2x the streams | choppier output: 36 commits over 43 s instead of 95, and a 3.36 s worst gap |
| Batch the decoder | the only change that lifts the per-replica ceiling properly | not shipped. Its correctness gate — byte-identical transcripts for eight staggered sessions — was never run, and silent corruption is the failure mode |

**Admission control is the cheap win.** The service now refuses past `CORE_MAX_SESSIONS` with a
structured error naming both limits, rather than accepting work it cannot do in time. A refused
client can retry or fail over; a silently starved one cannot.

## 10. What this test does not establish

* **Single language.** The sweep is Hindi. Per-language latency at N=1 is in `REPORT.md` §2;
  per-language behaviour *under load* is `NOT MEASURED`.
* **One box, one GPU.** No multi-replica or multi-GPU scaling was measured — the guidance
  above is inference from the resource shape, not a measurement.
* **Occupancy counters unavailable.** DCGM is not installed, so `SM_ACTIVE` and `SM_OCCUPANCY`
  are `NOT MEASURED`. "The decoder is launch-bound" remains consistent with the evidence rather
  than proven by it.
* **No percentile here rests on thousands of samples.** Three repeats per level, medians
  reported. Treat p95 as indicative of shape, not as a settled tail.
