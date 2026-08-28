# BENCHMARKS — `indic-transcribe-core` streaming

Generated 2026-08-28T05:47:48+00:00 by `bench/report.py` from the raw run data. Anything not actually run says `NOT MEASURED`; nothing here is inferred.

## Environment

| | |
|---|---|
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition (96 GB, sm_120) |
| Instance | AWS `g7e.2xlarge`, 8 vCPU |
| torch | 2.12.0+cu132 (CUDA 13.2) |
| torchaudio | 2.11.0+cpu — **no cu132 build exists**; see README |
| NeMo | 3.0.0 |
| Model | `indic-transcribe-core`, Canary 1.2 B, 1.2214 B params, bf16 |
| Languages | 25 (not the 27 the wrapper advertises) |
| Policy | AlignAtt, `alignatt_thr=8`, `waitk_lagging=1` |

### Run A — geometry sweep (the word-by-word question)

Each geometry in two arms. **`nemo`** is the budget upstream computes for itself, `10 * int(chunk_eff + right_eff)`; **`fixed`** is `max(4, round(...))`. A ⚠ marks the geometries where upstream's own formula yields **0** — every configuration below `chunk + right = 1.0 s`.

The `fixed` arm is shown in the main table. The `nemo` arm is broken out below, because what it actually does was **not** what was predicted.

| geometry | chunk/right eff | budget (nemo → fixed) | TTFP ms | partials | words/partial | max gap ms | delta lag p50/p95 ms | CER vs offline | NE | tick p95 ms | % budget |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **baseline** | 0.96 / 0.48 | 10 → 14 | 2538.6 | 7.6 | 2.97 | 987.4 | 85.2 / 102.6 | 0.030 | 0.00 | 123.2 | 12.8% |
| **T1** | 0.80 / 0.40 | 10 → 12 | 2331.6 | 9.3 | 2.62 | 967.0 | 79.3 / 97.9 | 0.030 | 0.00 | 104.0 | 13.0% |
| **T2** | 0.64 / 0.32 | 0 ⚠ → 10 | 2350.3 | 11.3 | 2.35 | 890.8 | 70.8 / 94.1 | 0.020 | 0.00 | 96.2 | 15.0% |
| **T3** | 0.48 / 0.32 | 0 ⚠ → 8 | 2105.0 | 14.8 | 1.81 | 757.1 | 62.0 / 76.9 | 0.030 | 0.00 | 90.2 | 18.8% |
| **T4** | 0.48 / 0.24 | 0 ⚠ → 7 | 2024.7 | 15.4 | 1.81 | 798.9 | 60.2 / 78.5 | 0.030 | 0.00 | 83.0 | 17.3% |
| **T5** | – | – | ERROR: AcceleratorError: CUDA error: an illegal memory access was e | | | | | | | | |
| **T6** | 0.32 / 0.16 | 0 ⚠ → 5 | 1915.9 | 20.4 | 1.53 | 680.4 | 54.9 / 71.3 | 0.030 | 0.00 | 74.7 | 23.3% |
| **T7** | 0.24 / 0.16 | 0 ⚠ → 4 | 1852.5 | 22.1 | 1.37 | 688.2 | 50.2 / 61.7 | 0.040 | 0.00 | 62.3 | 25.9% |

#### The `nemo` arm — a correction to the prediction

The prediction going in was that a budget of 0 makes
`disable_samples_mask = steps_per_inner_loop >= 0` true on the first inner-loop iteration, so the sample is disabled *before its first token* and emits nothing.

**That is not what happens.** The mask is `*= logical_not(is_last_chunk_batch)`, and `active_samples_inner_loop` is reset each outer step, so the sample is not killed — it is *throttled to exactly one token per tick*. Measured below: `words/partial` is **1.00 in every affected row**, never 0 and never anything else. The cost is real but it is accuracy, not silence:

| geometry | nemo budget | words/partial | CER vs offline | fixed-arm CER | CER penalty | TTFP ms | partials |
|---|---|---|---|---|---|---|---|
| baseline | 10 | 2.97 | 0.030 | 0.030 | +0.000 | 2538.7 | 7.6 |
| T1 | 10 | 2.62 | 0.030 | 0.030 | +0.000 | 2330.3 | 9.3 |
| T2 | 0 | 1.00 | 0.070 | 0.020 | +0.050 | 2311.9 | 11.8 |
| T3 | 0 | 1.00 | 0.050 | 0.030 | +0.020 | 2079.7 | 16.0 |
| T4 | 0 | 1.00 | 0.040 | 0.030 | +0.010 | 1999.2 | 16.4 |
| T5 | – | ERROR | | | | | |
| T6 | 0 | 1.00 | 0.050 | 0.030 | +0.020 | 1893.9 | 24.3 |
| T7 | 0 | 1.00 | 0.060 | 0.040 | +0.020 | 1834.8 | 30.8 |

So the defect is milder than predicted in kind and still real in effect: every sub-second geometry silently degrades to one-token-at-a-time emission with a worse transcript. The claim that the region was **untested** stands; the claim that it was **dead** does not, and is corrected here.


### Run B — concurrency

Real-time paced, staggered arrivals, open-loop timing from `t_sched`. `client_bound` rows fell behind their own send schedule and measure the harness, not the server.

| N | TTFP p50 | p95 | p99 | delta lag p95 | tick p95 ms | % chunk budget | sess/tick | client-bound | errors |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 2446.3 | 2446.3 | 2446.3 | 144.8 | 688.1 | 71.7% | 1.00 | 0 | 0 |
| 8 | 2388.4 | 3403.4 | 3403.4 | 141.4 | 119.7 | 12.5% | 1.05 | 0 | 0 |
| 16 | 2490.7 | 3699.6 | 3699.6 | 443.7 | 165.9 | 17.3% | 1.24 | 0 | 0 |
| 24 | 2807.9 | 3974.5 | 4981.6 | 1467.5 | 252.7 | 26.3% | 1.57 | 0 | 0 |
| 32 | 2929.5 | 5792.8 | 5870.1 | 3502.9 | 503.6 | 52.5% | 2.24 | 0 | 0 |

Head-of-line blocking suspected at: 8, 16, 24, 32

### Run C — batch-formation window (W)

| W ms | max_batch | N | sess/tick | batch hist | padding waste | wait p50 ms | tick p95 ms | % budget | TTFP p95 |
|---|---|---|---|---|---|---|---|---|---|
| 0 | 16 | 8 | 1.00 | `{'1': 29}` | 0.0% | 0.0 | 103.9 | 10.8% | 3409.1 |
| 0 | 16 | 16 | 1.74 | `{'1': 22, '2': 3, '3': 2, '4': 2, '5': 1, '7': 1}` | 0.0% | 0.0 | 328.1 | 34.2% | 3671.2 |
| 0 | 32 | 8 | 1.00 | `{'1': 29}` | 0.0% | 0.0 | 105.1 | 10.9% | 3401.0 |
| 0 | 32 | 16 | 1.74 | `{'1': 22, '2': 3, '3': 2, '4': 2, '5': 1, '7': 1}` | 0.0% | 0.0 | 327.1 | 34.1% | 3667.5 |
| 50 | 16 | 8 | 1.33 | `{'1': 14, '2': 7}` | 0.0% | 50.0 | 136.2 | 14.2% | 3576.3 |
| 50 | 16 | 16 | 3.33 | `{'1': 4, '2': 3, '3': 2, '4': 2, '5': 2, '6': 1, '10': 1}` | 0.0% | 50.0 | 381.5 | 39.7% | 3891.7 |
| 50 | 32 | 8 | 1.27 | `{'1': 16, '2': 6}` | 0.0% | 50.0 | 137.4 | 14.3% | 3574.4 |
| 50 | 32 | 16 | 3.38 | `{'1': 4, '2': 3, '3': 2, '4': 3, '5': 1, '6': 1, '7': 1, '8': 1}` | 0.0% | 50.0 | 501.0 | 52.2% | 3866.7 |
| 200 | 16 | 8 | 2.89 | `{'2': 2, '3': 6, '4': 1}` | 0.0% | 200.0 | 316.0 | 32.9% | 4002.0 |
| 200 | 16 | 16 | 6.57 | `{'3': 1, '5': 3, '6': 1, '11': 2}` | 0.0% | 200.0 | 563.1 | 58.7% | 4025.3 |
| 200 | 32 | 8 | 2.89 | `{'2': 2, '3': 6, '4': 1}` | 0.0% | 200.0 | 313.0 | 32.6% | 4002.9 |
| 200 | 32 | 16 | 6.57 | `{'3': 1, '5': 3, '6': 1, '11': 2}` | 0.0% | 200.0 | 757.3 | 78.9% | 4015.0 |

### Run D — soak (300s at N=8)

| metric | first half | second half | delta | min | max |
|---|---|---|---|---|---|
| vram_allocated_gb | 2.512 | 2.513 | +0.0 | 2.5 | 2.53 |
| vram_reserved_gb | 15.19 | 15.19 | +0.0 | 15.19 | 15.19 |
| tick_ms_p95 | 364.755 | 362.264 | -2.491 | 84.43 | 465.04 |
| avg_decode_ms | 127.289 | 126.994 | -0.295 | 0.0 | 152.83 |
| avg_encode_ms | 24.511 | 24.514 | +0.003 | 0.0 | 25.88 |

Waves 18, completed 144, errors 0. Session leak: **True** (sessions_active final = 1).

decoder_mems_list length is not exposed by /metrics; VRAM reserved is the proxy watched here. Direct measurement is NOT MEASURED.

### Run E — offline throughput reference

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

## Factors — what produces these numbers

**Re-encode amplification.** AlignAtt re-encodes the whole `left + chunk + right` buffer every
chunk. At the baseline 0.96 s chunk against an 11.44 s buffer that is ~11.9x the encoder work per
second of audio that offline decoding does. Run E measures that ceiling directly: RTFx 22.6 at
batch 1, rising to ~88-91 at batch 16. Smaller chunks make the amplification worse, which is the
real price of word-by-word — not accuracy.

**Decoder serialisation dominates capacity.** `_decode_one` is a Python loop over sessions, so
decoder work per chunk period scales with N regardless of how sessions are grouped. Run C shows
this cleanly: raising W from 0 to 200 ms lifts `avg_sessions_per_tick` from 1.00 to 2.89 at N=8,
but per-tick budget *rises* (10.8% -> 32.9%) because the same serial decoder work is concentrated
into fewer, longer ticks. Grouping recovers the encoder term only.

**Batch efficiency and padding.** The batch-size histogram at W=0 is overwhelmingly `{1: n}` --
sessions arrive at independent phases and most ticks carry exactly one. W is what turns that into
a distribution; padding waste stays low because buffers are near-equal length once a session is
past its first tick.

**Queue delay.** TTFP p95 degrades from 2446 ms at N=1 to 5793 ms at N=32 while the tick budget
stays at 52.5% -- the server is not saturated, so that growth is arrival queueing, not compute.

**Launch overhead — the prediction that could NOT be confirmed.** The stated prediction was
GPU-Util near 100% while SM occupancy stays low and tensor pipes idle, i.e. a launch-bound
decoder for which CUDA graphs are the right lever. DCGM is not installed on this host, so
`SM_ACTIVE`, `SM_OCCUPANCY`, `DRAM_ACTIVE` and `PIPE_TENSOR_ACTIVE` are **NOT MEASURED**.
`nvidia-smi` utilization alone measures kernel *residency*, not work, and cannot settle it.
The hypothesis is therefore neither confirmed nor refuted here, and ladder step 3 (CUDA-graphing
the inner decode step) is **not** justified on the evidence collected.

## Optimisation ladder — what was actually applied

| # | Change | Status | Measured effect |
|---|---|---|---|
| 1 | Batch-formation window W | **implemented**, runtime-settable | Run C: `sess/tick` 1.00 -> 2.89 at N=8 (W=200 ms), TTFP p95 +601 ms. A latency/grouping trade, not a free win. |
| 2 | CUDA events replacing two stats-only `synchronize()` | **implemented**, default on | Removes two mid-tick barriers between encoder and decoder. `CORE_CUDA_EVENTS=0` restores the old path for A/B. Isolated delta: `NOT MEASURED`. |
| 3 | CUDA-graph the inner decode step | **not attempted** | Its justification depends on counters this host cannot provide (see above). Attempting it would have been guessing. |
| 4 | Batch the decoder | **not attempted** | The only change that lifts the ~23-session ceiling, and the one with a real silent-corruption risk. Its gate -- byte-identical transcripts for 8 staggered sessions -- was not run, so it is not shipped. |

## Known defects, measured

* **T5 (0.32 / 0.24) fails on medium clips** with a CUDA illegal memory access inside the
  decoder's attention GEMM (`transformer_decoders.py:265` -> `transformer_modules.py:201`), on
  both arms. It **succeeds on short clips** (budget 6, 1.75 words/partial, CER 0.04), so the
  failure is duration-dependent. `decoder_mems_list` is concatenated without bound and grows with
  decode *steps* rather than committed tokens, which is consistent with outgrowing the decoder's
  1024-position limit on a small chunk over long audio. Its neighbours T4 and T6 both work.
  Not fixed; reported.

* **`max_generation_length` cannot simply be raised.** `initialize_aed_model_state` allocates
  `pred_tokens_ids` (and `tokens_frame_alignment`) at the state's default of 256. Assigning 512
  afterwards raises the loop bound without resizing the buffer, producing out-of-bounds device
  writes that surface later as an illegal memory access from an unrelated kernel. The engine now
  grows both tensors; without that, every geometry below baseline crashed.

## Discipline

Every number above is read from `results/*.json` by `bench/report.py`; nothing is transcribed by
hand. Percentiles come from the raw samples that produced them and are never averaged. Warm-up
batches are discarded in Run E. Runs and counters that were not collected say `NOT MEASURED`
rather than being estimated. Bootstrap CIs are computed for Run B TTFP p50; where intervals
overlap, no difference is claimed.

**What this campaign did not do:** no p99 here rests on >=1000 samples, and no headline config was
repeated 3x with spread reported. This was the agreed reduced first pass. Treat single-run
percentiles as indicative, not settled.
