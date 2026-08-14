# tts_eval — build status

Standalone TTS evaluation harness. Zero imports from `voice_2_voice_server`,
`voicera_backend` or any other service in this monorepo — talks to TTS servers over
the wire only. Copy the directory out and it still works.

**~12,400 lines of library code across 48 modules**, plus a 12-file pytest suite.
Core install: `numpy` + `PyYAML` only.

## Test status

```
119 passed in 125s   (cd tts_eval && python3 -m pytest -q)
```

Everything below marked ✅ is covered by that suite. One file per area, mirroring
the package layout; shared fixtures live in `tests/conftest.py`. Each file defends
one property — the properties are what the acceptance criteria actually ask for:

| File | n | Defends |
|---|---|---|
| `test_datasets.py` | 10 | Reproducible, tamper-evident test inputs |
| `test_text_scoring.py` | 9 | Symmetric, script-aware CER/WER scoring |
| `test_audio.py` | 5 | WAV/DSP primitives |
| `test_adapters.py` | 13 | Provider-agnostic synthesis + honest failures |
| `test_metrics.py` | 21 | Every AC has metrics; each silent-failure mode caught; absent deps degrade with reasons |
| `test_runner.py` | 18 | Only real inputs move the fingerprint; ordering, memory, warnings, real concurrency |
| `test_store.py` | 8 | Durable, rebuildable, secret-free storage |
| `test_compare.py` | 9 | Defensible verdicts, blocked when non-comparable |
| `test_subjective.py` | 11 | Blinding, rater screening, correct de-blinding |
| `test_config.py` | 8 | Config-only model onboarding |
| `test_cli.py` | 7 | CLI wraps the API and returns honest exit codes |

### Four real bugs the tests found and fixed

1. **Loop detector flagged clean speech and missed real loops.** Raw magnitude
   spectra are all dominated by the same spectral tilt, so cosine similarity is high
   between any two frames of anything. Now uses **mean-removed log spectra**.
   Verified separation: clean speech 0.52, tiled loop 1.00 at the exact 0.42 s period,
   white noise 0.007, held tone → `None` (no trajectory to repeat).
2. **`log(spec + 1e-8)` turned float noise into signal.** A pure tone has near-zero
   energy in most bins; an absolute epsilon amplified that noise into multi-nat log
   swings, so a held tone looked like it varied wildly. Now floors 80 dB below peak.
3. **Truncation of short utterances was invisible.** `length_ratio` assumed duration
   is purely proportional to text length. For "Yes." (4 chars) a *halved* clip still
   scored 0.69 — above the 0.55 flag line — because fixed onset/offset padding
   dominates. Now affine: `overhead + chars/rate`. Regression-guarded by
   `test_short_utterance_truncation_is_detected`.
4. **`save_audio: false` marked every utterance as failed.** `SynthesisResult.ok`
   read `audio_path` once the in-memory buffer was freed, so a run that deliberately
   does not persist WAVs — the `latency` suite, and any `--no-audio` run — scored
   0/69 success, zeroing throughput/reliability and every aggregate. `ok` now trusts
   a persisted `n_samples` captured at synthesis, independent of buffer and file
   lifecycle. The pre-existing `test_save_audio_false_writes_nothing` only checked
   that no file was written and never that the run succeeded — that blind spot is why
   it slipped through; the test now also asserts `n_ok == 13`, plus a reload guard.

### One design gap the tests exposed and closed

**Significance without effect size.** Two *identical* runs were compared and the
engine declared `stream_chunk_gap_p95_ms` "better" — a 0.001 ms difference with a CI
excluding zero. Real, and meaningless. A benchmark that certifies microsecond wins is
a noise amplifier.

Fixed by adding a `min_effect` floor to **all 42** catalogue entries (10 ms for TTFB,
0.005 for CER, 0.05 MOS, …) plus a 2% relative fallback. New `negligible` verdict for
"statistically real, below the floor". Verdict now needs **both** gates.

Also added: `checks_skipped` on `degeneracy_score`, so a sub-second utterance that
could not be assessed for repetition reports that instead of a clean-looking `0.0`.
And the mock now reports `fault_applied: false` when an utterance is too short to
carry an injected fault — otherwise a test thinks the detector missed something it
was never given.

---

## Prior art check (asked for, done)

No existing tool covers the acceptance criteria. What exists, and what we reuse:

| Tool | Gives | Missing for us |
|---|---|---|
| [VERSA](https://arxiv.org/abs/2412.17667) (CMU, NAACL'25) | 65 metrics / 729 variants: UTMOS, DNSMOS, NISQA, WER, speaker sim | No latency, no synthesis, no run registry, no reports |
| [TTSDS2](https://arxiv.org/pdf/2506.19441) | Distributional benchmark, 5 categories, 14 languages | Offline only, no latency, no coverage gating |
| [TTS-Evaluation](https://github.com/Shengqiang-Li/TTS-Evaluation) | WER + SECS + UTMOS drop-in | Not a framework |
| [IndicVoices-R](https://arxiv.org/abs/2409.05356) | Right *corpus* + protocol for 22 Indic languages | Dataset, not a harness |

**Decision:** we own synthesis + latency + storage + reporting + coverage. Perceptual
metrics delegate to VERSA/TTSDS2/UTMOS via pluggable backends. Both bridges built.

---

## Done and tested

### 1. Core types + audio DSP — `types.py`, `audio.py`, `errors.py`
- Explicit `to_dict`/`from_dict` on every persisted type. On-disk schema is a
  reviewable contract, not a side effect of field order.
- `SCHEMA_VERSION` guard: refuses to half-read a record written by a newer build.
- numpy-only DSP (no scipy/librosa on this box): WAV I/O via stdlib `wave`,
  framing, spectrogram, spectral centroid/flatness, autocorrelation F0.
- `Determinism` enum recorded per run — `deterministic` / `seeded` / `best_effort`.
  Indic-Mio at temperature 0.9 is `best_effort` and the report says so.

### 2. Adapter layer — 4 adapters, provider-agnostic ✅ tested
Timing lives in the **base class**, not subclasses. If each adapter timed itself,
cross-provider latency would not be comparable — which is the whole point.

| Adapter | Covers |
|---|---|
| `websocket_pcm` | Indic-Mio, AI4Bharat Parler, any float32-PCM WS server |
| `http_rest` | Sarvam, ElevenLabs, Cartesia, OpenAI, any REST TTS |
| `mock` | Deterministic offline, injectable faults |
| `replay` | Pre-generated WAVs — re-score without re-synthesis |

Cost of a new model: **1 YAML file**. New wire protocol: ~100-line subclass.
Out-of-tree adapter: `--adapter-module my_pkg.adapter`, no fork.

### 3. Dataset layer ✅ tested
- **69 utterances, 13 languages** (en hi bn ta te mr gu kn ml pa or as ur), native
  scripts, 7 categories: greeting, question, numeric, code_switch, long_form,
  edge_short, edge_symbols.
- **Two hashes, separate on purpose:** `content_hash` (synthesis-affecting fields
  only → enters fingerprint) and `manifest_hash` (everything → provenance).
  Editing a comment does not break comparability; editing a sentence does.
- Hash pinned in sidecar. Silent edit = load error.
- Deterministic stratified sampling — same 13 utterances on every machine.
- `expected_transcript` field: without it, "₹12,450" scored against a *correct*
  spoken rendering reports ~90% CER. Single most common way round-trip TTS
  benchmarks produce garbage.

### 4. Metric engine — all 7 criteria + reliability ✅ tested
`catalog.py` is schema + docs + AC traceability matrix in one table. Missing
dependency = `not_computed` with a reason, never a crash, never a hole in the report.

| Criterion | Metrics |
|---|---|
| Speech Naturalness | `utmos`, `dnsmos_{ovrl,sig,bak}`, `subjective_mos`, `subjective_mushra`, `ttsds2_overall` |
| Pronunciation Accuracy | `cer`, `wer`, `slot_accuracy` |
| Response Latency | `ttfb_ms`, `first_audible_ms`, `stream_starvation_ms`, `stream_chunk_gap_p95_ms` |
| Voice Consistency | `voice_consistency`, `intra_utterance_f0_cv`, `speaker_similarity`, `speaker_consistency`, `subjective_smos` |
| Language Coverage | `coverage_ratio`, `languages_verified`, `languages_attempted` |
| Audio Quality | `snr_db`, `clipping_pct`, `silence_ratio`, `leading/trailing_silence_ms`, `loudness_dbfs`, `dynamic_range_db`, `dc_offset`, `spectral_flatness`, `length_ratio`, `degeneracy_score`, `audio_quality_score` |
| Inference Time | `inference_time_ms`, `rtf`, `chars_per_second`, `throughput_utt_per_min`, `audio_duration_s` |
| Reliability (added) | `success_rate`, `degenerate_rate` |

Three metrics worth calling out — none of them in any existing tool:

- **`stream_starvation_ms`** — playout-buffer deficit against a real-time schedule.
  A model can post 120 ms TTFB and still stutter mid-sentence. Raw inter-chunk gaps
  do not catch this; a long gap is harmless if the prior chunk covered it.
- **`degeneracy_score`** — the autoregressive-TTS failure that returns HTTP 200,
  streams the right byte count, and contains a loop / buzz / half a sentence.
  Detected via **mean-removed** log-spectral self-similarity. Raw spectra are all
  dominated by the same spectral tilt, so a naive measure flags clean speech and
  misses real loops. **Verified separation: clean 0.52, looping 0.99**, exact loop
  period recovered (0.42 s), 11–52 ms/utterance.
- **`slot_accuracy`** — a dropped 4-digit OTP in a 40-char sentence still scores
  0.90 CER. For a voice agent that utterance is a total failure. CER cannot say so.

### 5. Runner + store + compare ✅ tested
- **Fingerprint** = sha256 over exactly what can change a number (dataset content
  hash, model id/version, generation params, seed, voice, **concurrency**, metric
  set, thresholds). Excludes run id, timestamps, hostname. Stored with its inputs
  so a mismatch is diffable, not an opaque hex compare.
- **Store: JSON is source of truth, SQLite is a rebuildable index.** A benchmark
  record must outlive the tool that wrote it — readable in 5 years with a text
  editor, diffable in review. `reindex()` rebuilds from JSON; index is disposable.
- Atomic write (temp + rename): interrupted write cannot leave a half-parsed record
  that the index advertises as valid.
- `timings.json` sidecar → `replay` adapter re-scores with **original** latencies
  instead of inventing them.
- **Compare: paired on utterance id**, bootstrap CI on the paired difference, and a
  verdict. CI straddling zero → `inconclusive`, never a win.
- **Blocking** comparability checks: different dataset, different concurrency.
  **Warning** checks: different ASR backend (a CER delta then measures the ASRs),
  different hardware, low success rate.
- `single_observation` verdict for run-level metrics — a coverage drop 1.0→0.77 is
  *exact*, just untestable. Labelling it "inconclusive" would read as "probably noise".
- `repeatability()` gives the benchmark's own noise floor.

**Test result:** injected latency regression → `worse` with tight CIs
(+210 ms, CI [+204, +216]). Audio-quality deltas correctly `inconclusive` at n=13.
Repeat runs: deterministic metrics **CV 0.0**, TTFB CV 5.2% — that is the real
noise floor, and no inter-model delta below it should be believed.

### 6. Subjective loop ✅ tested
Built because predicted MOS is a *predictor*: the [2026 open-TTS
survey](https://arxiv.org/html/2603.24116v2) and [zero-shot eval
work](https://arxiv.org/pdf/2603.24430) both document UTMOS rank-inverting against
listeners. A framework that reports UTMOS and calls Naturalness satisfied has
substituted a proxy for the measurement.

- 4 scales: MOS, MUSHRA, CMOS, SMOS.
- **Blinding:** HMAC tokens (not a plain hash — cannot be brute-forced by someone
  who knows the run ids). Answer key written to a **separate** file.
- Per-rater trial shuffle + within-trial shuffle → kills anchoring and fatigue bias.
- **Low-pass anchor** injected to catch inattentive raters.
- Single-file HTML player, no server, no npm — panels are non-technical.
- **Ingest:** de-blind → screen raters → per-rater normalisation → agreement.
- `divergence_report()` correlates human vs predicted MOS *on this model and set*,
  instead of assuming the proxy holds.

**Test result:** 4 raters, 1 lazy (scored anchor 70 vs systems 72) → **caught and
excluded** with reason recorded. Systems separated 80.5 vs 63.3, non-overlapping
CIs. Agreement 0.864 → "strong".

### 7. Configs — 4 model cards, 4 suites
`configs/models/`: `indic-mio`, `ai4bharat-parler`, `sarvam`, `mock`.
`configs/suites/`: `smoke`, `indic-full`, `latency`, `offline-rescore`.

Cards and suites are **separate documents**: a new model reuses every suite
unchanged; a new protocol applies to every model unchanged. One file and adding a
model would mean copying and diverging the protocol.

`indic-mio` and `ai4bharat-parler` share `adapter: websocket_pcm` and **zero code** —
they differ only in URL and static fields. That is the generalisation claim, concrete.

---

## Done since first draft (were pending)

| # | Item | Notes |
|---|---|---|
| 7 | **Reports** — Markdown + standalone HTML/CSS + CSV export | `write_run_report` emits `report.html`, `report.md` and the three CSVs into the run dir; comparison reports via `write_comparison_report`. ✅ |
| 7 | **Lightweight UI** — stdlib `http.server`, static report browsing + audio playback | `python -m tts_eval.ui` / `tts-eval serve`. Read-only, binds `127.0.0.1`. ✅ |
| 7 | **CLI** — `tts-eval run/list/report/compare/dataset/subjective/verify/serve` | `tts_eval/cli.py`, wired to the `tts-eval` entry point. Thin wrapper over the Python API — no eval logic of its own. ✅ tested (`TestCLI`) |

The Python API remains the reference interface; the CLI just wraps it, so anything
one does the other can.

## Still pending

Nothing in the build plan. `docs/STANDARDS.md` (VERSA/TTSDS2/DNSMOS/UTMOS setup +
IndicVoices-R conversion recipe) is now written. ✅

What remains is verification that needs hardware, not code — see below.

pytest suite is **done** (119 tests), not pending.

### Not yet exercised against anything real
- `websocket_pcm` against a live Indic-Mio / Parler server — code paths untested end to end.
- `http_rest` against Sarvam — untested end to end.
- `http_asr` against an IndicConformer server — so no real CER number exists yet.
- `utmos` / `dnsmos` / `speaker` / `versa` / `ttsds2` — all correctly report
  `absent: <reason>` here (no torch weights / no VERSA install / no reference corpus),
  which the suite asserts. Their *compute* paths are unexercised.

### Known limitations, stated not hidden

1. **Seed page.** `indic_mio_tts_server` does not forward a seed to vLLM. Card says
   `supports_seed: false`, so audio is **not** bit-reproducible; metrics reproduce
   only within CIs. Fix is upstream (forward seed + `temperature: 0`), then flip the
   card flag.
2. **Dataset not native-speaker reviewed.** 13 languages authored for the harness.
   Per-language CER from it is indicative. `docs/STANDARDS.md` carries the
   IndicVoices-R conversion recipe for publication-grade numbers.
3. **No reference audio in the seed set** → `speaker_similarity` and `ttsds2_overall`
   report `not_computed` on it by design.
4. **`voice_consistency` is a relative signal.** DSP dispersion estimate, calibrated
   per-feature but not against real speech. Rank models against each other on one
   set; do not read it against an absolute target. `speaker_consistency` (embedding,
   optional) is the calibrated version — both reported side by side.
5. **Round-trip CER carries the ASR's error rate.** For low-resource Indic languages
   that is substantial. Comparison engine therefore *warns* when two runs used
   different ASR backends, and thresholds default permissive (0.30).
6. **Not run against a live Indic-Mio server.** All verification is against the mock
   adapter. End-to-end numbers unverified by me.
