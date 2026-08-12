# Standards & external backends

`tts_eval` owns synthesis, latency, storage, reporting and coverage. Perceptual
naturalness and speaker metrics are **delegated to established tools** (UTMOS,
DNSMOS, VERSA, TTSDS2) rather than reimplemented, so the numbers are comparable to
what the rest of the field publishes. This document is the setup recipe for those
backends, plus how to build a publication-grade Indic test set.

Everything here is optional. With none of it installed a run still completes: each
absent backend reports `not_computed` with a reason, and the report shows the gap
instead of a blank. Install only the backends you intend to cite.

---

## Metric tiers

The suite's `metrics:` field selects a tier or an explicit backend list.

| Tier | Backends | Needs |
|---|---|---|
| `core` | latency, inference, audio-quality, voice-consistency, coverage, reliability | numpy only |
| `standard` | `core` + `intelligibility` (CER/WER/slot) | an ASR endpoint |
| `all` | `standard` + every installed plugin: `utmos`, `dnsmos`, `speaker`, `speaker_consistency`, `versa`, `ttsds2` | see below |

Per-backend settings go under `metric_options.<backend>` in the suite YAML. A
backend whose dependency or input is missing is skipped with a reason; it never
fails the run.

```yaml
# suite YAML
metrics: all
metric_options:
  utmos:   { device: cuda }
  ttsds2:  { reference_dir: /data/indic_ref_speech }
  versa:   { python: /opt/versa/venv/bin/python, score_config: configs/versa.yaml }
```

---

## Round-trip intelligibility (CER/WER) — ASR

Pronunciation accuracy is measured by transcribing the synthesised audio and
scoring it against `expected_transcript`. Point the suite at an ASR endpoint:

```yaml
asr:
  backend: http_asr           # or `whisper` for a local faster-whisper model
  url: http://localhost:8001/transcribe
  language_field: language
  transcript_path: transcript
```

For low-resource Indic languages the ASR carries its own error rate, so the CER is
round-trip, not absolute. The comparison engine **warns** when two runs used
different ASR backends — a CER delta then partly measures the ASRs, not the TTS.
Use one ASR across every run in a bake-off.

---

## Predicted MOS — UTMOS & DNSMOS

Reference-free naturalness predictors. Install the `mos` extra:

```bash
pip install 'tts-eval[mos]'      # torch + torchaudio
```

* **`utmos`** → `utmos` (predicted MOS 1–5). Loads `sarulab-speech/UTMOS22` via
  `torch.hub`. Options: `repo_dir` (a local checkout to avoid the network),
  `device` (`cuda`/`cpu`, auto-detected otherwise).
* **`dnsmos`** → `dnsmos_ovrl` / `dnsmos_sig` / `dnsmos_bak` (P.835). Runs the DNSMOS
  ONNX model via `onnxruntime`.

> **Predicted MOS is a predictor.** UTMOS is documented to rank-invert against real
> listeners on out-of-domain and zero-shot speech. Treat these as a screen, not a
> verdict, and confirm a close call with a human listening test
> (`tts-eval subjective`). `divergence_report()` correlates predicted vs human MOS
> *on your model and set* rather than assuming the proxy holds.

---

## Speaker similarity & consistency

Embedding metrics, preferring Resemblyzer (small, no torchaudio):

```bash
pip install 'tts-eval[speaker]'
```

* **`speaker`** → `speaker_similarity`: cosine similarity to a **per-case reference
  recording** (the target voice). Needs `reference_audio` on the test case (see
  below); reports `not_computed` without it.
* **`speaker_consistency`** → pairwise embedding similarity across a run's
  utterances. Needs no reference — this is the calibrated companion to the DSP-only
  `voice_consistency`, and both are reported side by side.

---

## VERSA (broad perceptual toolkit)

[VERSA](https://github.com/wavlab-speech/versa) (CMU WavLab, NAACL 2025) implements
~65 metrics / 729 variants (UTMOS, DNSMOS, NISQA, PESQ, speaker similarity, …).
It has a large, tiered dependency tree and is normally installed in **its own venv**.

1. Install VERSA per its README, in any interpreter.
2. Write a VERSA score config (`score_config`) listing the metrics to run.
3. Point the backend at that interpreter and config:

```yaml
metric_options:
  versa:
    python: /opt/versa/venv/bin/python     # the interpreter VERSA lives in
    score_config: configs/versa_score.yaml
    metric_map:                            # VERSA key -> our catalogue name
      utmos_score: utmos
      dnsmos_overall: dnsmos_ovrl
    # command: "{python} -m versa.bin.scorer ..."   # override if your VERSA CLI differs
    timeout_s: 3600
```

The bridge runs VERSA over the completed run's audio dir, then folds each mapped
metric back into the same record so it reports identically to a native metric. VERSA
importability is probed in the **target** interpreter, so a missing install degrades
to `not_computed` rather than crashing this process.

---

## TTSDS2 (distributional benchmark)

[TTSDS2](https://arxiv.org/pdf/2506.19441) scores the distributional distance
between your synthesised speech and a **corpus of real speech** across five
categories, in 14 languages.

```bash
pip install 'tts-eval[ttsds]'
```

```yaml
metric_options:
  ttsds2:
    reference_dir: /data/indic_ref_speech   # a directory of real .wav files
    min_utterances: 20                       # below this it reports not_computed
    system_name: indic-mio
```

Distributional, so it needs a **body** of reference speech, not a per-utterance
match. Reuse the reference corpus you build for IndicVoices-R below.

---

## Reference audio (for `speaker_similarity` and TTSDS2)

`speaker_similarity` needs a target recording per case; TTSDS2 needs a reference
corpus. Attach references in the dataset JSONL via `reference_audio`, a path
resolved relative to the dataset file:

```json
{"id": "hi-greet-01", "text": "नमस्ते ...", "language": "hi",
 "reference_audio": "refs/hi-greet-01.wav"}
```

Without references these two metrics report `not_computed` by design — the seed
`indic_conversational_v1` set ships none, so they are absent there.

---

## Publication-grade Indic test set — IndicVoices-R

The bundled `indic_conversational_v1` (69 utterances, 13 languages) is authored for
the harness and **not native-speaker reviewed**, so its per-language CER is
indicative. For numbers you intend to publish, build a set from
[IndicVoices-R](https://arxiv.org/abs/2409.05356) — the right corpus and protocol
for 22 Indic languages, with human transcripts and real reference audio.

Recipe (convert the corpus into a `tts_eval` JSONL):

1. **Select** a stratified subset per language (match the categories you care about:
   greetings, numeric, code-switch, long-form, edge cases).
2. **Map each item** to one JSONL line:
   - `id` — stable unique id (used for cross-run pairing); keep it deterministic.
   - `text` — the prompt sent to TTS.
   - `language` — the code your model/ASR expects (align with the model card).
   - `expected_transcript` — the human transcript, **spoken form** (digits and
     currency expanded), so CER is not falsely penalised on `₹12,450`.
   - `must_contain` — critical tokens (OTP, amounts, dates) for `slot_accuracy`.
   - `reference_audio` — path to the corpus's real recording, for speaker metrics
     and the TTSDS2 reference corpus.
3. **Pin it.** Load the JSONL once; the loader writes a content-hash sidecar. Commit
   both. A later silent edit then fails to load instead of quietly changing results.
4. **Cite** the corpus version and your subset's `content_hash` in the report.

```python
from tts_eval.datasets import dataset_from_cases
ds = dataset_from_cases(cases)          # cases = list of the dicts above
ds.write_jsonl("configs/datasets/indicvoices_r_v1.jsonl")
```

Point a suite at it with `dataset: configs/datasets/indicvoices_r_v1.jsonl` (a path,
not a bundled name) and run every model against that one pinned file.

---

## Interpreting the numbers — what is and isn't calibrated

* **`voice_consistency`** is a relative DSP signal; rank models against each other on
  one set, do not read it against an absolute target. `speaker_consistency`
  (embedding) is the calibrated version.
* **Round-trip CER** includes the ASR's error rate; thresholds default permissive
  (0.30) and cross-ASR comparisons are warned.
* **Predicted MOS** is a screen; a human listening test is the measurement.
* **Effect floors** gate every verdict: a statistically significant but sub-floor
  delta is reported `negligible`, not a win. Check `repeatability()` (the benchmark's
  own noise floor) before trusting any small delta.
