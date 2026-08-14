# TTS Eval — One-Stop Guide

Evaluate any TTS model. Config-driven, reproducible, provider-agnostic.

---

## 1. Install

```bash
cd /path/to/tts_eval

# Core only (numpy + PyYAML — runs on any laptop, no GPU)
pip install -e .

# Wire protocol deps (websockets, aiohttp — needed for real model adapters)
pip install -r requirements-adapters.txt

# Heavy ML metric deps (torch, faster-whisper, resemblyzer, etc. — optional)
# Metrics you don't install degrade to "not_computed" with reason, never crash.
pip install -r requirements-metrics.txt
```

> [!TIP]
> Don't need all metrics? Skip `requirements-metrics.txt`. Missing backends degrade gracefully — you get `not_computed` with explicit reason, not a crash.

---

## 2. Understand Config Architecture

Two separate YAML files control evaluation:

| File | Location | Purpose |
|------|----------|---------|
| **Model Card** | `configs/models/<name>.yaml` | *WHAT* is evaluated — model identity, wire protocol, connection URL, voices, languages |
| **Suite Config** | `configs/suites/<name>.yaml` | *HOW* it's evaluated — dataset, which metrics, concurrency, thresholds, ASR config |

This separation means: **connecting a new model with a supported protocol = one YAML file, zero Python.**

---

## 3. Model Card (Describe Your TTS Model)

Create `configs/models/my_model.yaml`:

### If your model speaks WebSocket (streaming PCM):
```yaml
model_id: my-model
model_version: "v1.0"
provider: MyCompany
adapter: websocket_pcm
adapter_config:
  url: ${MY_MODEL_URL:-ws://localhost:9000}   # env var expansion works
  encoding: float32                            # or int16
  field_map:                                   # maps tts_eval fields → your API fields
    text: prompt
    voice: speaker
    language: lang
sample_rate: 24000
voices: [voice_1, voice_2]
default_voice: voice_1
languages: [en, hi, ta]
determinism: best_effort    # or "deterministic" / "seeded"
```

### If your model speaks HTTP REST:
```yaml
model_id: my-rest-model
model_version: "v2.0"
provider: MyCompany
adapter: http_rest
adapter_config:
  url: ${MY_MODEL_URL:-http://localhost:8000/synthesize}
  method: POST
  headers:
    Authorization: "Bearer ${API_KEY}"    # env var for secrets
  response_type: streaming                # or "json_base64"
  # audio_path: "audios[0]"              # JSON path if response_type is json_base64
sample_rate: 24000
voices: [default]
default_voice: default
languages: [en]
determinism: best_effort
```

### Built-in adapters:

| Adapter | Protocol | Used by |
|---------|----------|---------|
| `websocket_pcm` | WebSocket binary PCM frames | Indic-Mio, AI4Bharat Parler |
| `http_rest` | HTTP POST/GET, streaming or JSON | Sarvam, ElevenLabs, OpenAI |
| `mock` | Offline deterministic signal | Testing (no server needed) |
| `replay` | Read pre-generated WAVs from disk | Re-scoring without re-synthesis |

### Env var expansion:
`${VAR:-default}` syntax works in adapter_config. Secrets auto-redacted before saving to disk.

---

## 4. Suite Config (Define Evaluation Protocol)

Use built-in suites or create your own.

### Built-in suites:

| Suite | File | Purpose |
|-------|------|---------|
| `smoke` | `configs/suites/smoke.yaml` | 13 utterances, concurrency 1, fast confidence check |
| `indic-full` | `configs/suites/indic-full.yaml` | 69 utterances, all metrics, concurrency 4 |
| `latency` | `configs/suites/latency.yaml` | Latency & capacity profiling only (no ASR/MOS overhead) |
| `offline-rescore` | `configs/suites/offline-rescore.yaml` | Re-score existing audio without re-synthesis |

### Custom suite example:
```yaml
suite_id: my-benchmark
dataset: indic_conversational_v1     # built-in 69-utterance multi-language set
metrics: standard                     # "core" | "standard" | "all", or explicit list
concurrency: 4                        # parallel synthesis tasks
seed: 1234                            # reproducibility (or null)
save_audio: true

asr:                                  # needed for CER/WER metrics
  backend: http                       # or "whisper" for local faster-whisper
  url: ${ASR_URL:-http://localhost:8001/transcribe}
  language_field: language
  transcript_path: transcript

thresholds:                           # pass/fail criteria
  cer_verified_max: 0.30
  success_verified_min: 0.90

generation_params:                    # fixed generation overrides
  temperature: 0.7
  top_p: 0.9
```

> [!IMPORTANT]
> `concurrency` is part of the run fingerprint. Runs at different concurrency levels cannot be directly compared.

---

## 5. Run Evaluation

Two equivalent front-ends — the `tts-eval` CLI and the Python API it wraps.

### CLI (fastest path):

```bash
tts-eval run --model my-model --suite smoke      # evaluate
tts-eval list                                    # stored runs
tts-eval report <run-id>                         # (re)build report.html/md + CSVs
tts-eval compare <baseline> <candidate>          # A/B with verdicts
tts-eval verify <run-id>                         # reproducibility drift
tts-eval serve                                   # web UI on 127.0.0.1:8765
tts-eval --help                                  # all commands and flags
```

Exit codes: `0` ok, `1` expected failure (bad config, unreachable server),
`2` a comparison/verify that ran but is not comparable / drifted.

### Python script (copy-paste ready):

```python
from tts_eval.config import load_model_card, load_suite
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore
from tts_eval.report import write_run_report

# 1. Init store
store = RunStore("runs")

# 2. Load configs
card  = load_model_card("my-model")   # looks in configs/models/
suite = load_suite("smoke")           # looks in configs/suites/

# 3. Build plan (validates everything, computes dataset hash)
plan = build_plan(card, suite, output_dir=store.root)

# 4. Run (synthesis + scoring)
record = run_sync(plan)

# 5. Save results
run_dir = store.save(record)

# 6. Generate reports
write_run_report(record, run_dir)

print(f"Done → {run_dir}")
```

Save as `eval_my_model.py`, run:
```bash
python eval_my_model.py
```

### For async code:
```python
import asyncio
from tts_eval.runner import execute

record = asyncio.run(execute(plan))
```

---

## 6. Available Metrics (42 total, 8 criteria)

Metrics degrade individually if backend missing. No crash.

### Latency & Speed
| Metric | Unit | Lower/Higher better |
|--------|------|---------------------|
| `ttfb_ms` | ms | Lower — time to first byte |
| `first_audible_ms` | ms | Lower — time to audible speech |
| `stream_starvation_ms` | ms | Lower — max playout buffer deficit |
| `stream_chunk_gap_p95_ms` | ms | Lower — 95th pctl chunk gap |
| `rtf` | x | Lower — real-time factor, must be < 1.0 for live |
| `inference_time_ms` | ms | Lower |
| `chars_per_second` | char/s | Higher — throughput |
| `throughput_utt_per_min` | utt/min | Higher — run-level |

### Audio Quality (no reference needed)
| Metric | What it catches |
|--------|-----------------|
| `snr_db` | Signal-to-noise ratio |
| `clipping_pct` | Samples at full scale |
| `silence_ratio` | Excess silence |
| `leading_silence_ms` / `trailing_silence_ms` | Delay before/after speech |
| `dynamic_range_db` | Loudness spread |
| `spectral_flatness` | Buzziness / noise-likeness |
| `degeneracy_score` | Autoregressive loops, held tones, silent truncation |
| `audio_quality_score` | Composite 0–1 index |

### Pronunciation (needs ASR backend)
| Metric | Description |
|--------|-------------|
| `cer` | Character Error Rate vs ground truth |
| `wer` | Word Error Rate vs ground truth |
| `slot_accuracy` | Required tokens (`must_contain`) found in transcript |

### Naturalness (optional ML backends)
| Metric | Backend | Description |
|--------|---------|-------------|
| `utmos` | UTMOSv2 | Predicted MOS 1–5 |
| `dnsmos_ovrl` / `sig` / `bak` | DNSMOS P.835 | Overall/Signal/Background quality |
| `subjective_mos` / `mushra` / `cmos` | Human raters | De-blinded human scores |
| `ttsds2_overall` | TTSDS2 | Distributional distance vs real speech |

### Voice & Speaker
| Metric | Description |
|--------|-------------|
| `voice_consistency` | DSP feature dispersion across utterances |
| `intra_utterance_f0_cv` | Pitch stability within utterance |
| `speaker_similarity` | Cosine sim vs reference embedding |
| `speaker_consistency` | Pairwise embedding sim across utterances |

### Coverage & Reliability
| Metric | Description |
|--------|-------------|
| `coverage_ratio` | Fraction of claimed languages verified |
| `success_rate` | Fraction returning usable audio |
| `degenerate_rate` | Fraction with degeneracy detected |

---

## 7. Test Set Format

Built-in dataset: `indic_conversational_v1` — 69 utterances across 13 Indic languages.

Format is **JSONL** (one JSON per line):

```json
{
  "id": "hi-num-01",
  "text": "आपका बकाया ₹12,450 है और अंतिम तिथि 5 मार्च 2026 है।",
  "language": "hi",
  "category": "numeric",
  "script": "Devanagari",
  "expected_transcript": "आपका बकाया बारह हजार चार सौ पचास रुपये है...",
  "must_contain": ["बारह हजार", "पांच मार्च"],
  "notes": "Tests Latin digits and currency inside Devanagari"
}
```

Key fields:
- `id` — unique utterance ID (used for cross-run pairing)
- `text` — text sent to TTS
- `language` — language code
- `expected_transcript` — spoken expansion (CER scored against this, not `text`)
- `must_contain` — required tokens for `slot_accuracy`

> [!IMPORTANT]
> `expected_transcript` is crucial for Indic languages. Prevents false CER penalties when TTS correctly verbalizes `₹12,450` as "बारह हजार चार सौ पचास रुपये".

---

## 8. Output

Run creates structured directory in `runs/<run_id>/`:

```
runs/<run_id>/
├── run.json          # Complete RunRecord (source of truth)
├── report.html       # Standalone HTML report with audio player
├── report.md         # Markdown summary (for PRs/docs)
├── utterances.csv    # Per-utterance raw data
├── aggregates.csv    # Run-level aggregates with bootstrap 95% CIs
├── coverage.csv      # Per-language coverage matrix
└── audio/
    ├── en-greet-01.wav
    ├── hi-num-01.wav
    └── timings.json  # TTFB, first-audible, total_ms per utterance
```

### Quick check after run:
```bash
# Summary stats
cat runs/*/aggregates.csv

# Full record
cat runs/*/run.json | python -m json.tool | head -50

# Open HTML report in browser
xdg-open runs/*/report.html
```

---

## 9. Adding Your Own Model

### Path A: Supported protocol (zero Python)

Model uses WebSocket or HTTP REST? → One YAML file in `configs/models/`. Done.

See Section 3 for templates.

### Path B: Custom wire protocol

Subclass `TTSAdapter`, use `@register_adapter`:

```python
from tts_eval.adapters.base import TTSAdapter, _Capture, register_adapter
from tts_eval.types import SynthesisRequest
from tts_eval.errors import SynthesisFailed

@register_adapter
class MyProtocolAdapter(TTSAdapter):
    name = "my_protocol"

    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        # Connect to your server
        # As audio frames arrive: capture.chunk(pcm_numpy_array)
        # When metadata available: capture.meta(sample_rate=24000)
        # On failure: raise SynthesisFailed("reason")
        pass
```

Load without modifying tts_eval — set in model card:
```yaml
adapter: my_protocol
adapter_module: my_package.my_adapter   # auto-imported at runtime
```

---

## 10. Compare Runs

```python
from tts_eval.compare import compare_runs
from tts_eval.store import RunStore

store = RunStore("runs")
baseline  = store.load("run_id_baseline")
candidate = store.load("run_id_candidate")

report = compare_runs(baseline, candidate)
# Pairs utterances by ID, bootstrap 95% CIs on paired differences
# Verdicts: better | worse | negligible | inconclusive
```

**Comparability rules:**
- `dataset_hash` and `concurrency` must match → else comparison **blocked**
- Different ASR backends or hardware → **warning** issued

---

## 11. Web UI

Browse runs, listen to audio, compare side-by-side:

```bash
python -m tts_eval.ui --runs runs --port 8765
# Open http://127.0.0.1:8765
```

---

## 12. Fingerprinting & Reproducibility

Every run gets a SHA256 `fingerprint` covering:
- Dataset content hash
- Model ID + version
- Adapter + voice
- Generation params + seed
- Concurrency
- Metric list + thresholds + ASR config

Find identical prior runs:
```python
store.find_repeats(fingerprint)
```

---

## 13. Quick-Start: Full Copy-Paste

### Evaluate with mock model (no server needed):
```bash
cd /path/to/tts_eval
pip install -e .

python - << 'EOF'
from tts_eval.config import load_model_card, load_suite
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore
from tts_eval.report import write_run_report

store = RunStore("runs")
card  = load_model_card("mock")
suite = load_suite("smoke")
plan  = build_plan(card, suite, output_dir=store.root)
record = run_sync(plan)
run_dir = store.save(record)
write_run_report(record, run_dir)
print(f"Results → {run_dir}")
EOF
```

### Evaluate your real model:
```bash
# 1. Create model card
cat > configs/models/my_model.yaml << 'YAML'
model_id: my-model
model_version: "v1"
provider: MyTeam
adapter: http_rest
adapter_config:
  url: http://localhost:8000/synthesize
  method: POST
  response_type: streaming
sample_rate: 24000
voices: [default]
default_voice: default
languages: [en, hi]
determinism: best_effort
YAML

# 2. Start your TTS server, then run:
python - << 'EOF'
from tts_eval.config import load_model_card, load_suite
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore
from tts_eval.report import write_run_report

store = RunStore("runs")
card  = load_model_card("my-model")
suite = load_suite("smoke")          # swap to "indic-full" for full benchmark
plan  = build_plan(card, suite, output_dir=store.root)
record = run_sync(plan)
run_dir = store.save(record)
write_run_report(record, run_dir)
print(f"Results → {run_dir}")
print("Open report.html in browser for full breakdown")
EOF

# 3. Check results
cat runs/*/aggregates.csv
```

---

## 14. Cheat Sheet

| Task | How |
|------|-----|
| Run eval | `tts-eval run -m <model> -s <suite>` (or API: `build_plan` → `run_sync` → `store.save`) |
| Quick test (no server) | `tts-eval run -m mock -s smoke` |
| Connect new model (HTTP/WS) | Create YAML in `configs/models/` |
| Connect model (custom protocol) | Subclass `TTSAdapter` + `@register_adapter` |
| Choose metrics | Suite config: `metrics: core\|standard\|all` or explicit list |
| Compare two runs | `compare_runs(baseline, candidate)` |
| Browse results in browser | `python -m tts_eval.ui --runs runs` |
| Reproducibility check | `store.find_repeats(fingerprint)` |
| Human listening test | `tts_eval.subjective.build_test(...)` |
| Re-score without re-synthesis | Use `replay` adapter + `offline-rescore` suite |
