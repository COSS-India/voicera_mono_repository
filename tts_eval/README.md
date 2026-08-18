# tts_eval

Standalone evaluation framework for TTS models (Indic-Mio, AI4Bharat Parler,
Sarvam, or any other provider). Zero imports from the rest of this monorepo —
it talks to TTS servers over the wire only, so this directory can be copied
out and it still works.

Two equivalent interfaces: a **`tts-eval` CLI** for day-to-day use and the
**Python API** it wraps. Anything one can do, the other can — pick whichever
fits. The CLI is the fast path:

```bash
tts-eval run -m mock -s smoke          # evaluate (mock model, no server/GPU)
tts-eval list                          # stored runs
tts-eval compare <baseline> <candidate>
tts-eval report <run-id>               # (re)generate report.html/md + CSVs
tts-eval verify <run-id>               # re-run and report reproducibility drift
tts-eval dataset show indic_conversational_v1
tts-eval serve                         # browse runs at http://127.0.0.1:8765
tts-eval --help                        # every command and flag
```

See `STATUS.md` for what's built/tested and known limitations.

## Install

```bash
cd tts_eval
pip install -e .                        # core: numpy + PyYAML only
pip install -r requirements-adapters.txt  # + websockets/aiohttp, for real servers
pip install -r requirements-metrics.txt   # + UTMOS/DNSMOS/ASR/speaker/TTSDS2 (optional, heavy)
```

Nothing below the core install is required to run the mock model and read a
full report. Everything heavier degrades to `not_computed` with a reason when
absent — it never crashes a run.

## 5-minute quickstart (no server, no GPU)

```python
from tts_eval.config import load_model_card, load_suite
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore
from tts_eval.report import write_run_report

store = RunStore("runs")                       # runs/ is created if missing
card = load_model_card("mock")                 # offline, deterministic model
suite = load_suite("smoke")                    # 13-utterance fast suite

record = run_sync(build_plan(card, suite, output_dir=store.root))
run_dir = store.save(record)
write_run_report(record, run_dir)

print(f"open {run_dir / 'report.html'}")
```

Open `report.html` in a browser — no server needed, it's one self-contained
file. That's the whole workflow. Everything else in this README is a variation
on these five calls.

## Evaluate a real model

Point a model card's `adapter_config.url` at your server (via env var or
directly), then run the same suite. The CLI is the fast path:

```bash
export INDIC_MIO_TTS_URL=ws://your-gpu-box:8003       # the indic-mio card reads this
tts-eval run --model indic-mio --suite indic-full --label mio-run-1
```

The equivalent Python API:

```python
from tts_eval.config import load_model_card, load_suite
from tts_eval.runner import build_plan, run_sync
from tts_eval.store import RunStore
from tts_eval.report import write_run_report

store = RunStore("runs")
card = load_model_card("indic-mio")            # reads INDIC_MIO_TTS_URL
suite = load_suite("indic-full")               # full 124-utterance set, all metrics

record = run_sync(build_plan(card, suite, output_dir=store.root))
run_dir = store.save(record)
write_run_report(record, run_dir)
```

Bundled model cards (`configs/models/`): `mock`, `indic-mio`, `ai4bharat-parler`,
`sarvam`. Bundled suites (`configs/suites/`): `smoke`, `indic-full`, `latency`,
`offline-rescore`.

### Server URLs are env vars

Each card resolves its endpoint from an environment variable (with a localhost
default), so the same card works against a tunnel, a published port, or an
in-cluster host without editing YAML:

| Card | Env var | Default |
|---|---|---|
| `indic-mio` | `INDIC_MIO_TTS_URL` | `ws://localhost:8003` |
| `ai4bharat-parler` | `AI4BHARAT_TTS_URL` | `ws://localhost:8002` |
| ASR (any suite's `asr:` block) | `TTS_EVAL_ASR_URL` | `http://localhost:8001/transcribe` |

### Reaching on-prem servers over an SSH tunnel

The TTS/STT servers usually run in Docker on a GPU box and aren't published to
the host, so forward each container's port to a local port and point the env
vars at `localhost`. Get a container's IP on the box:

```bash
# on the GPU box — repeat per server (stt, tts, mio)
docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' voicera-prod-stt-1
```

Open the tunnel (one `-L` per server; pick distinct local ports if two TTS
servers must be live at once):

```bash
ssh -i KEY.pem -N \
  -L 127.0.0.1:8003:<MIO_IP>:8003 \
  -L 127.0.0.1:8002:<PARLER_IP>:8002 \
  -L 127.0.0.1:8001:<STT_IP>:8001 \
  ubuntu@<gpu-box>
```

Then point the eval at the tunnel and run:

```bash
export INDIC_MIO_TTS_URL=ws://localhost:8003
export AI4BHARAT_TTS_URL=ws://localhost:8002
export TTS_EVAL_ASR_URL=http://localhost:8001/transcribe

tts-eval run --model indic-mio       --suite indic-full --label mio-1
tts-eval run --model ai4bharat-parler --suite indic-full --label parler-1
```

Verify the STT tunnel before a long run (a non-hang reply means it's reachable):

```bash
curl -s -m 10 -X POST http://localhost:8001/transcribe \
  -H 'Content-Type: application/json' -d '{"audio_b64":"","language_id":"hi"}'; echo
```

Keep the tunnel process alive for the whole run, and don't recreate the
containers mid-run — their IPs change and the tunnel targets go stale. Local
port must match the env var (e.g. if you forward MIO to local `8002`, set
`INDIC_MIO_TTS_URL=ws://localhost:8002`).

### Suite tiers (`metrics:` in the suite YAML)

| Tier | Needs | Gives you |
|---|---|---|
| `core` | nothing | Latency, inference time, audio quality, voice consistency, coverage (intelligibility unverified) |
| `standard` | an ASR endpoint (`asr:` block in the suite) | + Pronunciation Accuracy (CER/WER/slot accuracy), verified coverage |
| `all` | ASR + optional heavy deps | + predicted MOS (UTMOS/DNSMOS), speaker similarity, VERSA, TTSDS2 — whichever are installed |

Round-trip intelligibility needs an ASR endpoint. Point it at an existing
server:

```yaml
# in your suite YAML
asr:
  backend: http_asr          # or `whisper` for a local faster-whisper model
  url: http://localhost:8001/transcribe
  transcript_path: text
```

## Repeat runs for best-effort models (mean ± stdev)

Stochastic models (`determinism: best_effort`, e.g. Indic-Mio and Parler)
synthesise different audio each run, so a single run is one noisy sample — the
report's bootstrap CI captures variance *within* a run (across utterances), not
run-to-run synthesis noise. `scripts/multirun.py` runs a suite N times and
reports the **between-run** mean, stdev, min, max and CV% per metric — the error
bar to judge a go/no-go decision against.

```bash
# execute N fresh runs, then aggregate (writes the summary automatically):
./venv/bin/python scripts/multirun.py \
  --model indic-mio --suite indic-full --runs 5 --label mio-multi
```

Each run is labelled `<label>-01 … -NN`. It writes, next to the first run's dir
(or `--out`):

- `multirun_summary.csv` — mean, stdev, min, max, CV% (= stdev/mean), and how
  many runs produced each metric
- `multirun_summary.html` — sectioned report (`--pdf` also writes a PDF; needs
  Chrome/Chromium)

To (re)aggregate runs you already have — no re-synthesis — select them by label
or by directory glob:

```bash
# by label: matches the exact label and the LABEL-NN batch above
./venv/bin/python scripts/multirun.py --from-label mio-multi

# by directory glob:
./venv/bin/python scripts/multirun.py --from 'runs/20260818T10*'

# control output location + PDF:
./venv/bin/python scripts/multirun.py --from-label mio-multi \
  --out runs/mio-summary.csv --pdf
```

`--from-label` scans the whole runs root (`--runs-root`, default `runs`) and
reads each `run.json`, so it works even if the store index is stale. It matches
across all sessions, so keep batch labels unique if you reuse a prefix.

## Add a new model

**If it speaks a protocol we already support** (WebSocket float32 PCM, or REST
returning WAV/base64), this is the entire cost — one YAML file, no code:

```yaml
# configs/models/my-new-model.yaml
model_id: my-new-model
model_version: "2026-08"          # pin this to a build/checkpoint identifier
provider: MyVendor
adapter: websocket_pcm            # or http_rest
adapter_config:
  url: ${MY_MODEL_URL:-ws://localhost:9000}
  field_map: {text: prompt, voice: voice, language: language}
sample_rate: 24000
voices: [voice_a, voice_b]
languages: [en, hi, ta]
determinism: best_effort          # or `seeded` if it takes and honors a seed
```

Then:

```python
card = load_model_card("my-new-model")
record = run_sync(build_plan(card, load_suite("indic-full"), output_dir="runs"))
```

Same suite, same metrics, same report format, same comparison engine — nothing
else changes. `configs/models/indic-mio.yaml` and
`configs/models/ai4bharat-parler.yaml` are proof: both use `adapter:
websocket_pcm` and share zero code, differing only in URL and field names.

**If it's a genuinely new wire protocol**, subclass `TTSAdapter`
(`tts_eval/adapters/base.py`) — see `websocket_pcm.py` or `http_rest.py` for the
~100-line shape — and register it with `@register_adapter`. Keep it in your own
package and load it with no fork of this repo:

```yaml
adapter_module: my_pkg.custom_adapter   # imported before the adapter is built
adapter: my_protocol
```

## Compare two runs (is B actually better than A?)

```python
from tts_eval.compare import compare_runs
from tts_eval.report import write_comparison_report

baseline = store.load("<older-run-id>")
candidate = store.load("<newer-run-id>")

comparison = compare_runs(baseline, candidate)
print(comparison.summary_line())
# "model@2 vs model@1: 1 better, 2 worse, 1 negligible, 15 inconclusive (69 paired utterances)"

if not comparison.comparable:
    print("blocked:", comparison.blockers)   # e.g. different dataset or concurrency

write_comparison_report(comparison, "runs/compare-a-vs-b")
```

Comparison pairs utterances by id and requires both runs to have used the same
dataset content-hash and concurrency — otherwise it refuses (`blockers`), rather
than producing a number that isn't actually comparable. Verdicts:

- `better` / `worse` — significant **and** past the metric's minimum-effect floor
- `negligible` — real but too small to matter (a 0.001 ms difference is not a win)
- `inconclusive` — confidence interval straddles zero
- `single_observation` — a run-level metric (e.g. `coverage_ratio`) changed
  exactly, but has no variance estimate since there's only one value per run

Check repeatability (the benchmark's own noise floor) before trusting any small
delta:

```python
from tts_eval.compare import repeatability
print(repeatability([store.load(r) for r in ["run1", "run2", "run3"]]))
```

## Browse runs in a browser (lightweight UI)

```bash
python -m tts_eval.ui --runs runs --port 8765
```

Open `http://127.0.0.1:8765`. Lists all runs, click through to a report, play
back individual utterance audio, or use `/compare` to pick two runs and get a
side-by-side comparison. Hover a run to reveal a **Delete** button (removes that
run's directory and index rows); the run list also has a **Clear all runs**
link. Otherwise read-only, no auth, stdlib `http.server` only — binds to
`127.0.0.1` by default, don't expose it beyond your machine as-is.

## Human listening tests (MOS / MUSHRA / CMOS / SMOS)

Predicted MOS (UTMOS/DNSMOS) is documented to disagree with real listeners.
This generates a blinded, randomized listening test and ingests the results:

```python
from tts_eval.subjective import TestSpec, build_test, ingest_sheets, merge_into_run

# 1. Build a blinded test bundle from one or more runs (MUSHRA needs 2+, MOS needs 1)
manifest = build_test(
    [baseline, candidate], "runs/listening-test",
    TestSpec(scale="mushra", n_raters=5, n_trials=20),
)
# -> runs/listening-test/index.html   (send this + audio/ to raters, or use the CSV sheets)
# -> runs/listening-test/ANSWER_KEY.json  (DO NOT send this to raters)

# 2. After raters send back sheet_*.csv (or export from index.html):
report = ingest_sheets(
    sorted(Path("runs/listening-test").glob("sheet_*.csv")),
    "runs/listening-test/ANSWER_KEY.json",
)
print(report.excluded_raters)   # raters who failed the anchor-attention check
print(report.per_system)        # de-blinded per-system means + CIs
print(report.agreement)         # inter-rater agreement, i.e. can you trust the mean

# 3. Merge into the run records so they show up in the normal report
for run_id, scores in report.scores_by_run.items():
    merge_into_run(store.load(run_id), scores)
    store.save(store.load(run_id))
```

## Re-score existing audio (no re-synthesis)

Useful when you install a new metric backend after a run finished, or want to
benchmark a third party's audio drop on identical footing:

```python
from tts_eval.config import load_model_card, load_suite

card = load_model_card("mock")               # any card; only adapter fields matter
card.adapter = "replay"
card.adapter_config = {"audio_dir": str(store.audio_dir("<original-run-id>"))}

record = run_sync(build_plan(card, load_suite("offline-rescore"), output_dir="runs"))
```

If the original run's `audio/timings.json` sidecar is present, real latencies
are replayed; otherwise latency metrics report `not_applicable` instead of
being invented from file size.

## Export raw data

```python
from tts_eval.report import utterances_csv, aggregates_csv, coverage_csv

Path("out/utterances.csv").write_text(utterances_csv(record))
Path("out/aggregates.csv").write_text(aggregates_csv(record))
```

`write_run_report(record, run_dir)` writes all of `report.html`, `report.md`,
and the three CSVs at once.

## Repo layout

```
configs/
  models/          model cards — WHAT is being evaluated (one per model/version)
  suites/          suite configs — HOW it's evaluated (shared across all models)
tts_eval/
  adapters/        wire protocols: websocket_pcm, http_rest, mock, replay
  asr/             round-trip ASR backends for pronunciation accuracy
  datasets/        test-set loading + reproducibility hashing
  metrics/         metric backends (core + optional plugins) and the catalogue
  subjective/      blinded listening tests, ingestion, rater screening
  report/          Markdown / HTML / CSV rendering
  ui/              stdlib http.server UI
  runner.py        orchestration + fingerprinting
  store.py         run registry (JSON source of truth + SQLite index)
  compare.py       cross-run statistical comparison
tests/             pytest suite
```

Model cards and suites are deliberately separate files: a new model reuses
every suite unchanged, and a new benchmark protocol applies to every existing
model unchanged.
