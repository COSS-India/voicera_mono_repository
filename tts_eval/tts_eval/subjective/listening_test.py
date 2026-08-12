"""Blinded listening-test generation: MUSHRA, MOS, CMOS and SMOS.

Why this exists at all. Every objective naturalness metric available is a
*predictor*, and the literature documents them inverting against human listeners
— the 2026 open-TTS survey found human raters ranking a system differently from
UTMOS, and commercial A/B results have gone the same way. So a framework that
reports predicted MOS and calls the naturalness criterion satisfied is quietly
substituting a proxy for the measurement. This module produces the actual
measurement.

What blinding costs if you skip it: raters who can see which system produced a
clip score the one they expect to win higher, and a fixed presentation order lets
fatigue and anchoring load onto whichever system is always last. Both effects are
larger than the differences being measured. So:

*   system identity is replaced by an opaque token derived from an HMAC, and the
    key stays in a separate answer file the raters never receive;
*   trial order is shuffled per rater with a seeded RNG (reproducible, but
    different for each rater);
*   within a MUSHRA trial the systems are shuffled too;
*   an optional anchor (low-pass filtered reference) is injected so a rater who
    scores it highly can be identified as unreliable and excluded.

Output is a directory that can be zipped and handed to a panel: audio, a
per-rater CSV sheet, and a plain-HTML player that needs no server or install.
"""
from __future__ import annotations

import csv
import hashlib
import hmac
import html
import json
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..audio import read_wav, write_wav
from ..errors import ConfigError
from ..types import AudioBuffer, RunRecord

# Scales, with the ranges and instructions raters actually see.
SCALES: dict[str, dict[str, Any]] = {
    "mos": {
        "label": "MOS — absolute naturalness",
        "min": 1,
        "max": 5,
        "step": 1,
        "instruction": (
            "Rate how natural this voice sounds, as if a person were speaking. "
            "1 = clearly synthetic or hard to listen to, 5 = indistinguishable from a "
            "human recording."
        ),
        "single_system": True,
    },
    "mushra": {
        "label": "MUSHRA — side-by-side naturalness (0-100)",
        "min": 0,
        "max": 100,
        "step": 1,
        "instruction": (
            "All clips in a trial say the same sentence. Rate each from 0 to 100 for "
            "naturalness. Use the whole scale and compare the clips against each other; "
            "at least one clip in each trial should be near the top of your range."
        ),
        "single_system": False,
    },
    "cmos": {
        "label": "CMOS — comparative preference (-3..+3)",
        "min": -3,
        "max": 3,
        "step": 1,
        "instruction": (
            "Two clips say the same sentence. Score how much better B sounds than A: "
            "-3 = A is much better, 0 = they are equivalent, +3 = B is much better."
        ),
        "single_system": False,
    },
    "smos": {
        "label": "SMOS — speaker similarity to the reference (1-5)",
        "min": 1,
        "max": 5,
        "step": 1,
        "instruction": (
            "Compare each clip to the reference recording and rate how much they sound "
            "like the SAME speaker, ignoring what is being said and any difference in "
            "recording quality. 1 = clearly different people, 5 = clearly the same person."
        ),
        "single_system": False,
    },
}


@dataclass
class TestSpec:
    """Parameters of a listening test.

    Named ``Test*`` for domain reasons (it configures a listening *test*), which
    makes pytest try to collect it as a test class. ``__test__ = False`` opts out.
    """

    # Not a pytest test class.
    __test__ = False

    scale: str = "mushra"
    n_raters: int = 5
    # Utterances per rater. Panels lose reliability past ~30-40 trials, so the
    # default keeps a session under about 20 minutes.
    n_trials: int = 20
    seed: int = 4242
    # Insert a deliberately degraded anchor to catch inattentive raters.
    include_anchor: bool = True
    # Include the ground-truth recording as a hidden upper anchor when the dataset
    # provides one. Without it, MUSHRA scores have no ceiling to normalise against.
    include_reference: bool = True
    languages: tuple[str, ...] = ()
    # Blinding key. Generated and stored in the answer key if not supplied.
    blind_key: str | None = None

    def __post_init__(self) -> None:
        if self.scale not in SCALES:
            raise ConfigError(
                f"unknown scale {self.scale!r}; expected one of {', '.join(sorted(SCALES))}"
            )
        if self.n_raters < 1 or self.n_trials < 1:
            raise ConfigError("n_raters and n_trials must both be >= 1")
        if self.scale == "cmos" and self.include_anchor:
            # CMOS is a forced pairwise preference; a third anchor clip has nowhere
            # to go in the comparison.
            self.include_anchor = False


@dataclass
class Trial:
    """One screen for a rater: a sentence and the blinded clips to score."""

    trial_id: str
    utterance_id: str
    language: str
    text: str
    # blinded_token -> relative audio path
    clips: dict[str, str] = field(default_factory=dict)
    # blinded_token -> true system label (answer key only)
    key: dict[str, str] = field(default_factory=dict)
    reference_clip: str | None = None


def blind_token(key: str, run_id: str, utterance_id: str, role: str) -> str:
    """Stable, opaque per-(system, utterance) label.

    HMAC rather than a plain hash so the mapping cannot be brute-forced by a rater
    who knows the run ids — which would defeat blinding for anyone motivated enough
    to try.
    """
    mac = hmac.new(key.encode("utf-8"), f"{run_id}|{utterance_id}|{role}".encode("utf-8"), hashlib.sha256)
    return "sys_" + mac.hexdigest()[:10]


def _anchor(samples, sample_rate: int):
    """Low-pass anchor, per the MUSHRA convention (a 3.5 kHz-limited version).

    Implemented as a moving-average filter because it needs no scipy. It is a crude
    low-pass, which is fine: the anchor's only job is to be obviously worse than
    every system under test, so an inattentive rater who scores it highly can be
    excluded.
    """
    import numpy as np

    # Window length chosen so the -3 dB point lands near 3.5 kHz.
    window = max(3, int(sample_rate / 3500))
    kernel = np.ones(window, dtype=np.float64) / window
    filtered = np.convolve(samples.astype(np.float64), kernel, mode="same")
    return filtered.astype(np.float32)


def build_test(
    runs: Sequence[RunRecord],
    output_dir: str | Path,
    spec: TestSpec | None = None,
    *,
    dataset_texts: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Assemble a blinded listening test from one or more runs.

    One run -> absolute rating (MOS). Two or more -> side-by-side (MUSHRA/CMOS),
    restricted to utterances present and successful in *every* run, because a trial
    where one system is missing cannot be compared.
    """
    spec = spec or TestSpec()
    if not runs:
        raise ConfigError("no runs supplied")
    if SCALES[spec.scale]["single_system"] and len(runs) > 1:
        raise ConfigError(
            f"scale {spec.scale!r} rates one system at a time; pass a single run or use "
            "scale='mushra' to compare several"
        )
    if spec.scale == "cmos" and len(runs) != 2:
        raise ConfigError(f"CMOS compares exactly two systems, got {len(runs)}")

    out = Path(output_dir)
    audio_out = out / "audio"
    audio_out.mkdir(parents=True, exist_ok=True)

    key = spec.blind_key or hashlib.sha256(
        ("|".join(r.run_id for r in runs) + f"|{spec.seed}").encode("utf-8")
    ).hexdigest()[:24]

    # Utterances usable in every run.
    per_run_ok: list[dict[str, Any]] = []
    for record in runs:
        per_run_ok.append(
            {
                u.utterance_id: u
                for u in record.utterances
                if u.result.ok and u.result.audio_path and Path(u.result.audio_path).is_file()
            }
        )
    common = set(per_run_ok[0])
    for other in per_run_ok[1:]:
        common &= set(other)
    if spec.languages:
        wanted = set(spec.languages)
        common = {
            uid for uid in common if per_run_ok[0][uid].language in wanted
        }
    if not common:
        raise ConfigError(
            "no utterance has usable audio in every run — check that the runs saved audio "
            "and that they used the same test set"
        )

    # Stratify by language so a 20-trial sheet does not accidentally cover two
    # languages out of thirteen.
    rng = random.Random(f"{key}|{spec.seed}|selection")
    by_language: dict[str, list[str]] = {}
    for uid in sorted(common):
        by_language.setdefault(per_run_ok[0][uid].language, []).append(uid)
    for pool in by_language.values():
        rng.shuffle(pool)

    selected: list[str] = []
    languages = sorted(by_language)
    while len(selected) < min(spec.n_trials, len(common)):
        progressed = False
        for lang in languages:
            if by_language[lang]:
                selected.append(by_language[lang].pop())
                progressed = True
                if len(selected) >= min(spec.n_trials, len(common)):
                    break
        if not progressed:
            break

    texts = dict(dataset_texts or {})
    trials: list[Trial] = []
    for uid in selected:
        record0 = per_run_ok[0][uid]
        trial = Trial(
            trial_id=f"t{len(trials) + 1:03d}",
            utterance_id=uid,
            language=record0.language,
            text=texts.get(uid, record0.result.request.text),
        )
        for run_index, (record, ok_map) in enumerate(zip(runs, per_run_ok)):
            token = blind_token(key, record.run_id, uid, f"system{run_index}")
            source = Path(ok_map[uid].result.audio_path)
            dest_name = f"{trial.trial_id}_{token}.wav"
            shutil.copyfile(source, audio_out / dest_name)
            trial.clips[token] = f"audio/{dest_name}"
            trial.key[token] = f"{record.display_name} ({record.run_id})"

        if spec.include_anchor:
            token = blind_token(key, "anchor", uid, "anchor")
            buf = read_wav(Path(per_run_ok[0][uid].result.audio_path))
            dest_name = f"{trial.trial_id}_{token}.wav"
            write_wav(
                audio_out / dest_name,
                AudioBuffer(
                    samples=_anchor(buf.samples, buf.sample_rate), sample_rate=buf.sample_rate
                ),
            )
            trial.clips[token] = f"audio/{dest_name}"
            trial.key[token] = "ANCHOR (3.5 kHz low-pass — should score low)"

        reference = record0.result.request.reference_audio
        if spec.include_reference and reference and Path(reference).is_file():
            dest_name = f"{trial.trial_id}_reference.wav"
            shutil.copyfile(reference, audio_out / dest_name)
            trial.reference_clip = f"audio/{dest_name}"

        trials.append(trial)

    # Per-rater sheets, each with its own shuffle.
    sheets: list[str] = []
    for rater_index in range(1, spec.n_raters + 1):
        rater_id = f"rater{rater_index:02d}"
        rater_rng = random.Random(f"{key}|{spec.seed}|{rater_id}")
        ordered = list(trials)
        rater_rng.shuffle(ordered)
        sheets.append(_write_sheet(out, rater_id, ordered, spec, rater_rng))

    manifest = {
        "scale": spec.scale,
        "scale_label": SCALES[spec.scale]["label"],
        "instruction": SCALES[spec.scale]["instruction"],
        "range": [SCALES[spec.scale]["min"], SCALES[spec.scale]["max"]],
        "runs": [
            {"run_id": r.run_id, "model": r.display_name, "label": r.label} for r in runs
        ],
        "n_raters": spec.n_raters,
        "n_trials": len(trials),
        "languages": sorted({t.language for t in trials}),
        "seed": spec.seed,
        "include_anchor": spec.include_anchor,
        "sheets": sheets,
        "trials": [
            {
                "trial_id": t.trial_id,
                "utterance_id": t.utterance_id,
                "language": t.language,
                "clips": t.clips,
                "reference_clip": t.reference_clip,
            }
            for t in trials
        ],
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # The answer key is written SEPARATELY and must not be shipped to raters.
    answer_key = {
        "blind_key": key,
        "warning": "DO NOT SEND THIS FILE TO RATERS — it de-blinds the test.",
        "runs": {r.run_id: r.display_name for r in runs},
        "trials": {
            t.trial_id: {"utterance_id": t.utterance_id, "systems": t.key} for t in trials
        },
    }
    (out / "ANSWER_KEY.json").write_text(
        json.dumps(answer_key, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _write_player(out, trials, spec, manifest)
    return manifest


def _write_sheet(
    out: Path, rater_id: str, trials: Sequence[Trial], spec: TestSpec, rng: random.Random
) -> str:
    """One CSV per rater: the file they fill in and send back."""
    path = out / f"sheet_{rater_id}.csv"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["rater_id", "trial_id", "utterance_id", "language", "system_token", "clip", "score", "comment"]
        )
        for trial in trials:
            tokens = list(trial.clips)
            # Shuffle within the trial too, so system A is not always the first
            # column a rater hears.
            rng.shuffle(tokens)
            for token in tokens:
                writer.writerow(
                    [rater_id, trial.trial_id, trial.utterance_id, trial.language, token,
                     trial.clips[token], "", ""]
                )
    return path.name


def _write_player(out: Path, trials: Sequence[Trial], spec: TestSpec, manifest: Mapping[str, Any]) -> None:
    """Single-file HTML player.

    Plain HTML + CSS + a little vanilla JS, opened with a double-click. No server,
    no npm, no install: a listening panel is usually non-technical, and anything
    that needs a toolchain does not get used.
    """
    scale = SCALES[spec.scale]
    rows: list[str] = []
    for index, trial in enumerate(trials, start=1):
        clips = "".join(
            f"""
        <div class="clip">
          <span class="token">{html.escape(token)}</span>
          <audio controls preload="none" src="{html.escape(path)}"></audio>
          <input type="number" min="{scale['min']}" max="{scale['max']}" step="{scale['step']}"
                 data-trial="{html.escape(trial.trial_id)}" data-token="{html.escape(token)}"
                 placeholder="{scale['min']}-{scale['max']}">
        </div>"""
            for token, path in trial.clips.items()
        )
        reference = (
            f'<div class="reference"><span>Reference recording</span>'
            f'<audio controls preload="none" src="{html.escape(trial.reference_clip)}"></audio></div>'
            if trial.reference_clip
            else ""
        )
        rows.append(
            f"""
      <section class="trial">
        <h3>Trial {index} of {len(trials)} <small>{html.escape(trial.language)}</small></h3>
        <p class="text">{html.escape(trial.text)}</p>
        {reference}
        <div class="clips">{clips}</div>
      </section>"""
        )

    document = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Listening test — {html.escape(str(scale['label']))}</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ font: 15px/1.55 system-ui, -apple-system, "Segoe UI", sans-serif;
         max-width: 60rem; margin: 0 auto; padding: 2rem 1.25rem 6rem; }}
  h1 {{ font-size: 1.5rem; margin-bottom: .25rem; }}
  .instruction {{ background: #eef4ff; border-left: 4px solid #3b6fd4; padding: .9rem 1.1rem;
                  border-radius: 4px; margin: 1rem 0 2rem; }}
  @media (prefers-color-scheme: dark) {{ .instruction {{ background: #17233c; }} }}
  .trial {{ border: 1px solid #d7dbe3; border-radius: 8px; padding: 1rem 1.2rem; margin-bottom: 1.25rem; }}
  @media (prefers-color-scheme: dark) {{ .trial {{ border-color: #333a48; }} }}
  .trial h3 {{ margin: 0 0 .5rem; font-size: 1rem; }}
  .trial h3 small {{ font-weight: 400; opacity: .6; margin-left: .5rem; }}
  .text {{ font-size: 1.1rem; margin: .25rem 0 1rem; }}
  .clip, .reference {{ display: flex; align-items: center; gap: .75rem; margin-bottom: .5rem; flex-wrap: wrap; }}
  .token {{ font-family: ui-monospace, monospace; font-size: .8rem; opacity: .65; min-width: 8rem; }}
  .reference span {{ font-size: .8rem; font-weight: 600; min-width: 8rem; }}
  audio {{ height: 34px; flex: 1 1 18rem; }}
  input[type=number] {{ width: 5.5rem; padding: .35rem .5rem; border-radius: 4px;
                        border: 1px solid #b9c0cc; }}
  #bar {{ position: fixed; inset: auto 0 0 0; background: #111827; color: #fff;
          padding: .8rem 1.25rem; display: flex; gap: 1rem; align-items: center; justify-content: center; }}
  #bar input {{ padding: .4rem .6rem; border-radius: 4px; border: 0; }}
  button {{ background: #3b6fd4; color: #fff; border: 0; padding: .5rem 1.1rem;
            border-radius: 4px; font-size: .95rem; cursor: pointer; }}
  #count {{ font-variant-numeric: tabular-nums; opacity: .8; font-size: .9rem; }}
</style>
</head>
<body>
  <h1>Listening test</h1>
  <p><strong>{html.escape(str(scale['label']))}</strong> &mdash;
     {manifest['n_trials']} trials, {len(manifest['languages'])} language(s)</p>
  <div class="instruction">
    <p>{html.escape(str(scale['instruction']))}</p>
    <p>Use headphones in a quiet room. Play each clip at least once before scoring.
       Clip labels are meaningless codes: they do not indicate which system produced a clip.</p>
  </div>
  {''.join(rows)}
  <div id="bar">
    <label>Your rater id <input id="rater" placeholder="rater01" size="10"></label>
    <span id="count">0 scored</span>
    <button id="download">Download my scores (CSV)</button>
  </div>
<script>
// Scores are kept in the page and exported as the same CSV shape the sheets use,
// so `tts-eval subjective ingest` accepts either without a conversion step.
const inputs = Array.from(document.querySelectorAll('input[type=number]'));
const count = document.getElementById('count');
const meta = {json.dumps({t.trial_id: {"utterance_id": t.utterance_id, "language": t.language, "clips": t.clips} for t in trials})};
function refresh() {{
  count.textContent = inputs.filter(i => i.value !== '').length + ' of ' + inputs.length + ' scored';
}}
inputs.forEach(i => i.addEventListener('input', refresh));
refresh();
document.getElementById('download').addEventListener('click', () => {{
  const rater = (document.getElementById('rater').value || 'rater01').trim();
  const rows = [['rater_id','trial_id','utterance_id','language','system_token','clip','score','comment']];
  for (const i of inputs) {{
    if (i.value === '') continue;
    const t = i.dataset.trial, tok = i.dataset.token;
    rows.push([rater, t, meta[t].utterance_id, meta[t].language, tok, meta[t].clips[tok], i.value, '']);
  }}
  if (rows.length === 1) {{ alert('Nothing scored yet.'); return; }}
  const csv = rows.map(r => r.map(v => '"' + String(v).replace(/"/g, '""') + '"').join(',')).join('\\n');
  const url = URL.createObjectURL(new Blob([csv], {{type: 'text/csv'}}));
  const a = document.createElement('a');
  a.href = url; a.download = 'sheet_' + rater + '.csv'; a.click();
  URL.revokeObjectURL(url);
}});
</script>
</body>
</html>
"""
    (out / "index.html").write_text(document, encoding="utf-8")


def iter_scales() -> Iterable[tuple[str, str]]:
    return [(name, str(cfg["label"])) for name, cfg in sorted(SCALES.items())]


__all__ = ["SCALES", "TestSpec", "Trial", "blind_token", "build_test", "iter_scales"]
