"""Bridge to VERSA — the broadest existing speech-evaluation toolkit.

VERSA (CMU WavLab, NAACL 2025) implements ~65 metrics / 729 configured variants:
UTMOS, UTMOSv2, DNSMOS, NISQA, PLCMOS, SpeechBERTScore, PESQ/STOI, speaker
similarity, and more. If a perceptual metric is worth reporting, VERSA probably
already implements it correctly, so this harness drives it rather than
re-implementing any of it.

Integration is by **subprocess**, deliberately:

*   VERSA uses tiered optional dependencies and is installed from its own repo
    (``pip install -e .`` plus per-metric extras). Importing it in-process would
    couple this package's dependency resolution to its, and one conflicting pin
    would break the light core install that has to work everywhere.
*   Its own CLI (``scorer.py`` + a YAML score config) is the interface upstream
    supports and documents, so it is the interface least likely to break.

Because VERSA's CLI has evolved across versions, the command is a configurable
template rather than a hard-coded string, and the output parser accepts any
JSON/JSONL of ``{key: value}`` shape. Set ``metrics.versa.metric_map`` to rename
VERSA's keys onto catalogue names (e.g. ``utmos_score -> utmos``) so its numbers
land in the normal Naturalness section of the report instead of a side table.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from ...types import MetricStatus, MetricValue, UtteranceRecord
from ..base import MetricContext, RunBackend, make_value, missing_value, register_backend

# Default invocation per upstream's documented usage. Placeholders are filled in
# by :meth:`VersaBackend.compute`.
_DEFAULT_COMMAND = (
    "{python} -m versa.bin.scorer "
    "--score_config {score_config} "
    "--pred {pred_list} "
    "--output_file {output_file}"
)


@register_backend
class VersaBackend(RunBackend):
    """Runs VERSA over a completed run's audio and merges its metrics back in.

    Emits ``versa_metrics`` as a container value whose ``extra`` holds every metric
    VERSA returned (mean across utterances), plus catalogue-named metrics for
    anything listed in ``metric_map``.
    """

    name = "versa"
    provides = ("versa_metrics",)

    def available(self) -> tuple[bool, str]:
        score_config = self.options.get("score_config")
        if not score_config:
            return False, (
                "no score_config configured; VERSA needs a YAML listing the metrics to run. "
                "Set metrics.versa.score_config (see docs/STANDARDS.md)"
            )
        if not Path(str(score_config)).is_file():
            return False, f"score_config does not exist: {score_config}"

        python = str(self.options.get("python") or "python3")
        if shutil.which(python) is None:
            return False, f"interpreter {python!r} not found on PATH"

        # Probe importability in the target interpreter rather than this one: VERSA
        # is commonly installed in its own venv precisely because of its dep tree.
        probe = subprocess.run(
            [python, "-c", "import versa, sys; print(getattr(versa, '__version__', 'unknown'))"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if probe.returncode != 0:
            return False, (
                f"`import versa` failed under {python}: {probe.stderr.strip()[:200]} — "
                "install VERSA from https://github.com/wavlab-speech/versa"
            )
        return True, f"versa/{probe.stdout.strip() or 'unknown'}"

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        wavs = [
            (r.utterance_id, Path(r.result.audio_path))
            for r in records
            if r.result.ok and r.result.audio_path and Path(r.result.audio_path).is_file()
        ]
        if not wavs:
            return {
                "versa_metrics": missing_value(
                    "versa_metrics",
                    "no audio on disk to score (was the run started with --no-save-audio?)",
                )
            }

        workdir = ctx.workdir / "versa"
        workdir.mkdir(parents=True, exist_ok=True)
        # VERSA takes a Kaldi-style "<id> <path>" list.
        pred_list = workdir / "pred.scp"
        pred_list.write_text(
            "".join(f"{uid} {path.resolve()}\n" for uid, path in wavs), encoding="utf-8"
        )
        output_file = workdir / "versa_result.json"

        command = str(self.options.get("command") or _DEFAULT_COMMAND).format(
            python=str(self.options.get("python") or "python3"),
            score_config=str(self.options["score_config"]),
            pred_list=str(pred_list),
            output_file=str(output_file),
            n_utterances=len(wavs),
        )
        timeout = float(self.options.get("timeout_s") or 3600.0)

        proc = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout, cwd=str(workdir)
        )
        if proc.returncode != 0:
            return {
                "versa_metrics": MetricValue(
                    name="versa_metrics",
                    value=None,
                    unit="",
                    status=MetricStatus.ERROR,
                    detail=f"VERSA exited {proc.returncode}: {proc.stderr.strip()[-400:]}",
                    extra={"command": command},
                )
            }
        if not output_file.is_file():
            return {
                "versa_metrics": missing_value(
                    "versa_metrics",
                    f"VERSA completed but wrote no output at {output_file}; check the "
                    "`command` template against your VERSA version",
                )
            }

        means, n_rows = _mean_metrics(output_file)
        if not means:
            return {
                "versa_metrics": missing_value(
                    "versa_metrics", f"no numeric metrics found in {output_file}"
                )
            }

        out: dict[str, MetricValue] = {
            "versa_metrics": make_value(
                "versa_metrics",
                float(len(means)),
                unit="count",
                detail=f"{len(means)} VERSA metric(s) averaged over {n_rows} utterance(s)",
                extra={"means": means, "n_rows": n_rows, "command": command},
            )
        }
        # Promote mapped metrics to catalogue names so they appear in the normal
        # report sections rather than buried in an extra payload.
        for versa_key, catalogue_name in (self.options.get("metric_map") or {}).items():
            if versa_key in means:
                out[str(catalogue_name)] = make_value(
                    str(catalogue_name),
                    means[versa_key],
                    detail=f"computed by VERSA ({versa_key})",
                )
        return out


def _mean_metrics(path: Path) -> tuple[dict[str, float], int]:
    """Average every numeric field across VERSA's output rows.

    Accepts a JSON array, a JSON object of rows, or JSONL — upstream has used all
    three — so a version change degrades to "no metrics found" rather than a crash.
    """
    text = path.read_text(encoding="utf-8").strip()
    rows: list[Mapping[str, Any]] = []
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            rows = [r for r in parsed if isinstance(r, Mapping)]
        elif isinstance(parsed, Mapping):
            rows = [v for v in parsed.values() if isinstance(v, Mapping)] or [parsed]
    except json.JSONDecodeError:
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, Mapping):
                rows.append(row)

    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            sums[key] = sums.get(key, 0.0) + float(value)
            counts[key] = counts.get(key, 0) + 1
    return ({k: sums[k] / counts[k] for k in sums}, len(rows))


__all__ = ["VersaBackend"]
