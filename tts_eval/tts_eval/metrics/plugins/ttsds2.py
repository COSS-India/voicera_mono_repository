"""Bridge to TTSDS2 — the closest thing to a published multilingual TTS benchmark.

Why a bridge and not an implementation: TTSDS2 scores a *system* by comparing the
distribution of its output against distributions of real speech and noise across
prosody, speaker, intelligibility and generic-quality categories. That is a
genuinely different measurement from anything else in this harness (which scores
utterances individually), it is peer-reviewed, and it ships weights. Reproducing
it would mean re-deriving someone else's calibrated benchmark badly.

So this backend does the part the upstream library does not: it takes the audio a
run already produced, hands it to ``ttsds.BenchmarkSuite`` alongside a reference
corpus, and folds the resulting score back into the same run record as every other
metric — so TTSDS2 appears in the same report, the same comparison table and the
same CSV export as latency and CER.

Requirements, all reported as ``not_computed`` when unmet rather than failing:
  * ``pip install ttsds`` (plus its system deps: ffmpeg, automake)
  * a reference corpus of real speech: ``metrics.ttsds2.reference_dir``
  * audio persisted to disk, i.e. the run was not started with ``--no-save-audio``
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ...types import MetricValue, UtteranceRecord
from ..base import MetricContext, RunBackend, make_value, missing_value, register_backend


@register_backend
class TTSDS2Backend(RunBackend):
    name = "ttsds2"
    provides = ("ttsds2_overall",)

    def available(self) -> tuple[bool, str]:
        try:
            # s3prl (a ttsds dependency) calls torchaudio.set_audio_backend()
            # and imports torchaudio.sox_effects at import time, but both were
            # removed in torchaudio ≥ 2.1. Provide harmless stubs so the import
            # chain doesn't crash.
            try:
                import sys as _sys
                import types as _types
                import torchaudio
                if not hasattr(torchaudio, "set_audio_backend"):
                    torchaudio.set_audio_backend = lambda *_a, **_kw: None
                if "torchaudio.sox_effects" not in _sys.modules:
                    _sox = _types.ModuleType("torchaudio.sox_effects")
                    _sox.apply_effects_file = lambda *_a, **_kw: None  # type: ignore[attr-defined]
                    _sox.apply_effects_tensor = lambda *_a, **_kw: None  # type: ignore[attr-defined]
                    _sys.modules["torchaudio.sox_effects"] = _sox
            except ImportError:
                pass
            import ttsds  # noqa: F401
        except ImportError:
            return False, (
                "ttsds not installed — see docs/STANDARDS.md; needs system deps "
                "(ffmpeg, automake) before `pip install ttsds`"
            )
        except Exception as exc:  # noqa: BLE001
            return False, f"ttsds installed but failed to import: {exc}"
        reference_dir = self.options.get("reference_dir")
        if not reference_dir:
            return False, (
                "no reference_dir configured; TTSDS2 is distributional and needs a corpus "
                "of real speech to compare against (e.g. an IndicVoices-R subset). "
                "Set metrics.ttsds2.reference_dir"
            )
        if not Path(str(reference_dir)).is_dir():
            return False, f"reference_dir does not exist: {reference_dir}"
        return True, f"ttsds ({getattr(__import__('ttsds'), '__version__', 'unknown')})"

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        from ttsds import BenchmarkSuite
        from ttsds.util.dataset import DirectoryDataset

        paths = [
            Path(r.result.audio_path)
            for r in records
            if r.result.ok and r.result.audio_path and Path(r.result.audio_path).is_file()
        ]
        if len(paths) < int(self.options.get("min_utterances") or 20):
            return {
                "ttsds2_overall": missing_value(
                    "ttsds2_overall",
                    f"only {len(paths)} audio file(s) on disk; TTSDS2 is a distributional "
                    "benchmark and needs a substantial sample (>= 20) to be meaningful. "
                    "Check that the run saved audio.",
                )
            }

        # All of a run's WAVs already live in one directory, which is exactly the
        # shape DirectoryDataset expects — no copying or re-encoding needed.
        audio_dir = paths[0].parent
        system = DirectoryDataset(str(audio_dir), name=self.options.get("system_name") or "under_test")
        reference = DirectoryDataset(
            str(self.options["reference_dir"]), name="reference_real_speech"
        )

        suite_kwargs: dict[str, Any] = {
            "datasets": [system],
            "reference_datasets": [reference],
        }
        if self.options.get("multilingual", True):
            suite_kwargs["multilingual"] = True
        if self.options.get("category_weights"):
            suite_kwargs["category_weights"] = dict(self.options["category_weights"])
        if self.options.get("noise_dir"):
            suite_kwargs["noise_datasets"] = [
                DirectoryDataset(str(self.options["noise_dir"]), name="noise")
            ]

        suite = BenchmarkSuite(**suite_kwargs)
        suite.run()
        aggregated = suite.get_aggregated_results()

        overall, per_category = _extract_scores(aggregated)
        if overall is None:
            return {
                "ttsds2_overall": missing_value(
                    "ttsds2_overall",
                    "TTSDS2 ran but produced no overall score; inspect the raw results in "
                    "extra.raw",
                )
            }
        return {
            "ttsds2_overall": make_value(
                "ttsds2_overall",
                overall,
                extra={
                    "per_category": per_category,
                    "n_utterances": len(paths),
                    "audio_dir": str(audio_dir),
                    "reference_dir": str(self.options["reference_dir"]),
                },
                detail=(
                    "distributional score vs. the configured real-speech corpus; comparable "
                    "only against other runs using the SAME reference_dir"
                ),
            )
        }


def _extract_scores(aggregated: Any) -> tuple[float | None, dict[str, float]]:
    """Pull the overall and per-category scores out of TTSDS2's result object.

    Tolerant of shape because upstream returns a DataFrame in some versions and a
    dict in others; a version bump should degrade to "no overall score" with the
    raw payload attached, not crash a completed run.
    """
    per_category: dict[str, float] = {}
    overall: float | None = None

    # DataFrame-like
    if hasattr(aggregated, "to_dict"):
        try:
            as_dict = aggregated.to_dict()
        except Exception:  # noqa: BLE001
            as_dict = {}
    elif isinstance(aggregated, Mapping):
        as_dict = dict(aggregated)
    else:
        as_dict = {}

    def _walk(node: Any, prefix: str = "") -> None:
        nonlocal overall
        if isinstance(node, Mapping):
            for k, v in node.items():
                key = f"{prefix}{k}".lower()
                if isinstance(v, (int, float)):
                    if "overall" in key or key.endswith("score"):
                        if overall is None:
                            overall = float(v)
                    else:
                        per_category[str(k)] = float(v)
                else:
                    _walk(v, prefix=f"{k}.")

    _walk(as_dict)
    return overall, per_category


__all__ = ["TTSDS2Backend"]
