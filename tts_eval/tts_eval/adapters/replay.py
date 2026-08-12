"""Adapter that reads pre-generated WAV files instead of calling a model.

Three jobs:

*   **Re-scoring without re-synthesis.** A finished run keeps its audio, so a new
    metric (or a newly installed MOS backend) can be applied to the *exact* audio
    that was already reviewed. Without this, adding a metric would mean
    re-running an autoregressive model whose sampling is stochastic, and the new
    numbers would not be comparable with the old ones.
*   **Evaluating third-party audio.** A vendor or research group can hand over a
    directory of WAVs for the same test set and be benchmarked on identical
    footing, with no access to their server. This is also how TTSDS2-style
    corpus comparisons are fed.
*   **Auditability.** Anyone can reproduce a published report from the stored
    artefacts alone.

Latency is deliberately *not* invented here. If a sidecar ``timings.json`` from
the original run is present, real timings are replayed; otherwise latency metrics
are reported as ``not_computed`` rather than fabricated from file size.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ..audio import read_wav
from ..errors import AdapterUnavailable, ConfigError, SynthesisFailed
from ..types import Capabilities, Determinism, SynthesisRequest
from .base import TTSAdapter, _Capture, register_adapter


@register_adapter
class ReplayAdapter(TTSAdapter):
    name = "replay"
    requires = ()

    def __init__(self, config: Mapping[str, Any]):
        super().__init__(config)
        raw_dir = self.config.get("audio_dir")
        if not raw_dir:
            raise ConfigError("replay adapter requires 'audio_dir' in adapter_config")
        self._dir = Path(str(raw_dir)).expanduser()
        # ``{utterance_id}`` is the only required placeholder; ``{language}`` and
        # ``{voice}`` are available for vendor drops organised by folder.
        self._pattern = str(self.config.get("filename_pattern") or "{utterance_id}.wav")
        self._timings = self._load_timings()

    def _build_capabilities(self, config: Mapping[str, Any]) -> Capabilities:
        base = super()._build_capabilities(config)
        return Capabilities(
            streaming=False,
            voices=base.voices,
            languages=base.languages,
            supports_seed=False,
            supports_emotion=False,
            native_sample_rate=base.native_sample_rate,
            # Reading a file back is the only truly deterministic "synthesis".
            determinism=Determinism.DETERMINISTIC,
        )

    def _load_timings(self) -> dict[str, dict[str, Any]]:
        """Load real timings from the source run, if the drop includes them."""
        path = self._dir / "timings.json"
        if not path.is_file():
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        if isinstance(data, Mapping):
            return {str(k): dict(v) for k, v in data.items() if isinstance(v, Mapping)}
        return {}

    async def probe(self) -> None:
        if not self._dir.is_dir():
            raise AdapterUnavailable(f"replay audio_dir does not exist: {self._dir}")

    async def _synthesise(self, request: SynthesisRequest, capture: _Capture) -> None:
        rel = self._pattern.format(
            utterance_id=request.utterance_id,
            language=request.language,
            voice=request.voice or "",
        )
        path = self._dir / rel
        if not path.is_file():
            raise SynthesisFailed(f"no replay audio at {path}")

        try:
            buf = read_wav(path)
        except Exception as e:  # noqa: BLE001
            raise SynthesisFailed(f"could not read {path}: {type(e).__name__}: {e}") from e

        capture.sample_rate = buf.sample_rate
        capture.meta(sample_rate=buf.sample_rate, source_path=str(path), replayed=True)
        capture.chunk(buf.samples)

        # Overwrite the (meaningless) file-read timings with the originals when
        # available; leave them None otherwise so metrics say "not measured".
        recorded = self._timings.get(request.utterance_id)
        if recorded:
            capture.ttfb_ms = _opt_float(recorded.get("ttfb_ms"))
            capture.first_audible_ms = _opt_float(recorded.get("first_audible_ms"))
            capture.meta(replayed_timings=True)
        else:
            capture.ttfb_ms = None
            capture.first_audible_ms = None


def _opt_float(v: Any) -> float | None:
    try:
        return float(v) if v is not None else None
    except (TypeError, ValueError):
        return None


__all__ = ["ReplayAdapter"]
