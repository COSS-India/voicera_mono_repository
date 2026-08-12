"""Predicted-MOS backends: UTMOS and DNSMOS. Optional, GPU-friendly, skippable.

Both are *predictors*. The 2026 open-source-TTS survey and the zero-shot TTS
evaluation literature both document cases where UTMOS ranks systems differently
from human listeners — one survey found human raters placing a system third on
naturalness where UTMOS did not agree. Two consequences are built into this code
rather than left to a footnote:

1.  These metrics are never merged into a single "quality score". They sit in the
    Naturalness section next to ``subjective_mos``, and the report prints the
    caveat whenever predicted MOS is present without human ratings.
2.  UTMOS was trained on clips up to ~10 s, so score spread compresses on longer
    audio. Utterances beyond that get an explicit note on the value rather than a
    silently squashed score.

Neither backend is required. Absent dependencies produce ``not_computed`` values
naming the missing package, and the run proceeds.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from ...audio import resample
from ...datasets.loader import TestCase
from ...types import MetricValue, SynthesisResult
from ..base import MetricContext, UtteranceBackend, make_value, missing_value, register_backend

# Both predictors are defined at 16 kHz.
_MOS_SAMPLE_RATE = 16000
# Above this, UTMOS score spread compresses (training clips were <= 10 s).
_UTMOS_TRAINED_MAX_S = 10.0


@register_backend
class UTMOSBackend(UtteranceBackend):
    """UTMOS22 naturalness prediction (1-5) via torch.hub.

    Weights come from ``sarulab-speech/UTMOS22`` and are cached by torch.hub, so
    the first run needs network access. Set ``options.repo_dir`` to a local
    checkout for fully offline operation.
    """

    name = "utmos"
    provides = ("utmos",)

    def available(self) -> tuple[bool, str]:
        try:
            import torch  # noqa: F401
        except ImportError:
            return False, "torch not installed (pip install 'tts-eval[mos]')"
        return True, "utmos22_strong"

    def prepare(self, ctx: MetricContext) -> None:
        import torch

        repo = self.options.get("repo_dir") or "sarulab-speech/UTMOS22"
        source = "local" if self.options.get("repo_dir") else "github"
        self._device = self.options.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
        self._torch = torch
        self._model = torch.hub.load(
            repo, "utmos22_strong", source=source, trust_repo=True
        ).to(self._device)
        self._model.eval()

    def teardown(self) -> None:
        self._model = None

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None
        wav = resample(audio, _MOS_SAMPLE_RATE)
        tensor = self._torch.from_numpy(np.ascontiguousarray(wav.samples)).float()
        tensor = tensor.unsqueeze(0).to(self._device)

        with self._torch.no_grad():
            score = float(self._model(tensor, _MOS_SAMPLE_RATE).item())

        note = None
        if audio.duration_s > _UTMOS_TRAINED_MAX_S:
            note = (
                f"utterance is {audio.duration_s:.1f}s; UTMOS was trained on clips up to "
                f"{_UTMOS_TRAINED_MAX_S:.0f}s, so score spread is compressed here"
            )
        return {
            "utmos": make_value(
                "utmos",
                score,
                detail=note,
                extra={"duration_s": round(audio.duration_s, 3), "model": "utmos22_strong"},
            )
        }


@register_backend
class DNSMOSBackend(UtteranceBackend):
    """DNSMOS P.835 (signal / background / overall) via ONNX Runtime.

    Weights are not redistributable here; point ``options.model_path`` at the
    ``sig_bak_ovr.onnx`` from Microsoft's DNS-Challenge repo. Without it the
    backend reports unavailable with that instruction rather than failing mid-run.
    """

    name = "dnsmos"
    provides = ("dnsmos_ovrl", "dnsmos_sig", "dnsmos_bak")

    # DNSMOS scores fixed-length segments; the reference implementation uses 9 s
    # with a hop, and averages. Shorter audio is zero-padded, as upstream does.
    _SEGMENT_S = 9.0

    def available(self) -> tuple[bool, str]:
        try:
            import onnxruntime  # noqa: F401
        except ImportError:
            return False, "onnxruntime not installed (pip install onnxruntime)"
        path = self.options.get("model_path")
        if not path:
            return False, (
                "no model_path configured; download sig_bak_ovr.onnx from the "
                "DNS-Challenge repo and set metrics.dnsmos.model_path"
            )
        from pathlib import Path

        if not Path(str(path)).is_file():
            return False, f"model_path does not exist: {path}"
        return True, f"dnsmos-p835 ({Path(str(path)).name})"

    def prepare(self, ctx: MetricContext) -> None:
        import onnxruntime

        providers = self.options.get("providers") or ["CPUExecutionProvider"]
        self._session = onnxruntime.InferenceSession(str(self.options["model_path"]), providers=providers)
        self._input_name = self._session.get_inputs()[0].name

    def teardown(self) -> None:
        self._session = None

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None
        wav = resample(audio, _MOS_SAMPLE_RATE).samples
        seg_len = int(self._SEGMENT_S * _MOS_SAMPLE_RATE)
        if wav.size < seg_len:
            wav = np.pad(wav, (0, seg_len - wav.size))

        # Average over non-overlapping segments: a single leading segment would
        # miss a collapse that happens late in a long utterance.
        scores: list[np.ndarray] = []
        for start in range(0, wav.size - seg_len + 1, seg_len):
            chunk = wav[start : start + seg_len].astype(np.float32)[None, :]
            out = self._session.run(None, {self._input_name: chunk})[0]
            scores.append(np.asarray(out).reshape(-1)[:3])
        if not scores:
            return {
                name: missing_value(name, "audio too short to score") for name in self.provides
            }

        sig, bak, ovrl = np.mean(np.stack(scores), axis=0)[:3]
        extra = {"n_segments": len(scores)}
        return {
            "dnsmos_sig": make_value("dnsmos_sig", float(sig), extra=extra),
            "dnsmos_bak": make_value("dnsmos_bak", float(bak), extra=extra),
            "dnsmos_ovrl": make_value("dnsmos_ovrl", float(ovrl), extra=extra),
        }


__all__ = ["DNSMOSBackend", "UTMOSBackend"]
