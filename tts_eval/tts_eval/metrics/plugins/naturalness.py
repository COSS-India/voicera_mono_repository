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

    Weights come from ``tarepan/SpeechMOS`` -- the hub-packaged UTMOS22, whose
    ``hubconf.py`` exposes ``utmos22_strong`` and whose call signature is
    ``model(wave, sample_rate)``. (The original ``sarulab-speech/UTMOS22`` is the
    training repo; it ships no ``hubconf.py``, so ``torch.hub.load`` cannot use
    it -- pointing there fails every run with a missing-hubconf error.) Cached by
    torch.hub, so the first run needs network; set ``options.repo_dir`` to a local
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

        repo = self.options.get("repo_dir") or "tarepan/SpeechMOS"
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

    # DNSMOS scores fixed-length windows and averages. Match Microsoft's reference
    # ComputeScore exactly: 9.01 s windows, 1 s hop, audio repeated (not zero-
    # padded) until it fills one window, and the raw ONNX outputs mapped through
    # the published P.835 polynomial before averaging. Skipping that polyfit — as
    # an earlier version did — leaves scores on a different scale than published
    # DNSMOS numbers.
    _SEGMENT_S = 9.01
    _HOP_S = 1.0

    # Non-personalised P.835 polynomial coefficients from Microsoft's DNS-Challenge
    # dnsmos_local.py get_polyfit_val(). These map the sig_bak_ovr.onnx raw outputs
    # onto the calibrated 1-5 MOS scale.
    _P_SIG = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
    _P_BAK = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
    _P_OVR = np.poly1d([-0.06766283, 1.11546468, 0.04602535])

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
        hop_len = int(self._HOP_S * _MOS_SAMPLE_RATE)
        # Upstream repeats the clip (not zero-pad) until it fills a window; silence
        # padding would drag the noise/overall scores down on short utterances.
        while wav.size < seg_len:
            wav = np.concatenate([wav, wav])

        # Overlapping 1 s-hop windows, each mapped through the P.835 polyfit, then
        # averaged — the reference DNSMOS procedure. Overlap also catches a late
        # collapse a single leading window would miss.
        scores: list[np.ndarray] = []
        for start in range(0, wav.size - seg_len + 1, hop_len):
            chunk = wav[start : start + seg_len].astype(np.float32)[None, :]
            out = self._session.run(None, {self._input_name: chunk})[0]
            raw_sig, raw_bak, raw_ovr = np.asarray(out).reshape(-1)[:3]
            scores.append(
                np.array(
                    [self._P_SIG(raw_sig), self._P_BAK(raw_bak), self._P_OVR(raw_ovr)],
                    dtype=np.float64,
                )
            )
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
