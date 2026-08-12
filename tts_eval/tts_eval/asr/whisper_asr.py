"""Local Whisper ASR via faster-whisper. Optional.

Use this when you want CER/WER numbers **comparable to published TTS work**,
which almost universally reports round-trip WER measured with Whisper
large-v3. Use :mod:`tts_eval.asr.http_asr` against an IndicConformer-class
server when you want the tightest measurement floor for Indian languages.

The trade-off is stated in the run record rather than hidden: this backend's
identity (model size, compute type) is stored, and the comparison engine warns
when two runs used different ASR backends, because a CER delta between them is
not attributable to the TTS models.
"""
from __future__ import annotations

from typing import Any, Mapping

from ..audio import AudioBuffer, resample
from ..errors import MetricUnavailable
from .base import ASRBackend

# Whisper is trained at 16 kHz; feeding anything else silently degrades accuracy.
_WHISPER_SAMPLE_RATE = 16000


class WhisperASRBackend(ASRBackend):
    name = "whisper"

    def __init__(self, options: Mapping[str, Any] | None = None):
        super().__init__(options)
        # large-v3 is the default because it is what published numbers use; drop
        # to "medium" or "small" for a fast smoke run and say so in the report.
        self.model_size = str(self.options.get("model") or "large-v3")
        self.device = str(self.options.get("device") or "auto")
        self.compute_type = str(self.options.get("compute_type") or "default")
        self.beam_size = int(self.options.get("beam_size") or 5)
        # Language is forced from the test case rather than auto-detected: on a
        # badly mispronounced utterance auto-detection picks the wrong language and
        # the resulting CER measures language ID, not pronunciation.
        self.force_language = bool(self.options.get("force_language", True))
        self._model: Any | None = None

    def available(self) -> tuple[bool, str]:
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            return False, "faster-whisper not installed (pip install 'tts-eval[asr-local]')"
        return True, f"faster-whisper/{self.model_size}"

    def _describe_extra(self) -> dict[str, Any]:
        return {
            "model": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
        }

    def prepare(self) -> None:
        from faster_whisper import WhisperModel

        self._model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    def teardown(self) -> None:
        self._model = None

    def transcribe(self, audio: AudioBuffer, language: str) -> str:
        if self._model is None:
            raise MetricUnavailable("whisper model not loaded; prepare() was not called")
        pcm = resample(audio, _WHISPER_SAMPLE_RATE).samples
        segments, _info = self._model.transcribe(
            pcm,
            language=(language.split("-")[0] if self.force_language and language else None),
            beam_size=self.beam_size,
            # No VAD filter: it can trim a quiet or mispronounced start, which is
            # precisely the defect we are trying to detect.
            vad_filter=False,
        )
        return " ".join(segment.text.strip() for segment in segments).strip()


__all__ = ["WhisperASRBackend"]
