"""Speaker-embedding metrics: similarity to a reference, and embedding-based
cross-utterance consistency. Optional.

This is the SIM / SECS figure published TTS work reports. Two distinct uses:

*   ``speaker_similarity`` — cosine similarity between the synthesised voice and
    the *target* reference recording for that case. Needs
    ``TestCase.reference_audio``; reports ``not_computed`` without it rather than
    inventing a comparison.
*   ``speaker_consistency`` — mean pairwise similarity among utterances of the
    same voice. This is the embedding-based counterpart to the core
    ``voice_consistency`` metric, and both are reported: the DSP estimate always
    runs, the embedding version is more faithful when installed, and showing them
    together makes it obvious when the cheap proxy is misleading.

Backend preference is Resemblyzer (small, fast, no torchaudio) with a documented
fallback message if absent. Whichever ran is recorded, since cosine values from
different embedding models are not on the same scale and must not be compared.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from ...audio import read_wav, resample
from ...datasets.loader import TestCase
from ...types import AudioBuffer, MetricStatus, MetricValue, SynthesisResult, UtteranceRecord
from ..base import (
    MetricContext,
    RunBackend,
    UtteranceBackend,
    make_value,
    missing_value,
    register_backend,
)

# Resemblyzer's expected input rate.
_EMBED_SAMPLE_RATE = 16000
_MIN_GROUP = 3


class _EmbedderMixin:
    """Shared Resemblyzer loading so the two backends share one model instance."""

    def _load_embedder(self) -> Any:
        from resemblyzer import VoiceEncoder

        device = self.options.get("device")
        return VoiceEncoder(device=device) if device else VoiceEncoder()

    def _embed(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        from resemblyzer import preprocess_wav

        pcm = resample(AudioBuffer(samples=samples, sample_rate=sample_rate), _EMBED_SAMPLE_RATE).samples
        # preprocess_wav applies the VAD trim and loudness normalisation the
        # encoder was trained with; skipping it degrades embeddings noticeably.
        processed = preprocess_wav(pcm.astype(np.float32), source_sr=_EMBED_SAMPLE_RATE)
        return np.asarray(self._encoder.embed_utterance(processed), dtype=np.float64)

    @staticmethod
    def _availability() -> tuple[bool, str]:
        try:
            import resemblyzer  # noqa: F401
        except ImportError:
            return False, (
                "resemblyzer not installed (pip install resemblyzer) — install it for "
                "embedding-based speaker metrics; the core voice_consistency estimate "
                "runs regardless"
            )
        return True, "resemblyzer/VoiceEncoder"


@register_backend
class SpeakerSimilarityBackend(_EmbedderMixin, UtteranceBackend):
    name = "speaker"
    provides = ("speaker_similarity",)

    def available(self) -> tuple[bool, str]:
        return self._availability()

    def prepare(self, ctx: MetricContext) -> None:
        self._encoder = self._load_embedder()
        self._ref_cache: dict[str, np.ndarray] = {}

    def teardown(self) -> None:
        self._encoder = None

    def compute(
        self, case: TestCase, result: SynthesisResult, ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        audio = result.audio
        assert audio is not None

        reference_path = ctx.resolve_reference(case)
        if reference_path is None:
            reason = (
                "case has no reference_audio"
                if not case.reference_audio
                else f"reference audio not found: {case.reference_audio}"
            )
            # Still emit the embedding for the run-level consistency pass, which
            # needs no reference at all.
            base = missing_value("speaker_similarity", reason, MetricStatus.NOT_APPLICABLE)
            embedding = self._embed(audio.samples, audio.sample_rate)
            return {
                "speaker_similarity": MetricValue(
                    name=base.name,
                    value=None,
                    unit=base.unit,
                    status=base.status,
                    direction=base.direction,
                    detail=base.detail,
                    extra={"embedding": embedding.tolist(), "embedder": "resemblyzer"},
                )
            }

        key = str(reference_path)
        if key not in self._ref_cache:
            ref_audio = read_wav(reference_path)
            self._ref_cache[key] = self._embed(ref_audio.samples, ref_audio.sample_rate)

        embedding = self._embed(audio.samples, audio.sample_rate)
        similarity = _cosine(embedding, self._ref_cache[key])
        return {
            "speaker_similarity": make_value(
                "speaker_similarity",
                similarity,
                extra={
                    "embedding": embedding.tolist(),
                    "embedder": "resemblyzer",
                    "reference": str(reference_path),
                },
            )
        }


@register_backend
class SpeakerConsistencyBackend(RunBackend):
    """Mean pairwise embedding similarity within each voice.

    Reads embeddings the utterance backend already produced instead of re-reading
    audio, so this costs nothing beyond the similarity arithmetic.
    """

    name = "speaker_consistency"
    provides = ("speaker_consistency",)

    def available(self) -> tuple[bool, str]:
        return _EmbedderMixin._availability()

    def compute(
        self, records: Sequence[UtteranceRecord], ctx: MetricContext
    ) -> Mapping[str, MetricValue]:
        groups: dict[str, list[np.ndarray]] = {}
        for rec in records:
            mv = rec.metrics.get("speaker_similarity")
            emb = (mv.extra or {}).get("embedding") if mv is not None else None
            if not emb:
                continue
            voice = rec.result.request.voice or "(default)"
            groups.setdefault(str(voice), []).append(np.asarray(emb, dtype=np.float64))

        usable = {v: e for v, e in groups.items() if len(e) >= _MIN_GROUP}
        if not usable:
            return {
                "speaker_consistency": missing_value(
                    "speaker_consistency",
                    f"no voice has the {_MIN_GROUP} embeddings needed for a pairwise estimate",
                )
            }

        per_voice: dict[str, Any] = {}
        weighted, total = 0.0, 0
        for voice, embeddings in sorted(usable.items()):
            matrix = np.stack(embeddings)
            norms = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            sims = norms @ norms.T
            # Upper triangle excluding the diagonal: every unordered pair once.
            iu = np.triu_indices(len(embeddings), k=1)
            mean_sim = float(sims[iu].mean())
            per_voice[voice] = {
                "n": len(embeddings),
                "mean_pairwise_cosine": round(mean_sim, 4),
                "min_pairwise_cosine": round(float(sims[iu].min()), 4),
            }
            weighted += mean_sim * len(embeddings)
            total += len(embeddings)

        return {
            "speaker_consistency": make_value(
                "speaker_consistency",
                weighted / total,
                unit="cosine",
                extra={"per_voice": per_voice, "embedder": "resemblyzer"},
                detail="mean pairwise speaker-embedding similarity within each voice",
            )
        }


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denom) if denom > 1e-12 else 0.0


__all__ = ["SpeakerConsistencyBackend", "SpeakerSimilarityBackend"]
