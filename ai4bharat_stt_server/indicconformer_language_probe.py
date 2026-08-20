"""Experimental language-compatibility probe for IndicConformer.

The Hugging Face ``ai4bharat/indic-conformer-600m-multilingual`` remote code
uses one shared encoder and one shared 5,633-class CTC projection.  Language
selection is applied *after* that representation with ``language_masks``.
Consequently, all language hypotheses below reuse one encoder/CTC execution;
this is a compatibility probe, not a separately trained LID model.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
from typing import Any, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F


ScoringMethod = Literal[
    "mean_max_logprob",
    "non_blank_mass",
    "non_blank_confidence",
    "normalized_ctc_score",
]
_SCORING_METHODS: tuple[str, ...] = (
    "mean_max_logprob",
    "non_blank_mass",
    "non_blank_confidence",
    "normalized_ctc_score",
)


class IndicConformerProbeError(ValueError):
    """Raised when an incompatible or malformed IndicConformer input is used."""


@dataclass(frozen=True)
class LanguageProbeConfig:
    """Explicit experimental policy; thresholds are deliberately opt-in."""

    min_probe_duration_ms: float = 500.0
    confidence_threshold: float | None = None
    margin_threshold: float | None = None
    # Experimental no-speech guard. This is deliberately configurable and is
    # evaluated before encoder inference; it is not a language threshold.
    min_rms_energy: float | None = 1e-4
    scoring_method: ScoringMethod = "non_blank_mass"
    sample_rate: int = 16_000

    def __post_init__(self) -> None:
        if self.min_probe_duration_ms < 0:
            raise ValueError("min_probe_duration_ms must be non-negative")
        if self.sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if self.min_rms_energy is not None and self.min_rms_energy < 0:
            raise ValueError("min_rms_energy must be non-negative or None")
        if self.scoring_method not in _SCORING_METHODS:
            raise ValueError(f"Unknown scoring_method={self.scoring_method!r}; expected {_SCORING_METHODS}")


def normalize_ctc_output(ctc_output: torch.Tensor, *, output_kind: str = "logits") -> torch.Tensor:
    """Return stable float32 log-probabilities from a CTC tensor.

    The checked IndicConformer remote code calls ``.log_softmax(dim=-1)`` on
    the ONNX output after applying a language mask, so that output is raw
    logits despite being named ``logprobs`` by the ONNX graph.
    """

    if ctc_output.ndim != 3:
        raise IndicConformerProbeError(
            f"Expected CTC output shaped [B, T, V], got {tuple(ctc_output.shape)}"
        )
    values = ctc_output.float()
    if not torch.isfinite(values).all():
        raise IndicConformerProbeError("CTC output contains NaN or infinite values")
    if output_kind == "logits":
        return F.log_softmax(values, dim=-1)
    if output_kind == "log_probs":
        return values
    if output_kind == "probs":
        if (values < 0).any():
            raise IndicConformerProbeError("Probability CTC output contains negative values")
        return values.clamp_min(torch.finfo(values.dtype).tiny).log()
    raise IndicConformerProbeError(f"Unsupported CTC output kind: {output_kind!r}")


class IndicConformerLanguageProbe:
    """Score all model-provided language masks from one shared CTC output.

    This adapter targets the currently installed Hugging Face remote code. Its
    ``encode(wav)`` result is NumPy ``[B, 1024, T]`` and the shared CTC ONNX
    session returns raw logits ``[B, T, 5633]``.  The implementation still
    validates those contracts at runtime so a checkpoint/code revision fails
    clearly rather than silently producing an invalid result.
    """

    def __init__(
        self,
        model: Any,
        config: LanguageProbeConfig | None = None,
        languages: Sequence[str] | None = None,
    ) -> None:
        self.model = model
        self.config = config or LanguageProbeConfig()
        self._validate_model()
        available_languages = tuple(self.model.language_masks.keys())
        if languages is None:
            self.languages = available_languages
        else:
            requested_languages = tuple(languages)
            if not requested_languages:
                raise IndicConformerProbeError("At least one candidate language is required")
            if len(set(requested_languages)) != len(requested_languages):
                raise IndicConformerProbeError("Candidate languages must not contain duplicates")
            unsupported = sorted(set(requested_languages) - set(available_languages))
            if unsupported:
                raise IndicConformerProbeError(
                    f"Unknown candidate languages {unsupported}; supported: {', '.join(available_languages)}"
                )
            self.languages = requested_languages
        self._mask_cache: dict[tuple[str, str], torch.Tensor] = {}
        self._language_index_cache: dict[tuple[str, int], torch.Tensor] = {}

    def _validate_model(self) -> None:
        masks = getattr(self.model, "language_masks", None)
        if not isinstance(masks, dict) or not masks:
            raise IndicConformerProbeError("Model has no non-empty language_masks mapping")
        if not callable(getattr(self.model, "encode", None)):
            raise IndicConformerProbeError("Model has no callable encode(wav) method")
        models = getattr(self.model, "models", None)
        ctc = models.get("ctc_decoder") if isinstance(models, dict) else None
        if ctc is None or not callable(getattr(ctc, "run", None)):
            raise IndicConformerProbeError("Model has no compatible shared CTC ONNX head")
        if not hasattr(getattr(self.model, "config", None), "BLANK_ID"):
            raise IndicConformerProbeError("Model config has no BLANK_ID")
        lengths = {len(mask) for mask in masks.values()}
        if len(lengths) != 1 or next(iter(lengths)) == 0:
            raise IndicConformerProbeError("language_masks must be non-empty masks of one shared vocabulary")

    def _mask(self, language: str, device: torch.device, vocabulary_size: int) -> torch.Tensor:
        if language not in self.model.language_masks:
            raise IndicConformerProbeError(f"Unknown language {language!r}; supported: {', '.join(self.languages)}")
        key = (language, str(device))
        mask = self._mask_cache.get(key)
        if mask is None:
            source = self.model.language_masks[language]
            mask = torch.as_tensor(source, dtype=torch.bool, device=device)
            self._mask_cache[key] = mask
        if mask.numel() != vocabulary_size:
            raise IndicConformerProbeError(
                f"Mask for {language!r} has {mask.numel()} entries but CTC vocabulary has {vocabulary_size}"
            )
        if not bool(mask.any()):
            raise IndicConformerProbeError(f"Mask for {language!r} selects no CTC tokens")
        return mask

    @staticmethod
    def _as_lengths(encoded_lengths: Any, batch_size: int, max_frames: int, device: torch.device) -> torch.Tensor:
        lengths = torch.as_tensor(encoded_lengths, device=device).reshape(-1).long()
        if lengths.numel() != batch_size:
            raise IndicConformerProbeError(
                f"encoded_lengths has {lengths.numel()} entries for a batch of {batch_size}"
            )
        if (lengths <= 0).any() or (lengths > max_frames).any():
            raise IndicConformerProbeError(
                f"encoded_lengths must be in [1, {max_frames}], got {lengths.detach().cpu().tolist()}"
            )
        return lengths

    @staticmethod
    def _frame_mask(lengths: torch.Tensor, frames: int) -> torch.Tensor:
        return torch.arange(frames, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)

    def _blank_index(self, language: str, mask: torch.Tensor) -> int:
        """Resolve the language-local blank position from the checked model config."""

        blank_index = int(self.model.config.BLANK_ID)
        selected_count = int(mask.sum().item())
        if not 0 <= blank_index < selected_count:
            raise IndicConformerProbeError(
                f"Model BLANK_ID={blank_index} is invalid for {language!r}'s {selected_count}-token masked CTC vocabulary"
            )
        return blank_index

    def _language_indices(self, device: torch.device, vocabulary_size: int) -> torch.Tensor:
        """Return model-mask token indices shaped ``[languages, local_vocab]``.

        The current checkpoint has equal-size masks (257 selected positions),
        allowing one indexed gather and one batched log-softmax for every
        language.  This replaces 22 small Torch kernels on CPU/GPU while still
        deriving every index from ``model.language_masks[language]``.
        """

        key = (str(device), vocabulary_size)
        cached = self._language_index_cache.get(key)
        if cached is not None:
            return cached
        indices = [torch.nonzero(self._mask(language, device, vocabulary_size), as_tuple=False).squeeze(1) for language in self.languages]
        sizes = {index.numel() for index in indices}
        if len(sizes) != 1:
            raise IndicConformerProbeError(
                "This probe requires equal-size model language masks for batched scoring; "
                f"found selected sizes {sorted(sizes)}"
            )
        result = torch.stack(indices)
        blank_index = int(self.model.config.BLANK_ID)
        if not 0 <= blank_index < result.shape[1]:
            raise IndicConformerProbeError(
                f"Model BLANK_ID={blank_index} is invalid for {result.shape[1]}-token masked CTC vocabularies"
            )
        self._language_index_cache[key] = result
        return result

    def _score_all_languages(
        self, ctc_logits: torch.Tensor, lengths: torch.Tensor, method: ScoringMethod
    ) -> torch.Tensor:
        """Vectorized compatibility scores for all masks, returned as ``[B, L]``."""

        language_indices = self._language_indices(ctc_logits.device, ctc_logits.shape[-1])
        # Advanced indexing of [B, T, V] with [L, K] produces [B, T, L, K].
        language_logits = ctc_logits[:, :, language_indices]
        log_probs = F.log_softmax(language_logits.float(), dim=-1)
        if not torch.isfinite(log_probs).all():
            raise IndicConformerProbeError("CTC output contains NaN or infinite values")
        valid = self._frame_mask(lengths, log_probs.shape[1]).unsqueeze(-1)  # [B, T, 1]
        valid_count = lengths.to(log_probs.dtype).unsqueeze(1)  # [B, 1]
        best_logprob, best_token = log_probs.max(dim=-1)  # [B, T, L]
        blank_index = int(self.model.config.BLANK_ID)

        if method == "mean_max_logprob":
            return (best_logprob * valid).sum(dim=1) / valid_count
        if method == "non_blank_mass":
            blank_logprob = log_probs[..., blank_index]
            return ((1.0 - blank_logprob.exp()) * valid).sum(dim=1) / valid_count
        if method == "non_blank_confidence":
            non_blank = valid & (best_token != blank_index)
            count = non_blank.sum(dim=1)
            values = best_logprob.exp() * non_blank
            return torch.where(count > 0, values.sum(dim=1) / count.clamp_min(1), torch.zeros_like(valid_count).expand_as(count))
        if method == "normalized_ctc_score":
            changed = torch.ones_like(best_token, dtype=torch.bool)
            if best_token.shape[1] > 1:
                changed[:, 1:, :] = best_token[:, 1:, :] != best_token[:, :-1, :]
            kept = valid & changed & (best_token != blank_index)
            kept_count = kept.sum(dim=1)
            token_score = (best_logprob * kept).sum(dim=1) / kept_count.clamp_min(1)
            frame_score = (best_logprob * valid).sum(dim=1) / valid_count
            return torch.where(kept_count > 0, token_score, frame_score)
        raise AssertionError(f"Unexpected scoring method {method!r}")

    def project_ctc(self, encoder_outputs: Any) -> torch.Tensor:
        """Run the remote model's shared CTC ONNX projection exactly once."""

        encoded = np.asarray(encoder_outputs)
        if encoded.ndim != 3:
            raise IndicConformerProbeError(
                f"Expected encoder output shaped [B, H, T], got {tuple(encoded.shape)}"
            )
        if encoded.shape[0] < 1 or encoded.shape[1] < 1 or encoded.shape[2] < 1:
            raise IndicConformerProbeError("Encoder output is empty")
        try:
            output = self.model.models["ctc_decoder"].run(["logprobs"], {"encoder_output": encoded})[0]
        except Exception as exc:  # onnxruntime exposes several runtime-specific exception classes
            raise IndicConformerProbeError(f"Shared CTC projection failed: {exc}") from exc
        result = torch.as_tensor(output)
        if result.ndim != 3:
            raise IndicConformerProbeError(
                f"Expected CTC output shaped [B, T, V], got {tuple(result.shape)}"
            )
        if result.shape[0] != encoded.shape[0]:
            raise IndicConformerProbeError("CTC batch dimension differs from encoder output")
        return result

    def detect_language_from_encoder_output(
        self,
        encoder_outputs: Any,
        encoded_lengths: Any,
        *,
        ctc_output: torch.Tensor | np.ndarray | None = None,
        scoring_method: ScoringMethod | None = None,
        duration_ms: float | None = None,
        ctc_ms: float | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Rank every configured mask without encoding audio again.

        ``ctc_output`` lets callers pass a previously projected shared tensor;
        omitted means this method executes that projection once itself.
        """

        method = scoring_method or self.config.scoring_method
        if method not in _SCORING_METHODS:
            raise IndicConformerProbeError(f"Unknown scoring method: {method!r}")
        started = perf_counter()
        if duration_ms is not None and duration_ms < self.config.min_probe_duration_ms:
            return self._insufficient_audio_result(duration_ms, ctc_ms)
        if ctc_output is None:
            ctc_started = perf_counter()
            ctc_tensor = self.project_ctc(encoder_outputs)
            ctc_ms = (perf_counter() - ctc_started) * 1000
        else:
            ctc_tensor = torch.as_tensor(ctc_output)
        if ctc_tensor.ndim != 3:
            raise IndicConformerProbeError(f"Expected CTC [B, T, V], got {tuple(ctc_tensor.shape)}")
        lengths = self._as_lengths(encoded_lengths, ctc_tensor.shape[0], ctc_tensor.shape[1], ctc_tensor.device)
        if duration_ms is None:
            frame_duration_ms = float(getattr(self.model.config, "FRAME_DURATION_MS", 0.08)) * 1000
            if float(lengths.max()) * frame_duration_ms < self.config.min_probe_duration_ms:
                return self._insufficient_audio_result(float(lengths.max()) * frame_duration_ms, ctc_ms)

        scores = self._score_all_languages(ctc_tensor, lengths, method)  # [B, languages]
        return self._build_results(scores, method, ctc_ms, (perf_counter() - started) * 1000)

    def detect_language(self, wav: torch.Tensor, *, scoring_method: ScoringMethod | None = None) -> dict[str, Any] | list[dict[str, Any]]:
        """Encode once, project CTC once, then probe all language masks."""

        self._validate_audio(wav)
        duration_ms = wav.shape[-1] * 1000.0 / self.config.sample_rate
        if duration_ms < self.config.min_probe_duration_ms:
            return self._insufficient_audio_result(duration_ms, None)
        rms_energy = self._rms_energy(wav)
        if self._is_insufficient_speech_energy(rms_energy):
            return self._insufficient_speech_energy_result(duration_ms, rms_energy)
        started = perf_counter()
        encoder_started = perf_counter()
        encoder_outputs, encoded_lengths = self.model.encode(wav)
        encoder_ms = (perf_counter() - encoder_started) * 1000
        ctc_started = perf_counter()
        ctc_output = self.project_ctc(encoder_outputs)
        ctc_ms = (perf_counter() - ctc_started) * 1000
        result = self.detect_language_from_encoder_output(
            encoder_outputs,
            encoded_lengths,
            ctc_output=ctc_output,
            scoring_method=scoring_method,
            duration_ms=duration_ms,
            ctc_ms=ctc_ms,
        )
        return self._add_timing(result, encoder_ms, ctc_ms, (perf_counter() - started) * 1000)

    def transcribe_auto_language(
        self,
        wav: torch.Tensor,
        *,
        strategy: Literal["ctc", "rnnt"] = "rnnt",
        scoring_method: ScoringMethod | None = None,
        return_diagnostics: bool = True,
    ) -> dict[str, Any]:
        """Probe and decode, reusing the same encoded representation.

        CTC decoding consumes the already-projected output.  RNNT uses the
        remote code's existing ``_rnnt_decode(encoded, lengths, lang)`` call,
        which takes encoder output directly and therefore does not re-encode.
        """

        if strategy not in {"ctc", "rnnt"}:
            raise IndicConformerProbeError("strategy must be 'ctc' or 'rnnt'")
        self._validate_audio(wav)
        duration_ms = wav.shape[-1] * 1000.0 / self.config.sample_rate
        if duration_ms < self.config.min_probe_duration_ms:
            return {"language": None, "transcript": None, "reason": "insufficient_audio", "probe": self._insufficient_audio_result(duration_ms, None)}
        rms_energy = self._rms_energy(wav)
        if self._is_insufficient_speech_energy(rms_energy):
            return {
                "language": None,
                "transcript": None,
                "reason": "insufficient_speech_energy",
                "probe": self._insufficient_speech_energy_result(duration_ms, rms_energy),
            }

        total_started = perf_counter()
        encoder_started = perf_counter()
        encoder_outputs, encoded_lengths = self.model.encode(wav)
        encoder_ms = (perf_counter() - encoder_started) * 1000
        ctc_started = perf_counter()
        ctc_output = self.project_ctc(encoder_outputs)
        ctc_ms = (perf_counter() - ctc_started) * 1000
        probe = self.detect_language_from_encoder_output(
            encoder_outputs, encoded_lengths, ctc_output=ctc_output, scoring_method=scoring_method,
            duration_ms=duration_ms, ctc_ms=ctc_ms,
        )
        probe = self._add_timing(probe, encoder_ms, ctc_ms, None)
        if isinstance(probe, list):
            raise IndicConformerProbeError("transcribe_auto_language currently accepts a batch of one audio item")
        probe["total_ms"] = (perf_counter() - total_started) * 1000
        language = probe["language"]
        if language is None:
            return {"language": None, "transcript": None, "reason": "language_uncertain", "probe": probe}

        decode_started = perf_counter()
        if strategy == "ctc":
            transcript = self.decode_ctc_from_shared_output(ctc_output, encoded_lengths, language)
        else:
            decoder = getattr(self.model, "_rnnt_decode", None)
            if not callable(decoder):
                raise IndicConformerProbeError("Model does not expose _rnnt_decode(encoded, lengths, lang)")
            transcript = decoder(encoder_outputs, encoded_lengths, language)
        decode_ms = (perf_counter() - decode_started) * 1000
        probe["decode_ms"] = decode_ms
        probe["total_ms"] = (perf_counter() - total_started) * 1000
        response: dict[str, Any] = {"language": language, "transcript": transcript, "strategy": strategy}
        if return_diagnostics:
            response["probe"] = probe
        return response

    def decode_ctc_from_shared_output(self, ctc_output: torch.Tensor | np.ndarray, encoded_lengths: Any, language: str) -> str:
        """Greedy CTC decode equivalent to the remote code, without re-projecting."""

        ctc = torch.as_tensor(ctc_output)
        if ctc.ndim != 3 or ctc.shape[0] != 1:
            raise IndicConformerProbeError("CTC final decoding currently requires shared output shaped [1, T, V]")
        lengths = self._as_lengths(encoded_lengths, 1, ctc.shape[1], ctc.device)
        mask = self._mask(language, ctc.device, ctc.shape[-1])
        blank = self._blank_index(language, mask)
        log_probs = normalize_ctc_output(ctc[..., mask], output_kind="logits")
        tokens = log_probs[0, : int(lengths[0]), :].argmax(dim=-1)
        collapsed = torch.unique_consecutive(tokens)
        try:
            pieces: Sequence[str] = self.model.vocab[language]
            text = "".join(pieces[token] for token in collapsed.tolist() if token != blank)
        except (KeyError, IndexError, TypeError) as exc:
            raise IndicConformerProbeError(f"Invalid vocabulary for language {language!r}") from exc
        return text.replace("▁", " ").strip()

    def _build_results(self, scores: torch.Tensor, method: str, ctc_ms: float | None, probe_ms: float) -> dict[str, Any] | list[dict[str, Any]]:
        # Softmax over compatibility scores is only a normalized diagnostic,
        # not a calibrated probability of spoken language.
        confidence_values = torch.softmax(scores, dim=1)
        ranked_scores, ranked_indices = scores.sort(dim=1, descending=True)
        results: list[dict[str, Any]] = []
        for batch_index in range(scores.shape[0]):
            ranking = [
                {"language": self.languages[index], "score": float(ranked_scores[batch_index, rank])}
                for rank, index in enumerate(ranked_indices[batch_index].tolist())
            ]
            top_index = int(ranked_indices[batch_index, 0])
            top_score = float(ranked_scores[batch_index, 0])
            second_score = float(ranked_scores[batch_index, 1]) if len(self.languages) > 1 else None
            margin = top_score - second_score if second_score is not None else None
            confidence = float(confidence_values[batch_index, top_index])
            accepted = (
                (self.config.confidence_threshold is None or confidence >= self.config.confidence_threshold)
                and (self.config.margin_threshold is None or margin is None or margin >= self.config.margin_threshold)
            )
            results.append({
                "language": self.languages[top_index] if accepted else None,
                "top_language": self.languages[top_index],
                "score": top_score,
                "top_score": top_score,
                "second_score": second_score,
                "confidence": confidence,
                "margin": margin,
                "scoring_method": method,
                "candidates": ranking,
                "ctc_ms": ctc_ms,
                "probe_ms": probe_ms,
                "reason": None if accepted else "language_uncertain",
            })
        return results[0] if len(results) == 1 else results

    @staticmethod
    def _add_timing(result: dict[str, Any] | list[dict[str, Any]], encoder_ms: float, ctc_ms: float, total_ms: float | None) -> dict[str, Any] | list[dict[str, Any]]:
        results = result if isinstance(result, list) else [result]
        for item in results:
            item["encoder_ms"] = encoder_ms
            item["ctc_ms"] = ctc_ms
            if total_ms is not None:
                item["total_ms"] = total_ms
        return result

    def _insufficient_audio_result(self, duration_ms: float, ctc_ms: float | None) -> dict[str, Any]:
        return {
            "language": None,
            "top_language": None,
            "score": None,
            "top_score": None,
            "second_score": None,
            "confidence": None,
            "margin": None,
            "candidates": [],
            "duration_ms": duration_ms,
            "ctc_ms": ctc_ms,
            "reason": "insufficient_audio",
        }

    def _is_insufficient_speech_energy(self, rms_energy: float) -> bool:
        threshold = self.config.min_rms_energy
        return threshold is not None and rms_energy < threshold

    def _insufficient_speech_energy_result(self, duration_ms: float, rms_energy: float) -> dict[str, Any]:
        return {
            "language": None,
            "top_language": None,
            "score": None,
            "top_score": None,
            "second_score": None,
            "confidence": None,
            "margin": None,
            "candidates": [],
            "duration_ms": duration_ms,
            "rms_energy": rms_energy,
            "reason": "insufficient_speech_energy",
        }

    @staticmethod
    def _rms_energy(wav: torch.Tensor) -> float:
        return float(wav.float().square().mean().sqrt().item())

    @staticmethod
    def _validate_audio(wav: torch.Tensor) -> None:
        if not isinstance(wav, torch.Tensor):
            raise IndicConformerProbeError("wav must be a torch.Tensor")
        if wav.ndim not in {1, 2}:
            raise IndicConformerProbeError(f"wav must be [samples] or [channels, samples], got {tuple(wav.shape)}")
        if wav.numel() == 0 or wav.shape[-1] == 0:
            raise IndicConformerProbeError("Audio is empty")
        if not torch.is_floating_point(wav):
            raise IndicConformerProbeError("wav must use a floating-point dtype")
        if not torch.isfinite(wav).all():
            raise IndicConformerProbeError("Audio contains NaN or infinite values")


def probe_config_dict(config: LanguageProbeConfig) -> dict[str, Any]:
    """Convenience helper for emitting experiment metadata in benchmark JSON."""

    return asdict(config)
