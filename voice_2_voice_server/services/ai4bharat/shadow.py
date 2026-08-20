"""Disabled-by-default automatic-language shadow observation for AI4Bharat STT."""

from __future__ import annotations

import asyncio
import os
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from loguru import logger


PRELIMINARY_MARGIN_THRESHOLD = 0.050965
DEFAULT_MIN_DURATION_MS = 2_000
DEFAULT_MAX_DURATION_MS = 3_000
DEFAULT_CANDIDATE_LANGUAGES = ("hi", "kn", "mr", "ta", "te")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AutoLanguageShadowConfig:
    """Preliminary five-language shadow policy; never changes ASR state."""

    enabled: bool = False
    min_duration_ms: int = DEFAULT_MIN_DURATION_MS
    max_duration_ms: int = DEFAULT_MAX_DURATION_MS
    margin_threshold: float = PRELIMINARY_MARGIN_THRESHOLD
    candidate_languages: tuple[str, ...] = DEFAULT_CANDIDATE_LANGUAGES

    @classmethod
    def resolve(
        cls,
        *,
        enabled: Any = None,
        min_duration_ms: Any = None,
        max_duration_ms: Any = None,
        margin_threshold: Any = None,
        candidate_languages: Any = None,
    ) -> "AutoLanguageShadowConfig":
        resolved_enabled = _optional_bool(enabled)
        if resolved_enabled is None:
            resolved_enabled = _env_bool("ENABLE_AUTO_LANGUAGE_SHADOW", False)
        minimum = int(
            min_duration_ms
            if min_duration_ms is not None
            else os.getenv("AUTO_LANGUAGE_MIN_DURATION_MS", DEFAULT_MIN_DURATION_MS)
        )
        maximum = int(
            max_duration_ms
            if max_duration_ms is not None
            else os.getenv("AUTO_LANGUAGE_MAX_DURATION_MS", DEFAULT_MAX_DURATION_MS)
        )
        threshold = float(
            margin_threshold
            if margin_threshold is not None
            else os.getenv(
                "AUTO_LANGUAGE_MARGIN_THRESHOLD",
                PRELIMINARY_MARGIN_THRESHOLD,
            )
        )
        raw_languages = (
            candidate_languages
            if candidate_languages is not None
            else os.getenv(
                "AUTO_LANGUAGE_CANDIDATE_LANGUAGES",
                ",".join(DEFAULT_CANDIDATE_LANGUAGES),
            )
        )
        if isinstance(raw_languages, str):
            languages = tuple(
                item.strip() for item in raw_languages.split(",") if item.strip()
            )
        else:
            languages = tuple(str(item).strip() for item in raw_languages if str(item).strip())
        if minimum < 0:
            raise ValueError("auto_language_min_duration_ms must be non-negative")
        if maximum < minimum:
            raise ValueError(
                "auto_language_max_duration_ms must be >= auto_language_min_duration_ms"
            )
        if threshold < 0:
            raise ValueError("auto_language_margin_threshold must be non-negative")
        if not languages:
            raise ValueError("auto_language_candidate_languages must not be empty")
        return cls(
            enabled=resolved_enabled,
            min_duration_ms=minimum,
            max_duration_ms=maximum,
            margin_threshold=threshold,
            candidate_languages=languages,
        )


@dataclass
class AutoLanguageShadowStats:
    total_probe_events: int = 0
    accepted_probe_events: int = 0
    uncertain_probe_events: int = 0
    failed_probe_events: int = 0
    skipped_short_events: int = 0
    skipped_unsupported_language_events: int = 0
    agreement_events: int = 0
    predictions: Counter[str] = field(default_factory=Counter)
    margins: list[float] = field(default_factory=list)
    inference_latencies_ms: list[float] = field(default_factory=list)
    request_latencies_ms: list[float] = field(default_factory=list)


ProbeRequest = Callable[[bytes, str, str, tuple[str, ...]], Awaitable[dict[str, Any]]]


class AutoLanguageShadowObserver:
    """Runs probe requests in background and records session-local diagnostics."""

    def __init__(
        self,
        *,
        session_id: str,
        sample_rate: int,
        config: AutoLanguageShadowConfig,
        request_probe: ProbeRequest,
    ) -> None:
        self.session_id = session_id or "unknown"
        self.sample_rate = sample_rate
        self.config = config
        self._request_probe = request_probe
        self.stats = AutoLanguageShadowStats()
        self._tasks: set[asyncio.Task[None]] = set()

    def reset(self) -> None:
        self.stats = AutoLanguageShadowStats()
        self._tasks.clear()

    def observe(self, audio: bytes, current_language: str) -> bool:
        """Schedule one final-utterance observation without blocking transcription."""

        if not self.config.enabled:
            return False
        if current_language not in self.config.candidate_languages:
            self.stats.skipped_unsupported_language_events += 1
            logger.debug(
                "[AUTO-LANGUAGE-SHADOW] session={} decision=skip_unsupported_language "
                "current_lang={} candidate_languages={}",
                self.session_id,
                current_language,
                self.config.candidate_languages,
            )
            return False
        bytes_per_ms = self.sample_rate * 2 / 1000
        duration_ms = len(audio) / bytes_per_ms if bytes_per_ms else 0.0
        if duration_ms < self.config.min_duration_ms:
            self.stats.skipped_short_events += 1
            logger.debug(
                "[AUTO-LANGUAGE-SHADOW] session={} decision=skip_short "
                "duration_ms={:.0f} min_duration_ms={}",
                self.session_id,
                duration_ms,
                self.config.min_duration_ms,
            )
            return False

        max_bytes = int(self.sample_rate * self.config.max_duration_ms / 1000) * 2
        probe_audio = audio[-max_bytes:] if max_bytes and len(audio) > max_bytes else audio
        task = asyncio.create_task(
            self._run_probe(probe_audio, current_language),
            name=f"auto-language-shadow-{self.session_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run_probe(self, audio: bytes, current_language: str) -> None:
        request_started = time.perf_counter()
        try:
            result = await self._request_probe(
                audio,
                current_language,
                self.session_id,
                self.config.candidate_languages,
            )
            predicted = str(
                result.get("predicted_language")
                or result.get("top_language")
                or ""
            )
            margin_value = result.get("margin")
            margin = float(margin_value) if margin_value is not None else None
            accepted = margin is not None and margin >= self.config.margin_threshold
            decision = "accept" if accepted else "uncertain"
            duration_ms = len(audio) * 1000.0 / (self.sample_rate * 2)
            inference_ms = float(result.get("inference_ms") or 0.0)
            request_ms = (time.perf_counter() - request_started) * 1000

            self.stats.total_probe_events += 1
            self.stats.accepted_probe_events += int(accepted)
            self.stats.uncertain_probe_events += int(not accepted)
            self.stats.agreement_events += int(predicted == current_language)
            if predicted:
                self.stats.predictions[predicted] += 1
            if margin is not None:
                self.stats.margins.append(margin)
            self.stats.inference_latencies_ms.append(inference_ms)
            self.stats.request_latencies_ms.append(request_ms)

            logger.info(
                "[AUTO-LANGUAGE-SHADOW] session={} duration_ms={:.0f} "
                "current_lang={} predicted_lang={} top_score={} second_score={} "
                "margin={} confidence={} threshold={:.6f} decision={} "
                "probe_ms={:.2f} request_ms={:.2f}",
                self.session_id,
                duration_ms,
                current_language,
                predicted or None,
                result.get("top_score"),
                result.get("second_score"),
                margin,
                result.get("confidence"),
                self.config.margin_threshold,
                decision,
                inference_ms,
                request_ms,
            )
            top_candidates = list(result.get("top_candidates") or [])[:3]
            if top_candidates:
                logger.debug(
                    "[AUTO-LANGUAGE-SHADOW] session={} top3={}",
                    self.session_id,
                    top_candidates,
                )
        except Exception as exc:
            self.stats.failed_probe_events += 1
            logger.warning(
                "[AUTO-LANGUAGE-SHADOW] session={} decision=probe_failed error={}",
                self.session_id,
                exc,
            )

    async def drain(self) -> None:
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def log_summary(self, current_language: str) -> None:
        if not self.config.enabled:
            return
        total = self.stats.total_probe_events
        agreement = self.stats.agreement_events / total if total else 0.0
        average_margin = (
            sum(self.stats.margins) / len(self.stats.margins)
            if self.stats.margins
            else None
        )
        average_probe_ms = (
            sum(self.stats.inference_latencies_ms)
            / len(self.stats.inference_latencies_ms)
            if self.stats.inference_latencies_ms
            else None
        )
        logger.info(
            "[AUTO-LANGUAGE-SHADOW-SUMMARY] session={} current_lang={} probes={} "
            "accepted={} uncertain={} failed={} skipped_short={} skipped_unsupported={} "
            "predictions={} "
            "agreement_with_current={:.1%} average_margin={} average_probe_ms={}",
            self.session_id,
            current_language,
            total,
            self.stats.accepted_probe_events,
            self.stats.uncertain_probe_events,
            self.stats.failed_probe_events,
            self.stats.skipped_short_events,
            self.stats.skipped_unsupported_language_events,
            dict(self.stats.predictions),
            agreement,
            average_margin,
            average_probe_ms,
        )
