"""Per-session live automatic-language state and persistent event logging."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable

from loguru import logger

from services.ai4bharat.shadow import (
    DEFAULT_CANDIDATE_LANGUAGES,
    DEFAULT_MAX_DURATION_MS,
    DEFAULT_MIN_DURATION_MS,
    PRELIMINARY_MARGIN_THRESHOLD,
    _optional_bool,
)


DEFAULT_CONFIRMATION_COUNT = 2
DEFAULT_REPROBE_COOLDOWN_MS = 1_500
AUTO_LANGUAGE_UNKNOWN = "auto"


def is_language_unresolved(language_id: str | None) -> bool:
    return (language_id or "").strip().lower() == AUTO_LANGUAGE_UNKNOWN


def is_live_auto_language_enabled() -> bool:
    return os.getenv("ENABLE_AUTO_LANGUAGE", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def is_force_auto_stt_language() -> bool:
    """True when env overrides agent/frontend STT language to bootstrap auto."""
    return (
        is_live_auto_language_enabled()
        and os.getenv("AUTO_LANGUAGE_STT_LANGUAGE", "").strip().lower()
        == AUTO_LANGUAGE_UNKNOWN
    )


class AutoLanguageUnavailableError(RuntimeError):
    """The live GPU probe cannot serve requests safely."""


@dataclass(frozen=True)
class AutoLanguageConfig:
    enabled: bool = False
    device: str = "cuda"
    min_duration_ms: int = DEFAULT_MIN_DURATION_MS
    max_duration_ms: int = DEFAULT_MAX_DURATION_MS
    margin_threshold: float = PRELIMINARY_MARGIN_THRESHOLD
    candidate_languages: tuple[str, ...] = DEFAULT_CANDIDATE_LANGUAGES
    confirmation_count: int = DEFAULT_CONFIRMATION_COUNT
    reprobe_cooldown_ms: int = DEFAULT_REPROBE_COOLDOWN_MS
    event_log_path: str = "logs/auto_language_events.jsonl"

    @classmethod
    def resolve(
        cls,
        *,
        enabled: Any = None,
        device: Any = None,
        min_duration_ms: Any = None,
        max_duration_ms: Any = None,
        margin_threshold: Any = None,
        candidate_languages: Any = None,
        confirmation_count: Any = None,
        reprobe_cooldown_ms: Any = None,
        event_log_path: Any = None,
    ) -> "AutoLanguageConfig":
        resolved_enabled = _optional_bool(enabled)
        if resolved_enabled is None:
            resolved_enabled = os.getenv("ENABLE_AUTO_LANGUAGE", "false").strip().lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
        if not resolved_enabled:
            # Disabled mode must not parse, validate, allocate, log, or depend
            # on any experimental settings.
            return cls(enabled=False)
        resolved_device = str(
            device if device is not None else os.getenv("AUTO_LANGUAGE_DEVICE", "cuda")
        ).strip().lower()
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
            languages = tuple(item.strip() for item in raw_languages.split(",") if item.strip())
        else:
            languages = tuple(str(item).strip() for item in raw_languages if str(item).strip())
        confirmations = int(
            confirmation_count
            if confirmation_count is not None
            else os.getenv("AUTO_LANGUAGE_CONFIRMATION_COUNT", DEFAULT_CONFIRMATION_COUNT)
        )
        cooldown = int(
            reprobe_cooldown_ms
            if reprobe_cooldown_ms is not None
            else os.getenv(
                "AUTO_LANGUAGE_REPROBE_COOLDOWN_MS",
                DEFAULT_REPROBE_COOLDOWN_MS,
            )
        )
        log_path = str(
            event_log_path
            if event_log_path is not None
            else os.getenv(
                "AUTO_LANGUAGE_EVENT_LOG_PATH",
                "logs/auto_language_events.jsonl",
            )
        )
        if resolved_device != "cuda":
            raise ValueError("AUTO_LANGUAGE_DEVICE must be 'cuda' for live mode")
        if minimum < 0 or maximum < minimum:
            raise ValueError("Invalid automatic-language duration range")
        if threshold < 0:
            raise ValueError("AUTO_LANGUAGE_MARGIN_THRESHOLD must be non-negative")
        if confirmations < 1:
            raise ValueError("AUTO_LANGUAGE_CONFIRMATION_COUNT must be at least 1")
        if cooldown < 0:
            raise ValueError("AUTO_LANGUAGE_REPROBE_COOLDOWN_MS must be non-negative")
        if not languages:
            raise ValueError("AUTO_LANGUAGE_CANDIDATE_LANGUAGES must not be empty")
        return cls(
            enabled=resolved_enabled,
            device=resolved_device,
            min_duration_ms=minimum,
            max_duration_ms=maximum,
            margin_threshold=threshold,
            candidate_languages=languages,
            confirmation_count=confirmations,
            reprobe_cooldown_ms=cooldown,
            event_log_path=log_path,
        )


@dataclass
class AutoLanguageStats:
    total_probe_events: int = 0
    accepted_probe_events: int = 0
    uncertain_probe_events: int = 0
    failed_probe_events: int = 0
    skipped_short_events: int = 0
    skipped_cooldown_events: int = 0
    switch_events: int = 0
    predictions: Counter[str] = field(default_factory=Counter)
    margins: list[float] = field(default_factory=list)
    inference_latencies_ms: list[float] = field(default_factory=list)


ProbeRequest = Callable[
    [bytes, str, str, tuple[str, ...]],
    Awaitable[dict[str, Any]],
]
SwitchCallback = Callable[[str, str, dict[str, Any]], Awaitable[None]]
CurrentLanguage = Callable[[], str]


class AutoLanguageController:
    """Serializes probe decisions and updates only future utterance language."""

    _file_lock = threading.Lock()

    def __init__(
        self,
        *,
        session_id: str,
        sample_rate: int,
        config: AutoLanguageConfig,
        request_probe: ProbeRequest,
        current_language: CurrentLanguage,
        switch_language: SwitchCallback,
    ) -> None:
        self.session_id = session_id or "unknown"
        self.sample_rate = sample_rate
        self.config = config
        self._request_probe = request_probe
        self._current_language = current_language
        self._switch_language = switch_language
        self.stats = AutoLanguageStats()
        self.candidate_language: str | None = None
        self.confirmation_count = 0
        self.runtime_enabled = config.enabled
        self._state_version = 0
        self._last_probe_scheduled_at = 0.0
        self._tasks: set[asyncio.Task[None]] = set()
        self._probe_lock = asyncio.Lock()

    def reset(self) -> None:
        self.stats = AutoLanguageStats()
        self.candidate_language = None
        self.confirmation_count = 0
        self.runtime_enabled = self.config.enabled
        self._state_version = 0
        self._last_probe_scheduled_at = 0.0
        self._tasks.clear()

    def observe(self, audio: bytes, utterance_language: str) -> bool:
        if not self.runtime_enabled:
            return False
        if utterance_language not in self.config.candidate_languages:
            return False
        duration_ms = len(audio) * 1000.0 / (self.sample_rate * 2)
        if duration_ms < self.config.min_duration_ms:
            self.stats.skipped_short_events += 1
            return False
        now = time.monotonic()
        elapsed_ms = (now - self._last_probe_scheduled_at) * 1000
        if (
            self._last_probe_scheduled_at
            and elapsed_ms < self.config.reprobe_cooldown_ms
        ):
            self.stats.skipped_cooldown_events += 1
            return False
        self._last_probe_scheduled_at = now
        max_bytes = int(self.sample_rate * self.config.max_duration_ms / 1000) * 2
        probe_audio = audio[-max_bytes:] if max_bytes and len(audio) > max_bytes else audio
        state_version = self._state_version
        task = asyncio.create_task(
            self._run_probe(probe_audio, utterance_language, state_version),
            name=f"auto-language-live-{self.session_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return True

    async def _run_probe(
        self,
        audio: bytes,
        utterance_language: str,
        state_version: int,
    ) -> None:
        async with self._probe_lock:
            request_started = time.perf_counter()
            try:
                result = await self._request_probe(
                    audio,
                    utterance_language,
                    self.session_id,
                    self.config.candidate_languages,
                )
                if str(result.get("device") or "").lower() != "cuda":
                    raise AutoLanguageUnavailableError(
                        f"live probe returned non-CUDA device: {result.get('device')!r}"
                    )
                providers = list(result.get("providers") or [])
                if "CUDAExecutionProvider" not in providers:
                    raise AutoLanguageUnavailableError(
                        f"live probe lacks CUDAExecutionProvider: {providers}"
                    )
                await self._apply_result(
                    result,
                    audio,
                    utterance_language,
                    state_version,
                    (time.perf_counter() - request_started) * 1000,
                )
            except AutoLanguageUnavailableError as exc:
                self.runtime_enabled = False
                self.stats.failed_probe_events += 1
                self._persist(
                    {
                        "event": "AUTO_LANGUAGE_DISABLED",
                        "session_id": self.session_id,
                        "source": "auto_probe",
                        "reason": str(exc),
                    }
                )
                logger.error(
                    "[AUTO-LANGUAGE] session={} disabled_safely reason={}",
                    self.session_id,
                    exc,
                )
            except Exception as exc:
                self.stats.failed_probe_events += 1
                self._clear_candidate()
                self._persist(
                    {
                        "event": "AUTO_LANGUAGE_PROBE_FAILED",
                        "session_id": self.session_id,
                        "source": "auto_probe",
                        "current_language": self._current_language(),
                        "error": str(exc),
                    }
                )
                logger.warning(
                    "[AUTO-LANGUAGE] session={} decision=probe_failed error={}",
                    self.session_id,
                    exc,
                )

    async def _apply_result(
        self,
        result: dict[str, Any],
        audio: bytes,
        utterance_language: str,
        state_version: int,
        request_ms: float,
    ) -> None:
        predicted = str(result.get("predicted_language") or "")
        margin_raw = result.get("margin")
        margin = float(margin_raw) if margin_raw is not None else None
        accepted = margin is not None and margin >= self.config.margin_threshold
        current = self._current_language()
        stale = state_version != self._state_version
        decision = "uncertain"
        switched = False
        event_candidate: str | None = None
        event_confirmation_count = 0

        if stale:
            decision = "stale_after_explicit_switch"
            self._clear_candidate()
        elif not accepted or not predicted:
            self._clear_candidate()
        elif predicted == current:
            decision = "stay_current"
            self._clear_candidate()
        else:
            if predicted == self.candidate_language:
                self.confirmation_count += 1
            else:
                self.candidate_language = predicted
                self.confirmation_count = 1
            decision = "candidate"
            event_candidate = self.candidate_language
            event_confirmation_count = self.confirmation_count
            if self.confirmation_count >= self.config.confirmation_count:
                switch_event = {
                    "event": "AUTO_LANGUAGE_SWITCH",
                    "session_id": self.session_id,
                    "from_language": current,
                    "to_language": predicted,
                    "confirmations": self.confirmation_count,
                    "source": "auto_probe",
                    "margin": margin,
                }
                await self._switch_language(current, predicted, switch_event)
                self.stats.switch_events += 1
                self._state_version += 1
                self._persist(switch_event)
                logger.info(
                    "[AUTO_LANGUAGE_SWITCH] session={} from={} to={} confirmations={} "
                    "source=auto_probe margin={}",
                    self.session_id,
                    current,
                    predicted,
                    self.confirmation_count,
                    margin,
                )
                switched = True
                decision = "switch"
                self._clear_candidate()

        inference_ms = float(result.get("inference_ms") or 0.0)
        self.stats.total_probe_events += 1
        self.stats.accepted_probe_events += int(accepted)
        self.stats.uncertain_probe_events += int(not accepted)
        if predicted:
            self.stats.predictions[predicted] += 1
        if margin is not None:
            self.stats.margins.append(margin)
        self.stats.inference_latencies_ms.append(inference_ms)
        event = {
            "event": "AUTO_LANGUAGE_PROBE",
            "session_id": self.session_id,
            "source": "auto_probe",
            "audio_duration_ms": len(audio) * 1000.0 / (self.sample_rate * 2),
            "utterance_language": utterance_language,
            "current_language": current,
            "predicted_language": predicted or None,
            "top_score": result.get("top_score"),
            "second_score": result.get("second_score"),
            "margin": margin,
            "confidence": result.get("confidence"),
            "threshold": self.config.margin_threshold,
            "decision": decision,
            "candidate_language": event_candidate,
            "confirmation_count": event_confirmation_count,
            "switched": switched,
            "probe_latency_ms": inference_ms,
            "request_latency_ms": request_ms,
            "preprocessing_ms": result.get("preprocessing_ms"),
            "encoder_ms": result.get("encoder_ms"),
            "ctc_ms": result.get("ctc_ms"),
            "score_ms": result.get("probe_ms"),
            "gpu_device": result.get("gpu_device"),
            "providers": result.get("providers"),
            "top_candidates": list(result.get("top_candidates") or [])[:3],
        }
        self._persist(event)
        logger.info(
            "[AUTO-LANGUAGE] session={} current_lang={} predicted_lang={} margin={} "
            "threshold={:.6f} decision={} candidate={} confirmation_count={} "
            "probe_ms={:.2f} request_ms={:.2f} gpu_device={} source=auto_probe",
            self.session_id,
            current,
            predicted or None,
            margin,
            self.config.margin_threshold,
            decision,
            event_candidate,
            event_confirmation_count,
            inference_ms,
            request_ms,
            result.get("gpu_device"),
        )

    def record_bootstrap_probe(
        self,
        *,
        audio_duration_ms: float,
        predicted_language: str,
        result: dict[str, Any],
    ) -> None:
        if not self.config.enabled:
            return
        margin_raw = result.get("margin")
        margin = float(margin_raw) if margin_raw is not None else None
        self.stats.total_probe_events += 1
        self.stats.accepted_probe_events += 1
        self.stats.predictions[predicted_language] += 1
        if margin is not None:
            self.stats.margins.append(margin)
        inference_ms = float(result.get("inference_ms") or 0.0)
        self.stats.inference_latencies_ms.append(inference_ms)
        event = {
            "event": "AUTO_LANGUAGE_BOOTSTRAP_PROBE",
            "session_id": self.session_id,
            "source": "auto_probe_bootstrap",
            "audio_duration_ms": audio_duration_ms,
            "current_language": AUTO_LANGUAGE_UNKNOWN,
            "predicted_language": predicted_language,
            "top_score": result.get("top_score"),
            "second_score": result.get("second_score"),
            "margin": margin,
            "confidence": result.get("confidence"),
            "threshold": self.config.margin_threshold,
            "decision": "bootstrap_accept",
            "probe_latency_ms": inference_ms,
            "request_latency_ms": inference_ms,
            "gpu_device": result.get("gpu_device"),
            "providers": result.get("providers"),
            "top_candidates": list(result.get("top_candidates") or [])[:3],
        }
        self._persist(event)

    def explicit_language_changed(self, language: str, source: str) -> None:
        if not self.config.enabled:
            return
        self._state_version += 1
        self._clear_candidate()
        self._last_probe_scheduled_at = time.monotonic()
        self._persist(
            {
                "event": "AUTO_LANGUAGE_EXPLICIT_OVERRIDE",
                "session_id": self.session_id,
                "language": language,
                "source": source,
            }
        )

    def _clear_candidate(self) -> None:
        self.candidate_language = None
        self.confirmation_count = 0

    async def drain(self) -> None:
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def log_summary(self) -> None:
        if not self.config.enabled:
            return
        event = {
            "event": "AUTO_LANGUAGE_SUMMARY",
            "session_id": self.session_id,
            "current_language": self._current_language(),
            "source": "auto_probe",
            **asdict(self.stats),
            "predictions": dict(self.stats.predictions),
        }
        self._persist(event)
        logger.info(
            "[AUTO-LANGUAGE-SUMMARY] session={} current_lang={} probes={} accepted={} "
            "uncertain={} failed={} switches={} predictions={}",
            self.session_id,
            self._current_language(),
            self.stats.total_probe_events,
            self.stats.accepted_probe_events,
            self.stats.uncertain_probe_events,
            self.stats.failed_probe_events,
            self.stats.switch_events,
            dict(self.stats.predictions),
        )

    def _persist(self, event: dict[str, Any]) -> None:
        if not self.config.enabled:
            return
        record = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
        path = Path(self.config.event_log_path).expanduser()
        if not path.is_absolute():
            path = Path.cwd() / path
        try:
            with self._file_lock:
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        except Exception as exc:
            logger.error(
                "[AUTO-LANGUAGE] persistent_log_failed path={} error={}",
                path,
                exc,
            )
