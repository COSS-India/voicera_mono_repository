"""Collect per-call Pipecat MetricsFrame data for post-call latency analysis."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pipecat.frames.frames import InterimTranscriptionFrame, MetricsFrame, TranscriptionFrame
from pipecat.metrics.metrics import ProcessingMetricsData, TTFBMetricsData
from pipecat.observers.base_observer import BaseObserver, FramePushed


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


def _normalize_text(text: str) -> str:
    return " ".join((text or "").split())


def _is_same_utterance(previous: str, new: str) -> bool:
    """True when transcriptions are the same utterance (partial vs final)."""
    if not previous or not new:
        return False
    if previous == new:
        return True
    return new.startswith(previous) or previous.startswith(new)


def _pick_metric(
    existing: Optional[float], incoming: Optional[float], *, prefer_nonzero: bool = False
) -> Optional[float]:
    if incoming is None:
        return existing
    if existing is None:
        return incoming
    if prefer_nonzero and existing == 0 and incoming > 0:
        return incoming
    if prefer_nonzero and incoming == 0:
        return existing
    return max(existing, incoming)


def _merge_turn(into: Dict[str, Any], other: Dict[str, Any]) -> Dict[str, Any]:
    into["stt_ms"] = _pick_metric(into.get("stt_ms"), other.get("stt_ms"), prefer_nonzero=True)
    into["llm_ttfb_ms"] = _pick_metric(into.get("llm_ttfb_ms"), other.get("llm_ttfb_ms"))
    into["tts_first_chunk_ms"] = _pick_metric(
        into.get("tts_first_chunk_ms"), other.get("tts_first_chunk_ms")
    )
    full = _normalize_text(
        into.get("_full_text") or into.get("user_text_preview") or ""
    )
    other_full = _normalize_text(
        other.get("_full_text") or other.get("user_text_preview") or ""
    )
    longer = full if len(full) >= len(other_full) else other_full
    into["_full_text"] = longer
    into["user_text_preview"] = longer[:80] if longer else None
    return into


class CallMetricsObserver(BaseObserver):
    """Observes MetricsFrame and TranscriptionFrame; builds turn buckets after the call."""

    def __init__(
        self,
        *,
        stt_processor_name: str,
        llm_processor_name: str,
        tts_processor_name: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._stt_name = stt_processor_name
        self._llm_name = llm_processor_name
        self._tts_name = tts_processor_name
        self._turns: List[Dict[str, Any]] = []
        self._current: Optional[Dict[str, Any]] = None
        self._pending: Dict[str, Optional[float]] = {
            "stt_ms": None,
            "llm_ttfb_ms": None,
            "tts_first_chunk_ms": None,
        }

    def _matches(self, processor: str, expected: str) -> bool:
        if not processor or not expected:
            return False
        return processor == expected or expected in processor or processor in expected

    def _is_stt(self, processor: str) -> bool:
        return self._matches(processor, self._stt_name)

    def _is_llm(self, processor: str) -> bool:
        return self._matches(processor, self._llm_name)

    def _is_tts(self, processor: str) -> bool:
        return self._matches(processor, self._tts_name)

    def _set_metric(
        self,
        target: Dict[str, Any],
        field: str,
        value_ms: float,
    ) -> None:
        if target.get(field) is None:
            target[field] = value_ms

    def _apply_ttfb(self, processor: str, value_sec: float) -> None:
        value_ms = _ms(value_sec)
        if self._is_stt(processor):
            return
        if self._is_llm(processor):
            if self._current is not None:
                self._set_metric(self._current, "llm_ttfb_ms", value_ms)
            return
        if self._is_tts(processor):
            if self._current is not None:
                self._set_metric(self._current, "tts_first_chunk_ms", value_ms)
            return

    def _apply_processing(self, processor: str, value_sec: float) -> None:
        if not self._is_stt(processor):
            return
        value_ms = _ms(value_sec)
        if self._current is not None:
            self._set_metric(self._current, "stt_ms", value_ms)
        else:
            self._set_metric(self._pending, "stt_ms", value_ms)

    def _turn_is_meaningful(self, turn: Dict[str, Any]) -> bool:
        if turn.get("user_text_preview"):
            return True
        return any(
            turn.get(k) is not None and turn.get(k) != 0
            for k in ("stt_ms", "llm_ttfb_ms", "tts_first_chunk_ms")
        )

    def _close_current_turn(self) -> None:
        if not self._current:
            return
        if self._turn_is_meaningful(self._current):
            self._turns.append(self._current)
        self._current = None

    def _on_final_transcription(self, text: str) -> None:
        normalized = _normalize_text(text)
        if not normalized:
            return

        if self._current:
            old_full = _normalize_text(
                self._current.get("_full_text")
                or self._current.get("user_text_preview")
                or ""
            )
            if _is_same_utterance(old_full, normalized):
                self._current["_full_text"] = (
                    normalized if len(normalized) >= len(old_full) else old_full
                )
                self._current["user_text_preview"] = self._current["_full_text"][:80]
                if self._pending.get("stt_ms") is not None:
                    self._current["stt_ms"] = _pick_metric(
                        self._current.get("stt_ms"),
                        self._pending.get("stt_ms"),
                        prefer_nonzero=True,
                    )
                return

        self._close_current_turn()
        self._current = {
            "turn_index": len(self._turns) + 1,
            "_full_text": normalized,
            "user_text_preview": normalized[:80],
            "stt_ms": self._pending.get("stt_ms"),
            "llm_ttfb_ms": None,
            "tts_first_chunk_ms": None,
        }
        self._pending = {
            "stt_ms": None,
            "llm_ttfb_ms": None,
            "tts_first_chunk_ms": None,
        }

    def _dedupe_turns(self, turns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not turns:
            return []
        merged: List[Dict[str, Any]] = []
        for turn in turns:
            if merged:
                prev = merged[-1]
                prev_text = _normalize_text(
                    prev.get("_full_text") or prev.get("user_text_preview") or ""
                )
                cur_text = _normalize_text(
                    turn.get("_full_text") or turn.get("user_text_preview") or ""
                )
                if _is_same_utterance(prev_text, cur_text):
                    merged[-1] = _merge_turn(prev, turn)
                    continue
            merged.append(dict(turn))
        for idx, turn in enumerate(merged, start=1):
            turn["turn_index"] = idx
            turn.pop("_full_text", None)
        return merged

    async def on_push_frame(self, data: FramePushed):
        frame = data.frame

        if isinstance(frame, TranscriptionFrame) and not isinstance(
            frame, InterimTranscriptionFrame
        ):
            self._on_final_transcription(getattr(frame, "text", "") or "")
            return

        if not isinstance(frame, MetricsFrame):
            return

        for item in frame.data or []:
            if isinstance(item, TTFBMetricsData):
                self._apply_ttfb(item.processor, item.value)
            elif isinstance(item, ProcessingMetricsData):
                self._apply_processing(item.processor, item.value)

    def to_dict(self) -> Dict[str, Any]:
        self._close_current_turn()
        turns = self._dedupe_turns(self._turns)
        summary: Dict[str, Any] = {"turn_count": len(turns)}

        for field, key in (
            ("stt_ms", "avg_stt_ms"),
            ("llm_ttfb_ms", "avg_llm_ttfb_ms"),
            ("tts_first_chunk_ms", "avg_tts_first_chunk_ms"),
        ):
            values = [t[field] for t in turns if t.get(field) is not None]
            if values:
                summary[key] = round(sum(values) / len(values), 1)
                summary[f"max_{field}"] = max(values)

        return {"turns": turns, "summary": summary}
