"""Per-org call budget guardrails: concurrency, daily minutes, and call duration ceiling.

State is in-memory and per-process by design: it resets whenever the voice server
restarts, which aligns with the sandbox's daily operational schedule.
"""

import asyncio
import os

from loguru import logger

_lock = asyncio.Lock()
_active_calls_by_org: dict[str, int] = {}
_minutes_used_by_org: dict[str, float] = {}


def _max_concurrent_calls_per_org() -> int:
    return int(os.getenv("MAX_CONCURRENT_CALLS_PER_ORG", "5"))


def _max_minutes_per_org_per_day() -> float:
    return float(os.getenv("MAX_MINUTES_PER_ORG_PER_DAY", "60"))


def _max_call_duration_seconds() -> int:
    return int(os.getenv("MAX_CALL_DURATION_SECONDS", "600"))


def _has_capacity(org_id: str) -> bool:
    if not org_id:
        logger.warning(
            "usage_guard called with no org_id; pooling into a shared budget bucket"
        )
    return (
        _active_calls_by_org.get(org_id, 0) < _max_concurrent_calls_per_org()
        and _minutes_used_by_org.get(org_id, 0.0) < _max_minutes_per_org_per_day()
    )


async def try_acquire_call_slot(org_id: str) -> bool:
    """Atomically check concurrency + daily-minute budget and reserve a slot if both pass."""
    async with _lock:
        if not _has_capacity(org_id):
            return False
        _active_calls_by_org[org_id] = _active_calls_by_org.get(org_id, 0) + 1
        return True


async def release_call_slot(org_id: str, duration_seconds: float) -> None:
    """Release a previously acquired slot and record the minutes consumed."""
    async with _lock:
        _active_calls_by_org[org_id] = max(0, _active_calls_by_org.get(org_id, 0) - 1)
        _minutes_used_by_org[org_id] = (
            _minutes_used_by_org.get(org_id, 0.0) + duration_seconds / 60
        )


def clamp_call_timeout(requested_seconds: int) -> int:
    """Hard ceiling on call duration, regardless of agent-configured timeout."""
    return min(requested_seconds, _max_call_duration_seconds())


def peek_capacity_available(org_id: str) -> bool:
    """Non-mutating read of the same capacity check used by try_acquire_call_slot."""
    return _has_capacity(org_id)
