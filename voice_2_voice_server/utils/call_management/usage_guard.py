"""Per-org call budget guardrails: concurrency, daily minutes, and call duration ceiling.

State is in-memory and per-process by design: it resets whenever the voice server
restarts, which aligns with the sandbox's daily operational schedule.

Because the state is per-process, MAX_CONCURRENT_CALLS_PER_ORG and
MAX_MINUTES_PER_ORG_PER_DAY are treated as FLEET-WIDE limits and divided by
VOICE_SERVER_NUM_WORKERS. Without that division the configured value was
multiplied by the worker count, so the documented "6th concurrent call is
rejected" was really the 21st at 4 workers and the 61st at 12.
"""

import asyncio
import os

from loguru import logger

_lock = asyncio.Lock()
_active_calls_by_org: dict[str, int] = {}
_minutes_used_by_org: dict[str, float] = {}


def _worker_count() -> int:
    """Number of uvicorn worker processes this budget is split across.

    Must match api.server.run_server's VOICE_SERVER_NUM_WORKERS default.
    """
    try:
        return max(1, int(os.getenv("VOICE_SERVER_NUM_WORKERS", "4")))
    except ValueError:
        return 4


def _shard(limit: float) -> float:
    """Split a fleet-wide limit into this process's share.

    The counters above are per-process, so an unsharded limit silently multiplies
    by the worker count: MAX_CONCURRENT_CALLS_PER_ORG=5 admitted 20 concurrent
    calls at 4 workers and 60 at 12. Sharding makes the env var mean what it says.

    Distribution across workers is not perfectly even (the OS assigns each
    connection to whichever worker accepts it), so the effective fleet limit is
    approximate within about one worker's share. Set
    MAX_CONCURRENT_CALLS_FLEETWIDE=0 to restore the old per-process behaviour.
    """
    if os.getenv("MAX_CONCURRENT_CALLS_FLEETWIDE", "1").strip() == "0":
        return limit
    workers = _worker_count()
    if workers <= 1:
        return limit
    return max(1.0, limit / workers)


def _max_concurrent_calls_per_org() -> float:
    return _shard(float(os.getenv("MAX_CONCURRENT_CALLS_PER_ORG", "5")))


def _max_minutes_per_org_per_day() -> float:
    return _shard(float(os.getenv("MAX_MINUTES_PER_ORG_PER_DAY", "60")))


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
