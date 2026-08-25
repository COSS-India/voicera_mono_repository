"""Reachability probes for self-hosted model servers.

The agent forms only offer a self-hosted provider when its server actually
answers, so a deployment without the AI4Bharat or Qwen servers running does not
advertise models that would fail mid-call. Readiness-probe semantics: a
short-timeout HTTP GET, cached briefly so opening the form does not wait on a
probe per render.
"""

import asyncio
import os
from typing import Any, Dict, List

import aiohttp
from loguru import logger

# Provider key -> env var holding its base URL. An unset var means the model is
# not deployed in this environment.
SELF_HOSTED_PROBES = {
    "llm:qwen": "VLLM_BASE_URL",
    "stt:ai4bharat": "INDIC_STT_SERVER_URL",
    "tts:ai4bharat": "INDIC_TTS_SERVER_URL",
}

PROBE_TIMEOUT_SECONDS = 1.0
PROBE_CACHE_TTL_SECONDS = 30.0

_cache: Dict[str, Any] = {"expires_at": 0.0, "available": []}


async def probe(url: str) -> bool:
    """Return True if a model server at `url` answers a health check.

    Any response below 500 counts as up: a 404 still proves the process is
    alive and routing, unlike a bare TCP connect which a wedged process also
    passes. ws:// URLs are probed over http:// since those ports serve both.
    """
    probe_url = (
        url.replace("ws://", "http://").replace("wss://", "https://").rstrip("/")
    )
    timeout = aiohttp.ClientTimeout(total=PROBE_TIMEOUT_SECONDS)
    for candidate in (f"{probe_url}/health", probe_url):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(candidate) as resp:
                    if resp.status < 500:
                        return True
        except Exception as exc:
            logger.debug(f"Self-hosted probe failed for {candidate}: {exc}")
    return False


def _now() -> float:
    return asyncio.get_event_loop().time()


async def available_providers(use_cache: bool = True) -> Dict[str, Any]:
    """List self-hosted providers currently reachable, with a short TTL cache."""
    if use_cache and _now() < _cache["expires_at"]:
        return {"available": _cache["available"], "cached": True}

    configured = {
        key: os.getenv(env_var)
        for key, env_var in SELF_HOSTED_PROBES.items()
        if os.getenv(env_var)
    }
    results = await asyncio.gather(*(probe(url) for url in configured.values()))
    available: List[str] = [
        key for key, reachable in zip(configured, results) if reachable
    ]

    _cache["available"] = available
    _cache["expires_at"] = _now() + PROBE_CACHE_TTL_SECONDS
    return {"available": available, "cached": False}


def reset_cache() -> None:
    """Drop the cached probe result (used by tests and manual refreshes)."""
    _cache["expires_at"] = 0.0
    _cache["available"] = []
