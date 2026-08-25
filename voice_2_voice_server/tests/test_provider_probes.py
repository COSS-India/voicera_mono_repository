"""Self-check for self-hosted provider probes.

Runnable directly: python voice_2_voice_server/tests/test_provider_probes.py
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils import provider_probes as probes  # noqa: E402

# Port 1 is reserved and never served, so probes against it must fail.
DEAD_URL = "http://127.0.0.1:1"


def _clear_env():
    for env_var in probes.SELF_HOSTED_PROBES.values():
        os.environ.pop(env_var, None)


def test_unset_env_means_unavailable():
    """A provider with no URL configured is never offered."""
    _clear_env()
    probes.reset_cache()
    result = asyncio.run(probes.available_providers(use_cache=False))
    assert result["available"] == [], result


def test_configured_but_unreachable_is_unavailable():
    """A configured URL that nothing is listening on stays hidden."""
    _clear_env()
    os.environ["VLLM_BASE_URL"] = f"{DEAD_URL}/v1"
    probes.reset_cache()
    result = asyncio.run(probes.available_providers(use_cache=False))
    assert "llm:qwen" not in result["available"], result
    _clear_env()


def test_ws_url_probed_over_http():
    """ws:// TTS URLs must not reach the HTTP client unrewritten."""
    assert asyncio.run(probes.probe("ws://127.0.0.1:1")) is False


def test_cache_returned_without_reprobing():
    """A warm cache short-circuits the probe and says so."""
    _clear_env()

    async def check():
        probes._cache["available"] = ["stt:ai4bharat"]
        probes._cache["expires_at"] = (
            asyncio.get_event_loop().time() + probes.PROBE_CACHE_TTL_SECONDS
        )
        return await probes.available_providers()

    result = asyncio.run(check())
    assert result["cached"] is True, result
    assert result["available"] == ["stt:ai4bharat"], result
    probes.reset_cache()


def test_reset_cache_forces_reprobe():
    """After reset, a stale cached value is not served."""
    _clear_env()
    probes._cache["available"] = ["llm:qwen"]
    probes.reset_cache()
    result = asyncio.run(probes.available_providers())
    assert result["available"] == [], result


if __name__ == "__main__":
    test_unset_env_means_unavailable()
    test_configured_but_unreachable_is_unavailable()
    test_ws_url_probed_over_http()
    test_cache_returned_without_reprobing()
    test_reset_cache_forces_reprobe()
    print("all probe checks passed")
