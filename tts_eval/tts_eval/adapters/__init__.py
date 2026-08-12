"""Adapter package. Importing it registers every built-in adapter.

Built-ins are safe to import unconditionally: each one imports its transport
library lazily inside ``__init__`` (via ``TTSAdapter._require``), so a machine
without ``websockets`` or ``aiohttp`` can still list adapters, replay stored
audio, and run the mock — it only fails when it actually tries to construct the
adapter whose transport is missing, and then with an exact pip command.
"""
from __future__ import annotations

from .base import (  # noqa: F401
    TTSAdapter,
    available_adapters,
    build_adapter,
    get_adapter_class,
    iter_registered,
    load_adapter_module,
    register_adapter,
    resolve_voice,
)

# Import order is the order shown by `tts-eval adapters`.
from . import websocket_pcm  # noqa: F401,E402
from . import http_rest  # noqa: F401,E402
from . import mock  # noqa: F401,E402
from . import replay  # noqa: F401,E402

__all__ = [
    "TTSAdapter",
    "available_adapters",
    "build_adapter",
    "get_adapter_class",
    "iter_registered",
    "load_adapter_module",
    "register_adapter",
    "resolve_voice",
]
