"""Optional metric backends.

Importing this package registers them; it does **not** import their heavy
dependencies. Each backend's ``available()`` performs the import check and returns
a reason string, so a laptop install can list and select these backends and simply
receive ``not_computed`` values with an actionable message.
"""
from __future__ import annotations

from . import naturalness  # noqa: F401
from . import speaker  # noqa: F401
from . import ttsds2  # noqa: F401
from . import versa  # noqa: F401

__all__ = ["naturalness", "speaker", "ttsds2", "versa"]
