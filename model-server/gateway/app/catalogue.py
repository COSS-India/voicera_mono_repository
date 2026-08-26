"""The model catalogue, read from models.yaml.

One place describes every model the server can host. The gateway serves it so
the frontend stops shipping its own hardcoded lists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

log = logging.getLogger("gateway")

# Baked in next to the app at build time; overridable for local runs.
CATALOGUE_PATH = Path(__file__).resolve().parent / "models.yaml"

KINDS = ("stt", "tts", "llm")


def load(path: Path | None = None) -> list[dict[str, Any]]:
    """Flatten models.yaml into one list, each entry carrying its kind.

    A missing or unreadable catalogue is not fatal -- the gateway still routes
    traffic; it just cannot describe what it is routing to.
    """
    src = path or CATALOGUE_PATH
    try:
        raw = yaml.safe_load(src.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        log.warning("catalogue unavailable (%s): %s", src, exc)
        return []

    models: list[dict[str, Any]] = []
    for kind in KINDS:
        for entry in raw.get(kind) or []:
            models.append({"kind": kind, **entry})
    return models
