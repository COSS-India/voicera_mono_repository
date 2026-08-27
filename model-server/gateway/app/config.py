"""Gateway configuration.

A slot is deployed when a MODEL is named for it -- the same variable that drives
COMPOSE_PROFILES -- so the gateway and Compose can never disagree about what is
running. The upstream URL defaults to the Compose service name and only needs
setting to point at a different host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# Compose service names on the internal network. Overridden by <KIND>_UPSTREAM.
_DEFAULT_URL = {
    "stt": "http://stt:8001",
    "tts": "http://tts:8002",
    "llm": "http://llm:8003",
}


def _clean(name: str) -> str:
    return (os.getenv(name) or "").strip().rstrip("/")


@dataclass(frozen=True)
class Upstream:
    kind: str          # "stt" | "tts" | "llm"
    url: str           # "" when the slot is not deployed
    model: str         # model id, also the Compose profile name

    @property
    def enabled(self) -> bool:
        return bool(self.model and self.url)


@dataclass(frozen=True)
class Settings:
    stt: Upstream
    tts: Upstream
    llm: Upstream

    @staticmethod
    def _slot(kind: str) -> Upstream:
        model = _clean(f"{kind.upper()}_MODEL")
        url = _clean(f"{kind.upper()}_UPSTREAM") or (_DEFAULT_URL[kind] if model else "")
        return Upstream(kind, url, model)

    @classmethod
    def from_env(cls) -> Settings:
        return cls(*(cls._slot(k) for k in ("stt", "tts", "llm")))

    def all(self) -> list[Upstream]:
        return [self.stt, self.tts, self.llm]


settings = Settings.from_env()
