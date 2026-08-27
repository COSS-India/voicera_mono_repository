"""Shared request plumbing: typed access to application state."""
from __future__ import annotations

from fastapi import HTTPException, Request, status

from ..config import Settings
from ..engine import TTSEngine
from ..voices import Roster


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_engine(request: Request) -> TTSEngine:
    engine: TTSEngine = request.app.state.engine
    if not engine.ready:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "engine is still loading; poll GET /health until it reports ready",
        )
    return engine


def get_roster(request: Request) -> Roster:
    return request.app.state.roster
