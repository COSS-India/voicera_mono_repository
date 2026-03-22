"""Shared env loading for rag_system scripts (OpenAI key from voice_2_voice_server/.env)."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


def repo_root() -> Path:
    """rag_system/*.py → monorepo root (voicera_mono_repository)."""
    return Path(__file__).resolve().parents[2]


def load_rag_env() -> None:
    v2v = repo_root() / "voice_2_voice_server" / ".env"
    if v2v.is_file():
        load_dotenv(v2v)
    load_dotenv()
