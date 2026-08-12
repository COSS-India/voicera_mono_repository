"""Shared fixtures for the tts_eval test suite.

The suite is split by the module/behaviour under test (``test_<area>.py``), one
file per area, mirroring the package layout. Fixtures common to more than one file
live here so pytest injects them everywhere without imports; single-use helpers
stay local to the file that needs them.
"""
from __future__ import annotations

import pytest

from tts_eval.config import load_model_card, load_suite
from tts_eval.store import RunStore

# Built-in dataset every run defaults to; referenced by name in several files.
DATASET = "indic_conversational_v1"


@pytest.fixture
def mock_card():
    """The offline, bit-deterministic reference model — no server, no GPU."""
    return load_model_card("mock")


@pytest.fixture
def smoke_suite():
    """13-utterance fast suite used across most run-level tests."""
    return load_suite("smoke")


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs")
