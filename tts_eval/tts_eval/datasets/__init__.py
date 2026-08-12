"""Test-set loading and identity. See :mod:`tts_eval.datasets.loader`."""
from __future__ import annotations

from .loader import (  # noqa: F401
    BUILTIN_DIR,
    TestCase,
    TestDataset,
    dataset_from_cases,
    list_builtin,
    load_dataset,
)

__all__ = [
    "BUILTIN_DIR",
    "TestCase",
    "TestDataset",
    "dataset_from_cases",
    "list_builtin",
    "load_dataset",
]
