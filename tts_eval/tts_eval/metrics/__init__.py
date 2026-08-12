"""Metric engine, catalogue, core backends and optional plugins.

Importing this package registers every backend. Core backends are numpy-only;
plugins defer their heavy imports to ``available()``, so this import is cheap and
safe in a minimal install.
"""
from __future__ import annotations

from .aggregate import (  # noqa: F401
    aggregate_per_category,
    aggregate_per_language,
    aggregate_records,
    aggregate_run_values,
    summarise,
)
from .base import (  # noqa: F401
    CORE_BACKENDS,
    STANDARD_BACKENDS,
    MetricBackend,
    MetricContext,
    MetricEngine,
    RunBackend,
    Thresholds,
    UtteranceBackend,
    available_backends,
    build_backends,
    get_backend_class,
    iter_registered,
    make_value,
    missing_value,
    register_backend,
    resolve_backend_names,
)
from .catalog import CATALOG, ac_matrix, by_criterion, criteria_order, spec  # noqa: F401

# Core tier: always importable, always computable.
from . import audio_quality  # noqa: F401,E402
from . import coverage  # noqa: F401,E402
from . import intelligibility  # noqa: F401,E402
from . import latency  # noqa: F401,E402
from . import reliability  # noqa: F401,E402
from . import voice_consistency  # noqa: F401,E402

# Optional tier: registered here, dependency-checked at run time.
from . import plugins  # noqa: F401,E402

__all__ = [
    "CATALOG",
    "CORE_BACKENDS",
    "STANDARD_BACKENDS",
    "MetricBackend",
    "MetricContext",
    "MetricEngine",
    "RunBackend",
    "Thresholds",
    "UtteranceBackend",
    "ac_matrix",
    "aggregate_per_category",
    "aggregate_per_language",
    "aggregate_records",
    "aggregate_run_values",
    "available_backends",
    "build_backends",
    "by_criterion",
    "criteria_order",
    "get_backend_class",
    "iter_registered",
    "make_value",
    "missing_value",
    "register_backend",
    "resolve_backend_names",
    "spec",
    "summarise",
]
