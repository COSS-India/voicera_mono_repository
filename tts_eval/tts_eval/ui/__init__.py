"""Lightweight, dependency-free UI for browsing runs, reports and comparisons.

See :mod:`tts_eval.ui.server` for the route table and the reasoning behind
stdlib-only ``http.server`` instead of a web framework.
"""
from __future__ import annotations

from .server import (
    Handler, free_port, make_app, render_compare_form, render_configs_page,
    render_index, render_launch_page, serve, serve_forever,
)

__all__ = [
    "Handler",
    "free_port",
    "make_app",
    "render_compare_form",
    "render_configs_page",
    "render_index",
    "render_launch_page",
    "serve",
    "serve_forever",
]

