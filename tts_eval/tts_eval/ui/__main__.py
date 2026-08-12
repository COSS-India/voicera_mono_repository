"""``python -m tts_eval.ui [--runs DIR] [--host HOST] [--port PORT]``

Standalone entry point so the UI is usable before the full ``tts-eval`` CLI
exists, and remains usable afterwards as the thing the CLI's ``serve``
subcommand calls into.
"""
from __future__ import annotations

import argparse

from .server import serve_forever


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m tts_eval.ui", description="Browse tts_eval runs, reports and comparisons."
    )
    parser.add_argument("--runs", default="runs", help="Run store directory (default: ./runs)")
    parser.add_argument("--host", default="127.0.0.1", help="Bind address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port (default: 8765)")
    args = parser.parse_args()
    serve_forever(args.runs, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
