import contextlib
import socket
import sys
import threading
import time
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parent.parent
# Stubs stand in for the GPU stack; they must precede real packages.
sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
sys.path.insert(0, str(ROOT / "gateway"))


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def serve(app, port: int):
    """Run an ASGI app on a background thread and wait until it accepts.

    Shared by every test that needs a real socket rather than a test client:
    the gateway's whole job is streaming and cancellation, and ASGI test
    clients do not reproduce either.
    """
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(cfg)
    threading.Thread(target=server.run, daemon=True).start()
    for _ in range(100):
        time.sleep(0.05)
        with contextlib.suppress(OSError), socket.create_connection(("127.0.0.1", port), 0.1):
            return server
    raise RuntimeError(f"server on {port} never came up")
