"""`python -m orpheus_server` — start the server using config.yaml's host/port."""
from __future__ import annotations

import uvicorn

from .app import app
from .config import load_settings

if __name__ == "__main__":
    settings, _ = load_settings()
    uvicorn.run(app, host=settings.server.host, port=settings.server.port, log_config=None)
