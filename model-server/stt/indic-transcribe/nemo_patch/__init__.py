"""
Vendored NeMo fixes required by canary_bhili_ft.nemo. See README.md in this directory.

`ensure_installed()` copies the vendored tokenizer into the live NeMo package if it is not
already there, so the patch applies whether we run from a venv or inside the container.
"""
from __future__ import annotations

import shutil
from pathlib import Path

_MODULE = "canary_multilingual_tokenizer.py"


def ensure_installed() -> bool:
    """Returns True if it had to install the file, False if NeMo already had it."""
    import nemo.collections.common.tokenizers as pkg

    dest = Path(pkg.__file__).parent / _MODULE
    src = Path(__file__).parent / _MODULE
    if dest.exists() and dest.read_bytes() == src.read_bytes():
        return False
    shutil.copyfile(src, dest)
    return True


ensure_installed()
