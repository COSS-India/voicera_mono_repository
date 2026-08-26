import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# Stubs stand in for the GPU stack; they must precede real packages.
sys.path.insert(0, str(Path(__file__).resolve().parent / "stubs"))
sys.path.insert(0, str(ROOT / "gateway"))
