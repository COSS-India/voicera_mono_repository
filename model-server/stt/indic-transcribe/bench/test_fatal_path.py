#!/usr/bin/env python3
"""Prove the fatal-error path: an unrecoverable CUDA failure must kill the process, not linger.

The original outage was not the CUDA error itself -- it was that the worker caught it, logged it,
and kept spinning. The process stayed alive, `restart: unless-stopped` never fired, and /health
went on saying "ok" while every session failed. One bad session became an outage only a human
could clear.

The triggering bug is intermittent, so this does not wait for it. It injects the failure and
asserts the machinery: classify correctly, then exit(70) so the supervisor restarts us.
"""
import subprocess
import sys

sys.path.insert(0, "/app")
from core_engine import is_fatal_cuda  # noqa: E402

# ---- 1. classification -------------------------------------------------------------------
FATAL = [
    RuntimeError("CUDA error: an illegal memory access was encountered"),
    RuntimeError("CUBLAS_STATUS_INTERNAL_ERROR when calling cublasGemmStridedBatchedEx"),
    RuntimeError("CUDA error: device-side assert triggered"),
]
BENIGN = [
    ValueError("language 'bgc' is not supported by this checkpoint"),
    RuntimeError("session s3 is closed"),
    KeyError("prompt slot missing"),
    IndexError("list index out of range"),
]
bad = [repr(e) for e in FATAL if not is_fatal_cuda(e)] + \
      [repr(e) for e in BENIGN if is_fatal_cuda(e)]
print(f"[fatal] classifier: {len(FATAL)} fatal + {len(BENIGN)} benign -> "
      f"{'PASS' if not bad else 'FAIL ' + str(bad)}")

# ---- 2. the worker must terminate the process ---------------------------------------------
CHILD = '''
import sys, time, logging
sys.path.insert(0, "/app")
logging.basicConfig(level="CRITICAL")
from batcher import GpuWorker, BatcherConfig

class Sess:
    def ready(self): return True

class DeadEngine:
    """Behaves like an engine whose CUDA context has just been destroyed."""
    def __init__(self):
        self.sessions = {"s1": Sess()}
        self.fatal = None
        self.ready = True
    def tick(self):
        raise RuntimeError("CUDA error: an illegal memory access was encountered")

w = GpuWorker(DeadEngine(), BatcherConfig())
w.start()
time.sleep(10)          # if the worker swallowed it, we are still alive here -> wrong
print("WORKER SURVIVED A FATAL ERROR", flush=True)
sys.exit(0)
'''
r = subprocess.run([sys.executable, "-c", CHILD], capture_output=True, text=True, timeout=60)
died = r.returncode == 70
print(f"[fatal] worker on unrecoverable error: exit={r.returncode} "
      f"{'PASS (process replaced)' if died else 'FAIL (' + (r.stdout.strip() or 'no exit') + ')'}")

sys.exit(0 if (not bad and died) else 1)
