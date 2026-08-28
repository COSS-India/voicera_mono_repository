"""
The GPU worker: one thread, one CUDA consumer, and the batch-formation window.

Why this is a separate module
-----------------------------
The engine knows how to advance a set of sessions by one tick. It does not decide *when* to
tick, or which sessions to include. That decision is the experiment surface of this project, so
it lives here where it can be swept at runtime rather than being buried in the engine.

The batch-formation window (W)
------------------------------
Today's behaviour, which `W = 0` reproduces exactly, is: tick as soon as *any* session is ready.
That is why `avg_sessions_per_tick` measures 1.47 at N=8 and 2.33 at N=16 -- sessions arrive at
independent phases, so most ticks carry one session and pay a full encoder forward for it.

With `W > 0` the worker waits up to W ms after the first session becomes ready, letting others
join the same batch. The encoder batches nearly for free (30.8 ms at B=1 vs 35.6 ms at B=16), so
grouping is close to pure win *on the encoder*.

It is not a large win overall, and it is worth being honest about why. Fitting the inherited
measurements gives `tick_ms ~= 31 + sessions * 42`, and the decoder term is serial per session
regardless of grouping, because `_decode_one` is a Python loop over sessions. Per chunk period:

    N=16 -> encoder ~219 ms + decoder 16*42 = 672 ms -> 891 ms of a 960 ms budget

Grouping removes only the encoder term, so a perfect window moves capacity from ~16 to ~19-20.
The decoder alone caps it at 960/42 ~= 23 no matter how good the batching is. Only batching or
accelerating the decoder itself moves that ceiling.

W is also not free: it adds up to W ms of latency, and the budget it has to fit inside shrinks
with the chunk. At the baseline 0.96 s chunk there is room; at a 0.32 s word-by-word geometry the
whole tick budget is 320 ms, so W is a real tradeoff rather than a free parameter. Hence: swept,
not guessed.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

log = logging.getLogger("core.batcher")


@dataclass
class BatcherConfig:
    #: W. 0 reproduces "tick as soon as any session is ready" -- today's behaviour.
    batch_window_ms: float = 0.0
    #: Stop waiting early once this many sessions are ready; W is a ceiling, not a target.
    max_batch: int = 32
    #: Scale W down as the ready set grows, so a busy server does not pay the full wait.
    adaptive: bool = False
    #: How often the worker re-checks readiness while idle.
    poll_ms: float = 2.0

    def snapshot(self) -> dict:
        return {
            "batch_window_ms": self.batch_window_ms,
            "max_batch": self.max_batch,
            "adaptive": self.adaptive,
            "poll_ms": self.poll_ms,
        }


@dataclass
class BatcherStats:
    """Latency attribution. Every boundary is recorded so a number can be blamed on a stage
    rather than attributed by argument."""

    waits_ms: list = field(default_factory=list)      # time spent inside the batch window
    formed_sizes: list = field(default_factory=list)  # ready-count at the moment of ticking
    queue_depth: list = field(default_factory=list)
    idle_polls: int = 0
    ticks: int = 0

    def snapshot(self) -> dict:
        def pct(xs, q):
            if not xs:
                return None
            s = sorted(xs)
            return round(s[min(len(s) - 1, int(q * len(s)))], 2)

        return {
            "batcher_ticks": self.ticks,
            "batch_wait_ms_p50": pct(self.waits_ms, 0.50),
            "batch_wait_ms_p95": pct(self.waits_ms, 0.95),
            "formed_batch_p50": pct(self.formed_sizes, 0.50),
            "formed_batch_max": max(self.formed_sizes) if self.formed_sizes else None,
            "queue_depth_p95": pct(self.queue_depth, 0.95),
        }


class GpuWorker:
    """Runs `engine.tick()` on one thread and routes deltas to per-session sinks.

    One thread, always. The engine owns the GPU from a single context; a second worker would
    contend for the same device and interleave two sessions' decoder state.
    """

    def __init__(self, engine, cfg: BatcherConfig,
                 on_delta: Optional[Callable[[str, object], None]] = None):
        self.engine = engine
        self.cfg = cfg
        self.stats = BatcherStats()
        self._on_delta = on_delta
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sinks: dict[str, Callable] = {}
        self._lock = threading.Lock()

    # ---- sinks ----------------------------------------------------------------------
    def register(self, sid: str, sink: Callable) -> None:
        with self._lock:
            self._sinks[sid] = sink

    def unregister(self, sid: str) -> None:
        with self._lock:
            self._sinks.pop(sid, None)

    # ---- lifecycle ------------------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="gpu-worker", daemon=True)
        self._thread.start()
        log.info("gpu worker started (W=%.0f ms, max_batch=%d, adaptive=%s)",
                 self.cfg.batch_window_ms, self.cfg.max_batch, self.cfg.adaptive)

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    # ---- the loop -------------------------------------------------------------------
    def _ready_count(self) -> int:
        return sum(1 for s in list(self.engine.sessions.values()) if s.ready())

    def _run(self) -> None:
        poll = self.cfg.poll_ms / 1000.0
        while not self._stop.is_set():
            n = self._ready_count()
            if n == 0:
                self.stats.idle_polls += 1
                time.sleep(poll)
                continue

            wait_ms = 0.0
            W = self.cfg.batch_window_ms
            if W > 0 and n < self.cfg.max_batch:
                # Hold briefly so sessions at nearby phases land in the same encoder forward.
                if self.cfg.adaptive:
                    # A busy server already has a full batch forming; do not make it wait the
                    # full W to discover that.
                    W = W * max(0.0, 1.0 - n / max(1, self.cfg.max_batch))
                deadline = time.monotonic() + W / 1000.0
                while time.monotonic() < deadline and not self._stop.is_set():
                    if self._ready_count() >= self.cfg.max_batch:
                        break
                    time.sleep(min(poll, 0.001))
                wait_ms = W - max(0.0, (deadline - time.monotonic()) * 1000.0)
                n = self._ready_count()

            self.stats.waits_ms.append(round(wait_ms, 2))
            self.stats.formed_sizes.append(n)
            self.stats.queue_depth.append(len(self.engine.sessions))
            self.stats.ticks += 1

            try:
                deltas = self.engine.tick()
            except Exception as e:
                # A CUDA context failure is NOT transient. This used to be a blanket
                # `log; sleep; continue`, so an unrecoverable error left the worker spinning
                # forever: the process never exited, `restart: unless-stopped` never fired, and
                # the container went on reporting healthy while every session failed. A single
                # bad session became a total outage that only a human could clear.
                #
                # Now: fail loudly and DIE, so the supervisor replaces us with a clean context.
                from core_engine import is_fatal_cuda

                if is_fatal_cuda(e) or getattr(self.engine, "fatal", None):
                    self.engine.fatal = getattr(self.engine, "fatal", None) or \
                        f"{type(e).__name__}: {str(e)[:200]}"
                    self.engine.ready = False
                    log.critical("FATAL: CUDA context unrecoverable, exiting for restart: %s",
                                 self.engine.fatal)
                    self._stop.set()
                    self._die()
                    return
                log.exception("tick failed")
                time.sleep(poll)
                continue

            if deltas:
                with self._lock:
                    sinks = dict(self._sinks)
                for sid, d in deltas.items():
                    sink = sinks.get(sid)
                    if sink is None:
                        continue
                    try:
                        sink(d)
                    except Exception:
                        log.exception("sink failed for session %s", sid)
                if self._on_delta:
                    for sid, d in deltas.items():
                        self._on_delta(sid, d)

    @staticmethod
    def _die() -> None:
        """Replace the process. os._exit because a poisoned CUDA context makes an orderly
        shutdown unreliable -- atexit handlers and CUDA teardown can hang on the dead device,
        which would leave exactly the zombie this exists to prevent."""
        import os
        import sys

        sys.stderr.flush()
        sys.stdout.flush()
        time.sleep(0.3)          # let the log line reach the docker daemon
        os._exit(70)             # EX_SOFTWARE; compose `restart: unless-stopped` takes it from here

    # ---- introspection --------------------------------------------------------------
    def metrics(self) -> dict:
        m = self.stats.snapshot()
        m.update(self.cfg.snapshot())
        return m
