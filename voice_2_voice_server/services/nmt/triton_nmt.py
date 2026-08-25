"""Client for the hosted AI4Bharat IndicTrans2 NMT model (Triton, KServe v2).

The value here is the *coalescer*. The Triton model config reports
``dynamic_batching.max_queue_delay_microseconds: 0`` and one GPU instance, so
the server batches only what happens to be queued at the instant a pass starts.
Real batch width therefore has to be created client-side.

Every live-translation language worker calls :meth:`NmtClient.translate`
independently. A 5-language room fires 5 such calls within microseconds of each
other; a short coalescing window (default 8 ms) lets them leave as ONE Triton
request of 5 rows — one GPU pass instead of five. The window also merges rows
across *different* rooms, so N concurrent broadcasts do not mean N× requests.

Measured latency for the whole request (incl. RTT to ap-south-1):
    1 row  ~0.20 s | 3 rows ~0.38 s | 260-char paragraph ~0.43 s
i.e. the model returns the full translation for every language faster than a
typical LLM's time-to-first-token, so no output streaming is needed.

One request either fully succeeds or fully fails: a single unsupported pair or a
blank line makes Triton reject the entire batch. So a batch failure is contained
by re-submitting each row on its own, once (see ``_dispatch``).
"""

from __future__ import annotations

import asyncio
import os
from typing import List, Optional, Tuple

import aiohttp
from loguru import logger


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid {}={!r}, using {}", name, os.getenv(name), default)
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        logger.warning("Invalid {}={!r}, using {}", name, os.getenv(name), default)
        return default


class NmtError(Exception):
    """A row could not be translated (backend error, unsupported pair, timeout).

    ``retriable`` marks a *transient* failure (network blip, read timeout, 5xx)
    that a second attempt might clear — as opposed to a permanent one (an
    unsupported language pair, a model-rejected row), where retrying only wastes
    a request and adds latency to an already-doomed segment.
    """

    def __init__(self, message: str, *, retriable: bool = False):
        super().__init__(message)
        self.retriable = retriable


class _Row:
    """One (text, src, tgt) unit of work and the future awaiting its translation."""

    __slots__ = ("text", "src", "tgt", "future")

    def __init__(self, text: str, src: str, tgt: str, future: "asyncio.Future[str]"):
        self.text = text
        self.src = src
        self.tgt = tgt
        self.future = future


class NmtClient:
    """Process-global batching client for one Triton NMT model.

    Construct once (via :func:`get_nmt_client`) and share across all rooms. The
    background collector task and the aiohttp session are created lazily on the
    running event loop so import never touches the network.
    """

    def __init__(self) -> None:
        server_url = os.getenv("NMT_SERVER_URL")
        if not server_url:
            raise ValueError("NMT_SERVER_URL environment variable not set")
        self._base = server_url.strip().rstrip("/")
        self._model = os.getenv("NMT_MODEL_NAME", "nmt").strip() or "nmt"
        self._infer_url = f"{self._base}/v2/models/{self._model}/infer"
        self._ready_url = f"{self._base}/v2/models/{self._model}/ready"

        self._window_s = max(0.0, _env_int("NMT_BATCH_WINDOW_MS", 8) / 1000.0)
        self._max_batch = max(1, _env_int("NMT_MAX_BATCH", 64))
        self._max_inflight = max(1, _env_int("NMT_MAX_INFLIGHT", 4))
        self._timeout_s = _env_float("NMT_TIMEOUT_SECS", 8.0)
        # Readiness probe budget. Kept short so a bad NMT host fails the presenter
        # fast rather than hanging the connect, but env-tunable for a slow first
        # connect over a distant link.
        self._ready_timeout_s = _env_float("NMT_READY_TIMEOUT_SECS", 3.0)
        self._max_queue = max(1, _env_int("NMT_MAX_QUEUE", 2048))
        # Hard backstop on the caller's wait, so a lost future (collector death,
        # a dispatch that dies before resolving its rows) can never hang a
        # language for the whole broadcast. Covers queue wait + one batch + one
        # split-retry + margin; the http read timeout is the normal-case bound.
        self._await_timeout_s = self._timeout_s * 2 + self._window_s + 5.0

        self._queue: "asyncio.Queue[_Row]" = asyncio.Queue(maxsize=self._max_queue)
        self._session: Optional[aiohttp.ClientSession] = None
        self._collector: Optional[asyncio.Task] = None
        self._sema = asyncio.Semaphore(self._max_inflight)
        # Strong refs to in-flight dispatch tasks. asyncio only holds a weak ref
        # to a bare create_task() result, so without this an unawaited dispatch
        # can be garbage-collected mid-request, leaving its rows' futures unset
        # forever. Discarded via done-callback when each finishes.
        self._inflight_tasks: "set[asyncio.Task]" = set()
        self._started = False
        # Lightweight rolling stats, flushed every 100 batches.
        self._n_batches = 0
        self._n_rows = 0
        self._sum_latency = 0.0
        self._n_split_retries = 0

    # -- lifecycle ---------------------------------------------------------

    def _ensure_started(self) -> None:
        if self._started:
            return
        connector = aiohttp.TCPConnector(
            limit=self._max_inflight * 2, ttl_dns_cache=300, keepalive_timeout=60
        )
        timeout = aiohttp.ClientTimeout(
            total=None, connect=2, sock_read=self._timeout_s
        )
        self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        self._collector = asyncio.create_task(self._collect())
        self._started = True
        logger.info(
            "NMT client ready: {} (window={:.0f}ms max_batch={} inflight={})",
            self._infer_url,
            self._window_s * 1000,
            self._max_batch,
            self._max_inflight,
        )

    async def ready(self) -> bool:
        """Pre-flight probe used at presenter connect. True if the model serves."""
        self._ensure_started()
        assert self._session is not None
        try:
            async with self._session.get(
                self._ready_url,
                timeout=aiohttp.ClientTimeout(total=self._ready_timeout_s),
            ) as resp:
                if resp.status != 200:
                    logger.warning(
                        "NMT readiness probe {} returned HTTP {}",
                        self._ready_url,
                        resp.status,
                    )
                return resp.status == 200
        except Exception as e:
            # A bare TimeoutError str()s to "", which reads as a blank log line and
            # hides the actual cause. Name the exception type explicitly. A timeout
            # here almost always means the host cannot REACH the NMT box (firewall /
            # security group), not that the model is down.
            detail = str(e) or type(e).__name__
            logger.warning(
                "NMT readiness probe failed for {} ({}). Check egress/firewall to "
                "the NMT host.",
                self._ready_url,
                detail,
            )
            return False

    async def aclose(self) -> None:
        if self._collector is not None:
            self._collector.cancel()
            try:
                await self._collector
            except (asyncio.CancelledError, Exception):
                pass
            self._collector = None
        # Cancel any dispatch tasks still in flight so their sockets are torn down
        # with the session rather than left pending.
        for task in list(self._inflight_tasks):
            task.cancel()
        if self._inflight_tasks:
            await asyncio.gather(*self._inflight_tasks, return_exceptions=True)
            self._inflight_tasks.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None
        self._started = False

    # -- public API --------------------------------------------------------

    async def translate(self, text: str, src_code: str, tgt_code: str) -> str:
        """Translate one segment ``src_code -> tgt_code``. Raises :class:`NmtError`.

        Rows validated here never reach a batch: blank text and same-language
        pairs are the two inputs observed to poison an entire Triton batch, and
        a same-language pair also returns garbage. A same-language request is a
        no-op passthrough (the presenter chose a listener language equal to the
        source), returned without a network call.
        """
        self._ensure_started()
        stripped = text.strip()
        if not stripped:
            return ""
        if not src_code or not tgt_code:
            raise NmtError("missing source or target language code")
        if src_code == tgt_code:
            return text
        loop = asyncio.get_running_loop()
        future: "asyncio.Future[str]" = loop.create_future()
        row = _Row(stripped, src_code, tgt_code, future)
        try:
            self._queue.put_nowait(row)
        except asyncio.QueueFull:
            # Shed load rather than grow memory: the caller drops this segment,
            # keeping the stream close to live speech (same policy as the
            # upstream segment backlog).
            raise NmtError("NMT queue full")
        try:
            return await asyncio.wait_for(future, self._await_timeout_s)
        except asyncio.TimeoutError:
            # wait_for has already cancelled ``future``; a late-arriving result
            # from the batch path sees future.done() and is dropped cleanly.
            raise NmtError("NMT translate timed out", retriable=True)

    # -- batching internals ------------------------------------------------

    async def _collect(self) -> None:
        """Pull rows, coalesce within a short window, dispatch as batches.

        A single unexpected error must not kill the collector for good — that
        would leave every future ``translate`` waiting on the ``_await_timeout_s``
        backstop with no recovery. Log and continue instead; only cancellation
        (from :meth:`aclose`) ends the loop.
        """
        while True:
            try:
                first = await self._queue.get()
                batch: List[_Row] = [first]
                deadline = asyncio.get_running_loop().time() + self._window_s
                while len(batch) < self._max_batch:
                    remaining = deadline - asyncio.get_running_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        row = await asyncio.wait_for(self._queue.get(), remaining)
                    except asyncio.TimeoutError:
                        break
                    batch.append(row)
                # Dispatch without awaiting so the next window opens immediately;
                # the semaphore inside _dispatch bounds real concurrency. Keep a
                # strong ref until the task finishes (see _inflight_tasks).
                task = asyncio.create_task(self._dispatch(batch))
                self._inflight_tasks.add(task)
                task.add_done_callback(self._inflight_tasks.discard)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error("NMT collector loop error, continuing: {}", e)
                await asyncio.sleep(0.1)

    async def _dispatch(self, batch: List[_Row]) -> None:
        async with self._sema:
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            try:
                results = await self._infer([(r.text, r.src, r.tgt) for r in batch])
            except Exception as e:
                await self._handle_batch_failure(batch, e)
                return
            latency = loop.time() - t0
            self._record(len(batch), latency)
            if len(results) != len(batch):
                # A length mismatch means we cannot trust the index alignment;
                # mis-assigning would send one language's audio to another. Fail
                # the whole batch loudly instead.
                for r in batch:
                    if not r.future.done():
                        r.future.set_exception(
                            NmtError(
                                f"response rows {len(results)} != request rows {len(batch)}"
                            )
                        )
                return
            for r, out in zip(batch, results):
                if not r.future.done():
                    r.future.set_result(out)

    async def _handle_batch_failure(self, batch: List[_Row], error: Exception) -> None:
        """Contain a poisoned batch: retry each row alone, once.

        Triton fails the whole batch on any single bad row (unsupported pair,
        blank line). Re-submitting individually isolates the offender so other
        rooms' languages are unaffected.

        A batch of one has no offender to isolate, but a *transient* failure
        (network blip, read timeout, 5xx) still deserves the same single retry a
        row inside a bigger batch would get — otherwise a common single-language
        broadcast drops the segment on the first hiccup while a multi-language
        one recovers. A permanent failure (unsupported pair) is not retried.
        """
        if len(batch) <= 1:
            row = batch[0] if batch else None
            if row is not None and getattr(error, "retriable", False):
                if not row.future.done():
                    self._n_split_retries += 1
                    await self._retry_single(row)
                return
            for r in batch:
                if not r.future.done():
                    r.future.set_exception(NmtError(str(error)))
            return
        self._n_split_retries += 1
        logger.warning(
            "NMT batch of {} failed ({}); split-retrying rows individually",
            len(batch),
            error,
        )
        await asyncio.gather(
            *(self._retry_single(r) for r in batch), return_exceptions=True
        )

    async def _retry_single(self, row: _Row) -> None:
        async with self._sema:
            try:
                results = await self._infer([(row.text, row.src, row.tgt)])
                out = results[0] if results else ""
                if not row.future.done():
                    row.future.set_result(out)
            except Exception as e:
                if not row.future.done():
                    row.future.set_exception(NmtError(str(e)))

    async def _infer(self, rows: List[Tuple[str, str, str]]) -> List[str]:
        """One Triton infer call. Returns index-aligned OUTPUT_TEXT, or raises."""
        assert self._session is not None
        n = len(rows)
        payload = {
            "inputs": [
                {
                    "name": "INPUT_TEXT",
                    "shape": [n, 1],
                    "datatype": "BYTES",
                    "data": [r[0] for r in rows],
                },
                {
                    "name": "INPUT_LANGUAGE_ID",
                    "shape": [n, 1],
                    "datatype": "BYTES",
                    "data": [r[1] for r in rows],
                },
                {
                    "name": "OUTPUT_LANGUAGE_ID",
                    "shape": [n, 1],
                    "datatype": "BYTES",
                    "data": [r[2] for r in rows],
                },
            ]
        }
        try:
            async with self._session.post(self._infer_url, json=payload) as resp:
                body = await resp.json(content_type=None)
                if resp.status != 200 or (isinstance(body, dict) and "error" in body):
                    msg = body.get("error") if isinstance(body, dict) else f"HTTP {resp.status}"
                    # 5xx / no status = transient server-side; a 4xx or a model
                    # "error" body (e.g. unsupported pair) is permanent.
                    raise NmtError(str(msg), retriable=resp.status >= 500)
                for out in body.get("outputs", []):
                    if out.get("name") == "OUTPUT_TEXT":
                        return [str(x) for x in out.get("data", [])]
                raise NmtError("OUTPUT_TEXT missing from NMT response")
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            # Connection reset, read timeout, DNS blip — worth one more try.
            raise NmtError(f"NMT transport error: {e or type(e).__name__}", retriable=True)

    def _record(self, rows: int, latency: float) -> None:
        self._n_batches += 1
        self._n_rows += rows
        self._sum_latency += latency
        if self._n_batches % 100 == 0:
            logger.info(
                "NMT: {} batches | avg rows={:.1f} avg latency={:.2f}s split-retries={}",
                self._n_batches,
                self._n_rows / self._n_batches,
                self._sum_latency / self._n_batches,
                self._n_split_retries,
            )


# -- process-global singleton -------------------------------------------------

_CLIENT: Optional[NmtClient] = None
_CLIENT_LOCK = asyncio.Lock()


async def get_nmt_client() -> NmtClient:
    """Return the shared NMT client, creating it on first use."""
    global _CLIENT
    if _CLIENT is not None:
        return _CLIENT
    async with _CLIENT_LOCK:
        if _CLIENT is None:
            _CLIENT = NmtClient()
        return _CLIENT
