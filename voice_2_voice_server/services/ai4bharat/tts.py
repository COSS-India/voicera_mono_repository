import asyncio
import base64
import collections
import contextlib
import json
import os
from typing import AsyncGenerator

import aiohttp
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


class IndicParlerRESTTTSService(TTSService):

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 24000,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        server_url = os.getenv("INDIC_TTS_SERVER_URL")
        if not server_url:
            raise ValueError("INDIC_TTS_SERVER_URL environment variable not set")

        server_url = server_url.rstrip("/")
        if server_url.startswith("http://"):
            server_url = "ws://" + server_url[len("http://"):]
        elif server_url.startswith("https://"):
            server_url = "wss://" + server_url[len("https://"):]

        self._ws_url = f"{server_url}/ws"
        self._speaker = speaker
        self._description = description

        self._session: aiohttp.ClientSession | None = None
        self._ws: aiohttp.ClientWebSocketResponse | None = None
        self._recv_task: asyncio.Task | None = None

        self._connect_lock = asyncio.Lock()

        # FIFO of (queue, snippet) in wire-send order.
        # Used only to resolve the server-assigned request_id on the first chunk.
        # After that, _active[request_id] routes directly.
        self._pending_fifo: collections.deque = collections.deque()
        self._active: dict[int, asyncio.Queue] = {}
        self._tracking_lock = asyncio.Lock()
        # Serialises (append-to-fifo + ws.send) only — not audio streaming.
        self._send_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(self, frame: Frame):
        logger.info("Starting IndicParlerWS TTS service")
        connector = aiohttp.TCPConnector(limit=4, ttl_dns_cache=300)
        self._session = aiohttp.ClientSession(
            connector=connector,
            timeout=aiohttp.ClientTimeout(total=None, connect=10),
        )
        await self._ensure_connected()
        await super().start(frame)

    async def stop(self, frame: Frame):
        logger.info("Stopping IndicParlerWS TTS service")
        if self._recv_task:
            self._recv_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._recv_task
            self._recv_task = None
        if self._ws and not self._ws.closed:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._ws = None
        self._session = None
        await super().stop(frame)

    # ------------------------------------------------------------------
    # Connection management
    # ------------------------------------------------------------------

    async def _ensure_connected(self):
        async with self._connect_lock:
            if self._ws and not self._ws.closed:
                return
            logger.info(f"Connecting to TTS WebSocket: {self._ws_url}")
            self._ws = await self._session.ws_connect(
                self._ws_url,
                heartbeat=20,
                receive_timeout=None,
            )
            logger.info("WebSocket connected")

            if self._recv_task and not self._recv_task.done():
                self._recv_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._recv_task

            self._recv_task = asyncio.create_task(
                self._receiver_loop(), name="indic-parler-ws-recv"
            )

    # ------------------------------------------------------------------
    # Receiver loop
    # ------------------------------------------------------------------

    async def _receiver_loop(self):
        try:
            async for msg in self._ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)
                    except json.JSONDecodeError:
                        logger.warning(f"Non-JSON WS message: {msg.data!r}")
                        continue

                    if "error" in data:
                        await self._broadcast_error(data["error"])
                        continue

                    rid = data.get("request_id")
                    if rid is None:
                        logger.warning(f"Message with no request_id: {data}")
                        continue

                    async with self._tracking_lock:
                        if rid not in self._active:
                            if not self._pending_fifo:
                                logger.warning(f"Received request_id={rid} but no pending requests — dropping")
                                continue
                            queue, snippet = self._pending_fifo.popleft()
                            self._active[rid] = queue
                            logger.info(f"Resolved request_id={rid} → {snippet!r}")

                        queue = self._active.get(rid)

                    if queue:
                        await queue.put(data)

                elif msg.type in (aiohttp.WSMsgType.CLOSE, aiohttp.WSMsgType.CLOSED):
                    logger.warning("WebSocket closed by server")
                    break
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    logger.error(f"WebSocket error: {self._ws.exception()}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Receiver loop crashed: {e}")
        finally:
            await self._broadcast_error("WebSocket connection lost")

    async def _broadcast_error(self, message: str):
        async with self._tracking_lock:
            for q in self._active.values():
                await q.put({"error": message})
            for q, _ in self._pending_fifo:
                await q.put({"error": message})

    # ------------------------------------------------------------------
    # Core TTS
    # ------------------------------------------------------------------

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        full_description = f"{self._speaker}'s voice. {self._description}"
        queue: asyncio.Queue = asyncio.Queue()
        snippet = text[:40]

        try:
            await self._ensure_connected()
        except Exception as e:
            yield ErrorFrame(f"Failed to connect to TTS server: {e}")
            return

        async with self._send_lock:
            async with self._tracking_lock:
                self._pending_fifo.append((queue, snippet))
            try:
                await self._ws.send_str(json.dumps({"text": text, "description": full_description}))
                logger.info(f"Sent TTS request: {snippet!r}")
            except Exception as e:
                async with self._tracking_lock:
                    with contextlib.suppress(ValueError):
                        self._pending_fifo.remove((queue, snippet))
                yield ErrorFrame(f"Failed to send TTS request: {e}")
                return

        yield TTSStartedFrame()

        counter = 0
        resolved_rid = None
        try:
            while True:
                try:
                    data = await asyncio.wait_for(queue.get(), timeout=15.0)
                except asyncio.TimeoutError:
                    logger.warning(f"[req={resolved_rid or 'unresolved'}] Timeout waiting for audio chunk")
                    yield ErrorFrame("TTS chunk timeout")
                    break

                if "error" in data:
                    yield ErrorFrame(data["error"])
                    break

                if resolved_rid is None:
                    resolved_rid = data.get("request_id", "?")

                if "chunk" in data:
                    audio_bytes = base64.b64decode(data["chunk"])
                    sample_rate = data.get("sampling_rate", self.sample_rate)
                    logger.info(f"[req={resolved_rid}] chunk={counter} size={len(audio_bytes)}B sr={sample_rate}")
                    yield TTSAudioRawFrame(audio=audio_bytes, sample_rate=sample_rate, num_channels=1)
                    counter += 1

                if data.get("is_final"):
                    logger.info(f"[req={resolved_rid}] Complete — {counter} chunks")
                    break

        finally:
            if resolved_rid is not None:
                async with self._tracking_lock:
                    self._active.pop(resolved_rid, None)

        yield TTSStoppedFrame()