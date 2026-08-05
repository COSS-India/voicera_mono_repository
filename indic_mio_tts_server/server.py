"""WebSocket TTS server for SPRINGLab/Indic-Mio.

Speaks the exact same wire contract as the AI4Bharat (Parler) TTS server so the
voice pipeline adapter is a near-verbatim copy:

Client -> server, one JSON per utterance:
    {"prompt": "...", "voice": "...", "description": "...", "language": "..."}
  Only "prompt" is required. "voice" is an optional preset voice id selecting the
  speaker embedding (unknown/absent -> default voice). "description"/"language"
  are accepted for contract compatibility and are informational only.

Server -> client, in order:
  1. {"type":"meta","pid":...,"sample_rate":<SR>,"dtype":"float32","channels":1}
  2. binary frames: raw float32 mono PCM
  3. {"type":"done","pid":...}
  Errors: {"type":"error","message":...}

Unlike the Parler server this process does NOT hold the acoustic model or run a
hand-written batching loop. Token generation is delegated to vLLM (continuous
batching, high concurrency); this process only orchestrates I/O and the light
MioCodec decode. It is therefore a single async process, not multi-process.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid

import numpy as np
import websockets

from config import Config
from tts_engine import MioTTSEngine, TTSGenerationError

logging.basicConfig(
    level=os.getenv("MIO_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("indic_mio.server")


async def handle_client(websocket, engine: MioTTSEngine, config: Config) -> None:
    pid = uuid.uuid4().hex[:8]
    try:
        raw = await websocket.recv()
    except websockets.ConnectionClosed:
        return

    try:
        msg = json.loads(raw)
        prompt = msg["prompt"]
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        voice = msg.get("voice")
        if voice is not None and not isinstance(voice, str):
            raise TypeError("voice must be a string")
    except (json.JSONDecodeError, KeyError, TypeError) as e:
        await _safe_send(websocket, json.dumps({"type": "error", "message": f"bad request: {e}"}))
        return

    await _safe_send(
        websocket,
        json.dumps(
            {
                "type": "meta",
                "pid": pid,
                "sample_rate": engine.sample_rate,
                "dtype": "float32",
                "channels": 1,
            }
        ),
    )

    # Stream PCM as the engine produces it (incremental decode -> low TTFB),
    # re-slicing each decoded chunk into modest fixed-size frames for the socket.
    frame = max(1, config.frame_samples)
    try:
        async for chunk in engine.synthesize_stream(prompt, voice=voice):
            for start in range(0, chunk.size, frame):
                part = chunk[start : start + frame]
                if part.size == 0:
                    continue
                await websocket.send(part.astype(np.float32, copy=False).tobytes())
        await websocket.send(json.dumps({"type": "done", "pid": pid}))
    except websockets.ConnectionClosed:
        return
    except TTSGenerationError as e:
        logger.warning("[%s] generation failed: %s", pid, e)
        await _safe_send(websocket, json.dumps({"type": "error", "message": str(e)}))
        return
    except Exception as e:  # noqa: BLE001 - last-resort guard, keep socket clean
        logger.exception("[%s] unexpected error", pid)
        await _safe_send(websocket, json.dumps({"type": "error", "message": f"internal error: {e}"}))
        return


async def _safe_send(websocket, payload) -> None:
    try:
        await websocket.send(payload)
    except websockets.ConnectionClosed:
        pass


async def main_async(config: Config) -> None:
    engine = MioTTSEngine(config)
    engine.load_codec()
    await engine.start()

    async def _serve(ws):
        await handle_client(ws, engine, config)

    try:
        async with websockets.serve(_serve, config.host, config.port, max_size=None):
            logger.info(
                "Indic-Mio TTS server ws://%s:%d (vllm=%s model=%s codec=%s sr=%d)",
                config.host,
                config.port,
                config.llm_base_url,
                config.llm_model,
                config.codec_model_id,
                engine.sample_rate,
            )
            await asyncio.Future()
    finally:
        await engine.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Indic-Mio TTS WebSocket server (vLLM + MioCodec)")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = Config.from_env()
    if args.host:
        config = config.__class__(**{**config.__dict__, "host": args.host})
    if args.port:
        config = config.__class__(**{**config.__dict__, "port": args.port})

    asyncio.run(main_async(config))


if __name__ == "__main__":
    main()
