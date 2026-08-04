"""Smoke test. Requires vLLM (Indic-Mio) + this server running.

    vllm serve SPRINGLab/Indic-Mio --gpu-memory-utilization 0.5 --port 8100
    INDIC_MIO_VLLM_URL=http://localhost:8100/v1 python server.py
    python tests/ws_smoke.py
"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import websockets
from scipy.io import wavfile

URI = "ws://127.0.0.1:8003"
PAYLOAD = {
    "prompt": "नमस्ते, आप कैसे हैं? <happy>",
    "description": "A clear, natural voice.",
    "language": "hi",
}
OUT_DIR = Path(__file__).resolve().parent / "files"


def safe_filename(prompt: str, max_len: int = 80) -> str:
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", prompt.strip())
    s = re.sub(r"\s+", "_", s).strip("._") or "output"
    return s[:max_len]


async def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    async with websockets.connect(URI, max_size=None) as ws:
        await ws.send(json.dumps(PAYLOAD))
        meta = json.loads(await ws.recv())
        assert meta["type"] == "meta", meta
        sample_rate = int(meta["sample_rate"])

        chunks: list[np.ndarray] = []
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                body = json.loads(msg)
                assert body["type"] == "done", body
                break
            chunks.append(np.frombuffer(msg, dtype=np.float32))

        pcm = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        assert pcm.size > 0, "no audio received"

        out_path = OUT_DIR / f"{safe_filename(PAYLOAD['prompt'])}.wav"
        wavfile.write(out_path, sample_rate, pcm)
        print(f"ok -> {out_path} ({pcm.size} samples @ {sample_rate} Hz)")


if __name__ == "__main__":
    asyncio.run(main())
