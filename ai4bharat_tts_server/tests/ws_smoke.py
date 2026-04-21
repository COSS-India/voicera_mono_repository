"""Run `python server.py` first, then: python tests/ws_smoke.py"""
from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path

import numpy as np
import websockets
from scipy.io import wavfile

URI = "ws://127.0.0.1:8002"
PAYLOAD = {"prompt": "अरे, तुम आज कैसे हो? कैसे हो? कैसे हो? कैसे हो?", "description": "talk loud and easy"}

OUT_DIR = Path(__file__).resolve().parent / "files"


def safe_filename_from_prompt(prompt: str, max_len: int = 120) -> str:
    s = prompt.strip()
    s = re.sub(r'[<>:"/\\|?*\n\r\t]', "_", s)
    s = re.sub(r"\s+", "_", s)
    s = s.strip("._") or "output"
    return s[:max_len]


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    async with websockets.connect(URI) as ws:
        await ws.send(json.dumps(PAYLOAD))
        meta = json.loads(await ws.recv())
        assert meta["type"] == "meta"
        sample_rate = int(meta.get("sample_rate", 24000))

        chunks: list[np.ndarray] = []
        while True:
            msg = await ws.recv()
            if isinstance(msg, str):
                body = json.loads(msg)
                assert body["type"] == "done"
                break
            chunks.append(np.frombuffer(msg, dtype=np.float32))

        pcm = np.concatenate(chunks) if chunks else np.array([], dtype=np.float32)
        assert pcm.size > 0

        out_path = OUT_DIR / f"{safe_filename_from_prompt(PAYLOAD['prompt'])}.wav"
        wavfile.write(out_path, sample_rate, pcm)
        print(f"ok -> {out_path} ({pcm.size} samples @ {sample_rate} Hz)")


if __name__ == "__main__":
    asyncio.run(main())
