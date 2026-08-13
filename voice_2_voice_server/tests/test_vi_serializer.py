#!/usr/bin/env python3
"""Unit checks for ViFrameSerializer against the VI streaming spec."""

import asyncio
import base64
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pipecat.frames.frames import (  # noqa: E402
    EndFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    OutputAudioRawFrame,
)

from serializer.vi_serializer import (  # noqa: E402
    CHUNK_ALIGN_BYTES,
    FINAL_MARK_NAME,
    MAX_CHUNK_BYTES,
    MIN_CHUNK_BYTES,
    VI_SAMPLE_RATE,
    ViFrameSerializer,
)


class FakeWebSocket:
    """Captures out-of-band messages and auto-acknowledges the final mark."""

    def __init__(self, ack_marks=True):
        self.sent = []
        self._ack_marks = ack_marks
        self._serializer = None

    def bind(self, serializer):
        self._serializer = serializer

    async def send_text(self, text):
        message = json.loads(text)
        self.sent.append(message)
        if self._ack_marks and message.get("event") == "mark":
            await self._serializer.deserialize(json.dumps({
                "event": "mark",
                "room_id": message["room_id"],
                "mark": {"name": message["mark"]["name"]},
            }))

    def events(self):
        return [m["event"] for m in self.sent]


def make_serializer(websocket=None):
    serializer = ViFrameSerializer(
        room_id="ROOM1", call_id="CALL1", websocket=websocket
    )
    if websocket is not None:
        websocket.bind(serializer)
    return serializer


def audio_frame(num_bytes):
    return OutputAudioRawFrame(
        audio=b"\x01\x02" * (num_bytes // 2), sample_rate=VI_SAMPLE_RATE, num_channels=1
    )


def payload_size(message):
    return len(base64.b64decode(message["media"]["payload"]))


async def test_chunking_is_spec_compliant():
    serializer = make_serializer()
    emitted = []
    # 10 x 10ms chunks is what the transport sends for VI.
    for _ in range(25):
        result = await serializer.serialize(audio_frame(1600))
        if result:
            emitted.append(json.loads(result))

    assert emitted, "no media emitted"
    for message in emitted:
        size = payload_size(message)
        assert size % CHUNK_ALIGN_BYTES == 0, f"{size} not 160-aligned"
        assert MIN_CHUNK_BYTES <= size <= MAX_CHUNK_BYTES, f"{size} out of range"
        assert message["room_id"] == "ROOM1"
        assert message["event"] == "media"

    sequences = [m["sequence_number"] for m in emitted]
    assert sequences == sorted(sequences), "sequence numbers not monotonic"
    chunks = [m["media"]["chunk"] for m in emitted]
    assert chunks == list(range(1, len(chunks) + 1)), "chunk index not sequential"
    print(f"  chunking: {len(emitted)} messages, sizes ok, no buffering leftovers")


async def test_interruption_clears_buffer():
    serializer = make_serializer()
    await serializer.serialize(audio_frame(800))
    result = await serializer.serialize(InterruptionFrame())
    message = json.loads(result)
    assert message["event"] == "clear"
    assert message["room_id"] == "ROOM1"
    assert not serializer._output_buffer, "buffer survived interruption"
    print("  interruption: clear sent and buffer dropped")


async def test_exit_drains_tail_and_waits_for_mark():
    websocket = FakeWebSocket(ack_marks=True)
    serializer = make_serializer(websocket)
    # Leave a sub-minimum remainder in the buffer.
    await serializer.serialize(audio_frame(800))
    result = await serializer.serialize(EndFrame())
    message = json.loads(result)

    assert message["event"] == "exit"
    assert websocket.events() == ["media", "mark"], websocket.events()
    tail_size = payload_size(websocket.sent[0])
    assert tail_size == MIN_CHUNK_BYTES, f"tail padded to {tail_size}"
    assert websocket.sent[1]["mark"]["name"] == FINAL_MARK_NAME
    print("  exit: tail flushed with padding, mark sent and acknowledged")


async def test_exit_waits_for_playback_when_mark_ignored():
    websocket = FakeWebSocket(ack_marks=False)
    serializer = make_serializer(websocket)
    # ~1 second of audio queued for playback.
    for _ in range(5):
        await serializer.serialize(audio_frame(1600))

    start = asyncio.get_event_loop().time()
    await serializer.serialize(EndFrame())
    waited = asyncio.get_event_loop().time() - start

    assert waited > 0.4, f"exited after only {waited:.2f}s without a mark ack"
    print(f"  exit fallback: waited {waited:.2f}s for playout before exiting")


async def test_stop_suppresses_exit():
    websocket = FakeWebSocket()
    serializer = make_serializer(websocket)
    frame = await serializer.deserialize(json.dumps({
        "event": "stop", "room_id": "ROOM1", "stop": {"reason": "caller hangup"}
    }))
    assert isinstance(frame, EndFrame)
    result = await serializer.serialize(EndFrame())
    assert result is None, "exit sent to an already-stopped stream"
    assert websocket.sent == [], "traffic sent after stop"
    print("  stop: no exit written to a torn-down stream")


async def test_inbound_media_becomes_audio_frame():
    serializer = make_serializer()
    pcm = b"\x10\x20" * 800
    frame = await serializer.deserialize(json.dumps({
        "event": "media",
        "room_id": "ROOM1",
        "media": {"payload": base64.b64encode(pcm).decode()},
    }))
    assert isinstance(frame, InputAudioRawFrame)
    assert frame.sample_rate == VI_SAMPLE_RATE
    assert frame.audio == pcm
    print("  inbound media: decoded to 8 kHz InputAudioRawFrame")


async def main():
    tests = [
        test_chunking_is_spec_compliant,
        test_interruption_clears_buffer,
        test_exit_drains_tail_and_waits_for_mark,
        test_exit_waits_for_playback_when_mark_ignored,
        test_stop_suppresses_exit,
        test_inbound_media_becomes_audio_frame,
    ]
    failures = 0
    for test in tests:
        print(f"{test.__name__}:")
        try:
            await test()
        except AssertionError as e:
            failures += 1
            print(f"  FAILED: {e}")

    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
