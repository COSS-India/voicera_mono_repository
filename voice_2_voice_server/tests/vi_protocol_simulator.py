#!/usr/bin/env python3
"""Simulate a Vodafone Idea Voice Streaming client against the VI endpoint.

VI connects to us, so this script plays VI's role: it opens the WebSocket,
performs the connected/start handshake, streams caller audio, acknowledges
marks, and validates every message the server sends back against the VI spec.

Examples:
    python3 tests/vi_protocol_simulator.py \\
        --url ws://localhost:7860/vi/agent/mahavistaar --seconds 20

    # Exercise the proxy-stripped-path fallback
    python3 tests/vi_protocol_simulator.py \\
        --url ws://localhost:7860/ --agent-id mahavistaar

    # Send real caller audio instead of silence
    python3 tests/vi_protocol_simulator.py --url ws://localhost:7860/vi/stream \\
        --agent-id mahavistaar --wav sample_8k_mono.wav
"""

from __future__ import annotations

import argparse
import asyncio
import audioop
import base64
import json
import sys
import time
import uuid
import wave

try:
    import websockets
except ImportError:
    sys.exit("Install the websockets package first: pip install websockets")

VI_SAMPLE_RATE = 8000
CHUNK_BYTES = 1600  # 100 ms at 8 kHz mono PCM16
CHUNK_INTERVAL = CHUNK_BYTES / (VI_SAMPLE_RATE * 2)
MIN_CHUNK_BYTES = 1600
MAX_CHUNK_BYTES = 51200
CHUNK_ALIGN_BYTES = 160

VALID_INBOUND_EVENTS = {"media", "mark", "clear", "exit"}


class Report:
    """Collects protocol violations and traffic counters."""

    def __init__(self) -> None:
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.counts: dict[str, int] = {}
        self.audio_bytes = 0
        self.first_audio_latency: float | None = None

    def count(self, event: str) -> None:
        self.counts[event] = self.counts.get(event, 0) + 1

    def error(self, message: str) -> None:
        if message not in self.errors:
            self.errors.append(message)
            print(f"  [FAIL] {message}")

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            print(f"  [WARN] {message}")


def load_audio(path: str | None, seconds: float) -> bytes:
    """Return 8 kHz mono PCM16 audio, either from a WAV file or silence."""
    if not path:
        return b"\x00\x00" * int(VI_SAMPLE_RATE * seconds)

    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        pcm = wf.readframes(wf.getnframes())

    if width != 2:
        pcm = audioop.lin2lin(pcm, width, 2)
    if channels > 1:
        pcm = audioop.tomono(pcm, 2, 0.5, 0.5)
    if rate != VI_SAMPLE_RATE:
        pcm, _ = audioop.ratecv(pcm, 2, 1, rate, VI_SAMPLE_RATE, None)
        print(f"Resampled {path} from {rate} Hz to {VI_SAMPLE_RATE} Hz")
    return pcm


def validate_media(message: dict, room_id: str, report: Report) -> None:
    media = message.get("media")
    if not isinstance(media, dict):
        report.error("media message has no media object")
        return

    payload = media.get("payload")
    if not payload:
        report.error("media message has empty payload")
        return

    try:
        pcm = base64.b64decode(payload)
    except Exception:
        report.error("media payload is not valid base64")
        return

    size = len(pcm)
    report.audio_bytes += size

    if size % CHUNK_ALIGN_BYTES:
        report.error(f"chunk of {size} bytes is not a multiple of {CHUNK_ALIGN_BYTES}")
    if size < MIN_CHUNK_BYTES:
        report.error(f"chunk of {size} bytes is below the {MIN_CHUNK_BYTES} byte minimum")
    if size > MAX_CHUNK_BYTES:
        report.error(f"chunk of {size} bytes exceeds the {MAX_CHUNK_BYTES} byte maximum")

    if message.get("room_id") != room_id:
        report.error(f"media room_id mismatch: got {message.get('room_id')!r}")
    for field in ("chunk", "timestamp"):
        if field not in media:
            report.warn(f"media object is missing '{field}'")
    if "sequence_number" not in message:
        report.warn("media message is missing 'sequence_number'")


async def receiver(ws, room_id: str, report: Report, started: float, done: asyncio.Event) -> None:
    async for raw in ws:
        try:
            message = json.loads(raw)
        except json.JSONDecodeError:
            report.error(f"server sent non-JSON data: {str(raw)[:80]}")
            continue

        event = message.get("event")
        report.count(event or "<missing>")

        if event not in VALID_INBOUND_EVENTS:
            report.error(f"unexpected event from server: {event!r}")
            continue

        if event == "media":
            if report.first_audio_latency is None:
                report.first_audio_latency = time.monotonic() - started
                print(f"  first bot audio after {report.first_audio_latency:.2f}s")
            validate_media(message, room_id, report)

        elif event == "mark":
            name = (message.get("mark") or {}).get("name")
            print(f"  server sent mark '{name}' -> acknowledging")
            await ws.send(json.dumps({
                "event": "mark",
                "sequence_number": 999,
                "room_id": room_id,
                "mark": {"name": name},
            }))

        elif event == "clear":
            print("  server sent clear (barge-in)")

        elif event == "exit":
            print("  server sent exit -> session closing")
            done.set()
            return


async def run(args: argparse.Namespace) -> int:
    room_id = f"SIM_{uuid.uuid4().hex[:12]}"
    call_id = f"CALL_{uuid.uuid4().hex[:12]}"
    audio = load_audio(args.wav, args.seconds)
    report = Report()

    custom_parameters: dict[str, str] = {}
    if args.agent_id:
        custom_parameters["agent_id"] = args.agent_id
    if args.language:
        custom_parameters["language"] = args.language

    print(f"Connecting to {args.url}")
    print(f"  room_id={room_id} call_id={call_id} custom_parameters={custom_parameters}")

    async with websockets.connect(args.url, ping_interval=None, max_size=None) as ws:
        await ws.send(json.dumps({"event": "connected"}))
        await ws.send(json.dumps({
            "event": "start",
            "sequence_number": 1,
            "room_id": room_id,
            "start": {
                "room_id": room_id,
                "call_id": call_id,
                "cli": args.cli,
                "dni": args.dni,
                "custom_parameters": custom_parameters,
                "media_format": {
                    "encoding": "raw",
                    "sample_rate": str(VI_SAMPLE_RATE),
                    "bit_rate": "128000",
                },
            },
        }))
        print("  handshake sent (connected + start)")

        started = time.monotonic()
        done = asyncio.Event()
        reader = asyncio.create_task(receiver(ws, room_id, report, started, done))

        # Stream caller audio in real time, as VI would.
        sequence = 1
        chunk_index = 0
        next_send = time.monotonic()
        for offset in range(0, len(audio), CHUNK_BYTES):
            if done.is_set():
                break
            chunk = audio[offset:offset + CHUNK_BYTES]
            if len(chunk) < CHUNK_BYTES:
                chunk = chunk.ljust(CHUNK_BYTES, b"\x00")
            sequence += 1
            chunk_index += 1
            await ws.send(json.dumps({
                "event": "media",
                "sequence_number": sequence,
                "room_id": room_id,
                "media": {
                    "chunk": chunk_index,
                    "timestamp": str(int(chunk_index * CHUNK_INTERVAL * 1000)),
                    "payload": base64.b64encode(chunk).decode(),
                },
            }))
            next_send += CHUNK_INTERVAL
            await asyncio.sleep(max(0.0, next_send - time.monotonic()))

        print(f"  streamed {chunk_index} caller chunks; draining for {args.drain}s")
        try:
            await asyncio.wait_for(done.wait(), timeout=args.drain)
        except asyncio.TimeoutError:
            pass

        if not done.is_set():
            await ws.send(json.dumps({
                "event": "stop",
                "sequence_number": sequence + 1,
                "room_id": room_id,
                "stop": {"call_id": call_id, "reason": "simulator finished"},
            }))
            print("  sent stop")
            await asyncio.sleep(1.0)

        reader.cancel()

    print("\n=== Summary ===")
    print(f"events received : {report.counts or 'none'}")
    print(f"bot audio        : {report.audio_bytes} bytes "
          f"({report.audio_bytes / (VI_SAMPLE_RATE * 2):.1f}s)")
    if report.first_audio_latency is not None:
        print(f"first audio      : {report.first_audio_latency:.2f}s after start")

    if not report.counts.get("media"):
        report.error("server never sent any audio")

    if report.warnings:
        print(f"warnings         : {len(report.warnings)}")
    if report.errors:
        print(f"\nFAILED with {len(report.errors)} protocol violation(s)")
        return 1

    print("\nPASSED - server output conforms to the VI streaming spec")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="VI WebSocket endpoint")
    parser.add_argument("--agent-id", help="Sent as a custom parameter in start")
    parser.add_argument("--language", help="Optional language custom parameter")
    parser.add_argument("--wav", help="WAV file of caller audio (any rate)")
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="Seconds of silence when no WAV is given")
    parser.add_argument("--drain", type=float, default=25.0,
                        help="Seconds to keep listening after caller audio ends")
    parser.add_argument("--cli", default="919000000001", help="Caller number")
    parser.add_argument("--dni", default="919000000002", help="Called number")
    args = parser.parse_args()

    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
