"""Vodafone Idea (VI) Voice Streaming WebSocket serializer."""

from __future__ import annotations

import asyncio
import base64
import json
import time
from typing import Optional

from loguru import logger
from pydantic import BaseModel, Field
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.utils import create_stream_resampler
from pipecat.frames.frames import (
    AudioRawFrame,
    CancelFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType

VI_SAMPLE_RATE = 8000
MIN_CHUNK_BYTES = 1600  # 1.6 KB (~100 ms at 8 kHz mono 16-bit)
MAX_CHUNK_BYTES = 51200  # 50 KB
CHUNK_ALIGN_BYTES = 160  # 10 ms of audio
BYTES_PER_SECOND = VI_SAMPLE_RATE * 2

# VI acknowledges marks once the referenced audio has finished playing to the
# caller. We send one before `exit` so the closing message is not truncated.
FINAL_MARK_NAME = "voicera-final"
MAX_DRAIN_SECS = 30.0
DRAIN_GRACE_SECS = 2.0


class ViFrameSerializer(FrameSerializer):
    """Serializer for Vodafone Idea bidirectional Voice Streaming protocol."""

    class InputParams(BaseModel):
        """Configuration for ViFrameSerializer.

        Parameters:
            sample_rate: Pipeline audio rate; defaults to the StartFrame rate.
            auto_exit: Whether to emit `exit` when the pipeline ends.
            exit_parameters: Values VI passes back to the call flow after `exit`.
        """

        sample_rate: Optional[int] = None
        auto_exit: bool = True
        exit_parameters: dict = Field(default_factory=dict)

    def __init__(
        self,
        room_id: str,
        call_id: str,
        websocket=None,
        params: Optional[InputParams] = None,
    ):
        self._params = params or ViFrameSerializer.InputParams()
        self._sample_rate = self._params.sample_rate or VI_SAMPLE_RATE
        self._room_id = room_id
        self._call_id = call_id
        # Used to emit protocol messages that do not map 1:1 onto a pipeline
        # frame (tail flush and the closing mark before `exit`).
        self._websocket = websocket
        self._sequence_number = 0
        self._media_chunk_index = 0
        self._stream_start_time = time.monotonic()
        self._output_buffer = bytearray()
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._exit_sent = False
        self._stop_received = False
        # Monotonic time at which everything sent so far finishes playing.
        self._playout_deadline = time.monotonic()
        self._final_mark_event = asyncio.Event()

    @property
    def type(self) -> FrameSerializerType:
        """VI exchanges JSON text frames."""
        return FrameSerializerType.TEXT

    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate
        self._stream_start_time = time.monotonic()
        self._playout_deadline = time.monotonic()

    def set_exit_parameters(self, parameters: dict) -> None:
        """Attach parameters VI passes back to the call flow after `exit`."""
        self._params.exit_parameters = dict(parameters or {})

    def _next_sequence(self) -> int:
        self._sequence_number += 1
        return self._sequence_number

    def _timestamp_ms(self) -> str:
        elapsed_ms = int((time.monotonic() - self._stream_start_time) * 1000)
        return str(max(elapsed_ms, 0))

    @staticmethod
    def _align_down(size: int) -> int:
        if size <= 0:
            return 0
        return size - (size % CHUNK_ALIGN_BYTES)

    def _take_output_chunk(self, force: bool = False) -> Optional[bytes]:
        aligned_len = self._align_down(len(self._output_buffer))
        if aligned_len < MIN_CHUNK_BYTES and not force:
            return None
        if aligned_len <= 0 and not force:
            return None

        chunk_len = self._align_down(min(aligned_len, MAX_CHUNK_BYTES))
        if chunk_len < MIN_CHUNK_BYTES:
            if not force or not self._output_buffer:
                return None
            # VI rejects sub-1.6 KB frames, so pad the remainder with silence.
            self._output_buffer.extend(b"\x00" * (MIN_CHUNK_BYTES - len(self._output_buffer)))
            chunk_len = MIN_CHUNK_BYTES

        chunk = bytes(self._output_buffer[:chunk_len])
        del self._output_buffer[:chunk_len]
        return chunk

    def _track_playout(self, pcm_bytes: int) -> None:
        duration = pcm_bytes / BYTES_PER_SECOND
        self._playout_deadline = max(time.monotonic(), self._playout_deadline) + duration

    def _build_media_message(self, pcm_data: bytes) -> dict:
        self._media_chunk_index += 1
        self._track_playout(len(pcm_data))
        return {
            "event": "media",
            "sequence_number": self._next_sequence(),
            "room_id": self._room_id,
            "media": {
                "chunk": self._media_chunk_index,
                "timestamp": self._timestamp_ms(),
                "payload": base64.b64encode(pcm_data).decode("utf-8"),
            },
        }

    def _build_mark_message(self, name: str) -> dict:
        return {
            "event": "mark",
            "sequence_number": self._next_sequence(),
            "room_id": self._room_id,
            "mark": {"name": name},
        }

    def _build_clear_message(self) -> dict:
        return {
            "event": "clear",
            "room_id": self._room_id,
        }

    def _build_exit_message(self) -> dict:
        return {
            "event": "exit",
            "room_id": self._room_id,
            "exit": {"parameters": dict(self._params.exit_parameters)},
        }

    async def _send_out_of_band(self, message: dict) -> None:
        """Send a protocol message directly, outside the frame pipeline."""
        if self._websocket is None:
            return
        try:
            await self._websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.debug(f"VI out-of-band send failed: {e}")

    async def _wait_for_playback(self) -> None:
        """Block until VI confirms playback, or the audio would have finished."""
        remaining = max(0.0, self._playout_deadline - time.monotonic())
        timeout = min(remaining + DRAIN_GRACE_SECS, MAX_DRAIN_SECS)
        try:
            await asyncio.wait_for(self._final_mark_event.wait(), timeout=timeout)
            logger.debug("VI confirmed final playback via mark")
        except asyncio.TimeoutError:
            logger.warning(
                "VI did not acknowledge final mark within {:.1f}s; exiting anyway",
                timeout,
            )

    async def _drain_and_exit(self) -> str:
        self._exit_sent = True
        tail = self._take_output_chunk(force=True)
        if tail:
            await self._send_out_of_band(self._build_media_message(tail))
        if self._websocket is not None:
            await self._send_out_of_band(self._build_mark_message(FINAL_MARK_NAME))
            await self._wait_for_playback()
        return json.dumps(self._build_exit_message())

    async def _resample_to_vi_rate(self, audio: bytes, sample_rate: int) -> bytes:
        if sample_rate == VI_SAMPLE_RATE:
            return audio
        return await self._output_resampler.resample(audio, sample_rate, VI_SAMPLE_RATE)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            # Drop queued audio on both sides so barge-in feels immediate.
            self._output_buffer.clear()
            self._playout_deadline = time.monotonic()
            return json.dumps(self._build_clear_message())

        if isinstance(frame, (EndFrame, CancelFrame)):
            if self._exit_sent or not self._params.auto_exit:
                return None
            if self._stop_received:
                # VI already tore the stream down; sending `exit` is pointless.
                self._exit_sent = True
                self._output_buffer.clear()
                return None
            return await self._drain_and_exit()

        if isinstance(frame, AudioRawFrame):
            data = await self._resample_to_vi_rate(frame.audio, frame.sample_rate)
            if not data:
                return None
            self._output_buffer.extend(data)
            chunk = self._take_output_chunk(force=False)
            if chunk:
                return json.dumps(self._build_media_message(chunk))
            return None

        return None

    @staticmethod
    def _dtmf_digit_to_keypad(digit: str) -> Optional[KeypadEntry]:
        mapping = {
            "0": KeypadEntry.ZERO,
            "1": KeypadEntry.ONE,
            "2": KeypadEntry.TWO,
            "3": KeypadEntry.THREE,
            "4": KeypadEntry.FOUR,
            "5": KeypadEntry.FIVE,
            "6": KeypadEntry.SIX,
            "7": KeypadEntry.SEVEN,
            "8": KeypadEntry.EIGHT,
            "9": KeypadEntry.NINE,
            "*": KeypadEntry.STAR,
            "#": KeypadEntry.POUND,
        }
        return mapping.get(str(digit).strip())

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            message = json.loads(data)
        except json.JSONDecodeError:
            logger.warning("VI serializer failed to parse JSON message")
            return None

        event = message.get("event")
        if event in ("connected", "start"):
            return None

        if event == "media":
            media = message.get("media", {})
            payload_base64 = media.get("payload")
            if not payload_base64:
                return None
            payload = base64.b64decode(payload_base64)
            if not payload:
                return None
            pcm = payload
            if self._sample_rate != VI_SAMPLE_RATE:
                pcm = await self._input_resampler.resample(
                    payload, VI_SAMPLE_RATE, self._sample_rate
                )
            return InputAudioRawFrame(
                audio=pcm,
                num_channels=1,
                sample_rate=self._sample_rate,
            )

        if event == "dtmf":
            dtmf = message.get("dtmf", {})
            digit = dtmf.get("digit")
            keypad = self._dtmf_digit_to_keypad(digit) if digit is not None else None
            if keypad is not None:
                return InputDTMFFrame(button=keypad)
            return None

        if event == "clear":
            return InterruptionFrame()

        if event == "stop":
            reason = (message.get("stop") or {}).get("reason", "")
            logger.info(f"VI stop event received (reason={reason})")
            self._stop_received = True
            self._final_mark_event.set()
            return EndFrame()

        if event == "mark":
            name = (message.get("mark") or {}).get("name")
            if name == FINAL_MARK_NAME:
                self._final_mark_event.set()
            return None

        return None
