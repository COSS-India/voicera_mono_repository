"""Vodafone Idea (VI) Voice Streaming WebSocket serializer."""

from __future__ import annotations

import base64
import json
import time
from typing import Optional

from loguru import logger
from pydantic import Field
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
from pipecat.serializers.base_serializer import FrameSerializer

VI_SAMPLE_RATE = 8000
MIN_CHUNK_BYTES = 1600  # 1.6 KB (~100 ms at 8 kHz mono 16-bit)
MAX_CHUNK_BYTES = 51200  # 50 KB
CHUNK_ALIGN_BYTES = 160  # 10 ms of audio


class ViFrameSerializer(FrameSerializer):
    """Serializer for Vodafone Idea bidirectional Voice Streaming protocol."""

    class InputParams(FrameSerializer.InputParams):
        sample_rate: int = VI_SAMPLE_RATE
        auto_exit: bool = True
        exit_parameters: dict = Field(default_factory=dict)

    def __init__(
        self,
        room_id: str,
        call_id: str,
        params: Optional[InputParams] = None,
    ):
        super().__init__(params=params or ViFrameSerializer.InputParams())
        self._room_id = room_id
        self._call_id = call_id
        self._sequence_number = 0
        self._media_chunk_index = 0
        self._stream_start_time = time.monotonic()
        self._output_buffer = bytearray()
        self._input_resampler = create_stream_resampler()
        self._output_resampler = create_stream_resampler()
        self._exit_sent = False

    async def setup(self, frame: StartFrame):
        self._stream_start_time = time.monotonic()

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
        if aligned_len <= 0:
            return None

        chunk_len = min(aligned_len, MAX_CHUNK_BYTES)
        chunk_len = self._align_down(chunk_len)
        if chunk_len < MIN_CHUNK_BYTES and force:
            chunk_len = MIN_CHUNK_BYTES
            if len(self._output_buffer) < chunk_len:
                padding = chunk_len - len(self._output_buffer)
                self._output_buffer.extend(b"\x00" * padding)
        if chunk_len <= 0:
            return None

        chunk = bytes(self._output_buffer[:chunk_len])
        del self._output_buffer[:chunk_len]
        return chunk

    def _build_media_message(self, pcm_data: bytes) -> dict:
        self._media_chunk_index += 1
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

    def _build_clear_message(self) -> dict:
        return {
            "event": "clear",
            "room_id": self._room_id,
        }

    def _build_exit_message(self) -> dict:
        exit_body: dict = {"parameters": dict(self._params.exit_parameters)}
        return {
            "event": "exit",
            "room_id": self._room_id,
            "exit": exit_body,
        }

    async def _resample_to_vi_rate(self, audio: bytes, sample_rate: int) -> bytes:
        if sample_rate == VI_SAMPLE_RATE:
            return audio
        return await self._output_resampler.resample(audio, sample_rate, VI_SAMPLE_RATE)

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            self._output_buffer.clear()
            return json.dumps(self._build_clear_message())

        if (
            self._params.auto_exit
            and not self._exit_sent
            and isinstance(frame, (EndFrame, CancelFrame))
        ):
            self._output_buffer.clear()
            self._exit_sent = True
            return json.dumps(self._build_exit_message())

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
            if self._params.sample_rate != VI_SAMPLE_RATE:
                pcm = await self._input_resampler.resample(
                    payload, VI_SAMPLE_RATE, self._params.sample_rate
                )
            return InputAudioRawFrame(
                audio=pcm,
                num_channels=1,
                sample_rate=self._params.sample_rate,
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
            return EndFrame()

        if event == "mark":
            return None

        return None
