"""Ubona Media Stream WebSocket frame serializer."""

import base64
import json
import time
from typing import Optional

from loguru import logger
from pydantic import BaseModel

from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    OutputTransportMessageFrame,
    OutputTransportMessageUrgentFrame,
    StartFrame,
)
from pipecat.serializers.base_serializer import FrameSerializer, FrameSerializerType


class UbonaFrameSerializer(FrameSerializer):
    """Serializer for Ubona Media Stream WebSocket protocol (PCMU 8kHz)."""

    class InputParams(BaseModel):
        sample_rate: int = 8000

    def __init__(
        self,
        stream_id: str,
        call_id: str,
        params: Optional[InputParams] = None,
    ):
        self._stream_id = stream_id
        self._call_id = call_id
        self._params = params or self.InputParams()
        self._sample_rate = self._params.sample_rate
        self._seq_num = 0

    @property
    def type(self) -> FrameSerializerType:
        return FrameSerializerType.TEXT

    def _next_seq(self) -> int:
        self._seq_num += 1
        return self._seq_num

    def _ts(self) -> int:
        return int(time.time() * 1000)

    async def setup(self, frame: StartFrame):
        self._sample_rate = self._params.sample_rate or frame.audio_in_sample_rate

    async def serialize(self, frame: Frame) -> str | bytes | None:
        if isinstance(frame, InterruptionFrame):
            return json.dumps({
                "event": "clear",
                "seqNum": self._next_seq(),
                "streamId": self._stream_id,
            })

        if isinstance(frame, AudioRawFrame):
            # Convert PCM to μ-law if needed
            from pipecat.audio.utils import create_stream_resampler, pcm_to_ulaw
            
            if not hasattr(self, '_output_resampler'):
                self._output_resampler = create_stream_resampler()
            
            audio_data = await pcm_to_ulaw(
                frame.audio, frame.sample_rate, self._sample_rate, self._output_resampler
            )
            if not audio_data:
                return None

            return json.dumps({
                "event": "media",
                "seqNum": self._next_seq(),
                "streamId": self._stream_id,
                "media": {
                    "ts": self._ts(),
                    "payload": base64.b64encode(audio_data).decode("utf-8"),
                },
            })

        if isinstance(frame, (OutputTransportMessageFrame, OutputTransportMessageUrgentFrame)):
            return json.dumps(frame.message)

        return None

    async def deserialize(self, data: str | bytes) -> Frame | None:
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return None

        event = msg.get("event")

        if event == "media":
            payload = msg.get("media", {}).get("payload")
            if not payload:
                return None

            from pipecat.audio.utils import create_stream_resampler, ulaw_to_pcm
            
            if not hasattr(self, '_input_resampler'):
                self._input_resampler = create_stream_resampler()

            audio = await ulaw_to_pcm(
                base64.b64decode(payload),
                self._sample_rate,
                self._sample_rate,
                self._input_resampler,
            )
            if not audio:
                return None

            return InputAudioRawFrame(
                audio=audio, num_channels=1, sample_rate=self._sample_rate
            )

        if event == "dtmf":
            digit = msg.get("dtmf", {}).get("digit")
            if digit:
                try:
                    return InputDTMFFrame(KeypadEntry(digit))
                except ValueError:
                    logger.warning(f"Invalid DTMF: {digit}")

        if event == "ping":
            # Spec: pong must contain the same ts as ping for round-trip
            ping_ts = msg.get("ts", self._ts())
            self._pending_pong = {"event": "pong", "ts": ping_ts}

        return None

    def get_pending_pong(self) -> Optional[str]:
        """Get and clear any pending pong response."""
        if hasattr(self, '_pending_pong'):
            pong = json.dumps(self._pending_pong)
            del self._pending_pong
            return pong
        return None