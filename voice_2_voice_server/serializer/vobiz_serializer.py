import base64
import json
from loguru import logger
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame,
    InputAudioRawFrame,
    Frame,
)

class VobizFrameSerializer(PlivoFrameSerializer):
    """
    Vobiz is Plivo-compatible, but we override it to support 16kHz L16 (Raw PCM)
    as required by the Vobiz spec (μ-law is 8kHz only).
    """
    
    class InputParams(PlivoFrameSerializer.InputParams):
        def __init__(
            self,
            vobiz_sample_rate: int = 8000,
            sample_rate: int = None,
            auto_hang_up: bool = True
        ):
            super().__init__(
                plivo_sample_rate=vobiz_sample_rate,
                sample_rate=sample_rate,
                auto_hang_up=auto_hang_up
            )
    
    def __init__(
        self,
        stream_sid: str,
        call_sid: str,
        params: InputParams = None
    ):
        super().__init__(
            stream_id=stream_sid,
            call_id=call_sid,
            params=params or self.InputParams()
        )

    async def serialize(self, frame: Frame) -> str | bytes | None:
        # If we are in 16kHz mode, use L16 (Raw PCM) instead of μ-law
        if self._plivo_sample_rate == 16000 and isinstance(frame, AudioRawFrame):
            data = frame.audio
            # Resample to 16kHz if the frame is at a different rate
            if frame.sample_rate != 16000:
                data = await self._output_resampler.resample(data, frame.sample_rate, 16000)
            
            payload = base64.b64encode(data).decode("utf-8")
            answer = {
                "event": "playAudio",
                "media": {
                    "contentType": "audio/x-l16",
                    "sampleRate": 16000,
                    "payload": payload,
                },
                "streamId": self._stream_id,
            }
            return json.dumps(answer)
        
        # Fall back to base class (which handles 8kHz μ-law and other frames)
        return await super().serialize(frame)

    async def deserialize(self, data: str | bytes) -> Frame | None:
        event_name = None
        try:
            msg = json.loads(data) if isinstance(data, str) else json.loads(data.decode("utf-8"))
            event_name = msg.get("event", "<no event>")
            if event_name == "media":
                payload_len = len(msg.get("media", {}).get("payload") or "")
                logger.debug(f"🎤 Inbound message: event={event_name}, payload_b64_len={payload_len}")
            else:
                logger.debug(f"🎤 Inbound message: event={event_name}, keys={list(msg.keys())}")
        except (json.JSONDecodeError, TypeError):
            logger.warning("🎤 Inbound message: not JSON, len={}", len(data) if data else 0)

        # If we are in 16kHz mode, handle L16 (Raw PCM) input
        if self._plivo_sample_rate == 16000:
            try:
                message = json.loads(data) if isinstance(data, str) else json.loads(data.decode("utf-8"))
            except json.JSONDecodeError:
                logger.warning("🎤 Vobiz deserialize: invalid JSON in 16kHz mode")
                return None

            ev = message.get("event")
            if ev in ("media", "stream", "inbound"):
                media = message.get("media") or message
                payload_base64 = (media.get("payload") if isinstance(media, dict) else None) or message.get("payload")
                if not payload_base64:
                    logger.warning("🎤 Vobiz deserialize: %s event missing payload", ev)
                    return None

                payload = base64.b64decode(payload_base64)
                logger.info(
                    "🎤 Vobiz deserialize: parsed inbound media (16kHz L16), event=%s, %d bytes",
                    ev,
                    len(payload),
                )
                return InputAudioRawFrame(
                    audio=payload,
                    num_channels=1,
                    sample_rate=16000,
                )

            logger.debug("🎤 Vobiz deserialize: 16kHz mode, delegating event '%s' to Plivo", message.get("event"))

        # Fall back to base class (handles 8kHz μ-law and other events like DTMF)
        # If provider sends "stream" or "inbound" with same payload shape, normalize to "media" for Plivo
        deserialize_data = data
        try:
            msg = json.loads(data) if isinstance(data, str) else json.loads(data.decode("utf-8"))
            if msg.get("event") in ("stream", "inbound"):
                payload = (msg.get("media") or {}).get("payload") or msg.get("payload")
                if payload:
                    normalized = {"event": "media", "media": {"payload": payload}}
                    deserialize_data = json.dumps(normalized)
                    logger.debug("🎤 Vobiz deserialize: normalized event '%s' to 'media' for Plivo", msg.get("event"))
        except (json.JSONDecodeError, TypeError):
            pass

        result = await super().deserialize(deserialize_data)
        if event_name in ("media", "stream", "inbound") and result is not None:
            logger.info(
                "🎤 Vobiz deserialize: parsed inbound media (Plivo/8kHz), event=%s, frame type=%s",
                event_name,
                type(result).__name__,
            )
        return result