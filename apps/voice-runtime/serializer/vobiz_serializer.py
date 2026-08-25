import base64
import json
from pipecat.serializers.plivo import PlivoFrameSerializer
from pipecat.frames.frames import (
    AudioRawFrame, 
    InputAudioRawFrame, 
    Frame
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
        # If we are in 16kHz mode, handle L16 (Raw PCM) input
        if self._plivo_sample_rate == 16000:
            try:
                message = json.loads(data)
            except json.JSONDecodeError:
                return None

            if message.get("event") == "media":
                media = message.get("media", {})
                payload_base64 = media.get("payload")
                if not payload_base64:
                    return None
                
                payload = base64.b64decode(payload_base64)
                # In 16kHz L16 mode, the payload IS the raw PCM data (Linear16)
                return InputAudioRawFrame(
                    audio=payload, 
                    num_channels=1, 
                    sample_rate=16000
                )

        # Fall back to base class (handles 8kHz μ-law and other events like DTMF)
        return await super().deserialize(data)