"""Telnyx frame serializer wrapper.

Thin wrapper around pipecat's upstream TelnyxFrameSerializer so the rest of
the codebase has a single, consistent import path alongside
VobizFrameSerializer and UbonaFrameSerializer. Also lets us override or
patch Telnyx-specific behavior later without touching pipecat internals.
"""

from typing import Optional

from pipecat.serializers.telnyx import TelnyxFrameSerializer as _PipecatTelnyxSerializer


class TelnyxFrameSerializer(_PipecatTelnyxSerializer):
    """Telnyx WebSocket media stream serializer.

    Telnyx default codec is PCMU (mu-law) at 8kHz. Telnyx also supports L16
    at 16kHz when the TeXML Stream element requests audio/x-l16;rate=16000.
    We pass the encoding explicitly so the serializer can negotiate cleanly.
    """

    InputParams = _PipecatTelnyxSerializer.InputParams

    def __init__(
        self,
        stream_id: str,
        call_control_id: Optional[str] = None,
        outbound_encoding: str = "PCMU",
        inbound_encoding: str = "PCMU",
        api_key: Optional[str] = None,
        params: Optional["TelnyxFrameSerializer.InputParams"] = None,
    ):
        super().__init__(
            stream_id=stream_id,
            outbound_encoding=outbound_encoding,
            inbound_encoding=inbound_encoding,
            call_control_id=call_control_id,
            api_key=api_key,
            params=params or self.InputParams(),
        )
