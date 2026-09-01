import pytest
import base64
import json
from unittest.mock import AsyncMock, patch, MagicMock
from serializer.vobiz_serializer import VobizFrameSerializer
from pipecat.frames.frames import AudioRawFrame, InputAudioRawFrame, Frame

class TestVobizFrameSerializer:
    def test_initialization(self):
        serializer = VobizFrameSerializer(
            stream_sid="stream_123",
            call_sid="call_456"
        )
        assert serializer._stream_id == "stream_123"
        assert serializer._call_id == "call_456"
        assert serializer._plivo_sample_rate == 8000  # Default

    def test_custom_params_initialization(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(
            stream_sid="stream_123",
            call_sid="call_456",
            params=params
        )
        assert serializer._plivo_sample_rate == 16000

    @pytest.mark.asyncio
    async def test_serialize_16khz_matching_rate(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123", params=params)
        
        raw_audio = b"dummy_audio_bytes"
        frame = AudioRawFrame(audio=raw_audio, sample_rate=16000, num_channels=1)
        
        result = await serializer.serialize(frame)
        result_dict = json.loads(result)
        
        assert result_dict["event"] == "playAudio"
        assert result_dict["streamId"] == "str_123"
        assert result_dict["media"]["contentType"] == "audio/x-l16"
        assert result_dict["media"]["sampleRate"] == 16000
        assert result_dict["media"]["payload"] == base64.b64encode(raw_audio).decode("utf-8")

    @pytest.mark.asyncio
    async def test_serialize_16khz_different_rate(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123", params=params)
        
        serializer._output_resampler = AsyncMock()
        resampled_audio = b"resampled_bytes"
        serializer._output_resampler.resample.return_value = resampled_audio
        
        raw_audio = b"dummy_audio_bytes"
        frame = AudioRawFrame(audio=raw_audio, sample_rate=8000, num_channels=1)
        
        result = await serializer.serialize(frame)
        result_dict = json.loads(result)
        
        serializer._output_resampler.resample.assert_called_once_with(raw_audio, 8000, 16000)
        assert result_dict["media"]["payload"] == base64.b64encode(resampled_audio).decode("utf-8")

    @pytest.mark.asyncio
    @patch("pipecat.serializers.plivo.PlivoFrameSerializer.serialize")
    async def test_serialize_fallback(self, mock_super_serialize):
        mock_super_serialize.return_value = "fallback_result"
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123")
        
        # 8000Hz rate so it should fall back to super().serialize
        frame = AudioRawFrame(audio=b"dummy", sample_rate=8000, num_channels=1)
        result = await serializer.serialize(frame)
        
        assert result == "fallback_result"
        mock_super_serialize.assert_called_once_with(frame)

    @pytest.mark.asyncio
    async def test_deserialize_16khz_valid_media(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123", params=params)
        
        raw_audio = b"dummy_audio"
        encoded_audio = base64.b64encode(raw_audio).decode("utf-8")
        msg = {
            "event": "media",
            "media": {
                "payload": encoded_audio
            }
        }
        
        result = await serializer.deserialize(json.dumps(msg))
        assert isinstance(result, InputAudioRawFrame)
        assert result.audio == raw_audio
        assert result.sample_rate == 16000
        assert result.num_channels == 1

    @pytest.mark.asyncio
    async def test_deserialize_16khz_invalid_json(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123", params=params)
        
        result = await serializer.deserialize("not_json")
        assert result is None

    @pytest.mark.asyncio
    async def test_deserialize_16khz_missing_payload(self):
        params = VobizFrameSerializer.InputParams(vobiz_sample_rate=16000)
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123", params=params)
        
        msg = {
            "event": "media",
            "media": {}
        }
        result = await serializer.deserialize(json.dumps(msg))
        assert result is None

    @pytest.mark.asyncio
    @patch("pipecat.serializers.plivo.PlivoFrameSerializer.deserialize")
    async def test_deserialize_fallback(self, mock_super_deserialize):
        mock_super_deserialize.return_value = "fallback_frame"
        serializer = VobizFrameSerializer(stream_sid="str_123", call_sid="call_123")
        
        msg = {"event": "dtmf"}
        result = await serializer.deserialize(json.dumps(msg))
        assert result == "fallback_frame"
        mock_super_deserialize.assert_called_once()
