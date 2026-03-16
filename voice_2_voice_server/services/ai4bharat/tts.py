import asyncio
import os
from typing import AsyncGenerator, Optional

import grpc
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from . import tts_pb2, tts_pb2_grpc


class IndicParlerRESTTTSService(TTSService):
    """AI4Bharat TTS backed by the NVCF gRPC streaming endpoint."""

    DEFAULT_GRPC_TARGET = "grpc.nvcf.nvidia.com:443"
    DEFAULT_FUNCTION_ID = "a92982cc-6608-461f-8d10-dacefdd98516"
    DEFAULT_AUTH_TOKEN = "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"

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 24000,
        play_steps_in_s: float = 0.15,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        grpc_target = (
            os.getenv("INDIC_TTS_GRPC_TARGET")
            or os.getenv("INDIC_TTS_SERVER_URL")
            or self.DEFAULT_GRPC_TARGET
        ).strip()
        auth_token = (
            os.getenv("INDIC_TTS_AUTH_TOKEN")
            or os.getenv("NVCF_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or self.DEFAULT_AUTH_TOKEN
        )
        function_id = (
            os.getenv("INDIC_TTS_FUNCTION_ID")
            or self.DEFAULT_FUNCTION_ID
        )
        function_version_id = os.getenv("INDIC_TTS_FUNCTION_VERSION_ID")

        self._grpc_target = grpc_target
        self._auth_token = auth_token
        self._function_id = function_id
        self._function_version_id = function_version_id
        self._speaker = speaker
        self._description = description
        self._play_steps_in_s = play_steps_in_s
        self._request_timeout = float(os.getenv("INDIC_TTS_GRPC_TIMEOUT_SECONDS", "120"))

        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tts_pb2_grpc.TTSServiceStub] = None

    async def start(self, frame: Frame):
        logger.info(f"Starting IndicParler gRPC TTS service: {self._grpc_target}")
        self._channel = grpc.aio.secure_channel(
            self._grpc_target,
            grpc.ssl_channel_credentials(),
        )
        self._stub = tts_pb2_grpc.TTSServiceStub(self._channel)
        await super().start(frame)

    async def stop(self, frame: Frame):
        logger.info("Stopping IndicParler gRPC TTS service")
        if self._channel is not None:
            await self._channel.close()
            self._channel = None
            self._stub = None
        await super().stop(frame)

    def _build_metadata(self) -> list[tuple[str, str]]:
        metadata = [
            ("authorization", f"Bearer {self._auth_token}"),
            ("function-id", self._function_id),
        ]
        if self._function_version_id:
            metadata.append(("function-version-id", self._function_version_id))
        return metadata

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        if self._stub is None:
            logger.warning("IndicParler gRPC stub not initialized, creating temporary channel")
            temp_channel = grpc.aio.secure_channel(
                self._grpc_target,
                grpc.ssl_channel_credentials(),
            )
            stub = tts_pb2_grpc.TTSServiceStub(temp_channel)
            should_close = True
        else:
            temp_channel = None
            stub = self._stub
            should_close = False

        request = tts_pb2.TTSRequest(
            text=text,
            speaker=self._speaker,
            description=self._description,
            play_steps_in_s=self._play_steps_in_s,
        )

        try:
            yield TTSStartedFrame()

            chunk_count = 0
            async for chunk in stub.StreamTTS(
                request,
                metadata=self._build_metadata(),
                timeout=self._request_timeout,
            ):
                if not chunk.audio:
                    continue

                yield TTSAudioRawFrame(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate or self.sample_rate,
                    num_channels=1,
                )
                chunk_count += 1

            logger.info(f"IndicParler gRPC TTS streamed {chunk_count} chunks")
            yield TTSStoppedFrame()

        except grpc.aio.AioRpcError as e:
            logger.error(f"IndicParler gRPC error: {e.code()} {e.details()}")
            yield ErrorFrame(f"gRPC TTS error: {e.details() or e.code().name}")
        except asyncio.TimeoutError:
            yield ErrorFrame("TTS gRPC request timeout")
        except Exception as e:
            logger.error(f"IndicParler gRPC TTS error: {e}")
            yield ErrorFrame(f"TTS error: {e}")
        finally:
            if should_close and temp_channel is not None:
                await temp_channel.close()
