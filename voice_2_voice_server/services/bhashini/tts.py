import asyncio
import os
import time
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


class BhashiniTTSService(TTSService):
    """Bhashini TTS backed by the NVCF gRPC streaming endpoint."""

    DEFAULT_GRPC_TARGET = "grpc.nvcf.nvidia.com:443"
    DEFAULT_FUNCTION_ID = "a92982cc-6608-461f-8d10-dacefdd98516"

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        grpc_target = (
            os.getenv("BHASHINI_TTS_GRPC_TARGET")
            or self.DEFAULT_GRPC_TARGET
        ).strip()
        auth_token = (
            os.getenv("BHASHINI_NVCF_TTS_AUTH_TOKEN")
        )
        function_id = (
            os.getenv("BHASHINI_TTS_FUNCTION_ID")
            or self.DEFAULT_FUNCTION_ID
        )
        function_version_id = os.getenv("BHASHINI_TTS_FUNCTION_VERSION_ID")

        if not auth_token:
            raise ValueError(
                "Bhashini TTS auth token not set. Configure BHASHINI_NVCF_TTS_AUTH_TOKEN."
            )

        self._grpc_target = grpc_target
        self._auth_token = auth_token
        self._function_id = function_id
        self._function_version_id = function_version_id
        self._speaker = speaker
        self._description = description
        self._request_timeout = float(os.getenv("BHASHINI_TTS_TIMEOUT_SECONDS", "120"))
        self._play_steps_in_s = float(os.getenv("BHASHINI_TTS_PLAY_STEPS_IN_S", "0.5"))

        self._channel: Optional[grpc.aio.Channel] = None
        self._stub: Optional[tts_pb2_grpc.TTSServiceStub] = None

    async def start(self, frame: Frame):
        logger.info(
            "Starting Bhashini gRPC TTS service | target={} | function_id={} | version_id={} | sample_rate={}",
            self._grpc_target,
            self._function_id,
            self._function_version_id or "default",
            self.sample_rate,
        )
        self._channel = grpc.aio.secure_channel(
            self._grpc_target,
            grpc.ssl_channel_credentials(),
        )
        self._stub = tts_pb2_grpc.TTSServiceStub(self._channel)
        await super().start(frame)

    async def stop(self, frame: Frame):
        logger.info("Stopping Bhashini gRPC TTS service")
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

    def _masked_token(self) -> str:
        if not self._auth_token:
            return "missing"
        if len(self._auth_token) <= 10:
            return "***"
        return f"{self._auth_token[:8]}...{self._auth_token[-4:]}"

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            logger.debug("🔊 Bhashini TTS: run_tts skipped - empty text")
            return

        if self._stub is None:
            logger.warning("Bhashini gRPC stub not initialized, creating temporary channel")
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
            started_at = time.perf_counter()
            logger.info(
                "Bhashini gRPC TTS request started | chars={} | speaker={} | target={} | function_id={} | token={}",
                len(text),
                self._speaker,
                self._grpc_target,
                self._function_id,
                self._masked_token(),
            )
            yield TTSStartedFrame()

            chunk_count = 0
            total_bytes = 0
            first_chunk_ms = None
            async for chunk in stub.StreamTTS(
                request,
                metadata=self._build_metadata(),
                timeout=self._request_timeout,
            ):
                if not chunk.audio:
                    logger.debug("Bhashini gRPC returned empty audio chunk")
                    continue

                if first_chunk_ms is None:
                    first_chunk_ms = (time.perf_counter() - started_at) * 1000
                    logger.info(
                        "Bhashini gRPC first audio chunk received in {:.1f} ms | sample_rate={}",
                        first_chunk_ms,
                        chunk.sample_rate or self.sample_rate,
                    )

                yield TTSAudioRawFrame(
                    audio=chunk.audio,
                    sample_rate=chunk.sample_rate or self.sample_rate,
                    num_channels=1,
                )
                chunk_count += 1
                total_bytes += len(chunk.audio)

                if chunk_count <= 3 or chunk_count % 25 == 0:
                    logger.debug(
                        "Bhashini gRPC audio chunk {} | bytes={} | sample_rate={}",
                        chunk_count,
                        len(chunk.audio),
                        chunk.sample_rate or self.sample_rate,
                    )

            elapsed_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "Bhashini gRPC TTS completed | chunks={} | bytes={} | ttft_ms={} | total_ms={:.1f}",
                chunk_count,
                total_bytes,
                f"{first_chunk_ms:.1f}" if first_chunk_ms is not None else "n/a",
                elapsed_ms,
            )
            yield TTSStoppedFrame()

        except grpc.aio.AioRpcError as e:
            logger.error(f"Bhashini gRPC error: {e.code()} {e.details()}")
            yield ErrorFrame(f"gRPC TTS error: {e.details() or e.code().name}")
        except asyncio.TimeoutError:
            yield ErrorFrame("Bhashini TTS request timeout")
        except Exception as e:
            logger.error(f"Bhashini TTS error: {e}")
            yield ErrorFrame(f"Bhashini TTS error: {e}")
        finally:
            if should_close and temp_channel is not None:
                await temp_channel.close()
