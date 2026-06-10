import asyncio
import os
from typing import AsyncGenerator

import numpy as np
import grpc

import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import tts_pb2
import tts_pb2_grpc

from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService

from loguru import logger


class BhashiniTTSService(TTSService):

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        play_steps_in_s: float = 0.5,
        language: str = "hi",
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        self._function_id = os.getenv("BHASHINI_TTS_FUNCTION_ID")
        if not self._function_id:
            raise ValueError("BHASHINI_TTS_FUNCTION_ID environment variable not set")

        self._auth_token = os.getenv("BHASHINI_TTS_AUTH_TOKEN")
        if not self._auth_token:
            raise ValueError("BHASHINI_TTS_AUTH_TOKEN environment variable not set")

        self._speaker = speaker
        self._description = description
        self._play_steps_in_s = play_steps_in_s
        self._language = language

    def _full_description(self) -> str:
        if self._speaker:
            return f"{self._speaker} {self._description}"
        return self._description

    def can_generate_metrics(self) -> bool:
        return True

    @staticmethod
    def _to_pcm16_bytes(audio_chunk: np.ndarray) -> bytes:
        if np.issubdtype(audio_chunk.dtype, np.floating):
            return (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()
        if audio_chunk.dtype == np.int16:
            return audio_chunk.tobytes()
        if np.issubdtype(audio_chunk.dtype, np.integer):
            return np.clip(audio_chunk, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16).tobytes()
        return (np.clip(audio_chunk.astype(np.float32), -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        if not text.strip():
            return

        credentials = grpc.ssl_channel_credentials()
        metadata = [
            ("authorization", f"Bearer {self._auth_token}"),
            ("function-id", self._function_id),
        ]

        await self.start_ttfb_metrics()
        yield TTSStartedFrame()
        first_audio = True
        sample_rate = self.sample_rate

        try:
            async with grpc.aio.secure_channel("grpc.nvcf.nvidia.com:443", credentials) as channel:
                stub = tts_pb2_grpc.TTSServiceStub(channel)
                request = tts_pb2.SynthesizeRequest(
                    prompt=text,
                    description=self._full_description(),
                    language=self._language,
                )

                async for response in stub.Synthesize(request, metadata=metadata):
                    which = response.WhichOneof("payload")

                    if which == "meta":
                        sample_rate = response.meta.sample_rate

                    elif which == "audio":
                        if first_audio:
                            first_audio = False
                            await self.stop_ttfb_metrics()
                        arr = np.frombuffer(response.audio.pcm_data, dtype=np.float32)
                        audio_bytes = self._to_pcm16_bytes(arr)
                        logger.info(f"Audio chunk sent to Telephony: {len(audio_bytes)} bytes")
                        yield TTSAudioRawFrame(
                            audio=audio_bytes,
                            sample_rate=sample_rate,
                            num_channels=1,
                        )

                    elif which == "done":
                        break

        except grpc.aio.AioRpcError as e:
            yield ErrorFrame(f"gRPC error [{e.code()}]: {e.details()}")
        except asyncio.TimeoutError:
            yield ErrorFrame("Request timeout")
        except Exception as e:
            yield ErrorFrame(f"TTS error: {e}")

        yield TTSStoppedFrame()