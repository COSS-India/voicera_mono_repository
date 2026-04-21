"""Bhashini TTS service backed by NVIDIA Triton Inference Server over gRPC streaming.

Required environment variables (set in .env):
  BHASHINI_TRITON_URL          gRPC endpoint of the Triton server
                                e.g. grpc.nvcf.nvidia.com:443
  BHASHINI_TRITON_API_KEY      Bearer token / NVCF API key for authentication
  BHASHINI_TRITON_FUNCTION_ID  NVCF function-id header value

Optional environment variables:
  BHASHINI_TRITON_FUNCTION_VERSION_ID  NVCF function-version-id (if required)
  BHASHINI_TRITON_MODEL_NAME           Triton model name (default: indicparler_tts)
  BHASHINI_TRITON_USE_PLAINTEXT        Set to "true" to disable TLS (default: TLS on)
  BHASHINI_TRITON_TIMEOUT_S            Per-request timeout in seconds (default: 120)
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator

import numpy as np
import tritonclient.grpc as grpcclient
from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.tts_service import TTSService


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _require_env(name: str) -> str:
    """Return env var value or raise a clear error at service startup."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ValueError(
            f"BhashiniTTSService: required environment variable '{name}' is not set. "
            "Add it to your .env file."
        )
    return value


def _optional_env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _make_triton_inputs(prompt: str, description: str) -> list:
    """Build the two Triton InferInput tensors expected by indicparler_tts."""
    prompt_input = grpcclient.InferInput("PROMPT", [1], "BYTES")
    desc_input = grpcclient.InferInput("DESCRIPTION", [1], "BYTES")
    prompt_input.set_data_from_numpy(np.array([prompt], dtype=object))
    desc_input.set_data_from_numpy(np.array([description], dtype=object))
    return [prompt_input, desc_input]


def _decode(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _to_pcm16_bytes(audio_chunk: np.ndarray) -> bytes:
    """Convert Triton output audio tensor to mono PCM16 bytes."""
    # Handle common model output dtype (float waveform in [-1, 1]).
    if np.issubdtype(audio_chunk.dtype, np.floating):
        pcm = (np.clip(audio_chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        return pcm.tobytes()

    # Handle integer tensors directly while preventing wraparound.
    if audio_chunk.dtype == np.int16:
        return audio_chunk.tobytes()

    if np.issubdtype(audio_chunk.dtype, np.integer):
        pcm = np.clip(audio_chunk, np.iinfo(np.int16).min, np.iinfo(np.int16).max).astype(np.int16)
        return pcm.tobytes()

    # Fallback for unexpected dtypes.
    pcm = np.clip(audio_chunk.astype(np.float32), -1.0, 1.0)
    return (pcm * 32767.0).astype(np.int16).tobytes()


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------

class BhashiniTTSService(TTSService):
    """Text-to-speech via NVIDIA Triton gRPC streaming (indicparler_tts model).

    All connection parameters are read from environment variables so that no
    endpoint or credential is hard-coded in source.  See module docstring for
    the full list of variables.
    """

    def __init__(
        self,
        *,
        speaker: str = "Divya",
        description: str = "A clear, natural voice with good audio quality.",
        sample_rate: int = 44100,
        **kwargs,
    ):
        super().__init__(sample_rate=sample_rate, **kwargs)

        # --- Connection config from .env ---
        logger.info("TTS [INIT] Step 1/3 — Reading required environment variables")
        try:
            self._triton_url = _require_env("BHASHINI_TRITON_URL")
            logger.info("TTS [INIT] ✅ BHASHINI_TRITON_URL = {}", self._triton_url)
        except ValueError as e:
            logger.error("TTS [INIT] ❌ {}", e)
            raise

        try:
            self._api_key = _require_env("BHASHINI_TRITON_API_KEY")
            logger.info("TTS [INIT] ✅ BHASHINI_TRITON_API_KEY is set (length={})", len(self._api_key))
        except ValueError as e:
            logger.error("TTS [INIT] ❌ {}", e)
            raise

        try:
            self._function_id = _require_env("BHASHINI_TRITON_FUNCTION_ID")
            logger.info("TTS [INIT] ✅ BHASHINI_TRITON_FUNCTION_ID = {}", self._function_id)
        except ValueError as e:
            logger.error("TTS [INIT] ❌ {}", e)
            raise

        self._function_version_id = _optional_env("BHASHINI_TRITON_FUNCTION_VERSION_ID")
        self._model_name = _optional_env("BHASHINI_TRITON_MODEL_NAME", "indicparler_tts")
        self._use_tls = _optional_env("BHASHINI_TRITON_USE_PLAINTEXT", "false").lower() != "true"
        self._timeout_s = float(_optional_env("BHASHINI_TRITON_TIMEOUT_S", "120"))

        logger.info(
            "TTS [INIT] Step 2/3 — Optional config | model={} tls={} timeout={}s function_version_id='{}'",
            self._model_name, self._use_tls, self._timeout_s,
            self._function_version_id or "(not set)",
        )

        # --- Voice config ---
        self._speaker = speaker
        self._description = description

        logger.info(
            "TTS [INIT] Step 3/3 — Voice config | speaker='{}' sample_rate={}Hz description='{}'",
            self._speaker, sample_rate, self._description[:60] + "..." if len(self._description) > 60 else self._description,
        )
        logger.info(
            "TTS [INIT] ✅ BhashiniTTSService fully initialised | url={} model={} tls={} timeout={}s",
            self._triton_url, self._model_name, self._use_tls, self._timeout_s,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_headers(self) -> dict:
        headers = {
            "authorization": f"Bearer {self._api_key}",
            "function-id": self._function_id,
        }
        if self._function_version_id:
            headers["function-version-id"] = self._function_version_id
        return headers

    def _full_description(self) -> str:
        """Build the description string passed to the model."""
        if self._speaker:
            return f"{self._speaker}. {self._description}"
        return self._description

    # ------------------------------------------------------------------
    # TTSService implementation
    # ------------------------------------------------------------------

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        logger.info("TTS [RUN] ▶ run_tts called | text_len={} text='{}'",
                    len(text), text[:80] + "..." if len(text) > 80 else text)

        if not text.strip():
            logger.warning("TTS [RUN] Empty text received — skipping TTS call")
            return

        loop = asyncio.get_event_loop()
        result_queue: asyncio.Queue = asyncio.Queue()

        def _on_result(result, error):
            loop.call_soon_threadsafe(result_queue.put_nowait, (result, error))

        client: grpcclient.InferenceServerClient | None = None
        try:
            # Step 1: Create gRPC client
            logger.info(
                "TTS [RUN] Step 1/5 — Creating Triton gRPC client | url={} tls={}",
                self._triton_url, self._use_tls,
            )
            client = grpcclient.InferenceServerClient(
                url=self._triton_url,
                ssl=self._use_tls,
                verbose=False,
            )
            logger.info("TTS [RUN] ✅ Step 1/5 — gRPC client created")

            # Step 2: Start streaming with auth headers
            headers = self._build_headers()
            logger.info(
                "TTS [RUN] Step 2/5 — Starting gRPC stream | headers keys={}",
                list(headers.keys()),
            )
            client.start_stream(callback=_on_result, headers=headers)
            logger.info("TTS [RUN] ✅ Step 2/5 — gRPC stream started")

            # Step 3: Build inputs and send inference request
            full_desc = self._full_description()
            logger.info(
                "TTS [RUN] Step 3/5 — Sending inference request | model={} description='{}'",
                self._model_name,
                full_desc[:80] + "..." if len(full_desc) > 80 else full_desc,
            )
            inputs = _make_triton_inputs(text, full_desc)
            request_id = f"bhashini-tts-{id(text)}"
            client.async_stream_infer(
                model_name=self._model_name,
                inputs=inputs,
                request_id=request_id,
            )
            logger.info("TTS [RUN] ✅ Step 3/5 — Inference request sent | request_id={}", request_id)

            yield TTSStartedFrame()

            # Step 4: Stream audio chunks from result queue
            logger.info("TTS [RUN] Step 4/5 — Waiting for audio chunks (timeout={}s)", self._timeout_s)
            chunk_count = 0
            deadline = loop.time() + self._timeout_s

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    logger.error(
                        "TTS [RUN] ❌ Step 4/5 FAILED — Overall timeout of {}s exceeded after {} chunks",
                        self._timeout_s, chunk_count,
                    )
                    yield ErrorFrame("Bhashini TTS request timed out")
                    return

                try:
                    result, error = await asyncio.wait_for(
                        result_queue.get(), timeout=min(remaining, 10.0)
                    )
                except asyncio.TimeoutError:
                    logger.error(
                        "TTS [RUN] ❌ Step 4/5 FAILED — Timed out waiting for next chunk after {}s "
                        "(received {} chunks so far). Possible causes:\n"
                        "  → 1. BHASHINI_TRITON_API_KEY is invalid or expired\n"
                        "  → 2. BHASHINI_TRITON_FUNCTION_ID is wrong\n"
                        "  → 3. Network issue reaching {}\n"
                        "  → 4. Model '{}' is not deployed at that endpoint",
                        self._timeout_s, chunk_count, self._triton_url, self._model_name,
                    )
                    yield ErrorFrame("Bhashini TTS request timed out")
                    return

                if error is not None:
                    logger.error(
                        "TTS [RUN] ❌ Step 4/5 FAILED — Triton gRPC returned error after {} chunks: {}\n"
                        "  → Check BHASHINI_TRITON_API_KEY, BHASHINI_TRITON_FUNCTION_ID, and endpoint URL",
                        chunk_count, error,
                    )
                    yield ErrorFrame(f"Triton gRPC error: {error}")
                    return

                # Parse response fields
                try:
                    status = _decode(result.as_numpy("STATUS")[0])
                    is_final = bool(result.as_numpy("IS_FINAL")[0])
                    audio_chunk: np.ndarray = result.as_numpy("AUDIO_CHUNK")
                except Exception as parse_err:
                    logger.error(
                        "TTS [RUN] ❌ Step 4/5 — Failed to parse Triton response fields: {}", parse_err
                    )
                    yield ErrorFrame(f"Triton response parse error: {parse_err}")
                    return

                logger.debug(
                    "TTS [RUN] Chunk received | status='{}' is_final={} audio_size={}",
                    status, is_final,
                    audio_chunk.size if audio_chunk is not None else "None",
                )

                if status == "audio" and audio_chunk is not None and audio_chunk.size > 0:
                    pcm_bytes = _to_pcm16_bytes(audio_chunk)
                    chunk_count += 1
                    logger.info(
                        "TTS [RUN] 🔊 Audio chunk #{} | {} bytes | sample_rate={}Hz dtype={}",
                        chunk_count, len(pcm_bytes), self.sample_rate, str(audio_chunk.dtype),
                    )
                    yield TTSAudioRawFrame(
                        audio=pcm_bytes,
                        sample_rate=self.sample_rate,
                        num_channels=1,
                    )
                elif status != "audio":
                    logger.debug("TTS [RUN] Non-audio status frame: '{}'", status)

                if is_final:
                    if chunk_count == 0:
                        logger.warning(
                            "TTS [RUN] ⚠️ is_final=True but zero audio chunks received — "
                            "model may have rejected the input or returned an empty response"
                        )
                    else:
                        logger.info(
                            "TTS [RUN] ✅ Step 4/5 — Stream complete | total_chunks={} text_len={}",
                            chunk_count, len(text),
                        )
                    break

            # Step 5: Done
            logger.info("TTS [RUN] Step 5/5 — Yielding TTSStoppedFrame")
            yield TTSStoppedFrame()
            logger.info("TTS [RUN] ✅ run_tts finished successfully")

        except Exception as e:
            logger.error(
                "TTS [RUN] ❌ Unexpected exception in run_tts: {} — {}\n"
                "  → If this is a ValueError about env vars, check BHASHINI_TRITON_URL / "
                "BHASHINI_TRITON_API_KEY / BHASHINI_TRITON_FUNCTION_ID in your .env",
                type(e).__name__, e,
            )
            yield ErrorFrame(f"Bhashini TTS error: {e}")
        finally:
            if client is not None:
                try:
                    client.stop_stream()
                    logger.debug("TTS [RUN] gRPC stream stopped in finally block")
                except Exception as stop_err:
                    logger.warning("TTS [RUN] Error stopping gRPC stream: {}", stop_err)