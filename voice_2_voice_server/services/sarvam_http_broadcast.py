"""Sarvam HTTP TTS variant for the broadcast/translation path.

The translation room drives TTS by draining ``run_tts`` directly, outside a
pipeline, so it needs a service whose ``run_tts`` *yields* audio frames. Two stock
options don't fit:

* ``SarvamTTSService`` (websocket) pushes audio out-of-band via its receive task
  and only emits an end-of-utterance after a 2s idle timeout — the per-sentence
  drain never sees the audio and has no clean boundary.
* ``SarvamHttpTTSService`` yields audio, but its ``run_tts`` unconditionally reads
  ``_settings["pitch"|"pace"|"loudness"]``, which its ``__init__`` omits for
  ``bulbul:v3`` -> ``KeyError``. v3 also rejects pitch/loudness server-side.

This subclass rebuilds the request payload from whatever tuning keys are actually
present, so it works for ``bulbul:v2`` (pitch/pace/loudness) and ``bulbul:v3``
(temperature) alike. The caller sets ``_settings["language"]`` directly because
pipecat's enum map has no Assamese entry.
"""

import base64
import io
import wave
from typing import AsyncGenerator

from loguru import logger
from pipecat.frames.frames import (
    ErrorFrame,
    Frame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
)
from pipecat.services.sarvam.tts import SarvamHttpTTSService


class SarvamHttpBroadcastTTSService(SarvamHttpTTSService):
    """Sarvam HTTP TTS that tolerates any bulbul model in the broadcast drain."""

    async def run_tts(self, text: str) -> AsyncGenerator[Frame, None]:
        """Synthesise ``text`` in one HTTP call, yielding audio frames."""
        logger.debug(f"{self}: Generating TTS [{text}]")
        try:
            await self.start_ttfb_metrics()

            payload = {
                "text": text,
                "target_language_code": self._settings["language"],
                "speaker": self._voice_id,
                "sample_rate": self.sample_rate,
                "enable_preprocessing": self._settings.get("enable_preprocessing", False),
                "model": self._model_name,
            }
            # Only send tuning keys the model actually accepts: __init__ populates
            # pitch/pace/loudness for v2 and temperature for v3, never both.
            for key in ("pitch", "pace", "loudness", "temperature"):
                value = self._settings.get(key)
                if value is not None:
                    payload[key] = value

            headers = {
                "api-subscription-key": self._api_key,
                "Content-Type": "application/json",
            }
            url = f"{self._base_url}/text-to-speech"

            yield TTSStartedFrame()

            async with self._session.post(url, json=payload, headers=headers) as response:
                if response.status != 200:
                    error_text = await response.text()
                    yield ErrorFrame(error=f"Sarvam API error: {error_text}")
                    return
                response_data = await response.json()

            await self.start_tts_usage_metrics(text)

            audios = response_data.get("audios")
            if not audios:
                yield ErrorFrame(error="No audio data received")
                return

            raw = base64.b64decode(audios[0])
            # Sarvam returns WAV. Parse it properly instead of assuming a fixed
            # 44-byte header and trusting the requested sample_rate: the model
            # does not always honour the requested rate (observed: bulbul:v3
            # returning audio at a higher native rate regardless of a 16kHz
            # request), and a non-canonical WAV can carry extra chunks before
            # `data`. Getting either wrong plays audio at the wrong pitch/speed
            # and throws off every downstream duration/pacing calculation --
            # this is what was making segments take ~1.5x longer than their
            # text should and overrun the per-listener buffer.
            try:
                with wave.open(io.BytesIO(raw), "rb") as wav_file:
                    actual_rate = wav_file.getframerate()
                    audio_data = wav_file.readframes(wav_file.getnframes())
            except wave.Error:
                # Not a well-formed WAV (e.g. raw PCM already) -- fall back to
                # what we asked for rather than drop the segment outright.
                actual_rate = self.sample_rate
                audio_data = raw[44:] if raw.startswith(b"RIFF") else raw

            yield TTSAudioRawFrame(
                audio=audio_data,
                sample_rate=actual_rate,
                num_channels=1,
            )
        except Exception as e:
            yield ErrorFrame(error=f"Error generating TTS: {e}")
        finally:
            await self.stop_ttfb_metrics()
            yield TTSStoppedFrame()
