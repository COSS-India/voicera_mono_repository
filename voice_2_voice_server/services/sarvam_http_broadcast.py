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
from typing import AsyncGenerator, Optional

import numpy as np
import soxr
from loguru import logger
from pydub import AudioSegment
from pydub.effects import speedup as _pydub_speedup
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

    def __init__(self, *, model: str = "bulbul:v2", params=None, **kwargs):
        params = params or SarvamHttpTTSService.InputParams()
        # bulbul:v3/v3-beta accept no pace parameter at all server-side --
        # pipecat's own __init__ (below, via super()) only stores
        # pitch/pace/loudness in self._settings for non-v3 models, so a v3
        # request silently loses whatever pace was asked for: the dashboard
        # shows the saved value, but the broadcast plays at 1.0x regardless.
        # Capture the intended pace here, before that happens, so run_tts can
        # approximate the effect locally by resampling the returned audio.
        self._client_side_pace: Optional[float] = None
        if model in ("bulbul:v3-beta", "bulbul:v3"):
            requested_pace = params.pace
            if requested_pace is not None and abs(requested_pace - 1.0) > 1e-3:
                self._client_side_pace = requested_pace
        super().__init__(model=model, params=params, **kwargs)

    @staticmethod
    def _resample_for_pace(audio: bytes, rate: int, pace: float) -> bytes:
        """Fallback pace approximation: resample the waveform as though
        converting it to rate/pace, but leave it labelled at the original
        rate -- the classic 'cassette tape' trick: fewer samples played back
        at the same rate take proportionally less time (pace > 1 => faster,
        shorter, higher-pitched; pace < 1 => slower, longer, lower-pitched).
        Cheap and always works, but the pitch shift is very audible at
        anything beyond a subtle adjustment (sounds like a cassette tape sped
        up -- 'helium' at 1.7x). Used only when the proper time-stretch below
        can't run (e.g. a clip too short to chunk).
        """
        if not audio or not rate or pace <= 0:
            return audio
        try:
            samples = np.frombuffer(audio, dtype=np.int16)
            resampled = soxr.resample(samples, rate, rate / pace, quality="HQ")
            return resampled.astype(np.int16).tobytes()
        except Exception as e:
            logger.warning(f"sarvam broadcast tts: local pace adjustment failed, playing at 1.0x: {e}")
            return audio

    @classmethod
    def _stretch_for_pace(cls, audio: bytes, rate: int, pace: float) -> bytes:
        """Change playback speed while keeping pitch roughly natural, for a
        model (bulbul:v3/v3-beta) that gives us no server-side pace control.
        Speeds up by repeatedly dropping small (~ms-scale) slivers of audio
        between crossfaded chunks instead of resampling the whole waveform,
        so pitch stays close to the original -- unlike _resample_for_pace's
        'helium' effect. pydub's speedup() only supports pace > 1 (this is
        exactly the direction the dashboard's speed slider is normally used
        for); pace < 1 falls back to the resample trick, which is less
        jarring for a pitch *drop* than a pitch raise.
        """
        if not audio or not rate or pace <= 0:
            return audio
        if pace <= 1.0:
            return cls._resample_for_pace(audio, rate, pace)
        try:
            seg = AudioSegment(data=audio, sample_width=2, frame_rate=rate, channels=1)
            stretched = _pydub_speedup(seg, playback_speed=pace, chunk_size=150, crossfade=25)
            return stretched.raw_data
        except Exception as e:
            logger.warning(
                f"sarvam broadcast tts: time-stretch failed ({e}), falling back to resample"
            )
            return cls._resample_for_pace(audio, rate, pace)

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

            if self._client_side_pace:
                audio_data = self._stretch_for_pace(
                    audio_data, actual_rate, self._client_side_pace
                )

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
