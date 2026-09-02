"""
Streaming ASR session state.

Kept separate from the HTTP layer so both the native WebSocket endpoint and the
OpenAI-compatible API can drive the same session logic without a circular import.
"""
import asyncio
import hashlib
import json
import os
import time
import wave
from typing import Dict, Optional

import numpy as np
import torch
from fastapi import WebSocket

from asr_engine import (
    StepRequest,
    submit_mel,
    get_scheduler,
    get_initial_cache,
    get_streaming_params,
    prompt_index,
    resolve_model_and_prompt,
    DEVICE,
)

# Wire protocol only: the client ships 160 ms of int16 PCM per message. This is
# independent of the model's chunk size, which comes from encoder.streaming_cfg and
# is typically several of these frames. The server accumulates.
WIRE_CHUNK_SAMPLES = 2560
WIRE_CHUNK_BYTES = WIRE_CHUNK_SAMPLES * 2
INACTIVITY_TIMEOUT_SEC = 30.0

# Mel-extraction context, in mel frames. The preprocessor centres each frame and
# reflect-pads the edges, so frames near the boundary of a processing window are
# wrong unless real audio surrounds them. n_fft=512 => a frame needs +/-256 samples,
# i.e. 2 hops of 160. Three on the left (also covers pre-emphasis' 1-sample lookback)
# and two on the right make every committed frame bit-identical to offline extraction.
MEL_LEFT_CTX_FRAMES = 3
MEL_RIGHT_CTX_FRAMES = 2

SAMPLE_RATE = 16000

# Utterance endpointing. Without it one connection is a single ever-growing hypothesis
# that only finalises when the client disconnects, which voice agents cannot use.
# 400 ms sits above natural inter-word pauses while keeping end-of-turn detection --
# which is dead air in an agent's turn-taking loop -- as short as is safe.
VAD_SILENCE_MS = float(os.environ.get("ASR_VAD_SILENCE_MS", "400"))
VAD_ENABLED = os.environ.get("ASR_VAD", "1").lower() in ("1", "true", "yes")
# Measured on this model's log-mel: silence sits near -15.7 and speech's 10th
# percentile near -12.2, so ~3.4 of headroom. 2.5 puts the threshold between them
# with room for a noisier real microphone.
VAD_FLOOR_PERCENTILE = 0.1
VAD_MARGIN_DB = float(os.environ.get("ASR_VAD_MARGIN_DB", "2.5"))

# Live-mic input-rate sanity check. The server counts every 2560-sample frame as 160 ms
# of audio; if the browser is actually running a 48 kHz context, frames arrive 3x too
# fast and this ratio reads ~3.0. That single number distinguishes "the audio is wrong"
# from "the model is wrong" without needing to hear the audio.
# Deliberately wide: the fault this catches is a ~3x multiplier (48 kHz read as
# 16 kHz), while a healthy short session drifts low simply because the microphone
# takes a moment to ramp up. A narrow band only produces false alarms.
RATE_CHECK_MIN_AUDIO_SEC = 5.0
RATE_RATIO_MIN = 0.6
RATE_RATIO_MAX = 1.4

DEBUG_CAPTURE = os.environ.get("ASR_DEBUG_CAPTURE", "").lower() in ("1", "true", "yes")
CAPTURE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "debug")
CAPTURE_MAX_SECONDS = 30.0
CAPTURE_MAX_BYTES = int(CAPTURE_MAX_SECONDS * SAMPLE_RATE * 2)


def _build_id() -> str:
    """Stamp derived from the served assets, used to defeat the AudioWorklet cache."""
    h = hashlib.md5()
    for name in ("static/index.html", "static/audio-processor.js"):
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)
        try:
            with open(path, "rb") as f:
                h.update(f.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


BUILD_ID = _build_id()

sessions: Dict[str, "ASRSession"] = {}


class ASRSession:
    """
    One cache-aware streaming stream.

    Chunking mirrors NeMo's CacheAwareStreamingAudioBuffer.__iter__: the model is
    stepped with exactly streaming_cfg.chunk_size mel frames, and step 0 differs from
    every later step in chunk size, pre-encode cache, and drop_extra_pre_encoded.
    """

    def __init__(self, session_id: str, language: str = "hi", websocket: Optional[WebSocket] = None,
                 vad: Optional[bool] = None):
        self.session_id = session_id
        self.websocket = websocket
        # Whole-file transcription wants one transcript, not utterance segments; live
        # streams want the opposite.
        self.vad = VAD_ENABLED if vad is None else vad
        self.language_code = language.strip().lower()
        self.model, self.target_lang, self.is_bhili = resolve_model_and_prompt(self.language_code)
        if self.model is None:
            raise RuntimeError(f"No ASR model available for language {self.language_code!r}")

        p = get_streaming_params(self.model)
        self.chunk_size = p["chunk_size"]
        self.shift_size = p["shift_size"]
        self.pre_encode_cache_size = p["pre_encode_cache_size"]
        self.drop_extra_pre_encoded = p["drop_extra_pre_encoded"]
        self.sampling_frames = p["sampling_frames"]
        self.hop = p["hop_length"]
        self.feat_in = p["feat_in"]

        # Cap the backlog at two model chunks. If the GPU falls behind real time we
        # drop the oldest audio instead of accumulating unbounded lag.
        self.max_raw_samples = (self.chunk_size[1] * self.hop) * 2 + WIRE_CHUNK_SAMPLES

        self.last_active = time.time()

        # Input-rate accounting (survives reset_state: it describes the connection,
        # not the current utterance).
        self.first_audio_wall: Optional[float] = None
        self.samples_since_first = 0
        self.last_rate_ratio: Optional[float] = None
        self.rate_warned = False
        self.samples_dropped = 0
        self.drop_warned = False
        self.client_sample_rate: Optional[float] = None

        self.capture = bytearray() if DEBUG_CAPTURE else None
        self.detected_lang: Optional[str] = None
        # Describes the room, so it survives utterance boundaries.
        self.noise_floor: Optional[float] = None

        self.reset_state()

    # ---------- streaming state ----------

    def reset_state(self):
        self.cache_channel, self.cache_time, self.cache_len = get_initial_cache(1, self.is_bhili)
        self.prev_hyp = None
        self.prev_pred_out = None
        self.step_num = 0

        self.raw = np.zeros(0, dtype=np.float32)
        self.raw_offset = 0          # global sample index of raw[0]
        self.mel_done = 0            # count of globally-valid mel frames produced so far
        self.mel_buffer: Optional[torch.Tensor] = None
        self.mel_base = 0            # global frame index of mel_buffer[..., 0]
        self.buffer_idx = 0          # index into mel_buffer of the next unconsumed frame

        self.last_text = ""
        self.last_active = time.time()
        self.silence_ms = 0.0
        self.spoke = False

    @property
    def model_chunk_ms(self) -> int:
        return self.chunk_size[1] * self.hop * 1000 // SAMPLE_RATE

    def input_rate_ratio(self) -> Optional[float]:
        """
        Seconds of audio received per second of wall clock.

        ~1.0 for a correctly-configured live microphone. ~3.0 means the client is
        sending 48 kHz samples that the server is interpreting as 16 kHz; ~2.76 means
        44.1 kHz. Returns None until there is enough audio to measure meaningfully.
        """
        if self.first_audio_wall is None:
            return None
        audio_sec = self.samples_since_first / float(SAMPLE_RATE)
        if audio_sec < RATE_CHECK_MIN_AUDIO_SEC:
            return None
        elapsed = time.time() - self.first_audio_wall
        if elapsed <= 0.0:
            return None
        self.last_rate_ratio = audio_sec / elapsed
        return self.last_rate_ratio

    def implied_sample_rate(self, ratio: float) -> int:
        return int(round(SAMPLE_RATE * ratio / 100.0) * 100)

    def write_capture(self) -> Optional[str]:
        """Persist what the browser actually sent, plus a sidecar of signal stats."""
        if not self.capture:
            return None
        os.makedirs(CAPTURE_DIR, exist_ok=True)
        base = os.path.join(CAPTURE_DIR, self.session_id)
        pcm = bytes(self.capture)
        with wave.open(base + ".wav", "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(pcm)

        x = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        elapsed = (time.time() - self.first_audio_wall) if self.first_audio_wall else 0.0
        spec = np.abs(np.fft.rfft(x * np.hanning(len(x)))) if len(x) > 1 else np.zeros(1)
        freqs = np.fft.rfftfreq(len(x), 1.0 / SAMPLE_RATE) if len(x) > 1 else np.zeros(1)
        stats = {
            "session_id": self.session_id,
            "language": self.language_code,
            "client_sample_rate": self.client_sample_rate,
            "audio_seconds": len(x) / float(SAMPLE_RATE),
            "wall_clock_seconds": round(elapsed, 3),
            "input_rate_ratio": self.last_rate_ratio,
            "samples_dropped": self.samples_dropped,
            "implied_source_rate": self.implied_sample_rate(self.last_rate_ratio) if self.last_rate_ratio else None,
            "rms": float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0,
            "peak": float(np.max(np.abs(x))) if len(x) else 0.0,
            "clipped_fraction": float(np.mean(np.abs(x) > 0.99)) if len(x) else 0.0,
            "zero_crossing_rate": float(np.mean(np.abs(np.diff(np.sign(x))) > 0)) if len(x) > 1 else 0.0,
            "spectral_centroid_hz": float((spec * freqs).sum() / spec.sum()) if spec.sum() > 0 else 0.0,
        }
        with open(base + ".json", "w") as f:
            json.dump(stats, f, indent=2)
        print(f"[Capture] wrote {base}.wav  ({stats['audio_seconds']:.1f}s audio in "
              f"{stats['wall_clock_seconds']:.1f}s wall, ratio="
              f"{stats['input_rate_ratio']}, rms={stats['rms']:.4f})")
        return base + ".wav"

    # ---------- audio ingest ----------

    def append_audio(self, raw_bytes: bytes):
        audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        now = time.time()
        if self.first_audio_wall is None:
            # Start the clock at the first frame without counting it: that frame was
            # already buffered before it arrived, and counting it biases the ratio.
            self.first_audio_wall = now
        else:
            self.samples_since_first += len(audio)

        if self.capture is not None and len(self.capture) < CAPTURE_MAX_BYTES:
            self.capture.extend(raw_bytes[: CAPTURE_MAX_BYTES - len(self.capture)])

        self.raw = np.concatenate((self.raw, audio))
        if len(self.raw) > self.max_raw_samples:
            drop = len(self.raw) - self.max_raw_samples
            # Keep the drop hop-aligned so the mel frame grid never shifts.
            drop -= drop % self.hop
            if drop > 0:
                self.raw = self.raw[drop:]
                self.raw_offset += drop
                self.mel_done = max(self.mel_done, self.raw_offset // self.hop + MEL_LEFT_CTX_FRAMES)
                # Shedding audio is the intended response to falling behind real time,
                # but it silently truncates the transcript, so say so.
                self.samples_dropped += drop
                if not self.drop_warned:
                    self.drop_warned = True
                    print(f"[Backpressure] session {self.session_id}: dropping audio "
                          f"(input is outrunning inference); transcripts will have gaps")
        self.last_active = time.time()

    def _mel_window(self, final: bool):
        """
        The raw-audio window whose mel frames are now fully surrounded by context.

        Returns (segment, drop_left, keep) or None. Split out from _grow_mel so the
        extraction itself can be batched across streams on the live path.
        """
        first = max(0, self.mel_done - MEL_LEFT_CTX_FRAMES)
        w0 = first * self.hop
        if w0 < self.raw_offset:
            return None
        seg = self.raw[w0 - self.raw_offset:]
        if len(seg) == 0:
            return None

        n_frames = len(seg) // self.hop + 1
        drop_l = self.mel_done - first
        drop_r = 0 if final else MEL_RIGHT_CTX_FRAMES
        keep = n_frames - drop_l - drop_r
        if keep <= 0:
            return None
        return np.ascontiguousarray(seg), drop_l, keep

    def _commit_mel(self, mel: torch.Tensor, drop_l: int, keep: int):
        """Append the newly-valid frames and release raw audio nothing can need again."""
        new = mel[:, :, drop_l:drop_l + keep]
        if new.size(-1) == 0:
            return
        self.mel_buffer = new if self.mel_buffer is None else torch.cat((self.mel_buffer, new), dim=-1)
        self.mel_done += new.size(-1)

        keep_from = max(0, (self.mel_done - MEL_LEFT_CTX_FRAMES) * self.hop)
        if keep_from > self.raw_offset:
            cut = keep_from - self.raw_offset
            self.raw = self.raw[cut:]
            self.raw_offset += cut

    def _grow_mel(self, final: bool = False):
        """Synchronous extraction, for tests and whole-file transcription."""
        window = self._mel_window(final)
        if window is None:
            return
        seg, drop_l, keep = window
        seg_t = torch.from_numpy(seg).unsqueeze(0).to(DEVICE)
        seg_len = torch.tensor([len(seg)], device=DEVICE, dtype=torch.long)
        with torch.inference_mode():
            mel, _ = self.model.preprocessor(input_signal=seg_t, length=seg_len)
        self._commit_mel(mel, drop_l, keep)

    async def _agrow_mel(self, final: bool = False):
        """
        Extraction on the live path, batched across streams.

        A final window keeps its right edge, and zero-padding a batch changes the STFT's
        reflection padding there, so flushes fall back to the unbatched call to stay
        bit-identical with offline extraction.
        """
        if final:
            self._grow_mel(final=True)
            return
        window = self._mel_window(False)
        if window is None:
            return
        seg, drop_l, keep = window
        seg_t = torch.from_numpy(seg).to(DEVICE)
        future = submit_mel("bhili" if self.is_bhili else "indic", seg_t)
        mel = await asyncio.wrap_future(future)
        self._commit_mel(mel, drop_l, keep)

    # ---------- model stepping ----------

    def note_silence(self, frames_consumed: int):
        """
        Track trailing silence so an utterance can be closed on a pause.

        The threshold is relative to a noise floor tracked across the session, not to
        the current window: a window-local floor is self-referential, and an earlier
        version of this never fired because of it. The floor follows a quieter room
        immediately and creeps up slowly, so sustained speech cannot drag it along.
        """
        if not self.vad or self.mel_buffer is None or frames_consumed <= 0:
            return
        window = self.mel_buffer[:, :, max(0, self.buffer_idx - frames_consumed):self.buffer_idx]
        if window.size(-1) == 0:
            return

        energy = window.mean(dim=1)[0]                      # log-mel, per frame
        observed = float(torch.quantile(energy, VAD_FLOOR_PERCENTILE))
        if self.noise_floor is None or observed < self.noise_floor:
            self.noise_floor = observed
        else:
            self.noise_floor += min(0.05, (observed - self.noise_floor) * 0.02)

        speech = energy > (self.noise_floor + VAD_MARGIN_DB)
        frame_ms = self.hop / 16.0
        if bool(speech.any()):
            self.spoke = True
            last_speech = int(torch.nonzero(speech).flatten()[-1])
            self.silence_ms = (speech.numel() - 1 - last_speech) * frame_ms
        else:
            self.silence_ms += speech.numel() * frame_ms

    def endpoint_reached(self) -> bool:
        return self.vad and self.spoke and self.silence_ms >= VAD_SILENCE_MS

    def close_utterance(self) -> str:
        """
        Finalise the current utterance and start a fresh one on the same connection.

        Resets everything together. The earlier version called reset_state() and then
        restored mel_buffer and buffer_idx, which left mel_done and mel_base at zero
        while the buffer still held frames -- so the invariant
        `mel_buffer.size(-1) == mel_done - mel_base` broke and later frames were
        appended at the wrong offset.

        Discarding the buffered audio is correct here: the VAD only fires after
        VAD_SILENCE_MS of silence, so what is buffered is the silence that ended the
        utterance, not speech.
        """
        text = self.last_text
        self.reset_state()
        return text

    def _available_frames(self) -> int:
        if self.mel_buffer is None:
            return 0
        return self.mel_buffer.size(-1) - self.buffer_idx

    def has_full_chunk(self) -> bool:
        need = self.chunk_size[0] if self.step_num == 0 else self.chunk_size[1]
        return self._available_frames() >= need

    def _build_chunk(self):
        """
        Assemble the mel chunk for the next step, or None if there is not enough.

        Mirrors CacheAwareStreamingAudioBuffer.__iter__: the first step differs from
        every later one in chunk size, pre-encode cache, and drop_extra_pre_encoded.
        """
        first = self.step_num == 0
        chunk = self.chunk_size[0] if first else self.chunk_size[1]
        shift = self.shift_size[0] if first else self.shift_size[1]
        pre_cache = self.pre_encode_cache_size[0] if first else self.pre_encode_cache_size[1]
        # No cache exists before the first step, so there is nothing to drop yet.
        drop_extra = 0 if first else self.drop_extra_pre_encoded
        min_frames = self.sampling_frames[0] if first else self.sampling_frames[1]

        take = min(chunk, self._available_frames())
        if take < min_frames:
            return None

        with torch.inference_mode():
            audio_chunk = self.mel_buffer[:, :, self.buffer_idx:self.buffer_idx + take]
            if pre_cache == 0:
                chunk_with_cache = audio_chunk
            else:
                start = max(0, self.buffer_idx - pre_cache)
                cache_pre_encode = self.mel_buffer[:, :, start:self.buffer_idx]
                if cache_pre_encode.size(-1) < pre_cache:
                    pad = torch.zeros(
                        (1, self.feat_in, pre_cache - cache_pre_encode.size(-1)),
                        device=self.mel_buffer.device, dtype=self.mel_buffer.dtype)
                    cache_pre_encode = torch.cat((pad, cache_pre_encode), dim=-1)
                chunk_with_cache = torch.cat((cache_pre_encode, audio_chunk), dim=-1)
        return chunk_with_cache.contiguous(), shift, take, drop_extra

    def _request(self, chunk_with_cache, drop_extra, is_last_chunk):
        return StepRequest(
            model_key="bhili" if self.is_bhili else "indic",
            chunk=chunk_with_cache,
            cache_channel=self.cache_channel,
            cache_time=self.cache_time,
            cache_len=self.cache_len,
            prev_hyp=self.prev_hyp,
            prev_pred_out=self.prev_pred_out,
            prompt_index=prompt_index(self.model, self.target_lang),
            slice_lang=self.target_lang,
            drop_extra=drop_extra,
            keep_all_outputs=is_last_chunk,
        )

    def _apply(self, result, shift, take):
        self.cache_channel = result["cache_channel"]
        self.cache_time = result["cache_time"]
        self.cache_len = result["cache_len"]
        self.prev_hyp = result["prev_hyp"]
        self.prev_pred_out = result["prev_pred_out"]

        consumed = min(shift, take)
        self.buffer_idx += consumed
        self.step_num += 1

        # Must run before pruning: pruning rewinds buffer_idx, so measuring consumed
        # frames from a buffer_idx delta afterwards always yields zero.
        self.note_silence(consumed)

        # Keep only the frames a later step can still need as pre-encode context.
        retain = self.pre_encode_cache_size[1]
        if self.buffer_idx > retain:
            cut = self.buffer_idx - retain
            self.mel_buffer = self.mel_buffer[:, :, cut:].contiguous()
            self.mel_base += cut
            self.buffer_idx = retain

        if result["text"]:
            self.last_text = result["text"]
        return self.last_text

    def step(self, is_last_chunk: bool = False) -> str:
        """Synchronous step, for tests and offline tools. Blocks on the batcher."""
        built = self._build_chunk()
        if built is None:
            return self.last_text
        chunk_with_cache, shift, take, drop_extra = built
        result = get_scheduler().submit(
            self._request(chunk_with_cache, drop_extra, is_last_chunk)).result()
        return self._apply(result, shift, take)

    async def astep(self, is_last_chunk: bool = False) -> Optional[str]:
        """
        Await a batched step without holding a worker thread.

        wrap_future lets the event loop park here while the batcher thread runs one
        GPU step on behalf of this stream and many others.
        """
        built = self._build_chunk()
        if built is None:
            return None
        chunk_with_cache, shift, take, drop_extra = built
        future = get_scheduler().submit(self._request(chunk_with_cache, drop_extra, is_last_chunk))
        result = await asyncio.wrap_future(future)
        return self._apply(result, shift, take)

    def process_available(self) -> list[str]:
        """Extract mel, then step for every complete model chunk (synchronous)."""
        self._grow_mel(final=False)
        out = []
        while self.has_full_chunk():
            out.append(self.step(is_last_chunk=False))
        return out

    async def aprocess_available(self) -> list[str]:
        """
        Async drain: parks on the batcher instead of holding a worker thread.

        Returns partial transcripts. Utterances closed by silence are reported
        separately via `self.finals`.
        """
        await self._agrow_mel(final=False)
        self.finals: list = getattr(self, "finals", [])
        self.finals.clear()

        out = []
        while self.has_full_chunk():
            text = await self.astep(is_last_chunk=False)
            if text is None:
                break
            out.append(text)
            if self.endpoint_reached():
                final = self.close_utterance()
                if final:
                    self.finals.append(final)
                out = []
        return out

    async def aflush(self) -> str:
        """End of stream: commit the tail frames and drain whatever remains."""
        self._grow_mel(final=True)
        while self._available_frames() > 0:
            before = self.buffer_idx
            await self.astep(is_last_chunk=True)
            if self.buffer_idx == before:
                break
        return self.last_text

    def flush(self) -> str:
        """End of stream: commit the tail frames and drain whatever remains."""
        self._grow_mel(final=True)
        while self._available_frames() > 0:
            before = self.buffer_idx
            text = self.step(is_last_chunk=True)
            if self.buffer_idx == before:
                break
        return self.last_text
