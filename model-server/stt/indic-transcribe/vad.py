"""
Streaming voice-activity detection (Silero v5) for the ASR gateway.

Why this exists
---------------
1. **Concurrency.** AlignAtt re-encodes an 11.5s buffer every chunk, so a silent chunk costs
   exactly as much GPU as a speech chunk. Real conversational audio is mostly silence, so gating
   the forward pass on speech is the cheapest capacity win available.
2. **Hallucination suppression.** Attention encoder-decoder models are known to invent fluent text
   from silence -- the same failure mode that makes Whisper emit "Thanks for watching!" over a
   quiet track. Not feeding silence to the decoder is the most reliable defence.
3. **Endpointing.** A voice agent needs to know the speaker *stopped*, which is a different
   question from "is there speech in this frame". That is the hysteresis below.

Why not the `silero-vad` package
--------------------------------
`pip install silero-vad` pulls in `torchaudio`, and the available torchaudio wheel is built
against CUDA 13.0 while our torch is 13.2. Importing it raises

    RuntimeError: Detected that PyTorch and TorchAudio were compiled with different CUDA versions

and -- because NeMo imports torchaudio opportunistically -- that breaks NeMo itself, not just the
VAD. So we use the ONNX weights the package ships and drive them with onnxruntime on the CPU.
No torchaudio, no GPU contention with the ASR model, ~40us per 32ms frame.

The ONNX graph is inherently streaming: it takes and returns a `[2, batch, 128]` LSTM state, so
each session carries its own conversational context.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

log = logging.getLogger("asr.vad")

SAMPLE_RATE = 16000
FRAME = 512          # silero v5 requires exactly 512 samples @ 16 kHz (32 ms)
CONTEXT = 64         # ...plus 64 samples of carried-over context, prepended (see frame_probs)
STATE_SHAPE = (2, 1, 128)


_ONNX_NAMES = ("silero_vad_16k_op15.onnx", "silero_vad.onnx")


def _find_onnx(explicit: Optional[str] = None) -> Optional[Path]:
    """
    Locate the Silero ONNX WITHOUT importing `silero_vad`.

    This is the whole trick. `silero_vad/__init__.py` does `import torchaudio`, and we
    deliberately have no torchaudio (its wheel is CUDA-13.0 and breaks our CUDA-13.2 torch, which
    in turn breaks NeMo). So `import silero_vad` raises ModuleNotFoundError even though the .onnx
    we want is sitting on disk inside that very package. We only need the data file, so we go
    looking for it on sys.path instead of asking Python to load the package.
    """
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    import site
    import sys

    roots: list[Path] = []
    for getter in (getattr(site, "getsitepackages", None), getattr(site, "getusersitepackages", None)):
        try:
            got = getter() if getter else None
        except Exception:
            got = None
        if isinstance(got, str):
            roots.append(Path(got))
        elif got:
            roots.extend(Path(x) for x in got)
    roots.extend(Path(p) for p in sys.path if p)

    seen = set()
    for root in roots:
        d = root / "silero_vad" / "data"
        if d in seen:
            continue
        seen.add(d)
        for name in _ONNX_NAMES:
            if (d / name).exists():
                return d / name
    return None


class SileroVAD:
    """Frame-level speech probability. One instance is shared; state is per session."""

    def __init__(self, model_path: Optional[str] = None, num_threads: int = 1):
        import onnxruntime as ort

        path = _find_onnx(model_path or os.environ.get("ASR_VAD_ONNX"))
        if path is None:
            raise FileNotFoundError(
                "silero VAD onnx not found. Expected <site-packages>/silero_vad/data/"
                f"{{{','.join(_ONNX_NAMES)}}} (install with `pip install --no-deps silero-vad`) "
                "or set ASR_VAD_ONNX to an explicit path."
            )
        opts = ort.SessionOptions()
        # One thread: this runs on the request path alongside a GPU service on 8 vCPUs. Letting
        # onnxruntime spin up a pool per session would starve the gateway.
        opts.inter_op_num_threads = num_threads
        opts.intra_op_num_threads = num_threads
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.path = path
        log.info("silero VAD loaded from %s", path)

    @staticmethod
    def new_state() -> np.ndarray:
        return np.zeros(STATE_SHAPE, dtype=np.float32)

    @staticmethod
    def new_context() -> np.ndarray:
        return np.zeros((1, CONTEXT), dtype=np.float32)

    def frame_probs(
        self, pcm: np.ndarray, state: np.ndarray, context: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Score `pcm` (float32 mono 16 kHz) in 512-sample frames.

        Returns (probs_per_frame, new_state, new_context). A trailing partial frame is ignored --
        the caller keeps the remainder for next time.

        The `context` argument is essential and easy to miss. Silero v5 does NOT take a bare
        512-sample frame: its own OnnxWrapper feeds `concat([last_64_samples, frame_512])` = 576
        samples and carries the final 64 forward. Feeding bare 512-sample frames does not error --
        the graph accepts the shape and returns a speech probability of ~0.001 for even loud,
        clear speech. That silently disables VAD, which in turn silently disables both the silence
        gate and endpointing. Measured that the hard way.
        """
        n = (len(pcm) // FRAME) * FRAME
        if n == 0:
            return np.zeros(0, dtype=np.float32), state, context
        probs = np.empty(n // FRAME, dtype=np.float32)
        sr = np.array(SAMPLE_RATE, dtype=np.int64)
        for i in range(0, n, FRAME):
            frame = pcm[i : i + FRAME].reshape(1, FRAME).astype(np.float32)
            inp = np.concatenate([context, frame], axis=1)
            out, state = self.sess.run(
                None, {"input": inp, "state": state, "sr": sr}
            )
            context = inp[:, -CONTEXT:]
            probs[i // FRAME] = float(out[0][0])
        return probs, state, context


@dataclass
class VadConfig:
    enabled: bool = True
    threshold: float = 0.5              # speech if p >= threshold
    min_silence_ms: int = 500           # silence this long after speech => endpoint
    speech_pad_ms: int = 200            # keep this much audio around speech
    gate_gpu_on_silence: bool = True    # skip the forward pass while no speech seen yet


class VadGate:
    """
    Per-session speech/silence hysteresis over frame probabilities.

    Tracks two things the engine cares about:
      * `has_speech`  -- has this session contained speech at all? While False and no speech is
                         present, the engine can skip the GPU entirely.
      * `endpointed`  -- speech happened and has now been followed by `min_silence_ms` of quiet,
                         i.e. the speaker finished their turn and the transcript can be finalised.
    """

    __slots__ = ("cfg", "_vad", "_state", "_context", "_tail", "has_speech", "endpointed",
                 "_silence_ms", "_speech_ms", "last_prob")

    def __init__(self, vad: Optional[SileroVAD], cfg: VadConfig):
        self.cfg = cfg
        self._vad = vad
        self._state = SileroVAD.new_state()
        self._context = SileroVAD.new_context()
        self._tail = np.zeros(0, dtype=np.float32)   # samples not yet a whole frame
        self.has_speech = False
        self.endpointed = False
        self._silence_ms = 0.0
        self._speech_ms = 0.0
        self.last_prob = 0.0

    @property
    def active(self) -> bool:
        return self.cfg.enabled and self._vad is not None

    def observe(self, pcm: np.ndarray) -> None:
        """Feed the same audio that goes to the ASR buffer."""
        if not self.active:
            self.has_speech = True     # VAD off => treat everything as speech
            return
        buf = np.concatenate([self._tail, pcm]) if self._tail.size else pcm
        probs, self._state, self._context = self._vad.frame_probs(
            buf, self._state, self._context
        )
        used = len(probs) * FRAME
        self._tail = buf[used:]

        frame_ms = FRAME / SAMPLE_RATE * 1000
        for p in probs:
            self.last_prob = float(p)
            if p >= self.cfg.threshold:
                self.has_speech = True
                self.endpointed = False
                self._speech_ms += frame_ms
                self._silence_ms = 0.0
            else:
                self._silence_ms += frame_ms
                if self.has_speech and self._silence_ms >= self.cfg.min_silence_ms:
                    self.endpointed = True

    def should_skip_gpu(self) -> bool:
        """
        True when this chunk is safe to drop.

        Deliberately conservative: we only skip *before any speech has ever been seen*. Once a
        session has spoken, mid-utterance pauses must still be fed, because the decoder's state
        and the left-context buffer have to stay time-aligned with the audio -- dropping a chunk
        mid-stream would silently shift every later timestamp.
        """
        if not self.active or not self.cfg.gate_gpu_on_silence:
            return False
        return not self.has_speech

    def stats(self) -> dict:
        return {
            "vad_active": self.active,
            "has_speech": self.has_speech,
            "endpointed": self.endpointed,
            "speech_ms": round(self._speech_ms),
            "silence_ms": round(self._silence_ms),
            "last_prob": round(self.last_prob, 3),
        }
