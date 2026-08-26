"""Stub Parler runner emitting a deterministic ramp, so gain and the
float32 -> int16 conversion can be asserted exactly."""
import numpy as np

CHUNK = 128
NCHUNKS = 4


class TTSRequest:
    def __init__(self, prompt, description, pid=None):
        self.prompt, self.description, self.pid = prompt, description, pid


class ParlerTTSModelRunner:
    def __init__(self, checkpoint_path, play_steps=60, **k):
        self.checkpoint_path = checkpoint_path
        self.running_requests = {}
        self._emitted = {}

    def prefill(self, req):
        self.running_requests[req.pid] = req
        self._emitted[req.pid] = 0

    def step(self):
        pass

    def check_stopping_criteria(self):
        for pid in list(self.running_requests):
            if self._emitted[pid] >= NCHUNKS:
                del self.running_requests[pid]

    def audio_decode(self):
        out = {}
        for pid in list(self.running_requests):
            i = self._emitted[pid]
            if i < NCHUNKS:
                out[pid] = (np.arange(CHUNK, dtype=np.float32) / CHUNK - 0.5) * (1.0 - i * 0.1)
                self._emitted[pid] = i + 1
        return out
