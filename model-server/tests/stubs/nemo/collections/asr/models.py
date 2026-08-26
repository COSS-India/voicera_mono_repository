"""Stub ASR model. transcribe() echoes a fingerprint of the audio it was given,
so a test can assert the caller's samples arrived unaltered."""
import hashlib

import numpy as np


class _Model:
    def __init__(self, path):
        self.path = path
        self.cur_decoder = None

    def to(self, *a, **k):
        return self

    def freeze(self):
        return self

    def transcribe(self, audio=None, batch_size=None, language_id=None):
        out = []
        for arr in audio:
            a = np.asarray(arr, dtype=np.float32)
            digest = hashlib.md5(a.tobytes()).hexdigest()[:12]
            out.append(f"lang={language_id} n={a.size} md5={digest}")
        return [out]


class ASRModel:
    @staticmethod
    def restore_from(restore_path=None, map_location=None, **k):
        return _Model(restore_path)


class EncDecHybridRNNTCTCBPEModel:
    @staticmethod
    def restore_from(restore_path=None, map_location=None, **k):
        return _Model(restore_path)
