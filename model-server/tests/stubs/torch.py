"""Minimal torch stand-in: only what stt/server.py and tts/server.py touch."""


class _Device:
    def __init__(self, spec="cpu"):
        self.type = "cpu"
        self._spec = spec

    def __repr__(self):
        return f"device('{self._spec}')"


def device(spec="cpu"):
    return _Device(spec)


class cuda:
    @staticmethod
    def is_available():
        return False


class _NoGrad:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __call__(self, fn):
        return fn


def no_grad():
    return _NoGrad()
