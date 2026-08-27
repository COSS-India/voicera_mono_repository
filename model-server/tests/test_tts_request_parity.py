"""The description reaching the Parler runner must not change.

Before the revamp the Pipecat client composed "<speaker>. <description>" itself
and sent it as one string. The OpenAI endpoint takes them apart -- `voice` and
`instructions` -- and recomposes on the server. If that recomposition ever
drifts, the model gets a different prompt and the voice changes, which no other
test would catch.

SpeechRequest is loaded from the real server.py by AST so the test fails when
the source drifts rather than passing against a stale copy.
"""
import ast
from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

SERVER = Path(__file__).resolve().parent.parent / "tts" / "indic-parler" / "server.py"


def _load_speech_request():
    import typing

    from pydantic import Field
    tree = ast.parse(SERVER.read_text(encoding="utf-8"))
    node = next((n for n in tree.body
                 if isinstance(n, ast.ClassDef) and n.name == "SpeechRequest"), None)
    assert node is not None, "tts/server.py no longer defines SpeechRequest"
    ns = {"BaseModel": BaseModel, "Field": Field, "Optional": typing.Optional,
          "Literal": typing.Literal}
    exec(compile(ast.Module([node], []), "<server>", "exec"), ns)  # noqa: S102
    return ns["SpeechRequest"]


SpeechRequest = _load_speech_request()

# Exactly what the pre-revamp client sent as a single "description" field.
LEGACY_DESCRIPTION = "Divya. A clear, natural voice with good audio quality."


def test_voice_and_instructions_recompose_to_the_legacy_string():
    req = SpeechRequest(
        input="नमस्ते",
        voice="Divya",
        instructions="A clear, natural voice with good audio quality.",
    )
    assert req.description() == LEGACY_DESCRIPTION


def test_no_voice_means_instructions_alone():
    req = SpeechRequest(input="नमस्ते", instructions="A calm narrator.")
    assert req.description() == "A calm narrator."


def test_float32_is_the_default_format():
    # Anything else resamples or requantises on the server and changes the audio.
    assert SpeechRequest(input="x").response_format == "pcm_f32le"


@pytest.mark.parametrize("fmt", ["pcm_f32le", "pcm"])
def test_supported_formats_accepted(fmt):
    assert SpeechRequest(input="x", response_format=fmt).response_format == fmt


def test_unknown_format_rejected():
    # Pydantic raises ValidationError, but the test only cares that it refuses.
    with pytest.raises(ValidationError):
        SpeechRequest(input="x", response_format="mp3")
