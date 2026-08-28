"""Words must come back while the caller is still speaking.

This is a telephony requirement, not a feature. A transcriber that answers only
when the speaker stops turns every turn into a pause, and it has been true of
this pipeline since before the model-server existed: the client feeds audio
continuously and pushes an InterimTranscriptionFrame roughly every
AI4BHARAT_INTERIM_MS (600 ms) while a segment is open.

Nothing tested it. That absence is why the catalogue was allowed to say
`streaming: false` against indic-conformer and why that was then read back as
"this model waits for a full sentence" -- which is not what it meant and not
what happens. The flag described whether the *model* serves a WebSocket; the
partials come from the *client* either way.

So this pins the behaviour rather than the vocabulary:

* the emitter exists and is actually called from the audio path
* it fires on elapsed audio, not on the segment ending
* it is skipped while a transcription is already in flight, so a slow model
  cannot queue up a backlog of stale partials behind the live one
* both STT models go through it -- there is one client class, so a model that
  streams natively does not silently take a different path

The interval is read from the real source, so lengthening it to something a
caller would notice fails here.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
CLIENT = ROOT.parent / "voice_2_voice_server" / "services" / "ai4bharat" / "stt.py"

needs_client = pytest.mark.skipif(
    not CLIENT.is_file(), reason="voice_2_voice_server not present in this checkout"
)


def source() -> str:
    return CLIENT.read_text(encoding="utf-8")


def method(name: str) -> ast.FunctionDef:
    tree = ast.parse(source())
    cls = next((n for n in tree.body
                if isinstance(n, ast.ClassDef) and n.name == "ModelServerSTTService"), None)
    assert cls is not None, "stt.py no longer defines ModelServerSTTService"
    fn = next((n for n in cls.body
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name), None)
    assert fn is not None, f"ModelServerSTTService no longer defines {name}"
    return fn


@needs_client
def test_the_client_emits_partial_transcripts_at_all():
    """The frame type is the contract with Pipecat. Without it the pipeline sees
    nothing until the speaker stops, whichever model is deployed."""
    src = source()
    assert "InterimTranscriptionFrame" in src, \
        "the client no longer emits interim transcripts -- callers wait for silence"
    body = ast.dump(method("_maybe_emit_interim"))
    assert "InterimTranscriptionFrame" in body, \
        "_maybe_emit_interim no longer pushes an interim frame"


@needs_client
def test_partials_are_driven_from_the_audio_path():
    """An emitter nothing calls is the same as no emitter. It has to be reached
    from chunk handling, not only from the end-of-segment path."""
    handler = ast.dump(method("_handle_audio_chunk"))
    assert "_maybe_emit_interim" in handler, \
        "audio chunks no longer trigger interim transcripts"


@needs_client
def test_partials_fire_on_elapsed_audio_not_on_the_segment_ending():
    """The trigger must be 'enough new audio has arrived', so partials arrive
    mid-utterance. A trigger tied to the segment closing would make the
    behaviour indistinguishable from having no partials at all."""
    fn = ast.dump(method("_maybe_emit_interim"))
    assert "_bytes_since_last_interim" in fn, "the trigger is no longer elapsed audio"
    assert "_interim_interval_ms" in fn, "the interval is no longer consulted"


@needs_client
def test_the_interval_stays_inside_a_conversational_pause():
    """600 ms by default. Much beyond a second and the partials stop doing the
    job they exist for; this fails if someone quietly relaxes it."""
    m = re.search(r'AI4BHARAT_INTERIM_MS["\']\s*,\s*["\'](\d+)["\']', source())
    assert m, "the interim interval default is no longer readable from the source"
    assert int(m.group(1)) <= 1000, f"interim interval relaxed to {m.group(1)} ms"


@needs_client
def test_a_slow_model_cannot_queue_stale_partials():
    """If a transcription is already running, skip rather than wait. Waiting
    would serialise a backlog and deliver partials describing audio the caller
    finished saying seconds ago."""
    fn = ast.dump(method("_maybe_emit_interim"))
    assert "_transcribe_lock" in fn and "locked" in fn, \
        "the in-flight guard is gone; partials can now pile up behind a slow call"


@needs_client
def test_both_models_share_this_path():
    """One client class serves both STT models, so partials are not something a
    model can opt out of by being deployed. If a second construction site
    appears, one of the two models is on a different path and only the deployed
    one gets exercised."""
    services = ROOT.parent / "voice_2_voice_server" / "api" / "services.py"
    if not services.is_file():
        pytest.skip("services.py not present in this checkout")
    src = services.read_text(encoding="utf-8")
    assert src.count("ModelServerSTTService(") == 1


@needs_client
def test_the_catalogue_does_not_claim_a_model_lacks_partials():
    """The catalogue describes two different things and must not blur them:
    whether the model serves a WebSocket, and what the caller actually gets.
    Every STT model here returns partials; only some serve the route."""
    import yaml
    raw = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    for entry in raw.get("stt") or []:
        if entry["status"] != "ready":
            continue
        assert "streaming" not in entry, (
            f"{entry['id']}: the bare `streaming` key is ambiguous -- it was read as "
            f"'the caller waits for a full sentence', which is not what it meant. "
            f"Use partial_transcripts and streaming_endpoint."
        )
        assert entry.get("partial_transcripts") in {"native", "client-side"}, (
            f"{entry['id']}: every STT model returns partials mid-utterance; "
            f"say which way it does it"
        )
