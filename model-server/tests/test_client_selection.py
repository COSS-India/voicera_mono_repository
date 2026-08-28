"""A model has to be selectable, not just deployable.

`test_catalogue.py` reconciles two of the three places a model exists: the
catalogue and the folder on disk. This is the third, and it is the one that
fails silently.

    models.yaml            what /models advertises
    <kind>/<id>/           what Compose can build
    api/services.py        what an agent config is allowed to ask for   <- here

Deploying a model the voice server has never heard of does not raise at deploy
time. The container starts, /health goes green, /models lists it -- and then the
first agent that names it dies inside `create_*_service` with "Unknown ai4bharat
model", mid-call. Nothing before this test connected the two ends: the TTS slot
grew from one model to three without any check that the client could name them.

The convention being enforced is `<catalogue id>-<slot>`: folder `orpheus` in
`tts/` is `orpheus-tts` to an agent. It is a convention rather than a lookup
because the catalogue does not record the client-facing name; if that ever
becomes a real field, this test should read it instead of deriving it.
"""
import ast
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOGUE = ROOT / "models.yaml"
SERVICES = ROOT.parent / "voice_2_voice_server" / "api" / "services.py"

needs_client = pytest.mark.skipif(
    not SERVICES.is_file(), reason="voice_2_voice_server not present in this checkout"
)

# The catalogue has an llm kind, but the LLM slot does not select by model name
# the way STT and TTS do -- vLLM is asked for its served name, which
# test_llm_wiring.py pins across all four files it appears in.
SLOTS = ["stt", "tts"]


def ready_ids(kind: str) -> set[str]:
    raw = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    return {e["id"] for e in (raw.get(kind) or []) if e["status"] == "ready"}


def provider_branch(kind: str) -> list[ast.stmt]:
    """The body of the `provider == "AI4Bharat"` arm of `create_<kind>_service`.

    Scoping matters twice over. These functions dispatch over every provider
    VoicEra supports, so an unscoped walk also collects Sarvam's `bulbul:v3` and
    friends and then reports them as models the model-server fails to provide.

    And the arm's *body* is what is wanted, not the `If` node -- an `elif` chain
    is nested `If`s, so the AI4Bharat node carries every provider that follows it
    inside `orelse`. Returning `node` instead of `node.body` looks correct and
    quietly re-admits exactly what the scoping was for.
    """
    tree = ast.parse(SERVICES.read_text(encoding="utf-8"))
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef) and n.name == f"create_{kind}_service"), None)
    assert fn is not None, f"services.py no longer defines create_{kind}_service"

    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not isinstance(node.test, ast.Compare):
            continue
        test = node.test
        if not (isinstance(test.left, ast.Name) and test.left.id == "provider"):
            continue
        if any(isinstance(c, ast.Constant) and c.value == "AI4Bharat"
               for c in test.comparators):
            return node.body
    raise AssertionError(
        f'create_{kind}_service no longer has a `provider == "AI4Bharat"` branch'
    )


def selectable(kind: str) -> set[str]:
    """Model names the AI4Bharat branch will accept from an agent config.

    Read out of the real source by AST, so a branch that is deleted or renamed
    fails this rather than passing against a copy of what it used to say. Both
    spellings are collected -- `model == "x"` and `model in ("x", "y")` -- since
    the file uses the first for TTS and the second for STT.
    """
    names: set[str] = set()
    body = provider_branch(kind)
    for node in (n for stmt in body for n in ast.walk(stmt)):
        if not isinstance(node, ast.Compare):
            continue
        if not (isinstance(node.left, ast.Name) and node.left.id == "model"):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=True):
            if not isinstance(op, (ast.Eq, ast.In)):
                continue
            for const in ast.walk(comparator):
                if isinstance(const, ast.Constant) and isinstance(const.value, str):
                    names.add(const.value)
    assert names, f"no model names found in create_{kind}_service -- did the branch move?"
    return names


@needs_client
@pytest.mark.parametrize("kind", SLOTS)
def test_every_deployable_model_can_be_asked_for(kind):
    """Deploy-only is the failure that reaches a live call.

    The model builds, starts and reports healthy; only the agent naming it finds
    out, and it finds out by dropping the call.
    """
    missing = {f"{i}-{kind}" for i in ready_ids(kind)} - selectable(kind)
    assert not missing, (
        f"marked ready in models.yaml but no branch in create_{kind}_service: "
        f"{sorted(missing)} -- deployable, and unusable by any agent"
    )


@needs_client
@pytest.mark.parametrize("kind", SLOTS)
def test_nothing_selectable_is_missing_its_model(kind):
    """The other direction. A name the client accepts but nothing serves posts a
    real request into whichever model *is* deployed in that slot, and gets back
    plausible output from the wrong model rather than an error."""
    ready = {f"{i}-{kind}" for i in ready_ids(kind)}
    orphans = selectable(kind) - ready
    assert not orphans, (
        f"create_{kind}_service accepts {sorted(orphans)}, which no ready model "
        f"in models.yaml provides"
    )


@needs_client
def test_both_stt_models_share_one_client_class():
    """The slot's promise is that models behind it are interchangeable to the
    caller. Two client classes would mean two request shapes, and the second one
    only gets exercised when somebody switches STT_MODEL in production."""
    src = SERVICES.read_text(encoding="utf-8")
    assert "ModelServerSTTService" in src
    assert "IndicConformerRESTSTTService(" not in src, \
        "the model-specific class name is being constructed again"
    assert src.count("ModelServerSTTService(") == 1, \
        "a second construction site means the two models diverged"


@needs_client
def test_an_unknown_model_is_refused_rather_than_defaulted():
    """Falling back to a default here would be the worst outcome: a typo in an
    agent config would silently run a different model than the config names."""
    src = SERVICES.read_text(encoding="utf-8")
    for kind in SLOTS:
        segment = "\n".join(
            ast.get_source_segment(src, stmt) or "" for stmt in provider_branch(kind)
        )
        assert "Unknown ai4bharat" in segment, \
            f"create_{kind}_service no longer refuses an unrecognised model"
