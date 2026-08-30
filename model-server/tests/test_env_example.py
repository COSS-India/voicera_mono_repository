"""`.env.example` is the only documentation of what this thing needs to run.

Every variable Compose reads has a default, so a missing one never errors -- the
stack comes up and quietly does something other than what the operator intended.
`MIO_VLLM_GPU_MEMORY_UTILIZATION` is the sharp case: unset, the sidecar takes its
default fraction of a GPU shared with production through MPS, which does not
partition memory. Nothing fails. Production just gets slower.

The example file had drifted by six variables before this test existed, four of
them added the same week by overlays that nobody thought to document, because
nothing connected the two files.

Comments count. A commented-out line still tells the reader the variable exists
and what it is for, which is the whole job here; several of these should stay
commented, since setting them is the exception.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EXAMPLE = ROOT / ".env.example"

# Compose reads these itself; they are not ours to document as substitutions.
COMPOSE_BUILTINS = {"COMPOSE_PROFILES", "COMPOSE_FILE", "COMPOSE_PROJECT_NAME"}

# ${VAR}, ${VAR:-default}, ${VAR-default}, ${VAR:+alt}
SUBSTITUTION = re.compile(r"\$\{([A-Z][A-Z0-9_]*)[:\-+}]")


def compose_files() -> list[Path]:
    """The base file plus every overlay a model folder brings.

    Found by walking, not by a list, for the same reason setup.sh finds them by
    existence: a new model that brings an overlay is then covered on arrival.
    """
    found = sorted(ROOT.glob("compose.*.yml"))              # base and top-level overlays
    found += sorted(ROOT.glob("*/*/compose.*.yml"))          # whatever a model folder brings
    return [p for p in found if p.is_file()]


def referenced() -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for path in compose_files():
        for name in SUBSTITUTION.findall(path.read_text(encoding="utf-8")):
            out.setdefault(name, []).append(path.relative_to(ROOT).as_posix())
    return out


def documented() -> set[str]:
    text = EXAMPLE.read_text(encoding="utf-8")
    return set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]*)=", text, re.M))


def test_the_example_file_exists():
    assert EXAMPLE.is_file(), "there is nothing telling an operator what to configure"


def test_at_least_one_overlay_is_covered():
    """Guards the glob. If it stopped matching, the test below would pass by
    checking nothing -- which is how the drift happened in the first place."""
    overlays = [p for p in compose_files() if p.name == "compose.extra.yml"]
    assert overlays, "no model overlays found; the glob no longer matches"


def test_every_variable_compose_reads_is_documented():
    missing = {k: v for k, v in referenced().items()
               if k not in documented() and k not in COMPOSE_BUILTINS}
    assert not missing, (
        "read by Compose but absent from .env.example, so an operator cannot know "
        "it exists: " + "; ".join(f"{k} (in {', '.join(v)})" for k, v in sorted(missing.items()))
    )


def test_nothing_documented_has_quietly_stopped_being_used():
    """The other direction. A variable nobody reads is worse than undocumented:
    an operator sets it, sees no effect, and cannot tell whether it worked."""
    ref = set(referenced())
    # Some are consumed by a model's Dockerfile rather than by Compose.
    in_dockerfiles = set()
    for df in ROOT.glob("*/*/Dockerfile"):
        in_dockerfiles |= set(SUBSTITUTION.findall(df.read_text(encoding="utf-8")))
    orphans = documented() - ref - in_dockerfiles - COMPOSE_BUILTINS
    assert not orphans, (
        f".env.example documents {sorted(orphans)}, which nothing reads"
    )


@pytest.mark.parametrize("name", ["STT_MODEL", "TTS_MODEL", "LLM_MODEL", "GPU_DEVICE_IDS"])
def test_the_variables_an_operator_must_set_are_uncommented(name):
    """These are the ones you actually edit. A reader skims for live lines, so
    the slot selectors must not be hidden among the commented examples."""
    text = EXAMPLE.read_text(encoding="utf-8")
    assert re.search(rf"^{name}=", text, re.M), \
        f"{name} is commented out or missing; it is not optional"
