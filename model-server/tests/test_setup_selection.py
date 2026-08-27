"""setup.sh must offer the models, not assume them.

The requirement is that installing picks a model per slot from what is actually
in the repo. That regressed once already -- setup.sh asked "Enable STT? yes/no"
and then hardcoded `indic-conformer` -- so it is pinned here.

The menu is built by listing folders rather than parsing models.yaml, which
means adding a model folder makes it appear with no other edit. Everything below
checks the pieces that make that true.
"""
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
SETUP = ROOT.parent / "setup.sh"
SLOTS = ("stt", "tts", "llm")

needs_setup = pytest.mark.skipif(not SETUP.is_file(), reason="setup.sh not in this checkout")


def setup_source() -> str:
    # The repo may be checked out with CRLF on Windows; this is a shell script.
    return SETUP.read_text(encoding="utf-8").replace("\r\n", "\n")


@needs_setup
@pytest.mark.parametrize("slot", SLOTS)
def test_setup_offers_a_choice_for_every_slot(slot):
    assert re.search(rf"pick_model {slot}\b", setup_source()), (
        f"setup.sh never calls pick_model for the {slot} slot, so the user "
        f"cannot choose which {slot.upper()} model to host"
    )


@needs_setup
def test_the_menu_is_built_from_folders_not_a_hardcoded_list():
    src = setup_source()
    assert "list_slot_models()" in src, "the folder-listing helper is gone"
    assert "model-server/$1" in src or 'model-server/$slot' in src, \
        "list_slot_models no longer reads the slot's folder"


@needs_setup
@pytest.mark.parametrize("slot", SLOTS)
def test_no_model_id_is_assigned_without_asking(slot):
    """`STT_SEL="indic-conformer"` is the exact regression this guards."""
    for match in re.finditer(rf'{slot.upper()}_SEL=("?)([a-z0-9][\w.-]*)\1', setup_source()):
        value = match.group(2)
        assert value in {"__ask__", "yes", "no"} or value.startswith("$"), (
            f"setup.sh assigns {slot.upper()}_SEL={value!r} directly; "
            f"the model must come from the menu or the environment"
        )


@needs_setup
def test_selected_model_reaches_both_env_and_profiles():
    """The choice has to drive what Compose starts and what the gateway believes."""
    src = setup_source()
    for var in ("STT_MODEL", "TTS_MODEL", "LLM_MODEL", "COMPOSE_PROFILES"):
        assert f"^{var}=.*" in src, f"setup.sh never writes {var} into model-server/.env"


# ---------------------------------------------------------------- fetch.sh

def ready_models() -> list[tuple[str, str]]:
    raw = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    return [(kind, e["id"]) for kind in SLOTS
            for e in raw.get(kind) or [] if e["status"] == "ready"]


@pytest.mark.parametrize(("kind", "model_id"), ready_models())
def test_weight_fetchers_are_runnable(kind, model_id):
    """A model that needs weights ships its own fetch.sh, so setup.sh stays free
    of per-model download steps. Optional -- but broken is not allowed."""
    fetch = ROOT / kind / model_id / "fetch.sh"
    if not fetch.is_file():
        return
    if shutil.which("bash"):
        out = subprocess.run(["bash", "-n", str(fetch)], capture_output=True, text=True)
        assert out.returncode == 0, f"{kind}/{model_id}/fetch.sh does not parse:\n{out.stderr}"
    body = fetch.read_text(encoding="utf-8")
    assert "$(dirname" in body or "${BASH_SOURCE" in body, (
        f"{kind}/{model_id}/fetch.sh must resolve paths from its own location, "
        "or it downloads into whatever directory happened to be current"
    )


@needs_setup
def test_setup_runs_the_model_s_own_fetcher():
    assert "fetch.sh" in setup_source(), \
        "setup.sh no longer runs the selected model's fetch.sh"
