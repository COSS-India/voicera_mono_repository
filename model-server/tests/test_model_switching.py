"""Switching a model must be one environment variable and nothing else.

This is the whole premise of the layout: `stt/`, `tts/` and `llm/` each hold one
folder per model, and <KIND>_MODEL picks which folder the slot builds. If that
ever stops interpolating, adding a model silently keeps serving the old one --
which no other test would notice, because every other test reads source files
directly rather than going through Compose.

So this one renders the real Compose file with the real tool and reads back what
it resolved to.
"""
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "compose.model-server.yml"

SLOTS = [("stt", 8001), ("tts", 8002), ("llm", 8003)]

pytestmark = pytest.mark.skipif(
    shutil.which("docker") is None, reason="needs the docker CLI (no daemon required)"
)


def render(**env) -> dict:
    """`docker compose config` interpolates without needing a running daemon."""
    full = {"COMPOSE_PROFILES": "stt,tts,llm", "PATH": "/usr/bin:/bin:/usr/local/bin", **env}
    out = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE), "config"],
        capture_output=True, text=True, env=full, check=False,
    )
    assert out.returncode == 0, f"compose config failed:\n{out.stderr}"
    return yaml.safe_load(out.stdout)


@pytest.mark.parametrize(("slot", "_port"), SLOTS)
def test_named_model_selects_its_folder(slot, _port):
    cfg = render(**{f"{s.upper()}_MODEL": f"fake-{s}" for s, _ in SLOTS})
    svc = cfg["services"][slot]
    assert svc["build"]["context"].endswith(f"/{slot}/fake-{slot}"), svc["build"]["context"]
    assert svc["image"].endswith(f":fake-{slot}"), svc["image"]


def test_switching_a_model_leaves_the_service_name_and_port_alone():
    """The gateway addresses upstreams by service name, so those must not move."""
    before = render(LLM_MODEL="qwen3.5-4b")
    after = render(LLM_MODEL="gemma-3-4b")
    assert set(before["services"]) == set(after["services"])
    for slot, port in SLOTS:
        assert before["services"][slot]["environment"]["PORT"] == str(port)
        assert after["services"][slot]["environment"]["PORT"] == str(port)
    # Only the build context, image tag and bind mount should differ.
    assert before["services"]["llm"]["image"] != after["services"]["llm"]["image"]
    assert before["services"]["stt"] == after["services"]["stt"]


@pytest.mark.parametrize(("slot", "_port"), SLOTS)
def test_slot_is_off_when_its_profile_is_not_selected(slot, _port):
    cfg = render(COMPOSE_PROFILES="")
    assert slot not in cfg["services"], f"{slot} started without its profile"
    assert "gateway" in cfg["services"], "the gateway must run regardless"


def test_source_folder_is_the_one_mounted_into_the_container():
    """A build context and a bind mount pointing at different folders would run
    one model's weights against another model's code."""
    cfg = render(STT_MODEL="indic-conformer", TTS_MODEL="indic-parler")
    for slot in ("stt", "tts"):
        svc = cfg["services"][slot]
        binds = [v["source"] for v in svc["volumes"] if v.get("target") == "/app"]
        assert binds == [svc["build"]["context"]], f"{slot}: {binds} != {svc['build']['context']}"


def test_nemo_context_resolves_beside_the_repo_not_inside_it():
    """additional_contexts resolve against the compose file's directory, not the
    build context. Moving models a level deeper did not change that, and getting
    it wrong points the NeMo build context at a directory that does not exist."""
    cfg = render(STT_MODEL="indic-conformer")
    nemo = Path(cfg["services"]["stt"]["build"]["additional_contexts"]["nemo"])
    assert nemo.name == "ai4bharat_nemo"
    assert nemo.parent == ROOT.parent.parent, f"resolved to {nemo}"
