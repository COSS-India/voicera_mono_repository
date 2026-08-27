"""A model folder may bring services alongside its own container.

Indic-Mio is the first: token generation is a vLLM sidecar, because upstream
splits it that way and its Dockerfile pins a torch that vLLM's image would fight.
Rather than let that break the slot design, the folder declares the extra service
in `compose.extra.yml` and setup.sh picks the file up by existence.

What must stay true, and is easy to break silently:

* the slot still has exactly one service named after it, on its own port
* the sidecar publishes nothing on the host
* the overlay is actually applied when that model is selected
"""
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
COMPOSE = ROOT / "compose.model-server.yml"
SETUP = ROOT.parent / "setup.sh"
SLOTS = ("stt", "tts", "llm")


def models_with_extras() -> list[tuple[str, str]]:
    found = []
    for slot in SLOTS:
        root = ROOT / slot
        if not root.is_dir():
            continue
        for folder in sorted(p for p in root.iterdir() if p.is_dir()):
            if (folder / "compose.extra.yml").is_file():
                found.append((slot, folder.name))
    return found


needs_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="needs the docker CLI (no daemon required)"
)


def render(slot: str, model: str) -> dict:
    """Render the base compose plus the model's overlay, as setup.sh would."""
    out = subprocess.run(
        ["docker", "compose", "-f", str(COMPOSE),
         "-f", str(ROOT / slot / model / "compose.extra.yml"), "config"],
        capture_output=True, text=True, check=False,
        env={"COMPOSE_PROFILES": ",".join(SLOTS), f"{slot.upper()}_MODEL": model,
             "PATH": "/usr/bin:/bin:/usr/local/bin"},
    )
    assert out.returncode == 0, f"compose config failed:\n{out.stderr}"
    return yaml.safe_load(out.stdout)


def test_at_least_one_model_exercises_this():
    """If nothing declares extras the tests below vacuously pass, which would
    hide the feature rotting."""
    assert models_with_extras(), "no model declares compose.extra.yml any more"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), models_with_extras())
def test_the_overlay_is_valid_compose(slot, model):
    cfg = render(slot, model)
    assert slot in cfg["services"], f"the overlay dropped the {slot} service itself"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), models_with_extras())
def test_extra_services_publish_nothing_on_the_host(slot, model):
    """The gateway is the only published port. A sidecar that binds one would
    collide with whatever else is on the box -- production, in our case."""
    cfg = render(slot, model)
    for name, svc in cfg["services"].items():
        if name == "gateway":
            continue
        assert not svc.get("ports"), f"{name} publishes {svc.get('ports')}"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), models_with_extras())
def test_the_slot_service_keeps_its_name_and_port(slot, model):
    """The gateway addresses upstreams by service name. An overlay that renamed
    or moved the slot would break routing while looking fine locally."""
    expected_port = {"stt": "8001", "tts": "8002", "llm": "8003"}[slot]
    svc = render(slot, model)["services"][slot]
    assert svc["environment"]["PORT"] == expected_port
    assert svc["build"]["context"].endswith(f"/{slot}/{model}")


@needs_docker
@pytest.mark.parametrize(("slot", "model"), models_with_extras())
def test_extra_services_share_the_internal_network(slot, model):
    """A sidecar off the network is unreachable by the slot that needs it."""
    cfg = render(slot, model)
    for name, svc in cfg["services"].items():
        assert "model_net" in (svc.get("networks") or {}), f"{name} is not on model_net"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), models_with_extras())
def test_a_gpu_sidecar_does_not_grab_the_card(slot, model):
    """This GPU is shared with production through MPS, which does not partition
    memory. A vLLM sidecar left at its upstream default would reserve tens of
    gigabytes at startup and starve the prod workers."""
    cfg = render(slot, model)
    for name, svc in cfg["services"].items():
        if name in ("gateway", slot):
            continue
        for arg in svc.get("command") or []:
            if "gpu-memory-utilization" in str(arg):
                value = float(str(arg).split("=")[-1])
                assert value <= 0.15, (
                    f"{name} reserves {value:.0%} of the whole card at startup; "
                    f"on a shared GPU that is taken from production"
                )


@pytest.mark.skipif(not SETUP.is_file(), reason="setup.sh not in this checkout")
def test_setup_applies_overlays_by_existence_not_by_model_name():
    """Adding a model with a sidecar must not mean editing the installer."""
    src = SETUP.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert "compose.extra.yml" in src, "setup.sh no longer looks for model overlays"
    assert "COMPOSE_FILES" in src, "the overlay list is not built"
    for _slot, model in models_with_extras():
        assert model not in src.split("STT_SEL=")[0], (
            f"setup.sh hardcodes {model!r}; overlays must be found by file existence"
        )
