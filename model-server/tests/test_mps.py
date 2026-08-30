"""MPS is a property of the host, so it is detected rather than assumed.

A GPU in Exclusive Process mode can only be shared through an MPS daemon; a GPU
in the ordinary Default mode needs none of it. Both mistakes are silent:

* attach with no daemon behind the pipe directory and the client finds nothing
* skip it on an Exclusive Process GPU and the container never gets a context

This was originally wired into the base compose file with the pipe directory
written out as `/tmp/nvidia-mps-gpu1`, which was wrong twice over. It ignored
`GPU_DEVICE_IDS`, so selecting GPU 3 gave you GPU 3 with GPU 1's pipe -- a
mismatch nothing reports. And it applied on every host, including the laptops
and dedicated boxes where no daemon exists, which is what a teammate would hit
first.

Rendering is done by the real `docker compose config`, not by reading YAML, so
these check what Compose actually produces -- including whether an overlay's
`volumes` list quietly replaces the base service's mounts rather than adding to
them, which is the failure that would take the model's own source with it.
"""
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "compose.model-server.yml"
MPS = ROOT / "compose.mps.yml"
SLOTS = ["stt", "tts", "llm"]

needs_docker = pytest.mark.skipif(
    shutil.which("docker") is None, reason="needs the docker CLI (no daemon required)"
)

ENV = {"PATH": "/usr/bin:/bin:/usr/local/bin", "COMPOSE_PROFILES": ",".join(SLOTS),
       "STT_MODEL": "indic-conformer", "TTS_MODEL": "indic-parler",
       "LLM_MODEL": "qwen3.5-4b"}


def render(files: list[Path], **env) -> dict:
    args = ["docker", "compose"]
    for f in files:
        args += ["-f", str(f)]
    args += ["--project-directory", str(ROOT), "config"]
    out = subprocess.run(args, capture_output=True, text=True, check=False,
                         env={**ENV, **env})
    assert out.returncode == 0, f"compose config failed:\n{out.stderr}"
    return yaml.safe_load(out.stdout)


def mps_mounts(service: dict) -> list[str]:
    return [v["source"] for v in service.get("volumes", [])
            if "mps" in str(v.get("source", ""))]


# ------------------------------------------------------------------ the base

@needs_docker
@pytest.mark.parametrize("slot", SLOTS)
def test_the_base_stack_has_no_mps_at_all(slot):
    """A host without a daemon must get a stack that never mentions one.

    This is the teammate case: clone, run, no MPS anywhere on the machine.
    Nothing should be mounted from /tmp/nvidia-mps-*, and no CUDA_MPS_* variable
    should be set to a directory that will be empty.
    """
    svc = render([BASE])["services"][slot]
    assert not mps_mounts(svc), f"{slot} mounts an MPS pipe with no overlay applied"
    leaked = [k for k in svc.get("environment", {}) if k.startswith("CUDA_MPS")]
    assert not leaked, f"{slot} sets {leaked} without an MPS daemon being established"


@needs_docker
def test_the_base_stack_still_asks_for_a_gpu():
    """Removing MPS must not have removed the device reservation with it."""
    svc = render([BASE])["services"]["stt"]
    devices = svc["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["capabilities"] == ["gpu"]


# --------------------------------------------------------------- the overlay

@needs_docker
@pytest.mark.parametrize("gpu", ["0", "1", "3"])
@pytest.mark.parametrize("slot", SLOTS)
def test_the_pipe_directory_follows_the_selected_gpu(slot, gpu):
    """The original bug, pinned. Selecting a GPU must select its daemon."""
    svc = render([BASE, MPS], GPU_DEVICE_IDS=gpu)["services"][slot]
    assert mps_mounts(svc) == [f"/tmp/nvidia-mps-gpu{gpu}",
                               f"/tmp/nvidia-mps-log-gpu{gpu}"], \
        f"{slot} on GPU {gpu} is attached to another GPU's daemon"


@needs_docker
def test_an_unusual_daemon_location_can_be_named_outright():
    """The convention is one pipe directory per GPU, which stops meaning
    anything when GPU_DEVICE_IDS names several. MPS_PIPE_DIR is the way out."""
    svc = render([BASE, MPS], GPU_DEVICE_IDS="1,2",
                 MPS_PIPE_DIR="/var/run/mps", MPS_LOG_DIR="/var/log/mps")["services"]["stt"]
    assert mps_mounts(svc) == ["/var/run/mps", "/var/log/mps"]


@needs_docker
@pytest.mark.parametrize("slot", SLOTS)
def test_the_overlay_adds_to_the_service_rather_than_replacing_it(slot):
    """An overlay that carries `volumes:` can shadow the base list instead of
    extending it, which would unmount the model's own source and leave a
    container that starts and finds no code."""
    base = render([BASE])["services"][slot]
    with_mps = render([BASE, MPS])["services"][slot]

    base_targets = {v["target"] for v in base.get("volumes", [])}
    mps_targets = {v["target"] for v in with_mps.get("volumes", [])}
    assert base_targets <= mps_targets, \
        f"the MPS overlay dropped {base_targets - mps_targets} from {slot}"

    assert set(base["environment"]) <= set(with_mps["environment"]), \
        "the MPS overlay dropped environment the slot needs"
    assert with_mps.get("ipc") == "host", "an MPS client needs the host IPC namespace"


@needs_docker
def test_the_gateway_is_left_out():
    """Pure async I/O, no CUDA context. Giving it host IPC would be a needless
    loss of isolation for a process that cannot use MPS."""
    cfg = render([BASE, MPS])
    assert cfg["services"]["gateway"].get("ipc") != "host"
    assert not mps_mounts(cfg["services"]["gateway"])


# ------------------------------------------------------- models with sidecars

def sidecar_overlays() -> list[tuple[str, str]]:
    """Model folders that bring GPU sidecars needing their own MPS wiring."""
    return [(p.parent.parent.name, p.parent.name)
            for p in sorted(ROOT.glob("*/*/compose.mps.yml"))]


def test_the_sidecar_convention_is_exercised():
    """If no model brings one, the test below passes by checking nothing."""
    assert sidecar_overlays(), "no model declares compose.mps.yml any more"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), sidecar_overlays())
def test_a_models_sidecars_attach_to_the_same_daemon_as_its_slot(slot, model):
    """The slot service and the sidecar it talks to must reach one daemon.
    Different pipe directories would be two clients of two daemons, one of which
    does not exist."""
    files = [BASE, ROOT / slot / model / "compose.extra.yml",
             MPS, ROOT / slot / model / "compose.mps.yml"]
    cfg = render(files, GPU_DEVICE_IDS="5", **{f"{slot.upper()}_MODEL": model})

    slot_mounts = mps_mounts(cfg["services"][slot])
    assert slot_mounts == ["/tmp/nvidia-mps-gpu5", "/tmp/nvidia-mps-log-gpu5"]

    sidecars = [n for n in cfg["services"] if n not in {*SLOTS, "gateway"}]
    assert sidecars, f"{slot}/{model} declares no sidecar to attach"
    for name in sidecars:
        assert mps_mounts(cfg["services"][name]) == slot_mounts, \
            f"{name} is attached to a different daemon than the {slot} slot"


@needs_docker
@pytest.mark.parametrize(("slot", "model"), sidecar_overlays())
def test_a_sidecar_runs_without_mps_too(slot, model):
    """No daemon means no MPS overlay at all -- including the sidecar's. The
    sidecar must still be a complete, startable service on its own."""
    cfg = render([BASE, ROOT / slot / model / "compose.extra.yml"],
                 **{f"{slot.upper()}_MODEL": model})
    sidecars = [n for n in cfg["services"] if n not in {*SLOTS, "gateway"}]
    for name in sidecars:
        svc = cfg["services"][name]
        assert not mps_mounts(svc), f"{name} mounts an MPS pipe with no overlay applied"
        assert svc.get("image") or svc.get("build"), f"{name} is not startable on its own"


# ----------------------------------------------------------------- setup.sh

def setup_source() -> str:
    path = ROOT.parent / "setup.sh"
    if not path.is_file():
        pytest.skip("setup.sh not in this checkout")
    return path.read_text(encoding="utf-8").replace("\r", "")


def test_setup_detects_the_daemon_rather_than_assuming_it():
    """setup.sh reports the daemon to the operator; compose-files.sh acts on it.

    Both must check, and they check the same thing -- the control pipe. Keeping
    the assertion split this way is deliberate: setup.sh printing "MPS found"
    while the file list omits the overlay would be worse than either failing.
    """
    assert "/control" in setup_source(), \
        "setup.sh no longer checks for the daemon's control pipe"

    files_sh = COMPOSE_FILES_SH.read_text(encoding="utf-8").replace("\r", "")
    assert "/control" in files_sh, \
        "compose-files.sh no longer checks for the daemon"
    assert "compose.mps.yml" in files_sh, \
        "compose-files.sh no longer adds the MPS overlay"


def test_setup_says_something_when_there_is_no_daemon():
    """Silence here is the worst outcome: on an Exclusive Process GPU the
    containers will fail to get a context, and the operator needs to be pointed
    at `nvidia-smi -q | grep -i 'compute mode'` rather than left guessing."""
    src = setup_source()
    assert "compute mode" in src.lower(), \
        "the no-daemon path no longer tells the operator how to check the GPU"


def test_setup_writes_the_paths_out_rather_than_relying_on_the_default():
    """Nested substitution resolves here, but a deployed .env should not depend
    on the Compose version doing it."""
    src = setup_source()
    assert "MPS_PIPE_DIR=$MPS_PIPE_DIR" in src, \
        "setup.sh no longer records the resolved pipe directory in .env"


# --------------------------------------------- one file list, three callers

REPO = ROOT.parent
COMPOSE_FILES_SH = ROOT / "compose-files.sh"


def test_the_file_list_has_a_single_source():
    """`make up`, `make stop-all-ports` and setup.sh must launch the same stack.

    They did not. Both Makefile targets used the base compose file alone, which
    was survivable while MPS lived in the base and became silent breakage the
    moment it moved to an overlay: on an Exclusive Process GPU the containers
    start and never get a CUDA context, with nothing in the output to say why.
    """
    assert COMPOSE_FILES_SH.is_file(), "compose-files.sh is gone"

    makefile = REPO / "Makefile"
    if makefile.is_file():
        src = makefile.read_text(encoding="utf-8").replace("\r", "")
        assert "compose-files.sh" in src, \
            "the Makefile builds its own compose file list again"
        assert "-f model-server/compose.model-server.yml" not in src, \
            "a Makefile target still names the base compose file directly"

    setup = REPO / "setup.sh"
    if setup.is_file():
        src = setup.read_text(encoding="utf-8").replace("\r", "")
        assert "compose-files.sh" in src, "setup.sh builds its own list again"


@pytest.mark.parametrize(("selection", "expect"), [
    ({"TTS_MODEL": "indic-parler"}, []),
    ({"TTS_MODEL": "indic-mio"}, ["tts/indic-mio/compose.extra.yml"]),
    ({"STT_MODEL": "indic-transcribe"}, ["stt/indic-transcribe/compose.extra.yml"]),
])
def test_the_script_picks_up_what_the_selected_models_bring(selection, expect):
    """Driven from .env, which is what every caller actually has."""
    env = {"STT_MODEL": "indic-conformer", "TTS_MODEL": "indic-parler", **selection}
    real = ROOT / ".env"
    saved = real.read_text(encoding="utf-8") if real.is_file() else None
    try:
        real.write_text("".join(f"{k}={v}\n" for k, v in env.items()), encoding="utf-8")
        out = subprocess.run(["sh", str(COMPOSE_FILES_SH)], capture_output=True,
                             text=True, check=True).stdout
    finally:
        if saved is not None:
            real.write_text(saved, encoding="utf-8")
        else:
            real.unlink(missing_ok=True)

    listed = [f.replace(str(ROOT) + "/", "") for f in out.split() if f != "-f"]
    assert listed[0] == "compose.model-server.yml", "the base file must come first"
    for wanted in expect:
        assert wanted in listed, f"{wanted} missing for {selection}"
    # No daemon in this environment, so no MPS overlay should appear.
    assert not [f for f in listed if "mps" in f], \
        "an MPS overlay was added with no daemon present"


def run_file_list(env_lines: str) -> list[str]:
    """Run compose-files.sh against a given .env, return the files it names."""
    real = ROOT / ".env"
    saved = real.read_text(encoding="utf-8") if real.is_file() else None
    try:
        real.write_text(env_lines, encoding="utf-8")
        out = subprocess.run(["sh", str(COMPOSE_FILES_SH)], capture_output=True,
                             text=True, check=True).stdout
    finally:
        if saved is not None:
            real.write_text(saved, encoding="utf-8")
        else:
            real.unlink(missing_ok=True)
    return [f.replace(str(ROOT) + "/", "") for f in out.split() if f != "-f"]


def test_a_daemon_that_is_present_is_actually_attached_to(tmp_path):
    """The half that cannot be exercised without a daemon, so one is faked.

    Only the control pipe's existence is checked, which makes this honest to
    test: an empty file at $MPS_PIPE_DIR/control is exactly the fact the script
    reads. Without this, a change that stopped adding the overlay would pass --
    every other test here runs on a host with no daemon.
    """
    pipe = tmp_path / "pipe"
    pipe.mkdir()
    (pipe / "control").touch()

    listed = run_file_list(
        f"STT_MODEL=indic-conformer\nTTS_MODEL=indic-mio\nMPS_PIPE_DIR={pipe}\n"
    )
    assert "compose.mps.yml" in listed, \
        "a daemon is present and the MPS overlay was not added"
    assert "tts/indic-mio/compose.mps.yml" in listed, \
        "the selected model's sidecars were not attached to the daemon"
    # Ordering matters: an overlay can only add to a service the base defines.
    assert listed.index("compose.model-server.yml") == 0
    assert listed.index("tts/indic-mio/compose.extra.yml") < listed.index(
        "tts/indic-mio/compose.mps.yml"), \
        "the sidecar's MPS overlay is applied before the service it modifies exists"


def test_no_daemon_means_no_overlay_even_for_a_model_that_has_one(tmp_path):
    """The other half, and the teammate case: nothing on the host, nothing added."""
    listed = run_file_list(
        f"STT_MODEL=indic-conformer\nTTS_MODEL=indic-mio\nMPS_PIPE_DIR={tmp_path / 'absent'}\n"
    )
    assert not [f for f in listed if "mps" in f], f"MPS overlays added with no daemon: {listed}"
    assert "tts/indic-mio/compose.extra.yml" in listed, \
        "the model's own services were dropped along with MPS"
