"""The LLM's model id has to mean the same thing in four places.

vLLM rejects any request whose "model" field is not the name it was started
with, and it does so at call time with a 400 -- during a live phone call, not at
deploy. Four files have to agree on the string:

    llm/<model>/Dockerfile      SERVED_NAME=... and the folder name
    models.yaml                 the catalogue id the gateway reports at /models
    services/vllm_qwen/llm.py   VLLM_MODEL, what the client puts in the request
    config/llm_mappings.py      the default for provider "qwen"

Nothing else checks that, so this does. Source is read from the real files -- by
AST where it is Python -- so the test fails when one of them drifts.
"""
import ast
import os
import re
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
V2V = ROOT.parent / "voice_2_voice_server"

needs_v2v = pytest.mark.skipif(
    not V2V.is_dir(), reason="voice_2_voice_server not present in this checkout"
)


def dockerfile_env(path: Path) -> dict[str, str]:
    """Pull ENV assignments out of a Dockerfile, honouring line continuations."""
    text = re.sub(r"\\\r?\n", " ", path.read_text(encoding="utf-8"))
    env: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip().startswith("ENV "):
            continue
        for key, value in re.findall(r"(\w+)=(\"[^\"]*\"|\S+)", line[4:]):
            env[key] = value.strip('"')
    return env


def deployable_llms() -> list[str]:
    raw = yaml.safe_load((ROOT / "models.yaml").read_text(encoding="utf-8"))
    return [e["id"] for e in raw.get("llm") or [] if e["status"] == "ready"]


@pytest.mark.parametrize("model_id", deployable_llms())
def test_served_model_name_matches_the_folder_and_catalogue_id(model_id):
    env = dockerfile_env(ROOT / "llm" / model_id / "Dockerfile")
    assert env.get("SERVED_NAME") == model_id, (
        f"llm/{model_id}/Dockerfile serves {env.get('SERVED_NAME')!r}; "
        f"clients asking for {model_id!r} would get a 400"
    )
    assert env.get("PORT") == "8003", "the LLM slot is addressed as llm:8003"


@needs_v2v
def test_voice_server_asks_for_the_model_that_is_actually_served():
    src = (V2V / "services" / "vllm_qwen" / "llm.py").read_text(encoding="utf-8")
    match = re.search(r'^VLLM_MODEL = .*?"([^"]+)"\s*$', src, re.M)
    assert match, "VLLM_MODEL is no longer a module-level assignment"
    assert match.group(1) in deployable_llms(), (
        f"voice server defaults to {match.group(1)!r}, which no deployable LLM serves"
    )


@needs_v2v
def test_provider_mapping_agrees_with_the_voice_server_default():
    src = (V2V / "config" / "llm_mappings.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    mapping = next(
        (ast.literal_eval(n.value) for n in tree.body
         if isinstance(n, ast.Assign)
         and any(getattr(t, "id", None) == "LLM_DEFAULT_MODELS" for t in n.targets)),
        None,
    )
    assert mapping is not None, "LLM_DEFAULT_MODELS is gone"
    assert mapping["qwen"] in deployable_llms(), (
        f'llm_mappings sends model={mapping["qwen"]!r}, which no deployable LLM serves'
    )


# ---------------------------------------------------------------- base URL

def load_resolver():
    """Extract _resolve_base_url from the real source. Importing the module
    would drag in pipecat and the whole OpenAI client."""
    src = (V2V / "services" / "vllm_qwen" / "llm.py").read_text(encoding="utf-8")
    node = next((n for n in ast.parse(src).body
                 if isinstance(n, ast.FunctionDef) and n.name == "_resolve_base_url"), None)
    assert node is not None, "llm.py no longer defines _resolve_base_url"
    ns: dict = {"os": os}
    exec(compile(ast.Module([node], []), "<llm>", "exec"), ns)  # noqa: S102
    return ns["_resolve_base_url"]


@needs_v2v
@pytest.mark.parametrize(("env", "expected"), [
    ({"MODEL_SERVER_URL": "http://localhost:8100"}, "http://localhost:8100/v1"),
    ({"MODEL_SERVER_URL": "http://localhost:8100/"}, "http://localhost:8100/v1"),
    # Already carries /v1 -- must not become /v1/v1.
    ({"MODEL_SERVER_URL": "http://localhost:8100/v1"}, "http://localhost:8100/v1"),
    # Explicit override still wins when the gateway variable is absent.
    ({"VLLM_BASE_URL": "http://elsewhere:8003/v1"}, "http://elsewhere:8003/v1"),
    # Neither set: empty, so create_voice_llm raises instead of silently
    # falling through to api.openai.com.
    ({}, ""),
])
def test_base_url_resolution(monkeypatch, env, expected):
    resolve = load_resolver()
    for key in ("MODEL_SERVER_URL", "VLLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    assert resolve() == expected


@needs_v2v
def test_gateway_url_wins_over_the_legacy_override(monkeypatch):
    """STT and TTS both prefer MODEL_SERVER_URL; the LLM must not be the odd one."""
    monkeypatch.setenv("MODEL_SERVER_URL", "http://gateway:8000")
    monkeypatch.setenv("VLLM_BASE_URL", "http://100.64.1.16:8003/v1")
    assert load_resolver()() == "http://gateway:8000/v1"


@needs_v2v
def test_no_hardcoded_upstream_addresses_remain():
    """The old default was a Tailscale IP baked into the source."""
    src = (V2V / "services" / "vllm_qwen" / "llm.py").read_text(encoding="utf-8")
    code = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("#"))
    assert not re.search(r'"https?://\d+\.\d+\.\d+\.\d+', code), \
        "a literal IP address is back in the LLM service"
