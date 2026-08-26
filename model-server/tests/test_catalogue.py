"""The /models catalogue.

Guards the two ways this can go wrong quietly: a malformed models.yaml, and
/v1/models drifting into advertising something a caller would get a 503 from.
"""
import pytest
import yaml
from app import catalogue

CATALOGUE = catalogue.CATALOGUE_PATH.parent.parent.parent / "models.yaml"


def test_catalogue_file_is_valid():
    raw = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    assert set(raw) <= set(catalogue.KINDS), f"unexpected top-level keys: {set(raw) - set(catalogue.KINDS)}"
    for kind in catalogue.KINDS:
        for entry in raw.get(kind) or []:
            assert entry.get("id"), f"{kind} entry with no id"
            assert entry.get("status") in {"ready", "planned"}, f"{entry['id']}: bad status"


def test_ids_are_unique_within_a_kind():
    raw = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    for kind in catalogue.KINDS:
        ids = [e["id"] for e in raw.get(kind) or []]
        assert len(ids) == len(set(ids)), f"duplicate ids in {kind}: {ids}"


def test_load_flattens_and_tags_kind():
    models = catalogue.load(CATALOGUE)
    assert models, "catalogue loaded empty"
    assert all(m["kind"] in catalogue.KINDS for m in models)
    ready = [m["id"] for m in models if m["status"] == "ready"]
    assert "indic-conformer" in ready
    assert "indic-parler" in ready


def test_missing_catalogue_is_not_fatal(tmp_path):
    # The gateway must keep routing even if it cannot describe itself.
    assert catalogue.load(tmp_path / "nope.yaml") == []


@pytest.mark.parametrize("kind", catalogue.KINDS)
def test_every_deployable_model_has_a_compose_profile(kind):
    """A model marked ready must be startable, or `ready` is a lie."""
    compose = yaml.safe_load(
        (CATALOGUE.parent / "compose.model-server.yml").read_text(encoding="utf-8")
    )
    profiles = {p for svc in compose["services"].values() for p in svc.get("profiles", [])}
    raw = yaml.safe_load(CATALOGUE.read_text(encoding="utf-8"))
    for entry in raw.get(kind) or []:
        if entry["status"] == "ready":
            assert entry["id"] in profiles, (
                f"{entry['id']} is marked ready but has no compose profile"
            )
