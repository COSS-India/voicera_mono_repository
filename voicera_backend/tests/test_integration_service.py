"""
Unit tests for app.services.integration_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.integration_service import (
    create_integration,
    get_integration,
    get_openai_api_key_for_org,
    get_integrations_by_org,
    delete_integration,
)
from app.models.schemas import IntegrationCreate
from tests.helpers import make_mock_db

# ── Error message constants ───────────────────────────────────────────────────

ERR_INTEGRATION_NOT_FOUND = "Integration not found"

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"

INTEGRATION_DOC = {
    "org_id": ORG_ID,
    "model": "OpenAI",
    "api_key": "sk-test-key",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

OPENAI_CREATE = IntegrationCreate(org_id=ORG_ID, model="OpenAI", api_key="sk-test-key")


def _make_db(doc=None):
    coll = MagicMock()
    coll.find_one.return_value = doc
    db = make_mock_db(Integrations=coll)
    return db, coll


# ── TestCreateIntegration ─────────────────────────────────────────────────

class TestCreateIntegration:
    def test_success_returns_success_dict(self):
        db, coll = _make_db(doc=None)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = create_integration(OPENAI_CREATE)
        assert result["status"] == "success"

    def test_upsert_updates_existing_integration(self):
        db, coll = _make_db(doc=INTEGRATION_DOC)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = create_integration(OPENAI_CREATE)
        assert result["status"] == "success"

    def test_exception_returns_fail(self):
        db, coll = _make_db()
        coll.find_one.side_effect = Exception("DB error")
        with patch("app.services.integration_service.get_database", return_value=db):
            result = create_integration(OPENAI_CREATE)
        assert result["status"] == "fail"


# ── TestGetIntegration ────────────────────────────────────────────────────

class TestGetIntegration:
    def test_returns_doc_without_id_field(self):
        from bson import ObjectId
        doc_with_id = {**INTEGRATION_DOC, "_id": ObjectId()}
        db, _ = _make_db(doc=doc_with_id)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = get_integration(ORG_ID, "OpenAI")
        assert result is not None
        assert "_id" not in result
        assert result["model"] == "OpenAI"

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(doc=None)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = get_integration(ORG_ID, "Gemini")
        assert result is None

    def test_exception_returns_none(self):
        db = MagicMock()
        db.__getitem__.side_effect = Exception("DB down")
        with patch("app.services.integration_service.get_database", return_value=db):
            result = get_integration(ORG_ID, "OpenAI")
        assert result is None


# ── TestGetOpenaiApiKeyForOrg ─────────────────────────────────────────────

class TestGetOpenaiApiKeyForOrg:
    def test_finds_exact_openai_key(self):
        coll = MagicMock()
        coll.find_one.return_value = {**INTEGRATION_DOC, "model": "OpenAI"}
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            key = get_openai_api_key_for_org(ORG_ID)
        assert key == "sk-test-key"

    def test_finds_lowercase_openai_key(self):
        coll = MagicMock()
        # "OpenAI" not found, "openai" found
        coll.find_one.side_effect = [None, {**INTEGRATION_DOC, "model": "openai"}]
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            key = get_openai_api_key_for_org(ORG_ID)
        assert key == "sk-test-key"

    def test_no_key_returns_none(self):
        coll = MagicMock()
        coll.find_one.return_value = None
        coll.find.return_value = []  # empty scan
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            key = get_openai_api_key_for_org(ORG_ID)
        assert key is None

    def test_fallback_scan_finds_openai_key(self):
        """When get_integration returns None for both 'OpenAI'/'openai',
        falls back to scanning Integrations collection directly."""
        coll = MagicMock()
        coll.find_one.return_value = None
        # The fallback scan finds a doc with model='Open AI' (normalizes to 'openai')
        coll.find.return_value = [{"model": "Open AI", "api_key": "sk-fallback", "org_id": ORG_ID}]
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            key = get_openai_api_key_for_org(ORG_ID)
        assert key == "sk-fallback"

    def test_empty_api_key_continues_loop(self):
        """When api_key strips to empty string, loops to next model."""
        coll = MagicMock()
        # OpenAI doc has empty key → continues; openai doc has real key
        coll.find_one.side_effect = [
            {"model": "OpenAI", "api_key": "   "},  # empty after strip
            {"model": "openai", "api_key": "sk-real"},
        ]
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            key = get_openai_api_key_for_org(ORG_ID)
        assert key == "sk-real"


# ── TestGetIntegrationsByOrg ──────────────────────────────────────────────

class TestGetIntegrationsByOrg:
    def test_returns_list_without_id_fields(self):
        from bson import ObjectId
        doc_with_id = {**INTEGRATION_DOC, "_id": ObjectId()}
        coll = MagicMock()
        coll.find.return_value = [doc_with_id]
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = get_integrations_by_org(ORG_ID)
        assert len(result) == 1
        assert "_id" not in result[0]

    def test_exception_returns_empty_list(self):
        coll = MagicMock()
        coll.find.side_effect = Exception("DB error")
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = get_integrations_by_org(ORG_ID)
        assert result == []


# ── TestDeleteIntegration ─────────────────────────────────────────────────

class TestDeleteIntegration:
    def test_success_returns_success_dict(self):
        coll = MagicMock()
        coll.delete_one.return_value.deleted_count = 1
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = delete_integration(ORG_ID, "OpenAI")
        assert result["status"] == "success"

    def test_not_found_returns_fail(self):
        coll = MagicMock()
        coll.delete_one.return_value.deleted_count = 0
        db = make_mock_db(Integrations=coll)
        with patch("app.services.integration_service.get_database", return_value=db):
            result = delete_integration(ORG_ID, "Ghost")
        assert result["status"] == "fail"
        assert ERR_INTEGRATION_NOT_FOUND in result["message"]

    def test_exception_returns_fail(self):
        db = MagicMock()
        db.__getitem__.side_effect = Exception("DB error")
        with patch("app.services.integration_service.get_database", return_value=db):
            result = delete_integration(ORG_ID, "OpenAI")
        assert result["status"] == "fail"
        assert "DB error" in result["message"]
