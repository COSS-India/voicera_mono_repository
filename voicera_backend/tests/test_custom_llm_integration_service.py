"""
Unit tests for app.services.custom_llm_integration_service.
"""
import pytest
from unittest.mock import MagicMock, patch
from bson import ObjectId

from app.services.custom_llm_integration_service import (
    normalize_base_url,
    _mask_api_key,
    create_custom_llm_integration,
    get_custom_llm_integration,
    get_custom_llm_integration_for_bot,
    get_custom_llm_integrations_by_org,
    update_custom_llm_integration,
    delete_custom_llm_integration,
)
from app.models.schemas import CustomLLMIntegrationCreate, CustomLLMIntegrationUpdate
from tests.helpers import make_mock_db

# ── Error message constants ───────────────────────────────────────────────────

ERR_LLM_NOT_FOUND = "Custom LLM integration not found"
ERR_NO_FIELDS = "No fields"

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"
OBJECT_ID = ObjectId()
LLM_ID = str(OBJECT_ID)

LLM_DOC = {
    "_id": OBJECT_ID,
    "org_id": ORG_ID,
    "name": "My Custom LLM",
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4",
    "api_key": "sk-secret-key",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

CREATE_DATA = CustomLLMIntegrationCreate(
    org_id=ORG_ID,
    name="My Custom LLM",
    base_url="https://api.example.com/v1",
    model="gpt-4",
    api_key="sk-secret-key",
)

UPDATE_DATA = CustomLLMIntegrationUpdate(name="Updated LLM")


def _make_db(doc=None):
    coll = MagicMock()
    coll.find_one.return_value = doc
    db = make_mock_db(CustomLLMIntegrations=coll)
    return db, coll


# ── TestNormalizeBaseUrl ──────────────────────────────────────────────────

class TestNormalizeBaseUrl:
    def test_strips_chat_completions_suffix(self):
        result = normalize_base_url("https://api.example.com/v1/chat/completions")
        assert not result.endswith("/chat/completions")
        assert result.endswith("/v1")

    def test_adds_v1_when_missing(self):
        result = normalize_base_url("https://api.example.com")
        assert result.endswith("/v1")

    def test_already_ends_with_v1_unchanged(self):
        result = normalize_base_url("https://api.example.com/v1")
        assert result == "https://api.example.com/v1"

    def test_empty_raises_valueerror(self):
        with pytest.raises(ValueError):
            normalize_base_url("")

    def test_non_http_scheme_raises_valueerror(self):
        with pytest.raises(ValueError):
            normalize_base_url("ftp://api.example.com/v1")


# ── TestMaskApiKey ────────────────────────────────────────────────────────

class TestMaskApiKey:
    def test_shows_last_4_chars(self):
        result = _mask_api_key("sk-secret-key")
        assert result.endswith("-key")
        assert "sk-secret" not in result

    def test_short_key_returns_all_stars(self):
        result = _mask_api_key("abc")
        assert result == "****"

    def test_empty_key_returns_all_stars(self):
        result = _mask_api_key("")
        assert result == "****"


# ── TestCreateCustomLlmIntegration ────────────────────────────────────────

class TestCreateCustomLlmIntegration:
    def test_success_returns_integration_with_id(self):
        # The service returns unmasked key on create (mask_key=False default)
        db, coll = _make_db()
        coll.insert_one.return_value.inserted_id = OBJECT_ID
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = create_custom_llm_integration(CREATE_DATA)
        assert result["status"] == "success"
        assert "integration" in result
        integration = result["integration"]
        assert integration.get("id") == LLM_ID
        assert integration.get("name") == CREATE_DATA.name
        assert integration.get("base_url") == "https://api.example.com/v1"
        assert integration.get("model") == CREATE_DATA.model
        assert integration.get("org_id") == CREATE_DATA.org_id

    def test_invalid_url_returns_fail(self):
        bad_data = CustomLLMIntegrationCreate(
            org_id=ORG_ID,
            name="Bad",
            base_url="not-a-url",
            model="gpt-4",
            api_key="sk-key",
        )
        db, _ = _make_db()
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = create_custom_llm_integration(bad_data)
        assert result["status"] == "fail"

    def test_exception_returns_fail(self):
        db, coll = _make_db()
        coll.insert_one.side_effect = Exception("DB error")
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = create_custom_llm_integration(CREATE_DATA)
        assert result["status"] == "fail"


# ── TestGetCustomLlmIntegration ───────────────────────────────────────────

class TestGetCustomLlmIntegration:
    def test_returns_doc_with_id_field(self):
        # _doc_to_response default mask_key=False — key not masked unless explicit
        db, _ = _make_db(doc=LLM_DOC)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integration(ORG_ID, LLM_ID)
        assert result is not None
        assert result.get("id") == LLM_ID
        assert "_id" not in result

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(doc=None)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integration(ORG_ID, LLM_ID)
        assert result is None

    def test_invalid_id_returns_none(self):
        db, _ = _make_db()
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integration(ORG_ID, "not-an-objectid")
        assert result is None


# ── TestGetCustomLlmIntegrationForBot ─────────────────────────────────────

class TestGetCustomLlmIntegrationForBot:
    def test_returns_unmasked_api_key(self):
        db, _ = _make_db(doc=LLM_DOC)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integration_for_bot(ORG_ID, LLM_ID)
        assert result is not None
        assert result.get("api_key") == "sk-secret-key"  # unmasked

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(doc=None)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integration_for_bot(ORG_ID, LLM_ID)
        assert result is None


# ── TestGetCustomLlmIntegrationsByOrg ────────────────────────────────────

class TestGetCustomLlmIntegrationsByOrg:
    def test_returns_list_of_docs(self):
        coll = MagicMock()
        coll.find.return_value.sort.return_value = [LLM_DOC]
        db = make_mock_db(CustomLLMIntegrations=coll)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integrations_by_org(ORG_ID)
        assert len(result) == 1
        assert result[0].get("id") == LLM_ID

    def test_exception_returns_empty_list(self):
        coll = MagicMock()
        coll.find.side_effect = Exception("DB error")
        db = make_mock_db(CustomLLMIntegrations=coll)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = get_custom_llm_integrations_by_org(ORG_ID)
        assert result == []


# ── TestUpdateCustomLlmIntegration ────────────────────────────────────────

class TestUpdateCustomLlmIntegration:
    def test_success_returns_updated_doc(self):
        db, coll = _make_db(doc=LLM_DOC)
        updated = {**LLM_DOC, "name": "Updated LLM"}
        # update_custom_llm_integration calls find_one (existence check) then
        # update_one then find_one again to return the updated doc.
        coll.find_one.side_effect = [LLM_DOC, updated]
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = update_custom_llm_integration(ORG_ID, LLM_ID, UPDATE_DATA)
        assert result["status"] == "success"
        assert "integration" in result
        assert result["integration"]["name"] == "Updated LLM"
        assert result["integration"]["id"] == LLM_ID

    def test_not_found_returns_fail(self):
        # find_one returns None → existence check fails → not found
        db, _ = _make_db(doc=None)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = update_custom_llm_integration(ORG_ID, LLM_ID, UPDATE_DATA)
        assert result["status"] == "fail"
        assert ERR_LLM_NOT_FOUND in result["message"]

    def test_no_fields_to_update_returns_fail(self):
        # find_one returns doc (passes existence), but no update fields provided
        empty_update = CustomLLMIntegrationUpdate()
        db, _ = _make_db(doc=LLM_DOC)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = update_custom_llm_integration(ORG_ID, LLM_ID, empty_update)
        assert result["status"] == "fail"
        assert ERR_NO_FIELDS in result["message"]

    def test_invalid_id_returns_fail(self):
        db, _ = _make_db()
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = update_custom_llm_integration(ORG_ID, "bad-id", UPDATE_DATA)
        assert result["status"] == "fail"


# ── TestDeleteCustomLlmIntegration ────────────────────────────────────────

class TestDeleteCustomLlmIntegration:
    def test_success_returns_success_dict(self):
        coll = MagicMock()
        coll.delete_one.return_value.deleted_count = 1
        db = make_mock_db(CustomLLMIntegrations=coll)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = delete_custom_llm_integration(ORG_ID, LLM_ID)
        assert result["status"] == "success"

    def test_not_found_returns_fail(self):
        coll = MagicMock()
        coll.delete_one.return_value.deleted_count = 0
        db = make_mock_db(CustomLLMIntegrations=coll)
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = delete_custom_llm_integration(ORG_ID, LLM_ID)
        assert result["status"] == "fail"
        assert ERR_LLM_NOT_FOUND in result["message"]

    def test_invalid_id_returns_fail(self):
        db, _ = _make_db()
        with patch("app.services.custom_llm_integration_service.get_database", return_value=db):
            result = delete_custom_llm_integration(ORG_ID, "bad-id")
        assert result["status"] == "fail"
