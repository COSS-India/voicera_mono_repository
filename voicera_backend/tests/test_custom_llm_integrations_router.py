"""
Integration tests for /api/v1/custom-llm-integrations endpoints.
"""
import pytest
from unittest.mock import patch
from bson import ObjectId

BASE = "/api/v1/custom-llm-integrations"
LLM_ID = str(ObjectId())

LLM_DOC = {
    "id": LLM_ID,
    "org_id": "testorg1",
    "name": "My Custom LLM",
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4",
    "api_key": "****-key",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

CREATE_BODY = {
    "org_id": "testorg1",
    "name": "My Custom LLM",
    "base_url": "https://api.example.com/v1",
    "model": "gpt-4",
    "api_key": "sk-secret",
}

UPDATE_BODY = {"name": "Updated LLM"}
BOT_REQUEST = {"org_id": "testorg1", "custom_llm_id": LLM_ID}


# ── POST /custom-llm-integrations/bot/get-config ─────────────────────────

class TestBotGetConfig:
    def test_success_returns_200(self, client):
        with patch("app.services.custom_llm_integration_service.get_custom_llm_integration_for_bot",
                   return_value=LLM_DOC):
            resp = client.post(f"{BASE}/bot/get-config", json=BOT_REQUEST)
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        with patch("app.services.custom_llm_integration_service.get_custom_llm_integration_for_bot",
                   return_value=None):
            resp = client.post(f"{BASE}/bot/get-config", json=BOT_REQUEST)
        assert resp.status_code == 404

    def test_missing_api_key_returns_401(self, unauth_client):
        resp = unauth_client.post(f"{BASE}/bot/get-config", json=BOT_REQUEST)
        assert resp.status_code == 401


# ── GET /custom-llm-integrations ─────────────────────────────────────────

class TestListCustomLlmIntegrations:
    def test_success_returns_list(self, client):
        with patch("app.services.custom_llm_integration_service.get_custom_llm_integrations_by_org",
                   return_value=[LLM_DOC]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_empty_list_returns_200(self, client):
        with patch("app.services.custom_llm_integration_service.get_custom_llm_integrations_by_org",
                   return_value=[]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []


# ── POST /custom-llm-integrations ────────────────────────────────────────

class TestCreateCustomLlmIntegration:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "integration": LLM_DOC}
        with patch("app.services.custom_llm_integration_service.create_custom_llm_integration",
                   return_value=ok):
            resp = client.post(BASE, json=CREATE_BODY)
        assert resp.status_code == 201

    def test_wrong_org_returns_403(self, client):
        resp = client.post(BASE, json={**CREATE_BODY, "org_id": "otherorg9"})
        assert resp.status_code == 403

    def test_service_fail_returns_400(self, client):
        fail = {"status": "fail", "message": "Invalid URL"}
        with patch("app.services.custom_llm_integration_service.create_custom_llm_integration",
                   return_value=fail):
            resp = client.post(BASE, json=CREATE_BODY)
        assert resp.status_code == 400


# ── PUT /custom-llm-integrations/{id} ────────────────────────────────────

class TestUpdateCustomLlmIntegration:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "integration": LLM_DOC}
        with patch("app.services.custom_llm_integration_service.update_custom_llm_integration",
                   return_value=ok):
            resp = client.put(f"{BASE}/{LLM_ID}", json=UPDATE_BODY)
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        fail = {"status": "fail", "message": "Custom LLM integration not found"}
        with patch("app.services.custom_llm_integration_service.update_custom_llm_integration",
                   return_value=fail):
            resp = client.put(f"{BASE}/{LLM_ID}", json=UPDATE_BODY)
        assert resp.status_code == 404

    def test_bad_url_returns_400(self, client):
        fail = {"status": "fail", "message": "Invalid URL format"}
        with patch("app.services.custom_llm_integration_service.update_custom_llm_integration",
                   return_value=fail):
            resp = client.put(f"{BASE}/{LLM_ID}", json=UPDATE_BODY)
        assert resp.status_code == 400


# ── DELETE /custom-llm-integrations/{id} ─────────────────────────────────

class TestDeleteCustomLlmIntegration:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Deleted"}
        with patch("app.services.custom_llm_integration_service.delete_custom_llm_integration",
                   return_value=ok):
            resp = client.delete(f"{BASE}/{LLM_ID}")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        fail = {"status": "fail", "message": "Custom LLM integration not found"}
        with patch("app.services.custom_llm_integration_service.delete_custom_llm_integration",
                   return_value=fail):
            resp = client.delete(f"{BASE}/{LLM_ID}")
        assert resp.status_code == 404
