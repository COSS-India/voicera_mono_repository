"""
Integration tests for /api/v1/integrations endpoints.
"""
import pytest
from unittest.mock import patch

BASE = "/api/v1/integrations"

INTEGRATION = {
    "org_id": "testorg1",
    "model": "OpenAI",
    "api_key": "sk-test-key",
}

INTEGRATION_DOC = {
    **INTEGRATION,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

BOT_REQUEST = {"org_id": "testorg1", "model": "OpenAI"}


# ── POST /integrations/bot/get-api-key ────────────────────────────────────

class TestBotGetApiKey:
    def test_success_returns_integration(self, client):
        with patch("app.services.integration_service.get_integration",
                   return_value=INTEGRATION_DOC):
            resp = client.post(f"{BASE}/bot/get-api-key", json=BOT_REQUEST)
        assert resp.status_code == 200
        assert resp.json()["model"] == "OpenAI"

    def test_not_found_returns_404(self, client):
        with patch("app.services.integration_service.get_integration", return_value=None):
            resp = client.post(f"{BASE}/bot/get-api-key", json=BOT_REQUEST)
        assert resp.status_code == 404

    def test_missing_api_key_returns_401(self, unauth_client):
        resp = unauth_client.post(f"{BASE}/bot/get-api-key", json=BOT_REQUEST)
        assert resp.status_code == 401


# ── POST /integrations ────────────────────────────────────────────────────

class TestCreateIntegration:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "message": "Integration created successfully"}
        with patch("app.services.integration_service.create_integration", return_value=ok):
            resp = client.post(BASE, json=INTEGRATION)
        assert resp.status_code == 201

    def test_wrong_org_returns_403(self, client):
        resp = client.post(BASE, json={**INTEGRATION, "org_id": "otherorg9"})
        assert resp.status_code == 403

    def test_service_fail_returns_400(self, client):
        error = {"status": "fail", "message": "DB error"}
        with patch("app.services.integration_service.create_integration", return_value=error):
            resp = client.post(BASE, json=INTEGRATION)
        assert resp.status_code == 400


# ── GET /integrations/{model} ─────────────────────────────────────────────

class TestGetIntegrationByModel:
    def test_success_returns_200(self, client):
        with patch("app.services.integration_service.get_integration",
                   return_value=INTEGRATION_DOC):
            resp = client.get(f"{BASE}/OpenAI")
        assert resp.status_code == 200
        assert resp.json()["model"] == "OpenAI"

    def test_not_found_returns_404(self, client):
        with patch("app.services.integration_service.get_integration", return_value=None):
            resp = client.get(f"{BASE}/Ghost")
        assert resp.status_code == 404


# ── GET /integrations ─────────────────────────────────────────────────────

class TestGetAllIntegrations:
    def test_success_returns_list(self, client):
        with patch("app.services.integration_service.get_integrations_by_org",
                   return_value=[INTEGRATION_DOC]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_empty_list_returns_200(self, client):
        with patch("app.services.integration_service.get_integrations_by_org",
                   return_value=[]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []


# ── DELETE /integrations/{model} ─────────────────────────────────────────

class TestDeleteIntegration:
    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Integration deleted"}
        with patch("app.services.integration_service.delete_integration", return_value=ok):
            resp = client.delete(f"{BASE}/OpenAI")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        # Router calls get_integration first; returns 404 if None
        with patch("app.services.integration_service.get_integration", return_value=None):
            resp = client.delete(f"{BASE}/Ghost")
        assert resp.status_code == 404
