"""
Integration tests for /api/v1/agents endpoints.

Tests cover both:
  - Bot endpoints (X-API-Key authentication, covered by the `client` fixture's
    verify_api_key override)
  - Frontend endpoints (JWT authentication, covered by get_current_user override)

Org isolation (403 when org_id mismatches) is tested explicitly.
"""

import pytest
from unittest.mock import patch

BASE = "/api/v1/agents"

AGENT = {
    "agent_type": "sales_bot",
    "agent_id": "agent-001",
    "agent_config": {"prompt": "You are a sales assistant."},
    "org_id": "testorg1",
}

AGENT_DOC = {
    **AGENT,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}


# ── Bot endpoints ──────────────────────────────────────────────────────────

class TestBotGetAgentConfig:
    def test_get_by_agent_type_success(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC):
            resp = client.get(f"{BASE}/config/sales_bot")
        assert resp.status_code == 200
        assert resp.json()["agent_type"] == "sales_bot"

    def test_get_by_agent_type_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = client.get(f"{BASE}/config/ghost_agent")
        assert resp.status_code == 404

    def test_get_by_agent_id_success(self, client):
        with patch("app.services.agent_service.fetch_agent_config_by_id", return_value=AGENT_DOC):
            resp = client.get(f"{BASE}/config/id/agent-001")
        assert resp.status_code == 200
        assert resp.json()["agent_id"] == "agent-001"

    def test_get_by_agent_id_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config_by_id", return_value=None):
            resp = client.get(f"{BASE}/config/id/missing")
        assert resp.status_code == 404

    def test_get_by_phone_number_success(self, client):
        with patch("app.services.agent_service.fetch_agent_by_phone_number", return_value=AGENT_DOC):
            resp = client.get(f"{BASE}/by-phone/+911234567890")
        assert resp.status_code == 200

    def test_get_by_phone_number_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_by_phone_number", return_value=None):
            resp = client.get(f"{BASE}/by-phone/+910000000000")
        assert resp.status_code == 404


class TestBotAuthEnforcement:
    def test_missing_api_key_returns_401(self, unauth_client):
        resp = unauth_client.get(f"{BASE}/config/any_agent")
        assert resp.status_code == 401

    def test_wrong_api_key_returns_401(self, unauth_client):
        resp = unauth_client.get(
            f"{BASE}/config/any_agent", headers={"X-API-Key": "wrong-key"}
        )
        assert resp.status_code == 401


# ── Frontend: create agent ─────────────────────────────────────────────────

class TestCreateAgent:
    def test_create_success_returns_201(self, client):
        ok = {"status": "success", "message": "Agent type created successfully"}
        with patch("app.services.agent_service.create_agent", return_value=ok):
            resp = client.post(BASE, json=AGENT)
        assert resp.status_code == 201

    def test_create_wrong_org_returns_403(self, client):
        resp = client.post(BASE, json={**AGENT, "org_id": "otherorg9"})
        assert resp.status_code == 403

    def test_create_duplicate_agent_returns_400(self, client):
        error = {"status": "fail", "message": "Agent type already exists for this organization"}
        with patch("app.services.agent_service.create_agent", return_value=error):
            resp = client.post(BASE, json=AGENT)
        assert resp.status_code == 400

    def test_create_missing_required_field_returns_422(self, client):
        resp = client.post(BASE, json={"agent_type": "x", "org_id": "testorg1"})
        assert resp.status_code == 422


# ── Frontend: get agents by org ────────────────────────────────────────────

class TestGetAgentsByOrg:
    def test_get_agents_success(self, client):
        with patch("app.services.agent_service.fetch_agents_of_org", return_value=[AGENT_DOC]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) == 1

    def test_get_agents_returns_empty_list(self, client):
        with patch("app.services.agent_service.fetch_agents_of_org", return_value=[]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_get_agents_wrong_org_returns_403(self, client):
        resp = client.get(f"{BASE}/org/otherorg9")
        assert resp.status_code == 403


# ── Frontend: get, update, delete agent ───────────────────────────────────

class TestGetAgentConfig:
    def test_success(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC):
            resp = client.get(f"{BASE}/sales_bot")
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = client.get(f"{BASE}/ghost")
        assert resp.status_code == 404

    def test_wrong_org_returns_403(self, client):
        wrong_org_doc = {**AGENT_DOC, "org_id": "otherorg9"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=wrong_org_doc):
            resp = client.get(f"{BASE}/sales_bot")
        assert resp.status_code == 403


class TestUpdateAgent:
    UPDATE_BODY = {"agent_config": {"prompt": "Updated prompt."}}

    def test_update_success(self, client):
        ok = {"status": "success", "message": "Updated", "agent_type": "sales_bot"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.agent_service.update_agent_config", return_value=ok):
            resp = client.put(f"{BASE}/sales_bot", json=self.UPDATE_BODY)
        assert resp.status_code == 200
        assert resp.json()["status"] == "success"

    def test_update_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = client.put(f"{BASE}/ghost", json=self.UPDATE_BODY)
        assert resp.status_code == 404

    def test_update_wrong_org_returns_403(self, client):
        wrong = {**AGENT_DOC, "org_id": "otherorg9"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=wrong):
            resp = client.put(f"{BASE}/sales_bot", json=self.UPDATE_BODY)
        assert resp.status_code == 403

    def test_update_service_failure_returns_400(self, client):
        error = {"status": "fail", "message": "Duplicate agent type"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.agent_service.update_agent_config", return_value=error):
            resp = client.put(f"{BASE}/sales_bot", json=self.UPDATE_BODY)
        assert resp.status_code == 400


class TestDeleteAgent:
    def test_delete_success(self, client):
        ok = {"status": "success", "message": "Agent deleted successfully"}
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=AGENT_DOC), \
             patch("app.services.agent_service.delete_agent", return_value=ok):
            resp = client.delete(f"{BASE}/sales_bot")
        assert resp.status_code == 200

    def test_delete_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=None):
            resp = client.delete(f"{BASE}/ghost")
        assert resp.status_code == 404

    def test_delete_wrong_org_returns_403(self, client):
        wrong = {**AGENT_DOC, "org_id": "otherorg9"}
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=wrong):
            resp = client.delete(f"{BASE}/sales_bot")
        assert resp.status_code == 403

    def test_delete_service_failure_returns_400(self, client):
        error = {"status": "fail", "message": "Agent type not found"}
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=AGENT_DOC), \
             patch("app.services.agent_service.delete_agent", return_value=error):
            resp = client.delete(f"{BASE}/sales_bot")
        assert resp.status_code == 400


class TestDeleteAgentByQueryParam:
    def test_delete_by_query_success(self, client):
        ok = {"status": "success", "message": "Agent deleted successfully"}
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=AGENT_DOC), \
             patch("app.services.agent_service.delete_agent", return_value=ok):
            resp = client.delete(f"{BASE}?agent_type=sales_bot")
        assert resp.status_code == 200

    def test_delete_by_query_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=None):
            resp = client.delete(f"{BASE}?agent_type=ghost")
        assert resp.status_code == 404

    def test_delete_by_query_service_failure_returns_400(self, client):
        error = {"status": "fail", "message": "Delete failed"}
        with patch("app.services.agent_service.fetch_agent_config_for_org", return_value=AGENT_DOC), \
             patch("app.services.agent_service.delete_agent", return_value=error):
            resp = client.delete(f"{BASE}?agent_type=sales_bot")
        assert resp.status_code == 400

    def test_empty_agent_type_returns_400(self, client):
        resp = client.delete(f"{BASE}?agent_type=")
        assert resp.status_code == 400

    def test_missing_agent_type_returns_422(self, client):
        resp = client.delete(f"{BASE}")
        assert resp.status_code == 422
