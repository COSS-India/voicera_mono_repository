"""
Integration tests for /api/v1/phone-numbers endpoints.
"""
import json
import pytest
from unittest.mock import patch

BASE = "/api/v1/phone-numbers"

PHONE_DOC = {
    "phone_number": "+15550001234",
    "provider": "plivo",
    "agent_type": "sales_bot",
    "org_id": "testorg1",
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
    "last_link_action": "attached",
    "last_link_agent_type": "sales_bot",
    "last_link_by_email": "owner@example.com",
    "last_link_at": "2024-01-01T00:00:00",
}

AGENT_DOC = {
    "agent_type": "sales_bot",
    "org_id": "testorg1",
    "agent_id": "agent-001",
}

ATTACH_BODY = {
    "phone_number": "+15550001234",
    "provider": "plivo",
    "agent_type": "sales_bot",
}

DETACH_BODY = {"phone_number": "+15550001234"}


# ── GET /phone-numbers/org/{org_id} ──────────────────────────────────────

class TestGetAllPhoneNumbersByOrg:
    def test_success_returns_list(self, client):
        with patch("app.services.phone_number.get_all_phone_numbers_by_org",
                   return_value=[PHONE_DOC]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_wrong_org_returns_403(self, client):
        resp = client.get(f"{BASE}/org/otherorg9")
        assert resp.status_code == 403

    def test_empty_list_returns_200(self, client):
        with patch("app.services.phone_number.get_all_phone_numbers_by_org", return_value=[]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert resp.json() == []


# ── GET /phone-numbers/agent/{agent_type} ────────────────────────────────

class TestGetPhoneNumberByAgentType:
    def test_success_returns_phone_number(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.phone_number.get_phone_number_by_agent_type",
                   return_value=PHONE_DOC):
            resp = client.get(f"{BASE}/agent/sales_bot")
        assert resp.status_code == 200
        assert resp.json()["phone_number"] == "+15550001234"

    def test_agent_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = client.get(f"{BASE}/agent/ghost_bot")
        assert resp.status_code == 404

    def test_agent_wrong_org_returns_403(self, client):
        wrong_org_agent = {**AGENT_DOC, "org_id": "otherorg9"}
        with patch("app.services.agent_service.fetch_agent_config",
                   return_value=wrong_org_agent):
            resp = client.get(f"{BASE}/agent/sales_bot")
        assert resp.status_code == 403

    def test_no_phone_number_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.phone_number.get_phone_number_by_agent_type",
                   return_value=None):
            resp = client.get(f"{BASE}/agent/sales_bot")
        assert resp.status_code == 404


# ── POST /phone-numbers/attach ────────────────────────────────────────────

class TestAttachPhoneNumber:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "message": "Phone number attached"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.phone_number.attach_phone_number_to_agent", return_value=ok):
            resp = client.post(f"{BASE}/attach", json=ATTACH_BODY)
        assert resp.status_code == 201

    def test_agent_not_found_returns_404(self, client):
        with patch("app.services.agent_service.fetch_agent_config", return_value=None):
            resp = client.post(f"{BASE}/attach", json=ATTACH_BODY)
        assert resp.status_code == 404

    def test_agent_wrong_org_returns_403(self, client):
        wrong_org = {**AGENT_DOC, "org_id": "otherorg9"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=wrong_org):
            resp = client.post(f"{BASE}/attach", json=ATTACH_BODY)
        assert resp.status_code == 403

    def test_service_fail_returns_400(self, client):
        fail = {"status": "fail", "message": "Phone number already attached"}
        with patch("app.services.agent_service.fetch_agent_config", return_value=AGENT_DOC), \
             patch("app.services.phone_number.attach_phone_number_to_agent", return_value=fail):
            resp = client.post(f"{BASE}/attach", json=ATTACH_BODY)
        assert resp.status_code == 400


# ── DELETE /phone-numbers/detach ─────────────────────────────────────────

class TestDetachPhoneNumber:
    def _delete_with_body(self, client, body):
        # httpx 0.28 DELETE does not accept json=; pass as raw content with content-type
        return client.request(
            "DELETE",
            f"{BASE}/detach",
            content=json.dumps(body),
            headers={"Content-Type": "application/json"},
        )

    def test_success_returns_200(self, client):
        ok = {"status": "success", "message": "Phone number detached"}
        with patch("app.services.phone_number.detach_phone_number", return_value=ok):
            resp = self._delete_with_body(client, DETACH_BODY)
        assert resp.status_code == 200

    def test_not_found_returns_404(self, client):
        fail = {"status": "fail", "message": "Phone number not found"}
        with patch("app.services.phone_number.detach_phone_number", return_value=fail):
            resp = self._delete_with_body(client, DETACH_BODY)
        assert resp.status_code == 404

    def test_service_fail_returns_400(self, client):
        fail = {"status": "fail", "message": "Already detached"}
        with patch("app.services.phone_number.detach_phone_number", return_value=fail):
            resp = self._delete_with_body(client, {"phone_number": "+19999999999"})
        assert resp.status_code == 400
