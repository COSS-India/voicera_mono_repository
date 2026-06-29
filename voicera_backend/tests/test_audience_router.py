"""
Integration tests for /api/v1/audience endpoints.
"""
import pytest
from unittest.mock import patch

BASE = "/api/v1/audience"

AUDIENCE_DOC = {
    "audience_name": "VIP Customers",
    "phone_number": "+15550001234",
    "parameters": {"tier": "gold"},
}

CREATE_BODY = {
    "audience_name": "VIP Customers",
    "phone_number": "+15550001234",
    "parameters": {"tier": "gold"},
}


# ── POST /audience ────────────────────────────────────────────────────────

class TestCreateAudience:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "message": "Audience created"}
        with patch("app.services.audience_service.create_audience", return_value=ok):
            resp = client.post(BASE, json=CREATE_BODY)
        assert resp.status_code == 201

    def test_duplicate_name_returns_400(self, client):
        fail = {"status": "fail", "message": "Audience with this name already exists"}
        with patch("app.services.audience_service.create_audience", return_value=fail):
            resp = client.post(BASE, json=CREATE_BODY)
        assert resp.status_code == 400

    def test_missing_required_fields_returns_422(self, client):
        resp = client.post(BASE, json={"audience_name": "VIP"})
        assert resp.status_code == 422


# ── GET /audience/{audience_name} ─────────────────────────────────────────

class TestGetAudience:
    def test_success_returns_200(self, client):
        with patch("app.services.audience_service.get_audience_by_name",
                   return_value=AUDIENCE_DOC):
            resp = client.get(f"{BASE}/VIP Customers")
        assert resp.status_code == 200
        assert resp.json()["audience_name"] == "VIP Customers"

    def test_not_found_returns_404(self, client):
        with patch("app.services.audience_service.get_audience_by_name", return_value=None):
            resp = client.get(f"{BASE}/Unknown Audience")
        assert resp.status_code == 404


# ── GET /audience ─────────────────────────────────────────────────────────

class TestGetAllAudiences:
    def test_success_returns_list(self, client):
        with patch("app.services.audience_service.get_all_audiences",
                   return_value=[AUDIENCE_DOC]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)

    def test_phone_number_filter_forwarded(self, client):
        with patch("app.services.audience_service.get_all_audiences",
                   return_value=[]) as mock_svc:
            client.get(f"{BASE}?phone_number=%2B15550001234")
        # phone_number may be passed positionally or as kwarg depending on router impl
        args, kwargs = mock_svc.call_args
        all_values = list(args) + list(kwargs.values())
        assert "+15550001234" in all_values

    def test_empty_list_returns_200(self, client):
        with patch("app.services.audience_service.get_all_audiences", return_value=[]):
            resp = client.get(BASE)
        assert resp.status_code == 200
        assert resp.json() == []
