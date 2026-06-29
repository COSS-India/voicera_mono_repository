"""
Integration tests for /api/v1/campaigns endpoints.
"""

import pytest
from unittest.mock import patch

BASE = "/api/v1/campaigns"

CAMPAIGN = {
    "campaign_name": "Q1 Outreach",
    "org_id": "testorg1",
    "agent_type": "sales_bot",
    "status": "active",
}

CAMPAIGN_DOC = {**CAMPAIGN, "campaign_information": None}


# ── POST / ────────────────────────────────────────────────────────────────

class TestCreateCampaign:
    def test_success_returns_201(self, client):
        ok = {"status": "success", "message": "Campaign created successfully"}
        with patch("app.services.campaign_service.create_campaign", return_value=ok):
            resp = client.post(BASE, json=CAMPAIGN)
        assert resp.status_code == 201

    def test_org_id_injected_from_token_when_missing(self, client):
        """Router sets org_id from JWT when not provided in body."""
        ok = {"status": "success", "message": "Created"}
        with patch("app.services.campaign_service.create_campaign", return_value=ok) as mock_svc:
            resp = client.post(BASE, json={"campaign_name": "test"})
        assert resp.status_code == 201
        call_arg = mock_svc.call_args[0][0]
        assert call_arg.org_id == "testorg1"

    def test_wrong_org_returns_403(self, client):
        resp = client.post(BASE, json={**CAMPAIGN, "org_id": "otherorg9"})
        assert resp.status_code == 403

    def test_duplicate_campaign_returns_400(self, client):
        error = {"status": "fail", "message": "Campaign with this name already exists"}
        with patch("app.services.campaign_service.create_campaign", return_value=error):
            resp = client.post(BASE, json=CAMPAIGN)
        assert resp.status_code == 400

    def test_missing_campaign_name_returns_422(self, client):
        resp = client.post(BASE, json={"org_id": "testorg1"})
        assert resp.status_code == 422


# ── GET /org/{org_id} ──────────────────────────────────────────────────────

class TestGetCampaignsByOrg:
    def test_success_returns_list(self, client):
        with patch("app.services.campaign_service.get_all_campaigns", return_value=[CAMPAIGN_DOC]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert resp.json()[0]["campaign_name"] == "Q1 Outreach"

    def test_empty_list_returned(self, client):
        with patch("app.services.campaign_service.get_all_campaigns", return_value=[]):
            resp = client.get(f"{BASE}/org/testorg1")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_wrong_org_returns_403(self, client):
        resp = client.get(f"{BASE}/org/otherorg9")
        assert resp.status_code == 403


# ── GET /{campaign_name} ───────────────────────────────────────────────────

class TestGetCampaignByName:
    def test_success(self, client):
        with patch("app.services.campaign_service.get_campaign_by_name", return_value=CAMPAIGN_DOC):
            resp = client.get(f"{BASE}/Q1 Outreach")
        assert resp.status_code == 200
        assert resp.json()["campaign_name"] == "Q1 Outreach"

    def test_not_found_returns_404(self, client):
        with patch("app.services.campaign_service.get_campaign_by_name", return_value=None):
            resp = client.get(f"{BASE}/ghost-campaign")
        assert resp.status_code == 404

    def test_wrong_org_returns_403(self, client):
        wrong_org = {**CAMPAIGN_DOC, "org_id": "otherorg9"}
        with patch("app.services.campaign_service.get_campaign_by_name", return_value=wrong_org):
            resp = client.get(f"{BASE}/Q1 Outreach")
        assert resp.status_code == 403
