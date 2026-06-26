"""Tests for app/services/campaign_service.py"""

import pytest
from unittest.mock import patch, MagicMock

from app.services import campaign_service
from app.models.schemas import CampaignCreate
from tests.helpers import make_mock_db


CAMPAIGN_DATA = CampaignCreate(
    campaign_name="Test Campaign",
    org_id="testorg1",
    agent_type="sales_bot",
    status="active",
)


class TestCreateCampaign:
    def _make_db(self, existing=None):
        coll = MagicMock()
        coll.find_one.return_value = existing
        return make_mock_db(Campaigns=coll), coll

    def test_success(self):
        db, coll = self._make_db(existing=None)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.create_campaign(CAMPAIGN_DATA)
        assert result["status"] == "success"
        coll.insert_one.assert_called_once()

    def test_campaign_already_exists_returns_fail(self):
        existing = {"campaign_name": "Test Campaign"}
        db, coll = self._make_db(existing=existing)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.create_campaign(CAMPAIGN_DATA)
        assert result["status"] == "fail"
        assert "already exists" in result["message"]
        coll.insert_one.assert_not_called()

    def test_db_exception_returns_fail(self):
        db = MagicMock()
        db.__getitem__.side_effect = Exception("DB error")
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.create_campaign(CAMPAIGN_DATA)
        assert result["status"] == "fail"
        assert "DB error" in result["message"]

    def test_no_optional_fields(self):
        data = CampaignCreate(campaign_name="MinCampaign")
        coll = MagicMock()
        coll.find_one.return_value = None
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.create_campaign(data)
        assert result["status"] == "success"

    def test_campaign_information_included_when_set(self):
        data = CampaignCreate(
            campaign_name="InfoCampaign",
            org_id="testorg1",
            campaign_information={"goal": "outreach", "priority": 1},
        )
        coll = MagicMock()
        coll.find_one.return_value = None
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.create_campaign(data)
        assert result["status"] == "success"


class TestGetAllCampaigns:
    def test_returns_campaigns(self):
        docs = [{"campaign_name": "Camp1"}, {"campaign_name": "Camp2"}]
        coll = MagicMock()
        coll.find.return_value = iter(docs)
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_all_campaigns("testorg1")
        assert len(result) == 2

    def test_empty_returns_empty_list(self):
        coll = MagicMock()
        coll.find.return_value = iter([])
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_all_campaigns("testorg1")
        assert result == []

    def test_exception_returns_empty_list(self):
        db = MagicMock()
        db.__getitem__.side_effect = Exception("fail")
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_all_campaigns("testorg1")
        assert result == []


class TestGetCampaignByName:
    def test_found(self):
        doc = {"campaign_name": "MyCamp"}
        coll = MagicMock()
        coll.find_one.return_value = doc
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_campaign_by_name("MyCamp")
        assert result == doc

    def test_not_found_returns_none(self):
        coll = MagicMock()
        coll.find_one.return_value = None
        db = make_mock_db(Campaigns=coll)
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_campaign_by_name("ghost")
        assert result is None

    def test_exception_returns_none(self):
        db = MagicMock()
        db.__getitem__.side_effect = Exception("fail")
        with patch("app.services.campaign_service.get_database", return_value=db):
            result = campaign_service.get_campaign_by_name("any")
        assert result is None
