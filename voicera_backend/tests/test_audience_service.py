"""
Unit tests for app.services.audience_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.audience_service import (
    create_audience,
    get_audience_by_name,
    get_all_audiences,
)
from app.models.schemas import AudienceCreate

# ── Sample data ───────────────────────────────────────────────────────────────

AUDIENCE_DOC = {
    "audience_name": "VIP Customers",
    "phone_number": "+15550001234",
    "parameters": {"tier": "gold"},
}

CREATE_DATA = AudienceCreate(
    audience_name="VIP Customers",
    phone_number="+15550001234",
    parameters={"tier": "gold"},
)


def _make_db(doc=None):
    coll = MagicMock()
    coll.find_one.return_value = doc
    db = MagicMock()
    db.__getitem__.side_effect = lambda k: coll if k == "Audience" else MagicMock()
    return db, coll


# ── TestCreateAudience ────────────────────────────────────────────────────

class TestCreateAudience:
    def test_success_returns_success_dict(self):
        db, coll = _make_db(doc=None)
        with patch("app.services.audience_service.get_database", return_value=db):
            result = create_audience(CREATE_DATA)
        assert result["status"] == "success"
        coll.insert_one.assert_called_once()

    def test_duplicate_name_returns_fail(self):
        db, _ = _make_db(doc=AUDIENCE_DOC)
        with patch("app.services.audience_service.get_database", return_value=db):
            result = create_audience(CREATE_DATA)
        assert result["status"] == "fail"
        assert "already exists" in result["message"]

    def test_exception_returns_fail(self):
        db, coll = _make_db(doc=None)
        coll.insert_one.side_effect = Exception("DB error")
        with patch("app.services.audience_service.get_database", return_value=db):
            result = create_audience(CREATE_DATA)
        assert result["status"] == "fail"


# ── TestGetAudienceByName ─────────────────────────────────────────────────

class TestGetAudienceByName:
    def test_returns_doc_when_found(self):
        db, _ = _make_db(doc=AUDIENCE_DOC)
        with patch("app.services.audience_service.get_database", return_value=db):
            result = get_audience_by_name("VIP Customers")
        assert result["audience_name"] == "VIP Customers"

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(doc=None)
        with patch("app.services.audience_service.get_database", return_value=db):
            result = get_audience_by_name("Unknown Audience")
        assert result is None

    def test_returns_none_on_exception(self):
        db, coll = _make_db()
        coll.find_one.side_effect = Exception("DB error")
        with patch("app.services.audience_service.get_database", return_value=db):
            result = get_audience_by_name("VIP Customers")
        assert result is None


# ── TestGetAllAudiences ───────────────────────────────────────────────────

class TestGetAllAudiences:
    def test_returns_all_audiences(self):
        coll = MagicMock()
        coll.find.return_value = [AUDIENCE_DOC]
        db = MagicMock()
        db.__getitem__.side_effect = lambda k: coll if k == "Audience" else MagicMock()
        with patch("app.services.audience_service.get_database", return_value=db):
            result = get_all_audiences()
        assert len(result) == 1

    def test_filters_by_phone_number_when_provided(self):
        coll = MagicMock()
        coll.find.return_value = [AUDIENCE_DOC]
        db = MagicMock()
        db.__getitem__.side_effect = lambda k: coll if k == "Audience" else MagicMock()
        with patch("app.services.audience_service.get_database", return_value=db):
            get_all_audiences(phone_number="+15550001234")
        query = coll.find.call_args[0][0]
        assert query.get("phone_number") == "+15550001234"

    def test_exception_returns_empty_list(self):
        coll = MagicMock()
        coll.find.side_effect = Exception("DB error")
        db = MagicMock()
        db.__getitem__.side_effect = lambda k: coll if k == "Audience" else MagicMock()
        with patch("app.services.audience_service.get_database", return_value=db):
            result = get_all_audiences()
        assert result == []
