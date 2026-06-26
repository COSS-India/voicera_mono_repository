import pytest
from unittest.mock import patch, MagicMock
from app.services import phone_number

def test_last_link_fields():
    # Test with member_email
    fields = phone_number._last_link_fields("attached", "sales_bot", "user@test.com", "2024-01-01T00:00:00")
    assert fields == {
        "last_link_action": "attached",
        "last_link_agent_type": "sales_bot",
        "last_link_by_email": "user@test.com",
        "last_link_at": "2024-01-01T00:00:00",
    }
    
    # Test with no member_email
    fields = phone_number._last_link_fields("attached", "sales_bot", None, "2024-01-01T00:00:00")
    assert fields == {}
    
    # Test with no agent_type
    fields = phone_number._last_link_fields("attached", None, "user@test.com", "2024-01-01T00:00:00")
    assert fields["last_link_agent_type"] == ""


class TestGetAllPhoneNumbersByOrg:
    @patch("app.services.phone_number.get_database")
    def test_success(self, mock_get_db):
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.find.return_value = [{"phone_number": "123"}]
        mock_db.__getitem__.return_value = mock_table
        mock_get_db.return_value = mock_db
        
        result = phone_number.get_all_phone_numbers_by_org("org1")
        assert result == [{"phone_number": "123"}]
        mock_table.find.assert_called_once_with({"org_id": "org1"})

    @patch("app.services.phone_number.get_database")
    def test_exception_returns_empty_list(self, mock_get_db):
        mock_get_db.side_effect = Exception("DB Error")
        result = phone_number.get_all_phone_numbers_by_org("org1")
        assert result == []


class TestAttachPhoneNumberToAgent:
    @patch("app.services.phone_number.get_database")
    @patch("app.services.phone_number.agent_service.fetch_agent_config")
    def test_agent_not_found(self, mock_fetch, mock_get_db):
        mock_fetch.return_value = None
        result = phone_number.attach_phone_number_to_agent("123", "vobiz", agent_type="sales_bot")
        assert result["status"] == "fail"
        assert "not found" in result["message"].lower()

    @patch("app.services.phone_number.get_database")
    @patch("app.services.phone_number.agent_service.fetch_agent_config")
    def test_agent_has_no_org_id(self, mock_fetch, mock_get_db):
        mock_fetch.return_value = {"agent_type": "sales_bot"} # No org_id
        result = phone_number.attach_phone_number_to_agent("123", "vobiz", agent_type="sales_bot")
        assert result["status"] == "fail"
        assert "org_id" in result["message"]

    @patch("app.services.phone_number.get_database")
    def test_no_agent_no_org_returns_fail(self, mock_get_db):
        result = phone_number.attach_phone_number_to_agent("123", "vobiz")
        assert result["status"] == "fail"
        assert "Either agent_type or org_id" in result["message"]

    @patch("app.services.phone_number.get_database")
    @patch("app.services.phone_number.agent_service.fetch_agent_config")
    def test_update_existing_phone_number(self, mock_fetch, mock_get_db):
        mock_fetch.return_value = {"org_id": "org1"}
        mock_db = MagicMock()
        mock_phone_table = MagicMock()
        mock_agent_table = MagicMock()
        
        def mock_getitem(key):
            if key == "PhoneNumber": return mock_phone_table
            if key == "AgentConfig": return mock_agent_table
        mock_db.__getitem__.side_effect = mock_getitem
        mock_get_db.return_value = mock_db
        
        mock_phone_table.find_one.return_value = {"phone_number": "123"}
        mock_result = MagicMock()
        mock_result.modified_count = 1
        mock_phone_table.update_one.return_value = mock_result
        
        result = phone_number.attach_phone_number_to_agent("123", "vobiz", agent_type="sales_bot")
        assert result["status"] == "success"
        
        # Test no modifications
        mock_result.modified_count = 0
        result2 = phone_number.attach_phone_number_to_agent("123", "vobiz", agent_type="sales_bot")
        assert result2["status"] == "success"
        assert "already configured" in result2["message"].lower()

    @patch("app.services.phone_number.get_database")
    @patch("app.services.phone_number.agent_service.fetch_agent_config")
    def test_insert_new_phone_number(self, mock_fetch, mock_get_db):
        mock_fetch.return_value = {"org_id": "org1"}
        mock_db = MagicMock()
        mock_phone_table = MagicMock()
        mock_agent_table = MagicMock()
        
        def mock_getitem(key):
            if key == "PhoneNumber": return mock_phone_table
            if key == "AgentConfig": return mock_agent_table
        mock_db.__getitem__.side_effect = mock_getitem
        mock_get_db.return_value = mock_db
        
        mock_phone_table.find_one.return_value = None
        
        result = phone_number.attach_phone_number_to_agent("123", "vobiz", org_id="org1")
        assert result["status"] == "success"
        mock_phone_table.insert_one.assert_called_once()

    @patch("app.services.phone_number.get_database")
    def test_exception_returns_fail(self, mock_get_db):
        mock_get_db.side_effect = Exception("DB Crash")
        result = phone_number.attach_phone_number_to_agent("123", "vobiz", org_id="org1")
        assert result["status"] == "fail"
        assert "DB Crash" in result["message"]


class TestGetPhoneNumberByAgentType:
    @patch("app.services.phone_number.get_database")
    def test_success(self, mock_get_db):
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.find_one.return_value = {"phone_number": "123"}
        mock_db.__getitem__.return_value = mock_table
        mock_get_db.return_value = mock_db
        
        result = phone_number.get_phone_number_by_agent_type("sales_bot", "org1")
        assert result == {"phone_number": "123"}

    @patch("app.services.phone_number.get_database")
    def test_exception_returns_none(self, mock_get_db):
        mock_get_db.side_effect = Exception("Crash")
        result = phone_number.get_phone_number_by_agent_type("sales_bot", "org1")
        assert result is None


class TestDetachPhoneNumber:
    @patch("app.services.phone_number.get_database")
    def test_phone_not_found(self, mock_get_db):
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.find_one.return_value = None
        mock_db.__getitem__.return_value = mock_table
        mock_get_db.return_value = mock_db
        
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "fail"
        assert "not found" in result["message"]

    @patch("app.services.phone_number.get_database")
    def test_unauthorized_org(self, mock_get_db):
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.find_one.return_value = {"org_id": "other_org"}
        mock_db.__getitem__.return_value = mock_table
        mock_get_db.return_value = mock_db
        
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "fail"
        assert "authorized" in result["message"]

    @patch("app.services.phone_number.get_database")
    def test_no_agent_type(self, mock_get_db):
        mock_db = MagicMock()
        mock_table = MagicMock()
        mock_table.find_one.return_value = {"org_id": "org1"} # No agent_type
        mock_db.__getitem__.return_value = mock_table
        mock_get_db.return_value = mock_db
        
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "fail"
        assert "not attached" in result["message"]

    @patch("app.services.phone_number.get_database")
    def test_success_with_modifications(self, mock_get_db):
        mock_db = MagicMock()
        mock_phone_table = MagicMock()
        mock_agent_table = MagicMock()
        
        def mock_getitem(key):
            if key == "PhoneNumber": return mock_phone_table
            if key == "AgentConfig": return mock_agent_table
        mock_db.__getitem__.side_effect = mock_getitem
        mock_get_db.return_value = mock_db
        
        mock_phone_table.find_one.return_value = {"org_id": "org1", "agent_type": "sales_bot"}
        mock_result = MagicMock()
        mock_result.modified_count = 1
        mock_phone_table.update_one.return_value = mock_result
        
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "success"

    @patch("app.services.phone_number.get_database")
    def test_success_with_no_modifications(self, mock_get_db):
        mock_db = MagicMock()
        mock_phone_table = MagicMock()
        mock_agent_table = MagicMock()
        
        def mock_getitem(key):
            if key == "PhoneNumber": return mock_phone_table
            if key == "AgentConfig": return mock_agent_table
        mock_db.__getitem__.side_effect = mock_getitem
        mock_get_db.return_value = mock_db
        
        mock_phone_table.find_one.return_value = {"org_id": "org1", "agent_type": "sales_bot"}
        mock_result = MagicMock()
        mock_result.modified_count = 0
        mock_phone_table.update_one.return_value = mock_result
        
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "fail"
        assert "Failed to detach" in result["message"]

    @patch("app.services.phone_number.get_database")
    def test_exception_returns_fail(self, mock_get_db):
        mock_get_db.side_effect = Exception("DB Down")
        result = phone_number.detach_phone_number("123", "org1")
        assert result["status"] == "fail"
        assert "DB Down" in result["message"]
