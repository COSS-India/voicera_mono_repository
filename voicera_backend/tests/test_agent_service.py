"""
Unit tests for app.services.agent_service.
"""
import pytest
from unittest.mock import MagicMock, patch

from app.services.agent_service import (
    create_agent,
    fetch_agent_config,
    fetch_agent_config_for_org,
    fetch_agent_config_by_id,
    fetch_agents_of_org,
    update_agent_config,
    delete_agent,
    fetch_agent_by_phone_number,
)
from app.models.schemas import AgentConfigCreate, AgentConfigUpdate
from tests.helpers import make_mock_db

# ── Error message constants ───────────────────────────────────────────────────

ERR_AGENT_TYPE_EXISTS = "Agent type already exists"
ERR_AGENT_ID_EXISTS = "Agent ID already exists"
ERR_AGENT_NOT_FOUND = "not found"
ERR_ALREADY_EXISTS = "already exists"

# ── Sample data ───────────────────────────────────────────────────────────────

ORG_ID = "testorg1"

AGENT_CREATE = AgentConfigCreate(
    agent_type="sales_bot",
    agent_id="agent-001",
    agent_config={"prompt": "You are a sales assistant"},
    org_id=ORG_ID,
)

AGENT_DOC = {
    "agent_type": "sales_bot",
    "agent_id": "agent-001",
    "agent_config": {"prompt": "You are a sales assistant"},
    "org_id": ORG_ID,
    "created_at": "2024-01-01T00:00:00",
    "updated_at": "2024-01-01T00:00:00",
}

NON_CONV_CONFIG = {
    "interaction_mode": "non_conversational",
    "greeting_message": "Hello",
    "tts_model": {"name": "en-US"},
}

NON_CONV_AGENT_DOC = {
    **AGENT_DOC,
    "agent_config": NON_CONV_CONFIG,
}

UPDATE_DATA = AgentConfigUpdate(agent_config={"prompt": "Updated prompt"})
RENAME_UPDATE = AgentConfigUpdate(
    agent_type="new_bot", agent_config={"prompt": "Renamed"}
)


def _make_db(agent_doc=None):
    agents_coll = MagicMock()
    agents_coll.find_one.return_value = agent_doc
    db = make_mock_db(AgentConfig=agents_coll)
    return db, agents_coll


# ── TestCreateAgent ───────────────────────────────────────────────────────

class TestCreateAgent:
    def test_success_returns_success_dict(self):
        db, agents_coll = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = create_agent(AGENT_CREATE)
        assert result["status"] == "success"
        agents_coll.insert_one.assert_called_once()
        inserted_doc = agents_coll.insert_one.call_args[0][0]
        assert inserted_doc["org_id"] == ORG_ID
        assert inserted_doc["agent_type"] == AGENT_CREATE.agent_type
        assert inserted_doc["agent_id"] == AGENT_CREATE.agent_id

    def test_duplicate_agent_type_returns_fail(self):
        db, agents_coll = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = create_agent(AGENT_CREATE)
        assert result["status"] == "fail"
        assert ERR_AGENT_TYPE_EXISTS in result["message"]

    def test_duplicate_agent_id_returns_fail(self):
        agents_coll = MagicMock()
        # First find (type) → None, second find (id) → doc
        agents_coll.find_one.side_effect = [None, AGENT_DOC]
        db = make_mock_db(AgentConfig=agents_coll)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = create_agent(AGENT_CREATE)
        assert result["status"] == "fail"
        assert ERR_AGENT_ID_EXISTS in result["message"]

    def test_default_interaction_mode_is_conversational(self):
        db, agents_coll = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            create_agent(AGENT_CREATE)
        inserted = agents_coll.insert_one.call_args[0][0]
        assert inserted.get("agent_config", {}).get("interaction_mode") == "conversational"

    def test_greeting_message_punctuation_stripped(self):
        # greeting_message in agent_config passes _validate_agent_config_for_mode;
        # agent_data.greeting_message (top-level) is what gets punctuation-stripped
        # and stored back into agent_doc["agent_config"]["greeting_message"].
        data = AgentConfigCreate(
            agent_type="bot_a",
            agent_id="a-001",
            agent_config={
                "interaction_mode": "non_conversational",
                "tts_model": {"name": "en-US"},
                "greeting_message": "Hello, how are you!",  # passes validation
            },
            org_id=ORG_ID,
            greeting_message="Hello, how are you!",  # stripped and stored back
        )
        db, agents_coll = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            create_agent(data)
        inserted = agents_coll.insert_one.call_args[0][0]
        greeting = inserted.get("agent_config", {}).get("greeting_message", "")
        assert "!" not in greeting
        assert "," not in greeting

    def test_non_conversational_missing_greeting_returns_fail(self):
        data = AgentConfigCreate(
            agent_type="bot_b",
            agent_id="b-001",
            agent_config={
                "interaction_mode": "non_conversational",
                "tts_model": {"name": "en-US"},
            },
            org_id=ORG_ID,
        )
        db, agents_coll = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = create_agent(data)
        assert result["status"] == "fail"
        assert "Alert message" in result["message"]

    def test_non_conversational_missing_tts_returns_fail(self):
        data = AgentConfigCreate(
            agent_type="bot_c",
            agent_id="c-001",
            agent_config={
                "interaction_mode": "non_conversational",
                "greeting_message": "Hello",
            },
            org_id=ORG_ID,
        )
        db, agents_coll = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = create_agent(data)
        assert result["status"] == "fail"
        assert "TTS" in result["message"]


# ── TestFetchAgentConfig ───────────────────────────────────────────────────

class TestFetchAgentConfig:
    def test_returns_doc(self):
        db, _ = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_config("sales_bot")
        assert result["agent_type"] == "sales_bot"

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_config("ghost_bot")
        assert result is None

    def test_returns_none_on_exception(self):
        db, agents_coll = _make_db()
        agents_coll.find_one.side_effect = Exception("DB error")
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_config("sales_bot")
        assert result is None


# ── TestFetchAgentConfigForOrg ────────────────────────────────────────────

class TestFetchAgentConfigForOrg:
    def test_returns_doc(self):
        db, _ = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_config_for_org("sales_bot", ORG_ID)
        assert result["agent_type"] == "sales_bot"

    def test_queries_with_both_type_and_org(self):
        db, agents_coll = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            fetch_agent_config_for_org("sales_bot", ORG_ID)
        query = agents_coll.find_one.call_args[0][0]
        assert query.get("agent_type") == "sales_bot"
        assert query.get("org_id") == ORG_ID


# ── TestFetchAgentConfigById ──────────────────────────────────────────────

class TestFetchAgentConfigById:
    def test_returns_doc(self):
        db, _ = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_config_by_id("agent-001")
        assert result["agent_id"] == "agent-001"

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            assert fetch_agent_config_by_id("missing") is None


# ── TestFetchAgentsOfOrg ──────────────────────────────────────────────────

class TestFetchAgentsOfOrg:
    def test_returns_list_for_org(self):
        agents_coll = MagicMock()
        agents_coll.find.return_value.sort.return_value = [AGENT_DOC]
        db = make_mock_db(AgentConfig=agents_coll)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agents_of_org(ORG_ID)
        assert len(result) == 1
        assert result[0]["agent_type"] == "sales_bot"

    def test_returns_empty_list_on_exception(self):
        agents_coll = MagicMock()
        agents_coll.find.side_effect = Exception("DB error")
        db = make_mock_db(AgentConfig=agents_coll)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agents_of_org(ORG_ID)
        assert result == []


# ── TestUpdateAgentConfig ─────────────────────────────────────────────────

class TestUpdateAgentConfig:
    def _make_update_db(self, existing_doc=None, dup_check_doc=None):
        agents_coll = MagicMock()
        call_counter = {"n": 0}

        def find_one_side_effect(query):
            call_counter["n"] += 1
            if call_counter["n"] == 1:
                return existing_doc
            return dup_check_doc

        agents_coll.find_one.side_effect = find_one_side_effect
        update_result = MagicMock()
        update_result.matched_count = 1
        agents_coll.update_one.return_value = update_result

        db = make_mock_db(AgentConfig=agents_coll)
        return db, agents_coll

    def test_success_updates_doc(self):
        db, agents_coll = self._make_update_db(existing_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = update_agent_config("sales_bot", UPDATE_DATA, ORG_ID)
        assert result["status"] == "success"
        agents_coll.update_one.assert_called_once()

    def test_not_found_returns_fail(self):
        db, _ = self._make_update_db(existing_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = update_agent_config("ghost_bot", UPDATE_DATA, ORG_ID)
        assert result["status"] == "fail"
        assert "not found" in result["message"].lower()

    def test_rename_duplicate_target_returns_fail(self):
        db, _ = self._make_update_db(existing_doc=AGENT_DOC, dup_check_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = update_agent_config("sales_bot", RENAME_UPDATE, ORG_ID)
        assert result["status"] == "fail"
        assert ERR_ALREADY_EXISTS in result["message"].lower()

    def test_rename_cascades_to_related_collections(self):
        cascade_colls = {
            name: MagicMock()
            for name in ("PhoneNumber", "Meetings", "CallLogs", "CallRecordings",
                         "Campaigns", "Batches", "BatchContacts")
        }
        agents_coll = MagicMock()
        call_counter = {"n": 0}

        def fe(q):
            call_counter["n"] += 1
            return AGENT_DOC if call_counter["n"] == 1 else None

        agents_coll.find_one.side_effect = fe
        upd = MagicMock()
        upd.matched_count = 1
        agents_coll.update_one.return_value = upd

        all_colls = {"AgentConfig": agents_coll, **cascade_colls}
        db = make_mock_db(**all_colls)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = update_agent_config("sales_bot", RENAME_UPDATE, ORG_ID)
        assert result["status"] == "success"
        for coll_name, coll_mock in cascade_colls.items():
            coll_mock.update_many.assert_called_once()

    def test_non_conversational_mode_cannot_be_changed_back(self):
        db, _ = self._make_update_db(existing_doc=NON_CONV_AGENT_DOC)
        conv_update = AgentConfigUpdate(agent_config={"interaction_mode": "conversational"})
        with patch("app.services.agent_service.get_database", return_value=db):
            result = update_agent_config("sales_bot", conv_update, ORG_ID)
        assert result["status"] == "fail"
        assert "non-conversational" in result["message"].lower()


# ── TestDeleteAgent ───────────────────────────────────────────────────────

class TestDeleteAgent:
    def test_success_returns_success_dict(self):
        agents_coll = MagicMock()
        agents_coll.delete_one.return_value.deleted_count = 1
        db = make_mock_db(AgentConfig=agents_coll)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = delete_agent("sales_bot", org_id=ORG_ID)
        assert result["status"] == "success"

    def test_not_found_returns_fail(self):
        agents_coll = MagicMock()
        agents_coll.delete_one.return_value.deleted_count = 0
        db = make_mock_db(AgentConfig=agents_coll)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = delete_agent("ghost_bot", org_id=ORG_ID)
        assert result["status"] == "fail"
        assert ERR_AGENT_NOT_FOUND in result["message"].lower()


# ── TestFetchAgentByPhoneNumber ───────────────────────────────────────────

class TestFetchAgentByPhoneNumber:
    def test_returns_doc(self):
        db, _ = _make_db(agent_doc=AGENT_DOC)
        with patch("app.services.agent_service.get_database", return_value=db):
            result = fetch_agent_by_phone_number("+15550001234")
        assert result["agent_type"] == "sales_bot"

    def test_returns_none_when_not_found(self):
        db, _ = _make_db(agent_doc=None)
        with patch("app.services.agent_service.get_database", return_value=db):
            assert fetch_agent_by_phone_number("+10000000000") is None
