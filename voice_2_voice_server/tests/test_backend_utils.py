import pytest
import os
import requests
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime

from utils.backend_utils import (
    _get_backend_url,
    _get_api_key,
    _get_api_headers,
    fetch_integration_key,
    fetch_custom_llm_config,
    fetch_knowledge_chunks,
    fetch_agent_config_from_backend,
    create_meeting_in_backend,
    create_rejected_call_meeting,
    update_meeting_end_time,
    fetch_batch_agent_call_config,
    claim_next_batch_contact,
    report_batch_contact_result,
    finalize_batch_execution,
    fetch_agent_by_phone_number,
    submit_call_recording,
)


class TestBackendUtilsEnv:
    @patch.dict(os.environ, {}, clear=True)
    def test_get_backend_url_default(self):
        assert _get_backend_url() == "http://localhost:8000"

    @patch.dict(os.environ, {"VOICERA_BACKEND_URL": "https://test.backend"}, clear=True)
    def test_get_backend_url_custom(self):
        assert _get_backend_url() == "https://test.backend"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_key_none(self):
        assert _get_api_key() is None

    @patch.dict(os.environ, {"INTERNAL_API_KEY": "secret_key"}, clear=True)
    def test_get_api_key_present(self):
        assert _get_api_key() == "secret_key"

    @patch.dict(os.environ, {}, clear=True)
    def test_get_api_headers_no_key(self):
        headers = _get_api_headers()
        assert headers == {"Content-Type": "application/json"}

    @patch.dict(os.environ, {"INTERNAL_API_KEY": "secret_key"}, clear=True)
    def test_get_api_headers_with_key(self):
        headers = _get_api_headers()
        assert headers == {"Content-Type": "application/json", "X-API-Key": "secret_key"}


class TestBackendUtilsSyncAPI:
    @patch("requests.post")
    def test_fetch_integration_key_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"api_key": "openai_api_val"}
        mock_post.return_value = mock_resp

        key = fetch_integration_key("org_1", "OpenAI")
        assert key == "openai_api_val"
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_fetch_integration_key_404(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        key = fetch_integration_key("org_1", "OpenAI")
        assert key is None

    @patch("requests.post")
    def test_fetch_integration_key_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("conn error")
        key = fetch_integration_key("org_1", "OpenAI")
        assert key is None

    @patch("requests.post")
    def test_fetch_integration_key_generic_exception(self, mock_post):
        mock_post.side_effect = Exception("generic error")
        key = fetch_integration_key("org_1", "OpenAI")
        assert key is None

    @patch("requests.post")
    def test_fetch_custom_llm_config_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"provider": "custom_provider", "api_key": "custom_key"}
        mock_post.return_value = mock_resp

        config = fetch_custom_llm_config("org_1", "llm_1")
        assert config == {"provider": "custom_provider", "api_key": "custom_key"}

    @patch("requests.post")
    def test_fetch_custom_llm_config_404(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_post.return_value = mock_resp

        config = fetch_custom_llm_config("org_1", "llm_1")
        assert config is None

    @patch("requests.post")
    def test_fetch_custom_llm_config_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException()
        config = fetch_custom_llm_config("org_1", "llm_1")
        assert config is None

    @patch("requests.post")
    def test_fetch_custom_llm_config_generic_exception(self, mock_post):
        mock_post.side_effect = Exception()
        config = fetch_custom_llm_config("org_1", "llm_1")
        assert config is None

    @patch("requests.post")
    def test_fetch_knowledge_chunks_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"chunks": [{"text": "hello world", "score": 0.9}]}
        mock_post.return_value = mock_resp

        chunks = fetch_knowledge_chunks(org_id="org_1", question="what is this?", document_ids=["doc_1"])
        assert len(chunks) == 1
        assert chunks[0]["text"] == "hello world"

    @patch("requests.post")
    def test_fetch_knowledge_chunks_invalid_data(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = "not a dict"
        mock_post.return_value = mock_resp

        chunks = fetch_knowledge_chunks(org_id="org_1", question="what is this?")
        assert chunks == []

    @patch("requests.post")
    def test_fetch_knowledge_chunks_exception(self, mock_post):
        mock_post.side_effect = Exception("failed")
        chunks = fetch_knowledge_chunks(org_id="org_1", question="what is this?")
        assert chunks == []

    @patch("requests.post")
    def test_fetch_batch_agent_call_config_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"config_key": "val"}
        mock_post.return_value = mock_resp

        res = fetch_batch_agent_call_config("org_1", "agent_1")
        assert res == {"config_key": "val"}

    @patch("requests.post")
    def test_fetch_batch_agent_call_config_error(self, mock_post):
        mock_post.side_effect = Exception()
        res = fetch_batch_agent_call_config("org_1", "agent_1")
        assert res is None

    @patch("requests.post")
    def test_claim_next_batch_contact_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"contact": {"phone": "123"}}
        mock_post.return_value = mock_resp

        res = claim_next_batch_contact("org_1", "batch_1")
        assert res == {"phone": "123"}

    @patch("requests.post")
    def test_claim_next_batch_contact_invalid_json(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = ""
        mock_post.return_value = mock_resp

        res = claim_next_batch_contact("org_1", "batch_1")
        assert res is None

    @patch("requests.post")
    def test_claim_next_batch_contact_error(self, mock_post):
        mock_post.side_effect = Exception()
        res = claim_next_batch_contact("org_1", "batch_1")
        assert res is None

    @patch("requests.post")
    def test_report_batch_contact_result(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        # Should not raise exception
        report_batch_contact_result(org_id="org_1", batch_id="batch_1", row_number=1, ok=True)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_report_batch_contact_result_error(self, mock_post):
        mock_post.side_effect = Exception()
        # Should catch and log error without raising
        report_batch_contact_result(org_id="org_1", batch_id="batch_1", row_number=1, ok=True)

    @patch("requests.post")
    def test_finalize_batch_execution(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        finalize_batch_execution("org_1", "batch_1", False)
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_finalize_batch_execution_error(self, mock_post):
        mock_post.side_effect = Exception("finalize error")
        finalize_batch_execution("org_1", "batch_1", False)


class TestBackendUtilsAsyncAPI:
    @pytest.mark.asyncio
    @patch("requests.get")
    async def test_fetch_agent_config_from_backend_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "agent_config": {"model": "gpt-4"},
            "org_id": "org_1",
            "agent_type": "inbound",
        }
        mock_get.return_value = mock_resp

        res = await fetch_agent_config_from_backend("agent_1")
        assert res == {"model": "gpt-4", "org_id": "org_1", "agent_type": "inbound"}

    @pytest.mark.asyncio
    @patch("requests.get")
    async def test_fetch_agent_config_from_backend_request_exception(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException()
        res = await fetch_agent_config_from_backend("agent_1")
        assert res is None

    @pytest.mark.asyncio
    @patch("requests.get")
    async def test_fetch_agent_config_from_backend_generic_exception(self, mock_get):
        mock_get.side_effect = Exception()
        res = await fetch_agent_config_from_backend("agent_1")
        assert res is None

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_create_meeting_in_backend_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"meeting_id": "meet_1"}
        mock_post.return_value = mock_resp

        res = await create_meeting_in_backend({"call_sid": "123"})
        assert res == {"meeting_id": "meet_1"}

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_create_meeting_in_backend_request_exception(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException()
        res = await create_meeting_in_backend({"call_sid": "123"})
        assert res is None

    @pytest.mark.asyncio
    @patch("requests.post")
    async def test_create_meeting_in_backend_generic_exception(self, mock_post):
        mock_post.side_effect = Exception()
        res = await create_meeting_in_backend({"call_sid": "123"})
        assert res is None

    @pytest.mark.asyncio
    @patch("utils.backend_utils.fetch_agent_config_from_backend", new_callable=AsyncMock)
    @patch("requests.post")
    async def test_create_rejected_call_meeting_success(self, mock_post, mock_fetch_config):
        mock_fetch_config.return_value = {"org_id": "org_1"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        form_data = {
            "From": "08071387434",
            "To": "08071387435",
            "StartTime": "2026-01-14 17:04:30",
            "EndTime": "2026-01-14 17:05:00",
            "Direction": "inbound",
        }
        res = await create_rejected_call_meeting(
            call_uuid="call_uuid_1",
            agent_type="inbound",
            form_data=form_data,
        )
        assert res is True
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["org_id"] == "org_1"
        assert payload["from_number"] == "08071387434"
        assert payload["inbound"] is True
        assert payload["call_busy"] is True

    @pytest.mark.asyncio
    @patch("utils.backend_utils.fetch_agent_config_from_backend", new_callable=AsyncMock)
    @patch("requests.post")
    async def test_create_rejected_call_meeting_invalid_dates_and_missing_info(self, mock_post, mock_fetch_config):
        mock_fetch_config.return_value = {"org_id": "org_1"}
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        form_data = {
            "from_number": ["08071387434"],
            "to_number": "08071387435",
            "StartTime": "invalid_date",
            "EndTime": "invalid_date",
        }
        res = await create_rejected_call_meeting(
            call_uuid="call_uuid_1",
            agent_type="inbound",
            form_data=form_data,
            from_number="fallback_from",
            to_number="fallback_to",
        )
        assert res is True
        payload = mock_post.call_args[1]["json"]
        assert payload["from_number"] == "08071387434"
        assert payload["to_number"] == "08071387435"

    @pytest.mark.asyncio
    @patch("utils.backend_utils.fetch_agent_config_from_backend", new_callable=AsyncMock)
    async def test_create_rejected_call_meeting_no_agent_config(self, mock_fetch_config):
        mock_fetch_config.return_value = None
        res = await create_rejected_call_meeting("uuid", "inbound", {})
        assert res is False

    @pytest.mark.asyncio
    @patch("utils.backend_utils.fetch_agent_config_from_backend", new_callable=AsyncMock)
    @patch("requests.post")
    async def test_create_rejected_call_meeting_post_fails(self, mock_post, mock_fetch_config):
        mock_fetch_config.return_value = {"org_id": "org_1"}
        mock_post.side_effect = requests.exceptions.RequestException()
        res = await create_rejected_call_meeting("uuid", "inbound", {})
        assert res is False

    @pytest.mark.asyncio
    @patch("utils.backend_utils.fetch_agent_config_from_backend", new_callable=AsyncMock)
    @patch("requests.post")
    async def test_create_rejected_call_meeting_generic_exception(self, mock_post, mock_fetch_config):
        mock_fetch_config.return_value = {"org_id": "org_1"}
        mock_post.side_effect = Exception("error")
        res = await create_rejected_call_meeting("uuid", "inbound", {})
        assert res is False

    @pytest.mark.asyncio
    @patch("requests.patch")
    async def test_update_meeting_end_time_success(self, mock_patch):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_patch.return_value = mock_resp

        res = await update_meeting_end_time("call_123", "2026-06-24T12:00:00Z")
        assert res is True
        mock_patch.assert_called_once()

    @pytest.mark.asyncio
    @patch("requests.patch")
    async def test_update_meeting_end_time_fail(self, mock_patch):
        mock_patch.side_effect = requests.exceptions.RequestException()
        res = await update_meeting_end_time("call_123", "2026-06-24T12:00:00Z")
        assert res is False

    @pytest.mark.asyncio
    @patch("utils.backend_utils._get_api_key", return_value=None)
    @patch("requests.get")
    async def test_fetch_agent_by_phone_number_success(self, mock_get, mock_api_key):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"agent_type": "support"}
        mock_get.return_value = mock_resp

        # Test phone starts with '0'
        res = await fetch_agent_by_phone_number("08071387434")
        assert res == {"agent_type": "support"}
        mock_get.assert_called_with(
            "http://localhost:8000/api/v1/agents/by-phone/%2B918071387434",
            headers={"Content-Type": "application/json"},
            timeout=10,
        )

        # Test phone starts with '+'
        res = await fetch_agent_by_phone_number("+918071387434")
        assert res == {"agent_type": "support"}

        # Test phone starts without '+' or '0'
        res = await fetch_agent_by_phone_number("918071387434")
        assert res == {"agent_type": "support"}

    @pytest.mark.asyncio
    @patch("requests.get")
    async def test_fetch_agent_by_phone_number_fail(self, mock_get):
        mock_get.side_effect = requests.exceptions.RequestException()
        res = await fetch_agent_by_phone_number("08071387434")
        assert res is None

    @pytest.mark.asyncio
    @patch("requests.post")
    @patch("utils.backend_utils.update_meeting_end_time", new_callable=AsyncMock)
    async def test_submit_call_recording_success(self, mock_update_meeting, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp
        mock_update_meeting.return_value = True

        # Mock MinIO storage
        mock_storage = AsyncMock()
        mock_storage_resp = MagicMock()
        mock_storage_resp.read.return_value = b"test transcript text"
        mock_storage.get_object.return_value = mock_storage_resp

        await submit_call_recording(
            call_sid="call_123",
            agent_type="inbound",
            agent_config={"org_id": "org_1"},
            storage=mock_storage,
            call_start_time=100.0,
            start_time_utc="2026-06-24T12:00:00Z",
        )

        mock_storage.get_object.assert_called_once_with("transcripts", "call_123.txt")
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["org_id"] == "org_1"
        assert payload["transcript_content"] == "test transcript text"
        mock_update_meeting.assert_called_once()

    @pytest.mark.asyncio
    @patch("requests.post")
    @patch("utils.backend_utils.update_meeting_end_time", new_callable=AsyncMock)
    async def test_submit_call_recording_storage_fails(self, mock_update_meeting, mock_post):
        # Even if storage fails to read transcript, recording submission should proceed
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        mock_storage = AsyncMock()
        mock_storage.get_object.side_effect = Exception("MinIO error")

        await submit_call_recording(
            call_sid="call_123",
            agent_type="inbound",
            agent_config={},
            storage=mock_storage,
            call_start_time=100.0,
            start_time_utc="2026-06-24T12:00:00Z",
        )
        mock_post.assert_called_once()
        payload = mock_post.call_args[1]["json"]
        assert payload["transcript_content"] is None
        mock_update_meeting.assert_called_once()

    @pytest.mark.asyncio
    @patch("requests.post")
    @patch("utils.backend_utils.update_meeting_end_time", new_callable=AsyncMock)
    async def test_submit_call_recording_post_fails(self, mock_update_meeting, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException()
        mock_storage = AsyncMock()
        mock_storage_resp = MagicMock()
        mock_storage_resp.read.return_value = b"test transcript text"
        mock_storage.get_object.return_value = mock_storage_resp

        await submit_call_recording(
            call_sid="call_123",
            agent_type="inbound",
            agent_config={},
            storage=mock_storage,
            call_start_time=100.0,
            start_time_utc="2026-06-24T12:00:00Z",
        )
        # Should still try to update meeting end time
        mock_update_meeting.assert_called_once()
