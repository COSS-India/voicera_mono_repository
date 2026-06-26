import os
import socket
import json
import pytest
from unittest.mock import patch, MagicMock, AsyncMock, ANY
from fastapi.testclient import TestClient
from fastapi.websockets import WebSocketDisconnect

# Set environment variables before importing app to avoid raise or missing variables issues
os.environ["JOHNAIC_SERVER_URL"] = "http://mock-server"
os.environ["VOBIZ_API_BASE"] = "http://mock-vobiz"
os.environ["VOBIZ_CALLER_ID"] = "1234567890"

from api.server import (
    app,
    create_nodelay_websocket_protocol,
    _get_env_or_raise,
    _build_stream_xml,
    _resolve_call_identifier,
    make_outbound_call_vobiz,
    make_outbound_call_plivo,
    make_outbound_call_provider,
    log_meeting,
    run_server,
)

client = TestClient(app)


def test_get_env_or_raise_success():
    with patch.dict(os.environ, {"TEST_KEY": "val"}):
        assert _get_env_or_raise("TEST_KEY") == "val"


def test_get_env_or_raise_failure():
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError) as exc:
            _get_env_or_raise("TEST_KEY")
        assert "Missing required environment variable: TEST_KEY" in str(exc.value)


class DummyWebSocketProtocol:
    def connection_made(self, transport):
        pass


@patch("uvicorn.protocols.websockets.websockets_impl.WebSocketProtocol", DummyWebSocketProtocol)
def test_create_nodelay_websocket_protocol_success():
    proto_class = create_nodelay_websocket_protocol()
    assert proto_class is not None

    instance = proto_class()
    mock_transport = MagicMock()
    mock_socket = MagicMock()
    mock_transport.get_extra_info.return_value = mock_socket

    with patch.object(DummyWebSocketProtocol, 'connection_made') as mock_super_conn:
        instance.connection_made(mock_transport)
        mock_super_conn.assert_called_once_with(mock_transport)

    mock_socket.setsockopt.assert_called_once_with(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


@patch("uvicorn.protocols.websockets.websockets_impl.WebSocketProtocol", DummyWebSocketProtocol)
def test_create_nodelay_websocket_protocol_socket_exception():
    proto_class = create_nodelay_websocket_protocol()
    assert proto_class is not None

    instance = proto_class()
    mock_transport = MagicMock()
    mock_transport.get_extra_info.side_effect = Exception("socket error")

    with patch.object(DummyWebSocketProtocol, 'connection_made') as mock_super_conn:
        # Should not raise exception
        instance.connection_made(mock_transport)
        mock_super_conn.assert_called_once_with(mock_transport)


def test_create_nodelay_websocket_protocol_import_error():
    with patch("builtins.__import__", side_effect=ImportError):
        proto_class = create_nodelay_websocket_protocol()
        assert proto_class is None


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_make_outbound_call_vobiz_no_agent_config(mock_fetch_config):
    mock_fetch_config.return_value = None
    with pytest.raises(ValueError, match="Could not load agent config"):
        await make_outbound_call_vobiz("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_make_outbound_call_vobiz_no_org_id(mock_fetch_config):
    mock_fetch_config.return_value = {"agent_id": "agent_1"}
    with pytest.raises(ValueError, match="Agent has no org_id"):
        await make_outbound_call_vobiz("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
async def test_make_outbound_call_vobiz_no_credentials(mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.return_value = None
    with pytest.raises(ValueError, match="Vobiz Auth ID and Auth Token must be configured"):
        await make_outbound_call_vobiz("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
@patch.dict(os.environ, {"JOHNAIC_SERVER_URL": "http://server", "VOBIZ_API_BASE": "http://vobiz"})
async def test_make_outbound_call_vobiz_no_caller_id(mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.side_effect = lambda org_id, key: "some_val"
    env_mock = {"JOHNAIC_SERVER_URL": "http://server", "VOBIZ_API_BASE": "http://vobiz"}
    with patch.dict(os.environ, env_mock, clear=True):
        with pytest.raises(ValueError, match="No caller_id provided and VOBIZ_CALLER_ID not set"):
            await make_outbound_call_vobiz("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
@patch("requests.post")
@patch.dict(os.environ, {
    "JOHNAIC_SERVER_URL": "http://server",
    "VOBIZ_API_BASE": "http://vobiz",
    "VOBIZ_CALLER_ID": "12345"
})
async def test_make_outbound_call_vobiz_success(mock_post, mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.side_effect = lambda org, name: f"mocked_{name}"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"call_uuid": "vobiz-uuid-123"}
    mock_post.return_value = mock_resp

    res = await make_outbound_call_vobiz("9876543210", "agent_1")
    assert res == {"call_uuid": "vobiz-uuid-123"}

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert args[0] == "http://vobiz/Account/mocked_VobizAuthId/Call/"
    assert kwargs["json"]["to"] == "9876543210"
    assert kwargs["headers"]["X-Auth-ID"] == "mocked_VobizAuthId"


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_make_outbound_call_plivo_no_agent_config(mock_fetch_config):
    mock_fetch_config.return_value = None
    with pytest.raises(ValueError, match="Could not load agent config"):
        await make_outbound_call_plivo("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_make_outbound_call_plivo_no_org_id(mock_fetch_config):
    mock_fetch_config.return_value = {"agent_id": "agent_1"}
    with pytest.raises(ValueError, match="Agent has no org_id"):
        await make_outbound_call_plivo("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
async def test_make_outbound_call_plivo_no_credentials(mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.return_value = None
    with pytest.raises(ValueError, match="Plivo Auth ID and Auth Token must be configured"):
        await make_outbound_call_plivo("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
@patch.dict(os.environ, {"JOHNAIC_SERVER_URL": "http://server"})
async def test_make_outbound_call_plivo_no_caller_id(mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.side_effect = lambda org, name: "some_val"
    env_mock = {"JOHNAIC_SERVER_URL": "http://server"}
    with patch.dict(os.environ, env_mock, clear=True):
        with pytest.raises(ValueError, match="No caller_id provided and PLIVO_CALLER_ID not set"):
            await make_outbound_call_plivo("123", "agent_1")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.fetch_integration_key")
@patch("requests.post")
@patch.dict(os.environ, {
    "JOHNAIC_SERVER_URL": "http://server",
    "PLIVO_CALLER_ID": "12345"
})
async def test_make_outbound_call_plivo_success(mock_post, mock_fetch_key, mock_fetch_config):
    mock_fetch_config.return_value = {"org_id": "org_1"}
    mock_fetch_key.side_effect = lambda org, name: f"mocked_{name}"

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"request_uuid": "plivo-uuid-123"}
    mock_post.return_value = mock_resp

    res = await make_outbound_call_plivo("9876543210", "agent_1")
    assert res == {"request_uuid": "plivo-uuid-123"}

    mock_post.assert_called_once()
    args, kwargs = mock_post.call_args
    assert "api.plivo.com" in args[0]
    assert kwargs["json"]["to"] == "9876543210"
    assert kwargs["auth"] == ("mocked_PlivoAuthId", "mocked_PlivoAuthToken")


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.make_outbound_call_plivo", new_callable=AsyncMock)
@patch("api.server.make_outbound_call_vobiz", new_callable=AsyncMock)
async def test_make_outbound_call_provider_plivo(mock_vobiz, mock_plivo, mock_fetch_config):
    mock_fetch_config.return_value = {"telephony_provider": "plivo"}
    await make_outbound_call_provider("123", "agent_1")
    mock_plivo.assert_called_once_with("123", "agent_1", None)
    mock_vobiz.assert_not_called()


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.make_outbound_call_plivo", new_callable=AsyncMock)
@patch("api.server.make_outbound_call_vobiz", new_callable=AsyncMock)
async def test_make_outbound_call_provider_vobiz(mock_vobiz, mock_plivo, mock_fetch_config):
    mock_fetch_config.return_value = {"telephony_provider": "Vobiz"}
    await make_outbound_call_provider("123", "agent_1")
    mock_vobiz.assert_called_once_with("123", "agent_1", None)
    mock_plivo.assert_not_called()


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_make_outbound_call_provider_no_config(mock_fetch_config):
    mock_fetch_config.return_value = None
    with pytest.raises(ValueError, match="Could not load agent config"):
        await make_outbound_call_provider("123", "agent_1")


def test_build_stream_xml_8000():
    with patch.dict(os.environ, {"SAMPLE_RATE": "8000"}):
        xml = _build_stream_xml("ws://localhost")
        assert 'contentType="audio/x-mulaw;rate=8000"' in xml
        assert "ws://localhost" in xml


def test_build_stream_xml_16000():
    with patch.dict(os.environ, {"SAMPLE_RATE": "16000"}):
        xml = _build_stream_xml("ws://localhost")
        assert 'contentType="audio/x-l16;rate=16000"' in xml
        assert "ws://localhost" in xml


def test_resolve_call_identifier():
    assert _resolve_call_identifier({"CallUUID": "uuid1"}) == "uuid1"
    assert _resolve_call_identifier({"call_uuid": "uuid2"}) == "uuid2"
    assert _resolve_call_identifier({"call_id": "id3"}) == "id3"
    assert _resolve_call_identifier({"callId": "id4"}) == "id4"
    assert _resolve_call_identifier({"callSid": "sid5"}) == "sid5"
    assert _resolve_call_identifier({"CallSid": "sid6"}) == "sid6"
    assert _resolve_call_identifier({"request_uuid": "req7"}) == "req7"
    assert _resolve_call_identifier({"start": {"callId": "startid8"}}) == "startid8"
    assert _resolve_call_identifier({"start": {"callSid": "startsid9"}}) == "startsid9"
    assert _resolve_call_identifier({"other": "val"}) == "unknown"
    assert _resolve_call_identifier("not_a_dict") == "unknown"
    assert _resolve_call_identifier(None) == "unknown"


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.create_meeting_in_backend", new_callable=AsyncMock)
async def test_log_meeting_success(mock_create, mock_fetch_config):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}
    form_data = {"CallUUID": "uuid_123", "From": "111", "To": "222"}

    res = await log_meeting("agent_123", form_data)
    assert res == {"status": "success"}
    mock_create.assert_called_once()
    payload = mock_create.call_args[0][0]
    assert payload["meeting_id"] == "uuid_123"
    assert payload["from_number"] == "111"
    assert payload["to_number"] == "222"
    assert payload["inbound"] is False


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
@patch("api.server.create_meeting_in_backend", new_callable=AsyncMock)
async def test_log_meeting_busy(mock_create, mock_fetch_config):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}
    form_data = {
        "CallUUID": "uuid_123",
        "Direction": "inbound",
        "CallStatus": "busy"
    }

    res = await log_meeting("agent_123", form_data)
    assert res == {"status": "success"}
    payload = mock_create.call_args[0][0]
    assert payload["inbound"] is True
    assert payload["call_busy"] is True
    assert payload["end_time_utc"] != ""


@pytest.mark.asyncio
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
async def test_log_meeting_failure(mock_fetch_config):
    mock_fetch_config.side_effect = Exception("backend down")
    res = await log_meeting("agent_123", {})
    assert res["status"] == "error"
    assert "backend down" in res["message"]


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"service": "Telephony Server", "status": "running"}


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


@patch("api.server.make_outbound_call_provider", new_callable=AsyncMock)
def test_make_outbound_call_endpoint_success(mock_make_call):
    mock_make_call.return_value = {"call_uuid": "uuid_123"}
    payload = {
        "customer_number": "123456",
        "agent_id": "agent_123"
    }
    response = client.post("/outbound/call/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["result"] == {"call_uuid": "uuid_123"}


@patch("api.server.make_outbound_call_provider", new_callable=AsyncMock)
def test_make_outbound_call_endpoint_value_error(mock_make_call):
    mock_make_call.side_effect = ValueError("Invalid provider")
    payload = {
        "customer_number": "123456",
        "agent_id": "agent_123"
    }
    response = client.post("/outbound/call/", json=payload)
    assert response.status_code == 400
    assert "Invalid provider" in response.json()["detail"]


@patch("api.server.make_outbound_call_provider", new_callable=AsyncMock)
def test_make_outbound_call_endpoint_generic_error(mock_make_call):
    mock_make_call.side_effect = Exception("Unknown error")
    payload = {
        "customer_number": "123456",
        "agent_id": "agent_123"
    }
    response = client.post("/outbound/call/", json=payload)
    assert response.status_code == 500
    assert "Unknown error" in response.json()["detail"]


@patch("api.server.log_meeting", new_callable=AsyncMock)
@patch.dict(os.environ, {"JOHNAIC_WEBSOCKET_URL": "ws://socket_url"})
def test_vobiz_answer_webhook_start_app(mock_log):
    response = client.post("/answer?agent_id=agent_123", data={"Event": "StartApp", "CallUUID": "uuid_123"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"
    assert "<Stream" in response.text
    assert "ws://socket_url/agent/agent_123" in response.text
    mock_log.assert_called_once_with("agent_123", {"Event": "StartApp", "CallUUID": "uuid_123"})


@patch("api.server.log_meeting", new_callable=AsyncMock)
def test_vobiz_answer_webhook_hangup(mock_log):
    response = client.post("/answer?agent_id=agent_123", data={"Event": "Hangup", "HangupCause": "USER_BUSY"})
    assert response.status_code == 200
    mock_log.assert_called_once_with("agent_123", {"Event": "Hangup", "HangupCause": "USER_BUSY"})


@patch("api.server.log_meeting", new_callable=AsyncMock)
def test_vobiz_answer_webhook_other_event(mock_log):
    response = client.post("/answer?agent_id=agent_123", data={"Event": "Other"})
    assert response.status_code == 200
    mock_log.assert_not_called()


@patch("api.server.log_meeting", new_callable=AsyncMock)
@patch.dict(os.environ, {"JOHNAIC_WEBSOCKET_URL": "ws://socket_url"})
def test_plivo_answer_webhook(mock_log):
    response = client.post("/plivo/answer?agent_id=agent_123", data={"CallUUID": "uuid_123"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/xml"
    assert "ws://socket_url/plivo/agent/agent_123" in response.text
    mock_log.assert_called_once_with("agent_123", {"CallUUID": "uuid_123"})


@patch("api.server.log_meeting", new_callable=AsyncMock)
def test_plivo_hangup_webhook(mock_log):
    response = client.post("/plivo/hangup?agent_id=agent_123", data={"CallUUID": "uuid_123"})
    assert response.status_code == 200
    mock_log.assert_called_once_with("agent_123", {"CallUUID": "uuid_123"})


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_vobiz_success(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    with client.websocket_connect("/agent/agent_123") as ws:
        ws.send_text(json.dumps({
            "event": "start",
            "start": {"callSid": "call_123", "streamSid": "stream_123"}
        }))

    mock_bot.assert_called_once_with(
        ANY, "stream_123", "call_123", "inbound", {"agent_type": "inbound", "org_id": "org_1"}, provider="vobiz"
    )


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_vobiz_config_not_found_none(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = None

    with client.websocket_connect("/agent/agent_123") as ws:
        pass

    mock_bot.assert_not_called()


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_vobiz_config_not_found_exception(mock_fetch_config):
    mock_fetch_config.side_effect = FileNotFoundError("config file not found")

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/agent/agent_123") as ws:
            ws.receive_text()
    assert exc.value.code == 1008


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_vobiz_not_start_event(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    with client.websocket_connect("/agent/agent_123") as ws:
        ws.send_text(json.dumps({
            "event": "not_start"
        }))

    mock_bot.assert_not_called()


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_vobiz_generic_exception(mock_fetch_config):
    mock_fetch_config.side_effect = Exception("db connection failed")

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/agent/agent_123") as ws:
            ws.receive_text()


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_plivo_success(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    with client.websocket_connect("/plivo/agent/agent_123") as ws:
        pass

    mock_bot.assert_called_once_with(
        ANY, stream_sid=None, call_sid=None, agent_type="inbound", agent_config={"agent_type": "inbound", "org_id": "org_1"}, provider="plivo"
    )


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_plivo_config_not_found_none(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = None

    with client.websocket_connect("/plivo/agent/agent_123") as ws:
        pass

    mock_bot.assert_not_called()


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_plivo_config_not_found_exception(mock_fetch_config):
    mock_fetch_config.side_effect = FileNotFoundError("config file not found")

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/plivo/agent/agent_123") as ws:
            ws.receive_text()
    assert exc.value.code == 1008


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_plivo_generic_exception(mock_fetch_config):
    mock_fetch_config.side_effect = Exception("db connection failed")

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/plivo/agent/agent_123") as ws:
            ws.receive_text()


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_success(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    with client.websocket_connect("/browser/agent/agent_123") as ws:
        ws.send_text(json.dumps({
            "event": "start",
            "start": {"callSid": "call_123", "streamSid": "stream_123"}
        }))

    mock_bot.assert_called_once()
    kwargs = mock_bot.call_args[1]
    assert kwargs["provider"] == "browser"
    assert kwargs["sample_rate"] == 16000
    assert callable(kwargs["transcript_callback"])


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_transcript_callback(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    async def mock_bot_call(*args, **kwargs):
        callback = kwargs.get("transcript_callback")
        if callback:
            await callback("user", "Hello browser", "12:00:00")

    mock_bot.side_effect = mock_bot_call

    with client.websocket_connect("/browser/agent/agent_123") as ws:
        ws.send_text(json.dumps({
            "event": "start",
            "start": {"callSid": "call_123", "streamSid": "stream_123"}
        }))

        response_text = ws.receive_text()
        data = json.loads(response_text)
        assert data["event"] == "transcript"
        assert data["role"] == "user"
        assert data["content"] == "Hello browser"
        assert data["timestamp"] == "12:00:00"


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_config_not_found_none(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = None

    with client.websocket_connect("/browser/agent/agent_123") as ws:
        pass

    mock_bot.assert_not_called()


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_config_not_found_exception(mock_fetch_config):
    mock_fetch_config.side_effect = FileNotFoundError("config file not found")

    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/browser/agent/agent_123") as ws:
            ws.receive_text()
    assert exc.value.code == 1008


@patch("api.server.bot", new_callable=AsyncMock)
@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_not_start_event(mock_fetch_config, mock_bot):
    mock_fetch_config.return_value = {"agent_type": "inbound", "org_id": "org_1"}

    with client.websocket_connect("/browser/agent/agent_123") as ws:
        ws.send_text(json.dumps({
            "event": "not_start"
        }))

    mock_bot.assert_not_called()


@patch("api.server.fetch_agent_config_from_backend", new_callable=AsyncMock)
def test_websocket_browser_generic_exception(mock_fetch_config):
    mock_fetch_config.side_effect = Exception("db connection failed")

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/browser/agent/agent_123") as ws:
            ws.receive_text()


@patch("uvicorn.Server")
@patch("uvicorn.Config")
@patch("api.server.create_nodelay_websocket_protocol")
def test_run_server(mock_create_protocol, mock_config, mock_server):
    mock_create_protocol.return_value = MagicMock()

    run_server(host="1.2.3.4", port=9999, log_level="warning")

    mock_config.assert_called_once_with(
        app,
        host="1.2.3.4",
        port=9999,
        log_level="warning",
        loop="auto",
        http="auto",
        ws="websockets"
    )

    instance_config = mock_config.return_value
    assert instance_config.ws_protocol_class == mock_create_protocol.return_value

    mock_server.assert_called_once_with(instance_config)
    mock_server.return_value.run.assert_called_once()


@patch("uvicorn.Server")
@patch("uvicorn.Config")
@patch("api.server.create_nodelay_websocket_protocol")
def test_run_server_no_protocol(mock_create_protocol, mock_config, mock_server):
    mock_create_protocol.return_value = None

    run_server(host="1.2.3.4", port=9999, log_level="warning")

    instance_config = mock_config.return_value
    # If None, ws_protocol_class shouldn't be set or should be set to None.
    # In api/server.py:
    # if nodelay_protocol:
    #     config.ws_protocol_class = nodelay_protocol
    # So it doesn't modify it if None. Let's make sure it was called anyway.
    mock_server.assert_called_once_with(instance_config)


@patch("utils.batching.BatchWorker.run")
def test_batch_run_route(mock_worker_run):
    mock_worker_run.return_value = {"status": "success"}
    payload = {
        "org_id": "org_123",
        "batch_id": "batch_123",
        "agent_type": "inbound",
        "concurrency": 5
    }
    response = client.post("/outbound/batch/run/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_worker_run.assert_called_once_with(
        org_id="org_123",
        batch_id="batch_123",
        agent_type="inbound",
        concurrency=5
    )


@patch("utils.batching.BatchWorker.stop")
def test_batch_stop_route(mock_worker_stop):
    mock_worker_stop.return_value = {"status": "success"}
    payload = {
        "org_id": "org_123",
        "batch_id": "batch_123"
    }
    response = client.post("/outbound/batch/stop/", json=payload)
    assert response.status_code == 200
    assert response.json() == {"status": "success"}
    mock_worker_stop.assert_called_once_with(
        batch_id="batch_123"
    )
