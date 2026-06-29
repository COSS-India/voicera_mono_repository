import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from app.services import plivo
from app.config import settings

@pytest.fixture
def mock_httpx():
    with patch("app.services.plivo.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_get_auth():
    with patch("app.services.plivo._get_plivo_auth_for_org") as mock_auth:
        mock_auth.return_value = ("test_auth_id", "test_auth_token")
        yield mock_auth

@pytest.fixture
def anyio_backend():
    return 'asyncio'


@pytest.mark.anyio
class TestCreatePlivoApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"app_id": "app_123"}
        mock_httpx.post.return_value = mock_response

        result = await plivo.create_plivo_application("org1", "sales_bot", "http://answer.url/plivo/answer")

        assert result["status"] == "success"
        assert result["app_id"] == "app_123"
        args, kwargs = mock_httpx.post.call_args
        assert kwargs["json"]["app_name"] == "sales_bot"
        assert kwargs["json"]["answer_url"] == "http://answer.url/plivo/answer"
        assert kwargs["json"]["hangup_url"] == "http://answer.url/plivo/hangup"
        assert kwargs["auth"] == ("test_auth_id", "test_auth_token")
        assert kwargs["headers"]["Content-Type"] == "application/json"

    async def test_missing_auth_returns_fail(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await plivo.create_plivo_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Plivo Auth ID" in result["message"]

    async def test_http_status_error_returns_fail(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Invalid Request"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)

        result = await plivo.create_plivo_application("org1", "sales_bot", "http://answer.url")

        assert result["status"] == "fail"
        assert "Invalid Request" in result["message"]

    async def test_request_error_returns_fail(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Network Down", request=mock_request)
        result = await plivo.create_plivo_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Network Down" in result["message"]

    async def test_generic_exception_returns_fail(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = ValueError("Unexpected issue")
        result = await plivo.create_plivo_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Unexpected issue" in result["message"]


@pytest.mark.anyio
class TestDeletePlivoApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.delete.return_value = mock_response

        result = await plivo.delete_plivo_application("org1", "app_123")
        assert result["status"] == "success"

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await plivo.delete_plivo_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Not Found"
        error_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await plivo.delete_plivo_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await plivo.delete_plivo_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.delete.side_effect = Exception("Boom")
        result = await plivo.delete_plivo_application("org1", "app_123")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestLinkNumberToApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = await plivo.link_number_to_application("org1", "+123 456 7890", "app_123")
        assert result["status"] == "success"
        args, kwargs = mock_httpx.post.call_args
        assert "%2B123%20456%207890" in args[0] # Checks if quoted correctly

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await plivo.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await plivo.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await plivo.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = Exception("Crash")
        result = await plivo.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestUnlinkNumberFromApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = await plivo.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "success"
        args, kwargs = mock_httpx.post.call_args
        assert kwargs["json"]["app_id"] == ""

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await plivo.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await plivo.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await plivo.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = Exception("Crash")
        result = await plivo.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestGetPlivoNumbers:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"objects": [{"number": "+1234567890"}, {"number": ""}, {"missing": "number"}]}
        mock_httpx.get.return_value = mock_response

        result = await plivo.get_plivo_numbers("org1")
        assert result["status"] == "success"
        assert result["numbers"] == ["+1234567890"]

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await plivo.get_plivo_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.get.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await plivo.get_plivo_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.get.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await plivo.get_plivo_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

class TestGetPlivoAuthForOrg:
    @patch("app.services.plivo._integration_service.get_integration")
    def test_success(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "PlivoAuthId":
                return {"api_key": "id_123"}
            if integration_type == "PlivoAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = plivo._get_plivo_auth_for_org("org1")
        assert result == ("id_123", "tok_123")

    @patch("app.services.plivo._integration_service.get_integration")
    def test_missing_id(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "PlivoAuthId":
                return None
            if integration_type == "PlivoAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = plivo._get_plivo_auth_for_org("org1")
        assert result is None

    @patch("app.services.plivo._integration_service.get_integration")
    def test_missing_token(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "PlivoAuthId":
                return {"api_key": "id_123"}
            if integration_type == "PlivoAuthToken":
                return None
        mock_get_integration.side_effect = side_effect
        
        result = plivo._get_plivo_auth_for_org("org1")
        assert result is None

    @patch("app.services.plivo._integration_service.get_integration")
    def test_empty_keys(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "PlivoAuthId":
                return {"api_key": " "}
            if integration_type == "PlivoAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = plivo._get_plivo_auth_for_org("org1")
        assert result is None

class TestAuthHeaders:
    def test_auth_headers_no_content_type(self):
        headers = plivo._auth_headers()
        assert headers == {"Accept": "application/json"}
    
    def test_auth_headers_with_content_type(self):
        headers = plivo._auth_headers(include_content_type=True)
        assert headers == {"Accept": "application/json", "Content-Type": "application/json"}
