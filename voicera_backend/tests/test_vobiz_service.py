import pytest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
from app.services import vobiz
from app.config import settings

@pytest.fixture
def mock_httpx():
    with patch("app.services.vobiz.httpx.AsyncClient") as mock_client_class:
        mock_client = AsyncMock()
        mock_client_class.return_value.__aenter__.return_value = mock_client
        yield mock_client

@pytest.fixture
def mock_get_auth():
    with patch("app.services.vobiz._get_vobiz_auth_for_org") as mock_auth:
        mock_auth.return_value = ("test_auth_id", "test_auth_token")
        yield mock_auth

@pytest.fixture
def anyio_backend():
    return 'asyncio'



@pytest.mark.anyio
class TestCreateVobizApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"app_id": "app_123"}
        mock_httpx.post.return_value = mock_response

        # Act
        result = await vobiz.create_vobiz_application("org1", "sales_bot", "http://answer.url")

        # Assert
        assert result["status"] == "success"
        assert result["app_id"] == "app_123"
        mock_httpx.post.assert_called_once()
        args, kwargs = mock_httpx.post.call_args
        assert kwargs["json"]["app_name"] == "sales_bot"
        assert kwargs["json"]["answer_url"] == "http://answer.url"

    async def test_missing_auth_returns_fail(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.create_vobiz_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Vobiz Auth ID" in result["message"]

    async def test_http_status_error_returns_fail(self, mock_httpx, mock_get_auth):
        # Arrange
        error_response = MagicMock()
        error_response.text = "Invalid Request"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)

        # Act
        result = await vobiz.create_vobiz_application("org1", "sales_bot", "http://answer.url")

        # Assert
        assert result["status"] == "fail"
        assert "Invalid Request" in result["message"]

    async def test_request_error_returns_fail(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Network Down", request=mock_request)
        result = await vobiz.create_vobiz_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Network Down" in result["message"]

    async def test_generic_exception_returns_fail(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = ValueError("Unexpected issue")
        result = await vobiz.create_vobiz_application("org1", "sales_bot", "http://answer.url")
        assert result["status"] == "fail"
        assert "Unexpected issue" in result["message"]


@pytest.mark.anyio
class TestDeleteVobizApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.delete.return_value = mock_response

        result = await vobiz.delete_vobiz_application("org1", "app_123")
        assert result["status"] == "success"

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.delete_vobiz_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Not Found"
        error_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await vobiz.delete_vobiz_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await vobiz.delete_vobiz_application("org1", "app_123")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.delete.side_effect = Exception("Boom")
        result = await vobiz.delete_vobiz_application("org1", "app_123")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestUpdateVobizApplicationName:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = await vobiz.update_vobiz_application_name("org1", "app_123", "new_name")
        assert result["status"] == "success"

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.update_vobiz_application_name("org1", "app_123", "new_name")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await vobiz.update_vobiz_application_name("org1", "app_123", "new_name")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await vobiz.update_vobiz_application_name("org1", "app_123", "new_name")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = Exception("Crash")
        result = await vobiz.update_vobiz_application_name("org1", "app_123", "new_name")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestLinkNumberToApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.post.return_value = mock_response

        result = await vobiz.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "success"

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.post.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await vobiz.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.post.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await vobiz.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.post.side_effect = Exception("Crash")
        result = await vobiz.link_number_to_application("org1", "+1234567890", "app_123")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestUnlinkNumberFromApplication:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_httpx.delete.return_value = mock_response

        result = await vobiz.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "success"

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await vobiz.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.delete.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await vobiz.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"

    async def test_generic_exception(self, mock_httpx, mock_get_auth):
        mock_httpx.delete.side_effect = Exception("Crash")
        result = await vobiz.unlink_number_from_application("org1", "+1234567890")
        assert result["status"] == "fail"


@pytest.mark.anyio
class TestGetVobizNumbers:
    async def test_success(self, mock_httpx, mock_get_auth):
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.json.return_value = {"items": [{"e164": "+1234567890"}, {"e164": ""}, {"missing": "e164"}]}
        mock_httpx.get.return_value = mock_response

        result = await vobiz.get_vobiz_numbers("org1")
        assert result["status"] == "success"
        assert result["numbers"] == ["+1234567890"]

    async def test_missing_auth(self, mock_get_auth):
        mock_get_auth.return_value = None
        result = await vobiz.get_vobiz_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

    async def test_http_status_error(self, mock_httpx, mock_get_auth):
        error_response = MagicMock()
        error_response.text = "Conflict"
        error_request = MagicMock()
        mock_httpx.get.side_effect = httpx.HTTPStatusError("Error", request=error_request, response=error_response)
        result = await vobiz.get_vobiz_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

    async def test_request_error(self, mock_httpx, mock_get_auth):
        mock_request = MagicMock()
        mock_httpx.get.side_effect = httpx.RequestError("Timeout", request=mock_request)
        result = await vobiz.get_vobiz_numbers("org1")
        assert result["status"] == "fail"
        assert result["numbers"] == []

class TestGetVobizAuthForOrg:
    @patch("app.services.vobiz._integration_service.get_integration")
    def test_success(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "VobizAuthId":
                return {"api_key": "id_123"}
            if integration_type == "VobizAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = vobiz._get_vobiz_auth_for_org("org1")
        assert result == ("id_123", "tok_123")

    @patch("app.services.vobiz._integration_service.get_integration")
    def test_missing_id(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "VobizAuthId":
                return None
            if integration_type == "VobizAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = vobiz._get_vobiz_auth_for_org("org1")
        assert result is None

    @patch("app.services.vobiz._integration_service.get_integration")
    def test_missing_token(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "VobizAuthId":
                return {"api_key": "id_123"}
            if integration_type == "VobizAuthToken":
                return None
        mock_get_integration.side_effect = side_effect
        
        result = vobiz._get_vobiz_auth_for_org("org1")
        assert result is None

    @patch("app.services.vobiz._integration_service.get_integration")
    def test_empty_keys(self, mock_get_integration):
        def side_effect(org_id, integration_type):
            if integration_type == "VobizAuthId":
                return {"api_key": " "}
            if integration_type == "VobizAuthToken":
                return {"api_key": "tok_123"}
        mock_get_integration.side_effect = side_effect
        
        result = vobiz._get_vobiz_auth_for_org("org1")
        assert result is None
