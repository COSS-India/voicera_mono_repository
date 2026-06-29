"""Tests for app/routers/vobiz.py"""

from unittest.mock import patch, AsyncMock

BASE = "/api/v1/vobiz"

SUCCESS_APP = {"status": "success", "message": "Created", "app_id": "vapp_123"}
FAIL_APP = {"status": "fail", "message": "Bad credentials"}
SUCCESS_NUMS = {"status": "success", "numbers": ["+9876543210"]}
FAIL_NUMS = {"status": "fail", "message": "No vobiz integration"}
SUCCESS_DEL = {"status": "success", "message": "Deleted"}
SUCCESS_LINK = {"status": "success", "message": "Linked"}
SUCCESS_UNLINK = {"status": "success", "message": "Unlinked"}


class TestCreateVobizApplicationEndpoint:
    def test_success_returns_201(self, client):
        with patch("app.routers.vobiz.vobiz.create_vobiz_application", new_callable=AsyncMock, return_value=SUCCESS_APP):
            resp = client.post(BASE + "/application", json={"agent_type": "sales_bot", "answer_url": "http://x.com/answer"})
        assert resp.status_code == 201
        assert resp.json()["app_id"] == "vapp_123"

    def test_exception_in_service_returns_500(self, client):
        with patch("app.routers.vobiz.vobiz.create_vobiz_application", new_callable=AsyncMock, side_effect=RuntimeError("crash")):
            resp = client.post(BASE + "/application", json={"agent_type": "sales_bot", "answer_url": "http://x.com"})
        assert resp.status_code == 500


class TestGetVobizNumbersEndpoint:
    def test_success_returns_numbers(self, client):
        with patch("app.routers.vobiz.vobiz.get_vobiz_numbers", new_callable=AsyncMock, return_value=SUCCESS_NUMS):
            resp = client.get(BASE + "/numbers")
        assert resp.status_code == 200
        assert resp.json()["numbers"] == ["+9876543210"]

    def test_fail_status_returns_400(self, client):
        with patch("app.routers.vobiz.vobiz.get_vobiz_numbers", new_callable=AsyncMock, return_value=FAIL_NUMS):
            resp = client.get(BASE + "/numbers")
        assert resp.status_code == 400


class TestDeleteVobizApplicationEndpoint:
    def test_success_returns_200(self, client):
        with patch("app.routers.vobiz.vobiz.delete_vobiz_application", new_callable=AsyncMock, return_value=SUCCESS_DEL):
            resp = client.delete(BASE + "/application/vapp_123")
        assert resp.status_code == 200

    def test_fail_status_returns_400(self, client):
        with patch("app.routers.vobiz.vobiz.delete_vobiz_application", new_callable=AsyncMock, return_value=FAIL_APP):
            resp = client.delete(BASE + "/application/vapp_123")
        assert resp.status_code == 400

    def test_exception_returns_500(self, client):
        with patch("app.routers.vobiz.vobiz.delete_vobiz_application", new_callable=AsyncMock, side_effect=RuntimeError("crash")):
            resp = client.delete(BASE + "/application/vapp_123")
        assert resp.status_code == 500


class TestLinkVobizNumberEndpoint:
    PAYLOAD = {"phone_number": "+9876543210", "application_id": "vapp_123"}

    def test_success_returns_201(self, client):
        with patch("app.routers.vobiz.vobiz.link_number_to_application", new_callable=AsyncMock, return_value=SUCCESS_LINK):
            resp = client.post(BASE + "/numbers/link", json=self.PAYLOAD)
        assert resp.status_code == 201

    def test_fail_status_returns_400(self, client):
        with patch("app.routers.vobiz.vobiz.link_number_to_application", new_callable=AsyncMock, return_value=FAIL_APP):
            resp = client.post(BASE + "/numbers/link", json=self.PAYLOAD)
        assert resp.status_code == 400

    def test_exception_returns_500(self, client):
        with patch("app.routers.vobiz.vobiz.link_number_to_application", new_callable=AsyncMock, side_effect=RuntimeError("crash")):
            resp = client.post(BASE + "/numbers/link", json=self.PAYLOAD)
        assert resp.status_code == 500


class TestUnlinkVobizNumberEndpoint:
    PAYLOAD = {"phone_number": "+9876543210"}

    def test_success_returns_200(self, client):
        with patch("app.routers.vobiz.vobiz.unlink_number_from_application", new_callable=AsyncMock, return_value=SUCCESS_UNLINK):
            resp = client.request("DELETE", BASE + "/numbers/unlink", json=self.PAYLOAD)
        assert resp.status_code == 200

    def test_fail_status_returns_400(self, client):
        with patch("app.routers.vobiz.vobiz.unlink_number_from_application", new_callable=AsyncMock, return_value=FAIL_APP):
            resp = client.request("DELETE", BASE + "/numbers/unlink", json=self.PAYLOAD)
        assert resp.status_code == 400

    def test_exception_returns_500(self, client):
        with patch("app.routers.vobiz.vobiz.unlink_number_from_application", new_callable=AsyncMock, side_effect=RuntimeError("crash")):
            resp = client.request("DELETE", BASE + "/numbers/unlink", json=self.PAYLOAD)
        assert resp.status_code == 500
