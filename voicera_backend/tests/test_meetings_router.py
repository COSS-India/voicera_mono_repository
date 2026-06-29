"""
Integration tests for /api/v1/meetings endpoints.

Bot endpoints (create/update meeting) use internal API key auth.
Frontend endpoints (read meetings) use JWT auth.
Both are mocked in the session `client` fixture.
"""

import pytest
from unittest.mock import patch

BASE = "/api/v1/meetings"

MEETING_DOC = {
    "meeting_id": "meet-001",
    "agent_type": "sales_bot",
    "org_id": "testorg1",
    "inbound": False,
    "from_number": "+910000000001",
    "to_number": "+910000000002",
    "start_time_utc": "2024-01-01T10:00:00",
    "end_time_utc": "2024-01-01T10:05:00",
    "duration": 300.0,
    "created_at": "2024-01-01T10:00:00",
    "call_busy": False,
}

PAGINATED_RESPONSE = {
    "items": [MEETING_DOC],
    "total": 1,
    "page": 1,
    "limit": 50,
}

FILTER_OPTIONS = {
    "agent_types": ["sales_bot", "support_bot"],
    "from_numbers": ["+910000000001"],
    "to_numbers": ["+910000000002"],
}


# ── Bot: POST / (create meeting) ───────────────────────────────────────────

class TestCreateMeeting:
    CREATE_BODY = {
        "meeting_id": "meet-001",
        "agent_type": "sales_bot",
        "org_id": "testorg1",
    }

    def test_success_returns_201(self, client):
        with patch("app.services.meeting_service.setup_meeting_id", return_value=MEETING_DOC):
            resp = client.post(BASE, json=self.CREATE_BODY)
        assert resp.status_code == 201

    def test_service_failure_returns_400(self, client):
        error = {"status": "fail", "message": "Invalid meeting data"}
        with patch("app.services.meeting_service.setup_meeting_id", return_value=error):
            resp = client.post(BASE, json=self.CREATE_BODY)
        assert resp.status_code == 400

    def test_missing_meeting_id_returns_422(self, client):
        resp = client.post(BASE, json={"agent_type": "sales_bot"})
        assert resp.status_code == 422

    def test_missing_agent_type_returns_422(self, client):
        resp = client.post(BASE, json={"meeting_id": "m-001"})
        assert resp.status_code == 422

    def test_inbound_call_creates_meeting(self, client):
        inbound_body = {**self.CREATE_BODY, "inbound": True, "from_number": "+910000000099"}
        with patch("app.services.meeting_service.setup_meeting_id", return_value=MEETING_DOC):
            resp = client.post(BASE, json=inbound_body)
        assert resp.status_code == 201

    def test_busy_call_creates_meeting(self, client):
        busy_body = {**self.CREATE_BODY, "call_busy": True}
        with patch("app.services.meeting_service.setup_meeting_id", return_value=MEETING_DOC):
            resp = client.post(BASE, json=busy_body)
        assert resp.status_code == 201

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.post(BASE, json=self.CREATE_BODY)
        assert resp.status_code in (401, 403)


# ── Bot: PATCH /{meeting_id} (end meeting) ─────────────────────────────────

class TestUpdateMeeting:
    def test_update_success_returns_200(self, client):
        ok = {"status": "success", "meeting_id": "meet-001"}
        with patch("app.services.meeting_service.update_meeting_end_time", return_value=ok):
            resp = client.patch(
                f"{BASE}/meet-001",
                json={"end_time_utc": "2024-01-01T10:05:00"},
            )
        assert resp.status_code == 200

    def test_update_not_found_returns_404(self, client):
        error = {"status": "fail", "message": "Meeting not found"}
        with patch("app.services.meeting_service.update_meeting_end_time", return_value=error):
            resp = client.patch(
                f"{BASE}/ghost-meet",
                json={"end_time_utc": "2024-01-01T10:05:00"},
            )
        assert resp.status_code == 404

    def test_missing_end_time_returns_422(self, client):
        resp = client.patch(f"{BASE}/meet-001", json={})
        assert resp.status_code == 422

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.patch(f"{BASE}/meet-001", json={"end_time_utc": "2024-01-01T10:05:00"})
        assert resp.status_code in (401, 403)

# ── Frontend: GET /filter-options ─────────────────────────────────────────

class TestGetFilterOptions:
    def test_returns_filter_options(self, client):
        with patch("app.services.meeting_service.fetch_meeting_filter_options", return_value=FILTER_OPTIONS):
            resp = client.get(f"{BASE}/filter-options")
        assert resp.status_code == 200
        data = resp.json()
        assert "agent_types" in data
        assert "from_numbers" in data
        assert "to_numbers" in data

    def test_empty_filter_options(self, client):
        empty = {"agent_types": [], "from_numbers": [], "to_numbers": []}
        with patch("app.services.meeting_service.fetch_meeting_filter_options", return_value=empty):
            resp = client.get(f"{BASE}/filter-options")
        assert resp.status_code == 200


# ── Frontend: GET / (paginated list) ──────────────────────────────────────

class TestGetMeetingsPaginated:
    def test_returns_paginated_response(self, client):
        with patch("app.services.meeting_service.fetch_meetings_paginated", return_value=PAGINATED_RESPONSE):
            resp = client.get(BASE)
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert data["total"] == 1

    def test_pagination_params_forwarded(self, client):
        with patch("app.services.meeting_service.fetch_meetings_paginated", return_value=PAGINATED_RESPONSE) as mock_svc:
            resp = client.get(f"{BASE}?page=2&limit=10")
        assert resp.status_code == 200
        _, kwargs = mock_svc.call_args
        assert kwargs.get("page") == 2
        assert kwargs.get("limit") == 10

    def test_agent_type_filter_forwarded(self, client):
        with patch("app.services.meeting_service.fetch_meetings_paginated", return_value=PAGINATED_RESPONSE) as mock_svc:
            client.get(f"{BASE}?agent_type=sales_bot")
        _, kwargs = mock_svc.call_args
        assert kwargs.get("agent_type") == "sales_bot"

    def test_inbound_filter_forwarded(self, client):
        with patch("app.services.meeting_service.fetch_meetings_paginated", return_value=PAGINATED_RESPONSE) as mock_svc:
            client.get(f"{BASE}?inbound=true")
        _, kwargs = mock_svc.call_args
        assert kwargs.get("inbound") is True

    def test_page_less_than_1_returns_422(self, client):
        resp = client.get(f"{BASE}?page=0")
        assert resp.status_code == 422

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.get(BASE)
        assert resp.status_code in (401, 403)


# ── Frontend: GET /{meeting_id} ────────────────────────────────────────────

class TestGetMeetingById:
    def test_success_returns_meeting(self, client):
        transformed = {**MEETING_DOC, "transcript": [], "recording_url": None}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=MEETING_DOC), \
             patch("app.services.meeting_service.transform_meeting_for_frontend", return_value=transformed):
            resp = client.get(f"{BASE}/meet-001")
        assert resp.status_code == 200
        assert resp.json()["meeting_id"] == "meet-001"

    def test_not_found_returns_404(self, client):
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=None):
            resp = client.get(f"{BASE}/ghost-meet")
        assert resp.status_code == 404

    def test_wrong_org_returns_403(self, client):
        wrong_org = {**MEETING_DOC, "org_id": "otherorg9"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=wrong_org):
            resp = client.get(f"{BASE}/meet-001")
        assert resp.status_code == 403

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.get(f"{BASE}/meet-001")
        assert resp.status_code in (401, 403)


# ── Frontend: GET /{meeting_id}/recording ──────────────────────────────────

class TestGetMeetingRecording:
    def test_success_returns_streaming_response(self, client):
        meeting = {**MEETING_DOC, "recording_url": "minio://bucket/record.wav"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=meeting), \
             patch("app.routers.meetings.MinIOStorage.parse_minio_url", return_value=("bucket", "record.wav")), \
             patch("app.routers.meetings.MinIOStorage.object_exists", return_value=True):
             
            from unittest.mock import AsyncMock, MagicMock
            mock_response = MagicMock()
            mock_response.headers = {"content-length": "100"}
            mock_response.stream.return_value = [b"audio", b"data"]
            
            with patch("app.routers.meetings.MinIOStorage.get_object", new_callable=AsyncMock, return_value=mock_response):
                resp = client.get(f"{BASE}/meet-001/recording")
        
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "audio/wav"
        assert resp.content == b"audiodata"

    def test_not_found_returns_404(self, client):
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=None):
            resp = client.get(f"{BASE}/ghost-meet/recording")
        assert resp.status_code == 404

    def test_wrong_org_returns_403(self, client):
        wrong_org = {**MEETING_DOC, "org_id": "otherorg9"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=wrong_org):
            resp = client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code == 403

    def test_no_recording_url_returns_404(self, client):
        no_rec = {**MEETING_DOC, "recording_url": None}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=no_rec):
            resp = client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code == 404

    def test_http_url_returns_400(self, client):
        http_rec = {**MEETING_DOC, "recording_url": "https://example.com/audio.wav"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=http_rec):
            resp = client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code == 400

    def test_invalid_minio_url_returns_400(self, client):
        bad_rec = {**MEETING_DOC, "recording_url": "minio://invalid"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=bad_rec), \
             patch("app.routers.meetings.MinIOStorage.parse_minio_url", return_value=None):
            resp = client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code == 400

    def test_object_does_not_exist_returns_404(self, client):
        meeting = {**MEETING_DOC, "recording_url": "minio://bucket/record.wav"}
        with patch("app.services.meeting_service.fetch_meeting_details", return_value=meeting), \
             patch("app.routers.meetings.MinIOStorage.parse_minio_url", return_value=("bucket", "record.wav")), \
             patch("app.routers.meetings.MinIOStorage.object_exists", return_value=False):
            resp = client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code == 404

    def test_unauthenticated_returns_4xx(self, unauth_client):
        resp = unauth_client.get(f"{BASE}/meet-001/recording")
        assert resp.status_code in (401, 403)
