"""VI CPaaS Outbound Campaign (OBD) API client."""

from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

import requests

FLOW_ID = "Or5NHOa3hND98TLJ/1pyCQ=="
CPAAS_HOST = "https://cts.myvi.in:8443"
CPAAS_API_ROOT = f"{CPAAS_HOST}/Cpaas/api/v1"
OBD_BASE = f"{CPAAS_API_ROOT}/obdcampaignapi"
TOKEN_TTL_SECS = 86400
TOKEN_REFRESH_MARGIN_SECS = 3600
REQUEST_TIMEOUT_SECS = 30
IST = ZoneInfo("Asia/Kolkata")


class ViObdError(Exception):
    """VI OBD API error."""


def get_dni_from_env() -> str:
    """Return the flow's attached outbound caller ID from VI_DNI."""
    dni = os.environ.get("VI_DNI", "").strip()
    if not dni:
        raise ViObdError(
            "Missing VI_DNI in environment / .env — "
            "set this to the phone number attached to the VI flow."
        )
    return dni


def _normalize_cpaas_base(base: str, default: str) -> str:
    """Ensure override URLs include the /Cpaas/api/v1/ prefix."""
    value = (base or "").strip().rstrip("/") or default
    if "/Cpaas/" in value or "/cpaas/" in value.lower():
        return value
    if "/api/v1/" in value:
        return value.replace("/api/v1/", "/Cpaas/api/v1/", 1)
    return default


class ViObdClient:
    """Client for VI CPaaS OBD endpoints."""

    def __init__(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        obd_base: str = OBD_BASE,
    ):
        self._username = username
        self._password = password
        self._obd_base = _normalize_cpaas_base(obd_base, OBD_BASE)
        self._token: Optional[str] = None
        self._token_fetched_at: float = 0.0
        self._last_request_url: Optional[str] = None

    @classmethod
    def from_env(cls) -> "ViObdClient":
        username = os.environ.get("VI_OBD_USERNAME", "").strip()
        password = os.environ.get("VI_OBD_PASSWORD", "").strip()
        if not username or not password:
            raise ViObdError(
                "Missing VI_OBD_USERNAME or VI_OBD_PASSWORD in environment / .env"
            )
        obd_base = _normalize_cpaas_base(
            os.environ.get("VI_OBD_BASE_URL", OBD_BASE), OBD_BASE
        )
        return cls(username=username, password=password, obd_base=obd_base)

    @staticmethod
    def _token_is_fresh(token: Optional[str], fetched_at: float) -> bool:
        if not token:
            return False
        age = time.monotonic() - fetched_at
        return age < (TOKEN_TTL_SECS - TOKEN_REFRESH_MARGIN_SECS)

    @staticmethod
    def _parse_response(response: requests.Response) -> Any:
        try:
            return response.json()
        except ValueError:
            return {"_raw_text": response.text}

    def _request(
        self,
        path: str,
        *,
        token: Optional[str] = None,
        json_body: Optional[dict] = None,
    ) -> tuple[Any, int]:
        url = f"{self._obd_base.rstrip('/')}/{path.lstrip('/')}"
        self._last_request_url = url
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        response = requests.post(
            url,
            json=json_body,
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECS,
        )
        return self._parse_response(response), response.status_code

    def get_auth_token(
        self,
        username: Optional[str] = None,
        password: Optional[str] = None,
        *,
        force_refresh: bool = False,
    ) -> tuple[str, dict]:
        """Authenticate against obdcampaignapi/AuthToken."""
        if not force_refresh and self._token_is_fresh(self._token, self._token_fetched_at):
            return self._token, {
                "idToken": self._token,
                "expiresIn": TOKEN_TTL_SECS,
                "_cached": True,
                "_request_url": f"{self._obd_base.rstrip('/')}/AuthToken",
            }

        user = username or self._username
        pwd = password or self._password
        if not user or not pwd:
            raise ViObdError("username and password are required for authentication")

        body, status = self._request(
            "AuthToken",
            json_body={"username": user, "password": pwd},
        )
        body = body if isinstance(body, dict) else {"_raw": body}
        body["_request_url"] = self._last_request_url

        if status != 200:
            raise ViObdError(
                f"AuthToken failed (HTTP {status}) at {self._last_request_url}: {body}"
            )

        id_token = body.get("idToken")
        if not id_token:
            raise ViObdError(
                f"AuthToken response missing idToken at {self._last_request_url}: {body}"
            )

        self._token = id_token
        self._token_fetched_at = time.monotonic()
        return id_token, body

    @staticmethod
    def _campaign_window(window_hours: float = 1.0) -> dict[str, str]:
        now = datetime.now(IST)
        end = now + timedelta(hours=window_hours)
        return {
            "fromdate": now.strftime("%Y-%m-%d"),
            "todate": now.strftime("%Y-%m-%d"),
            "fromtime": now.strftime("%H:%M:%S"),
            "totime": end.strftime("%H:%M:%S"),
        }

    def create_campaign(
        self,
        token: str,
        flow_id: str = FLOW_ID,
        *,
        name: Optional[str] = None,
        description: str = "test campaign from terminal script",
        window_hours: float = 1.0,
        dialtimeout: Optional[int] = None,
        retryintervaltype: int = 0,
        retryintervalvalue: int = 5,
        retrycount: int = 1,
    ) -> dict:
        """Step 1: create an OBD campaign with a near-immediate time window."""
        if dialtimeout is None:
            dialtimeout = int(os.environ.get("VI_OBD_DIAL_TIMEOUT", "30"))

        if not name:
            name = f"test-call-{datetime.now(IST).strftime('%Y%m%d-%H%M%S')}"

        window = self._campaign_window(window_hours)
        payload = {
            "flowid": flow_id,
            **window,
            "dialtimeout": dialtimeout,
            "name": name,
            "description": description,
            "retryintervaltype": retryintervaltype,
            "retryintervalvalue": retryintervalvalue,
            "retrycount": retrycount,
        }

        body, status = self._request(
            "createCampaign",
            token=token,
            json_body=payload,
        )
        body = body if isinstance(body, dict) else {"_raw": body}
        body["_request_url"] = self._last_request_url

        if status != 200:
            raise ViObdError(
                f"createCampaign failed (HTTP {status}) at {self._last_request_url}: {body}"
            )
        if body.get("status") != 1:
            raise ViObdError(
                f"createCampaign returned non-success status at {self._last_request_url}: {body}"
            )
        body["_request_payload"] = payload
        return body

    @staticmethod
    def _campain_key_from_response(create_response: dict) -> str:
        """Return campainKey (raw) — confirmed working campaign_ID for ingestion."""
        campain_key = create_response.get("campainKey") or create_response.get(
            "campaignKey"
        )
        if not campain_key:
            raise ViObdError(
                f"createCampaign response missing campainKey: {create_response}"
            )
        return str(campain_key)

    @staticmethod
    def _build_ingestion_payload(
        campaign_id: str,
        dni: Any,
        msisdn: Any,
        *,
        shape: Literal["nested", "nested_and_top_level", "flat"],
    ) -> dict:
        """Build staticCampaignDataIngestion JSON body.

        VI expects dni/msisdn inside each Records[] object (nested shape).
        Flat top-level dni/msisdn alone returns 409 on this account — kept
        only as a documented reference shape, not used in the retry chain.
        """
        record = {"dni": dni, "msisdn": msisdn}
        if shape == "nested":
            return {"campaign_ID": campaign_id, "Records": [record]}
        if shape == "nested_and_top_level":
            return {
                "campaign_ID": campaign_id,
                "dni": dni,
                "msisdn": msisdn,
                "Records": [record],
            }
        # flat — doc table shape; rejected with 409 "dni is missing in Records array JSON"
        return {
            "campaign_ID": campaign_id,
            "dni": dni,
            "msisdn": msisdn,
            "Records": [{}],
        }

    def upload_call_list(
        self,
        token: str,
        campaign_id: str,
        dni: Any,
        msisdn: Any,
        *,
        payload_shape: Literal["nested", "nested_and_top_level", "flat"] = "nested",
    ) -> tuple[dict, int]:
        """Step 2: ingest one number into the campaign."""
        payload = self._build_ingestion_payload(
            campaign_id, dni, msisdn, shape=payload_shape
        )
        body, status = self._request(
            "staticCampaignDataIngestion",
            token=token,
            json_body=payload,
        )
        body = body if isinstance(body, dict) else {"_raw": body}
        body["_request_url"] = self._last_request_url
        body["_request_payload"] = payload
        body["_payload_shape"] = payload_shape
        return body, status

    @staticmethod
    def _ingestion_succeeded(body: dict, status: int) -> bool:
        if status != 200:
            return False
        data = body.get("data") or {}
        if data.get("status") == "success":
            return True
        # Nested Records shape returns rowsAffected instead of status
        rows = data.get("rowsAffected")
        return isinstance(rows, int) and rows >= 1

    def upload_call_list_with_fallback(
        self,
        token: str,
        create_response: dict,
        dni: Any,
        msisdn: Any,
    ) -> tuple[dict, str]:
        """Ingest using campainKey (raw) and nested Records payload shapes."""
        campaign_id = self._campain_key_from_response(create_response)
        # nested first; belt-and-suspenders if VI still returns 409
        shapes: list[Literal["nested", "nested_and_top_level"]] = [
            "nested",
            "nested_and_top_level",
        ]

        attempts: list[dict] = []
        for shape in shapes:
            body, status = self.upload_call_list(
                token, campaign_id, dni, msisdn, payload_shape=shape
            )
            attempt_info = {
                "payload_shape": shape,
                "campaign_ID": campaign_id,
                "request_payload": body.get("_request_payload"),
                "http_status": status,
                "request_url": body.get("_request_url"),
                "response": {
                    k: v
                    for k, v in body.items()
                    if k not in ("_request_payload", "_payload_shape")
                },
            }
            attempts.append(attempt_info)

            if self._ingestion_succeeded(body, status):
                body["_successful_attempt"] = attempt_info
                body["_all_attempts"] = attempts
                return body, shape

            if status == 409:
                continue

            body["_all_attempts"] = attempts
            raise ViObdError(
                f"staticCampaignDataIngestion failed with HTTP {status} "
                f"using payload_shape={shape!r}: {body}"
            )

        raise ViObdError(
            "staticCampaignDataIngestion failed for all payload shapes. "
            f"Attempts: {attempts}"
        )

    def get_campaign_status(
        self,
        token: str,
        campaign_id: Any,
        *,
        id_kind: str = "campaign_Ref_ID",
    ) -> tuple[dict, int]:
        """Query campaignstatus for a campaign."""
        payload = {"campaign_ID": campaign_id}
        body, status = self._request(
            "campaignstatus",
            token=token,
            json_body=payload,
        )
        body = body if isinstance(body, dict) else {"_raw": body}
        body["_request_url"] = self._last_request_url
        body["_request_payload"] = payload
        body["_campaign_id_kind"] = id_kind
        return body, status

    def get_campaign_status_with_fallback(
        self,
        token: str,
        campaign_ref_id: str | int,
        campain_key: Optional[str] = None,
    ) -> tuple[dict, str]:
        """Query status — campaign_Ref_ID (numeric) first, campainKey if 400/406."""
        ref_value: Any = campaign_ref_id
        if isinstance(campaign_ref_id, str) and campaign_ref_id.isdigit():
            ref_value = int(campaign_ref_id)

        attempts: list[dict] = []
        body, status = self.get_campaign_status(
            token, ref_value, id_kind="campaign_Ref_ID"
        )
        attempts.append(self._status_attempt(body, status))

        if status == 200:
            body["_successful_attempt"] = attempts[-1]
            body["_all_attempts"] = attempts
            return body, "campaign_Ref_ID"

        if status in (400, 406) and campain_key:
            body, status = self.get_campaign_status(
                token, campain_key, id_kind="campainKey"
            )
            attempts.append(self._status_attempt(body, status))
            if status == 200:
                body["_successful_attempt"] = attempts[-1]
                body["_all_attempts"] = attempts
                return body, "campainKey"

        body["_all_attempts"] = attempts
        raise ViObdError(
            f"campaignstatus failed (HTTP {status}) for campaign_Ref_ID={campaign_ref_id!r}"
            + (f"; campainKey fallback also failed" if campain_key else "")
            + f": {body}"
        )

    @staticmethod
    def _status_attempt(body: dict, status: int) -> dict:
        return {
            "campaign_id_kind": body.get("_campaign_id_kind"),
            "request_payload": body.get("_request_payload"),
            "http_status": status,
            "request_url": body.get("_request_url"),
            "response": {
                k: v
                for k, v in body.items()
                if not k.startswith("_") or k == "_raw_text"
            },
        }
