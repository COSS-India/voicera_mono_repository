"""Plivo native call recording via Record API."""

import asyncio
import os
from typing import Optional, Tuple

import requests
from loguru import logger

from .backend_utils import fetch_integration_key

DEFAULT_POLL_ATTEMPTS = 10
DEFAULT_POLL_INTERVAL_SECS = 2.0


def _get_plivo_api_base() -> str:
    return os.environ.get("PLIVO_API_BASE", "https://api.plivo.com/v1").rstrip("/")


def _get_plivo_auth(org_id: str) -> Optional[Tuple[str, str]]:
    auth_id = fetch_integration_key(org_id, "PlivoAuthId")
    auth_token = fetch_integration_key(org_id, "PlivoAuthToken")
    if not auth_id or not auth_token:
        logger.error(f"Plivo credentials not found for org_id={org_id}")
        return None
    return auth_id, auth_token


def _extract_recording_id(data: dict) -> Optional[str]:
    if not isinstance(data, dict):
        return None
    return (
        data.get("recording_id")
        or data.get("recording_uuid")
        or data.get("recordingId")
        or data.get("uuid")
    )


def _extract_recording_url(metadata: dict) -> Optional[str]:
    if not isinstance(metadata, dict):
        return None
    return (
        metadata.get("recording_url")
        or metadata.get("recordingUrl")
        or metadata.get("url")
    )


async def start_plivo_call_recording(
    call_uuid: str,
    org_id: str,
    time_limit_secs: int,
) -> Optional[str]:
    """Start Plivo call recording. Returns recording_id or None on failure."""
    auth = _get_plivo_auth(org_id)
    if not auth:
        return None

    auth_id, auth_token = auth
    url = f"{_get_plivo_api_base()}/Account/{auth_id}/Call/{call_uuid}/Record/"
    payload = {
        "time_limit": time_limit_secs,
        "file_format": "mp3",
    }

    def _post():
        return requests.post(
            url,
            json=payload,
            auth=(auth_id, auth_token),
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            timeout=30,
        )

    try:
        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        recording_id = _extract_recording_id(data)
        logger.info(
            f"Started Plivo recording: call_uuid={call_uuid} recording_id={recording_id}"
        )
        return recording_id
    except Exception as e:
        logger.error(f"Failed to start Plivo recording for {call_uuid}: {e}")
        return None


async def fetch_plivo_recording_metadata(
    recording_id: str,
    org_id: str,
) -> Optional[dict]:
    """Fetch recording metadata from Plivo API."""
    auth = _get_plivo_auth(org_id)
    if not auth:
        return None

    auth_id, auth_token = auth
    url = f"{_get_plivo_api_base()}/Account/{auth_id}/Recording/{recording_id}/"

    def _get():
        return requests.get(
            url,
            auth=(auth_id, auth_token),
            headers={"Accept": "application/json"},
            timeout=30,
        )

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.debug(f"Plivo recording metadata fetch failed for {recording_id}: {e}")
        return None


async def list_plivo_recordings_for_call(
    call_uuid: str,
    org_id: str,
) -> Optional[dict]:
    """List recordings for a call UUID (fallback when recording_id is unknown)."""
    auth = _get_plivo_auth(org_id)
    if not auth:
        return None

    auth_id, auth_token = auth
    url = f"{_get_plivo_api_base()}/Account/{auth_id}/Recording/"
    params = {"call_uuid": call_uuid, "limit": 1}

    def _get():
        return requests.get(
            url,
            params=params,
            auth=(auth_id, auth_token),
            headers={"Accept": "application/json"},
            timeout=30,
        )

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        data = response.json()
        objects = data.get("objects") if isinstance(data, dict) else None
        if objects:
            return objects[0]
        return None
    except Exception as e:
        logger.debug(f"Plivo recording list failed for call {call_uuid}: {e}")
        return None


async def download_plivo_recording(recording_url: str, org_id: str) -> Optional[bytes]:
    """Download recording file bytes from Plivo URL (Basic auth if required)."""
    auth = _get_plivo_auth(org_id)
    if not auth:
        return None

    auth_id, auth_token = auth

    def _get():
        return requests.get(recording_url, auth=(auth_id, auth_token), timeout=120)

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download Plivo recording from {recording_url}: {e}")
        return None


async def wait_and_download_plivo_recording(
    recording_id: Optional[str],
    org_id: str,
    call_uuid: Optional[str] = None,
    max_attempts: int = DEFAULT_POLL_ATTEMPTS,
    interval_secs: float = DEFAULT_POLL_INTERVAL_SECS,
) -> Optional[bytes]:
    """Poll until recording_url is ready, then download once."""
    resolved_id = recording_id

    for attempt in range(1, max_attempts + 1):
        metadata = None
        if resolved_id:
            metadata = await fetch_plivo_recording_metadata(resolved_id, org_id)
        elif call_uuid and attempt == 1:
            metadata = await list_plivo_recordings_for_call(call_uuid, org_id)
            if metadata:
                resolved_id = _extract_recording_id(metadata)

        if metadata:
            recording_url = _extract_recording_url(metadata)
            if recording_url:
                audio_bytes = await download_plivo_recording(recording_url, org_id)
                if audio_bytes:
                    logger.info(
                        f"Downloaded Plivo recording {resolved_id or call_uuid} "
                        f"({len(audio_bytes)} bytes)"
                    )
                    return audio_bytes

        if attempt < max_attempts:
            logger.debug(
                f"Plivo recording not ready (attempt {attempt}/{max_attempts}), retrying..."
            )
            await asyncio.sleep(interval_secs)

    logger.warning(
        f"Plivo recording not ready after {max_attempts} attempts: "
        f"recording_id={recording_id} call_uuid={call_uuid}"
    )
    return None
