"""Vobiz native call recording via Record API."""

import asyncio
import os
from typing import Optional, Tuple

import requests
from loguru import logger

from .backend_utils import fetch_integration_key

DEFAULT_POLL_ATTEMPTS = 10
DEFAULT_POLL_INTERVAL_SECS = 2.0


def _get_vobiz_api_base() -> str:
    base = os.environ.get("VOBIZ_API_BASE")
    if not base:
        raise ValueError("Missing required environment variable: VOBIZ_API_BASE")
    return base.rstrip("/")


def _get_vobiz_auth(org_id: str) -> Optional[Tuple[str, str]]:
    auth_id = fetch_integration_key(org_id, "VobizAuthId")
    auth_token = fetch_integration_key(org_id, "VobizAuthToken")
    if not auth_id or not auth_token:
        logger.error(f"Vobiz credentials not found for org_id={org_id}")
        return None
    return auth_id, auth_token


def _vobiz_headers(org_id: str) -> Optional[dict]:
    auth = _get_vobiz_auth(org_id)
    if not auth:
        return None
    auth_id, auth_token = auth
    return {
        "X-Auth-ID": auth_id,
        "X-Auth-Token": auth_token,
        "Content-Type": "application/json",
    }


async def start_vobiz_call_recording(
    call_sid: str,
    org_id: str,
    time_limit_secs: int,
) -> Optional[str]:
    """Start Vobiz call recording. Returns recording_id or None on failure."""
    headers = _vobiz_headers(org_id)
    if not headers:
        return None

    auth_id = headers["X-Auth-ID"]
    url = f"{_get_vobiz_api_base()}/Account/{auth_id}/Call/{call_sid}/Record/"
    payload = {
        "time_limit": time_limit_secs,
        "file_format": "mp3",
        "record_channel_type": "mono",
    }

    def _post():
        return requests.post(url, json=payload, headers=headers, timeout=30)

    try:
        response = await asyncio.to_thread(_post)
        response.raise_for_status()
        data = response.json()
        recording_id = (
            data.get("recording_id")
            or data.get("recording_uuid")
            or data.get("uuid")
            or data.get("recordingId")
        )
        logger.info(
            f"Started Vobiz recording: call_sid={call_sid} recording_id={recording_id}"
        )
        return recording_id
    except Exception as e:
        logger.error(f"Failed to start Vobiz recording for {call_sid}: {e}")
        return None


async def fetch_vobiz_recording_metadata(
    recording_id: str,
    org_id: str,
) -> Optional[dict]:
    """Fetch recording metadata from Vobiz API."""
    headers = _vobiz_headers(org_id)
    if not headers:
        return None

    auth_id = headers["X-Auth-ID"]
    url = f"{_get_vobiz_api_base()}/Account/{auth_id}/Recording/{recording_id}/"

    def _get():
        return requests.get(url, headers=headers, timeout=30)

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.debug(f"Vobiz recording metadata fetch failed for {recording_id}: {e}")
        return None


async def download_vobiz_recording(recording_url: str, org_id: str) -> Optional[bytes]:
    """Download recording file bytes from Vobiz URL."""
    headers = _vobiz_headers(org_id)
    if not headers:
        return None

    download_headers = {
        "X-Auth-ID": headers["X-Auth-ID"],
        "X-Auth-Token": headers["X-Auth-Token"],
    }

    def _get():
        return requests.get(recording_url, headers=download_headers, timeout=120)

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        return response.content
    except Exception as e:
        logger.error(f"Failed to download Vobiz recording from {recording_url}: {e}")
        return None


async def wait_and_download_vobiz_recording(
    recording_id: str,
    org_id: str,
    max_attempts: int = DEFAULT_POLL_ATTEMPTS,
    interval_secs: float = DEFAULT_POLL_INTERVAL_SECS,
) -> Optional[bytes]:
    """Poll until recording_url is ready, then download once."""
    for attempt in range(1, max_attempts + 1):
        metadata = await fetch_vobiz_recording_metadata(recording_id, org_id)
        if metadata:
            recording_url = metadata.get("recording_url") or metadata.get("url")
            if recording_url:
                audio_bytes = await download_vobiz_recording(recording_url, org_id)
                if audio_bytes:
                    logger.info(
                        f"Downloaded Vobiz recording {recording_id} ({len(audio_bytes)} bytes)"
                    )
                    return audio_bytes

        if attempt < max_attempts:
            logger.debug(
                f"Vobiz recording not ready (attempt {attempt}/{max_attempts}), retrying..."
            )
            await asyncio.sleep(interval_secs)

    logger.warning(
        f"Vobiz recording not ready after {max_attempts} attempts: {recording_id}"
    )
    return None
