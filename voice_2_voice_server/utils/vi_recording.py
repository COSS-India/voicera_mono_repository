"""Download VI CPaaS call recordings from flow-builder webhook URLs."""

from __future__ import annotations

import asyncio
import mimetypes
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from loguru import logger

from storage.minio_client import MinIOStorage
from utils.call_recording_utils import patch_call_recording_url
from utils.vi_recording_webhook import extract_call_id_from_recording_url

DEFAULT_RETRY_ATTEMPTS = 6
DEFAULT_RETRY_INTERVAL_SECS = 2.0

CONTENT_TYPE_TO_EXT = {
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mp4": "m4a",
    "application/octet-stream": "wav",
}


def guess_extension(recording_url: str, content_type: Optional[str]) -> str:
    """Infer file extension from response Content-Type or URL path."""
    if content_type:
        normalized = content_type.split(";", 1)[0].strip().lower()
        ext = CONTENT_TYPE_TO_EXT.get(normalized)
        if ext:
            return ext
        guessed = mimetypes.guess_extension(normalized)
        if guessed:
            return guessed.lstrip(".")

    path = urlparse(recording_url).path.lower()
    for ext in ("mp3", "wav", "m4a"):
        if path.endswith(f".{ext}"):
            return ext
    return "wav"


async def download_vi_recording(recording_url: str) -> Tuple[Optional[bytes], str]:
    """Download recording bytes from VI signed fetch URL (auth is in query params)."""

    def _get():
        return requests.get(recording_url, timeout=120)

    try:
        response = await asyncio.to_thread(_get)
        response.raise_for_status()
        content_type = response.headers.get("content-type")
        ext = guess_extension(recording_url, content_type)
        audio_bytes = response.content
        if not audio_bytes:
            logger.warning("VI recording download returned empty body for {}", recording_url)
            return None, ext
        logger.info(
            "Downloaded VI recording ({} bytes, ext={}) from {}",
            len(audio_bytes),
            ext,
            recording_url.split("?", 1)[0],
        )
        return audio_bytes, ext
    except Exception as e:
        logger.error("Failed to download VI recording from {}: {}", recording_url.split("?", 1)[0], e)
        return None, "wav"


async def wait_and_download_vi_recording(
    recording_url: str,
    max_attempts: int = DEFAULT_RETRY_ATTEMPTS,
    interval_secs: float = DEFAULT_RETRY_INTERVAL_SECS,
) -> Tuple[Optional[bytes], str]:
    """Retry VI download; recording may not be ready immediately after hangup."""
    ext = "wav"
    for attempt in range(1, max_attempts + 1):
        audio_bytes, ext = await download_vi_recording(recording_url)
        if audio_bytes:
            return audio_bytes, ext
        if attempt < max_attempts:
            logger.debug(
                "VI recording not ready (attempt {}/{}), retrying in {}s...",
                attempt,
                max_attempts,
                interval_secs,
            )
            await asyncio.sleep(interval_secs)

    logger.warning(
        "VI recording not available after {} attempts for {}",
        max_attempts,
        recording_url.split("?", 1)[0],
    )
    return None, ext


async def ingest_vi_recording_from_url(
    recording_url: str,
    call_sid: Optional[str] = None,
) -> bool:
    """
    Download VI recording, save to MinIO, and patch backend recording_url.

    Returns True when the provider recording is stored and backend is updated.
    """
    resolved_call_sid = call_sid or extract_call_id_from_recording_url(recording_url)
    if not resolved_call_sid:
        logger.error("VI recording ingest: could not resolve call_sid from URL or payload")
        return False

    audio_bytes, ext = await wait_and_download_vi_recording(recording_url)
    if not audio_bytes:
        logger.error("VI recording ingest failed for call_sid={}", resolved_call_sid)
        return False

    storage = MinIOStorage.from_env()
    await storage.save_recording_bytes(resolved_call_sid, audio_bytes, ext)
    minio_url = f"minio://recordings/{resolved_call_sid}.{ext}"

    updated = await patch_call_recording_url(resolved_call_sid, minio_url)
    if updated:
        logger.info(
            "VI recording ingest complete: call_sid={} url={}",
            resolved_call_sid,
            minio_url,
        )
    return updated
