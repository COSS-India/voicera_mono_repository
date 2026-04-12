"""
Agent assets API: org-scoped non-conversational audio upload.
"""
from __future__ import annotations

import io
import uuid
import wave
from typing import Any, Dict

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydub import AudioSegment

from app.auth import get_current_user
from app.storage.minio_client import MinIOStorage

router = APIRouter(prefix="/agent-assets", tags=["agent-assets"])

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


def _wav_meta(data: bytes) -> Dict[str, Any]:
    with wave.open(io.BytesIO(data), "rb") as wf:
        frames = wf.getnframes()
        sample_rate = wf.getframerate() or 0
        duration_ms = int((frames / sample_rate) * 1000) if sample_rate else 0
        return {
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "mime_type": "audio/wav",
            "num_channels": wf.getnchannels(),
        }


def _to_wav(raw: bytes, filename: str, content_type: str | None) -> bytes:
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    is_wav = name.endswith(".wav") or ctype in ("audio/wav", "audio/x-wav")
    is_mp3 = name.endswith(".mp3") or ctype in ("audio/mpeg", "audio/mp3")

    if not (is_wav or is_mp3):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only WAV and MP3 files are supported",
        )

    if is_wav:
        try:
            _wav_meta(raw)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid WAV file",
            ) from None
        return raw

    # MP3 -> WAV conversion using ffmpeg+pydub.
    try:
        segment = AudioSegment.from_file(io.BytesIO(raw), format="mp3")
        out = io.BytesIO()
        segment.export(out, format="wav")
        wav_data = out.getvalue()
        _wav_meta(wav_data)
        return wav_data
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to decode MP3. Please upload a valid WAV or MP3 file.",
        ) from None


@router.post("/upload")
async def upload_non_conversational_audio(
    file: UploadFile = File(...),
    org_id: str = Form(...),
    agent_id: str = Form(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Upload non-conversational audio asset. Stores as WAV in org-scoped MinIO path.
    """
    if org_id != current_user.get("org_id"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized for this organization",
        )

    if not agent_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="agent_id is required",
        )

    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large (max {MAX_UPLOAD_BYTES // (1024 * 1024)} MB)",
        )

    wav_data = _to_wav(content, file.filename or "", file.content_type)
    meta = _wav_meta(wav_data)

    asset_id = str(uuid.uuid4())
    object_name = f"non_conversational/{org_id}/{agent_id}/{asset_id}.wav"
    storage = MinIOStorage()
    await storage.put_object_bytes(
        bucket_name="agent-assets",
        object_name=object_name,
        data=wav_data,
        content_type="audio/wav",
    )
    audio_url = f"minio://agent-assets/{object_name}"

    return {
        "audio_url": audio_url,
        "audio_meta": meta,
        "asset_id": asset_id,
    }

