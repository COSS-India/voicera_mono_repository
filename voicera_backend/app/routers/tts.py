"""
TTS router — /api/v1/tts

Endpoints:
  POST   /synthesize          Synthesize speech (proxies to OmniVoice server)
  POST   /ref-audio            Upload + store reference audio in MinIO
  GET    /ref-audio            List stored reference audios for the current org
  GET    /ref-audio/{key}      Download a stored reference audio
  DELETE /ref-audio/{key}      Delete a stored reference audio
  GET    /languages            Supported language list
  GET    /voice-designs        Predefined voice design presets
"""

from __future__ import annotations

import io
import os
import uuid
from typing import Any, Dict, List, Optional

import aiohttp
from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from fastapi.responses import Response, StreamingResponse

from app.auth import get_current_user
from app.storage.minio_client import MinIOStorage

router = APIRouter(prefix="/tts", tags=["tts"])

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OMNIVOICE_URL: str = (
    os.getenv("OMNIVOICE_SERVER_URL", "http://localhost:8001").rstrip("/")
)
REF_AUDIO_BUCKET = "tts-ref-audio"

_minio = MinIOStorage()

# ---------------------------------------------------------------------------
# Language map (display name → ISO 639-3)
# ---------------------------------------------------------------------------

OMNIVOICE_LANG_MAP: dict[str, str] = {
    "English": "en",
    "English (India)": "en",
    "English (United States)": "en",
    "Hindi": "hi",
    "Bengali": "bn",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Punjabi": "pa",
    "Odia": "or",
    "Assamese": "as",
    "Urdu": "ur",
    "Nepali": "ne",
    "Sanskrit": "sa",
    "Bodo": "brx",
    "Dogri": "doi",
    "Konkani": "kok",
    "Kashmiri": "ks",
    "Maithili": "mai",
    "Manipuri": "mni",
    "Santali": "sat",
    "Sindhi": "sd",
    "Chinese": "zh",
    "Japanese": "ja",
    "Korean": "ko",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Portuguese": "pt",
    "Arabic": "ar",
    "Russian": "ru",
    "Italian": "it",
    "Dutch": "nl",
    "Turkish": "tr",
    "Polish": "pl",
    "Vietnamese": "vi",
    "Thai": "th",
    "Indonesian": "id",
    "Malay": "ms",
    "Swahili": "sw",
    "Ukrainian": "uk",
    "Czech": "cs",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Greek": "el",
    "Hebrew": "he",
    "Finnish": "fi",
    "Swedish": "sv",
    "Danish": "da",
    "Norwegian": "no",
}

# ---------------------------------------------------------------------------
# Voice design presets
# ---------------------------------------------------------------------------

VOICE_DESIGN_PRESETS: list[dict] = [
    {"id": "male_neutral",    "label": "Male · Neutral",         "instruct": "male, young adult, moderate pitch"},
    {"id": "female_neutral",  "label": "Female · Neutral",       "instruct": "female, young adult, moderate pitch"},
    {"id": "male_deep",       "label": "Male · Deep",            "instruct": "male, middle-aged, low pitch"},
    {"id": "female_soft",     "label": "Female · Soft",          "instruct": "female, young adult, high pitch"},
    {"id": "male_elderly",    "label": "Male · Elderly",         "instruct": "male, elderly, low pitch"},
    {"id": "female_elderly",  "label": "Female · Elderly",       "instruct": "female, elderly, moderate pitch"},
    {"id": "child",           "label": "Child",                  "instruct": "child, high pitch"},
    {"id": "whisper",         "label": "Whisper",                "instruct": "whisper"},
    {"id": "male_indian",     "label": "Male · Indian Accent",   "instruct": "male, young adult, indian accent"},
    {"id": "female_indian",   "label": "Female · Indian Accent", "instruct": "female, young adult, indian accent"},
    {"id": "male_british",    "label": "Male · British",         "instruct": "male, young adult, british accent"},
    {"id": "female_british",  "label": "Female · British",       "instruct": "female, young adult, british accent"},
    {"id": "male_american",   "label": "Male · American",        "instruct": "male, young adult, american accent"},
    {"id": "female_american", "label": "Female · American",      "instruct": "female, young adult, american accent"},
    {"id": "male_teen",       "label": "Male · Teen",            "instruct": "male, teenager, moderate pitch"},
    {"id": "female_teen",     "label": "Female · Teen",          "instruct": "female, teenager, high pitch"},
]

# ---------------------------------------------------------------------------
# MinIO helpers
# ---------------------------------------------------------------------------

def _ensure_bucket() -> None:
    try:
        if not _minio.client.bucket_exists(REF_AUDIO_BUCKET):
            _minio.client.make_bucket(REF_AUDIO_BUCKET)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Storage unavailable: {exc}",
        )


def _minio_object_name(org_id: str, key: str) -> str:
    return f"{org_id}/{key}"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get("/languages")
async def list_languages(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[dict]:
    """Return all supported languages with display name and ISO code."""
    return [{"name": name, "code": code} for name, code in OMNIVOICE_LANG_MAP.items()]


@router.get("/voice-designs")
async def list_voice_designs(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[dict]:
    """Return predefined voice design presets."""
    return VOICE_DESIGN_PRESETS


@router.post("/synthesize")
async def synthesize(
    text: str = Form(...),
    language: str = Form("English"),
    ref_text: Optional[str] = Form(None),
    ref_audio_key: Optional[str] = Form(None),
    instruct: Optional[str] = Form(None),
    speed: float = Form(1.0),
    duration: Optional[float] = Form(None),
    ref_audio: Optional[UploadFile] = File(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> Response:
    """
    Synthesize speech and return WAV audio.

    Modes:
      - Voice clone: provide ref_audio (file upload) OR ref_audio_key (previously uploaded)
      - Voice design: provide instruct (e.g. "male, young adult, moderate pitch")
      - Neutral: neither ref_audio nor instruct → server default voice

    Returns WAV bytes with headers:
      X-Audio-Duration  (seconds)
      X-Synth-Time      (seconds)
      X-RTF             (synthesis time / audio duration)
    """
    org_id = current_user.get("org_id", "default")
    lang_code = OMNIVOICE_LANG_MAP.get(language, language)

    # Build multipart form for OmniVoice server
    form = aiohttp.FormData()
    form.add_field("text", text)
    form.add_field("language_id", lang_code)
    if ref_text:
        form.add_field("ref_text", ref_text)
    if instruct:
        form.add_field("instruct", instruct)
    if speed != 1.0:
        form.add_field("speed", str(speed))
    if duration is not None:
        form.add_field("duration", str(duration))

    ref_audio_bytes: Optional[bytes] = None
    ref_audio_filename = "ref.wav"

    if ref_audio is not None:
        ref_audio_bytes = await ref_audio.read()
        ref_audio_filename = ref_audio.filename or "ref.wav"
    elif ref_audio_key:
        # Fetch from MinIO
        try:
            obj = await _minio.get_object(REF_AUDIO_BUCKET, _minio_object_name(org_id, ref_audio_key))
            ref_audio_bytes = obj.read()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Reference audio not found: {exc}",
            )

    if ref_audio_bytes is not None:
        form.add_field(
            "ref_audio",
            ref_audio_bytes,
            filename=ref_audio_filename,
            content_type="audio/wav",
        )

    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                f"{OMNIVOICE_URL}/tts",
                data=form,
                timeout=aiohttp.ClientTimeout(total=120),
            ) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    raise HTTPException(
                        status_code=resp.status,
                        detail=f"OmniVoice server error: {body}",
                    )
                audio_bytes = await resp.read()
                audio_duration = resp.headers.get("X-Audio-Duration", "")
                synth_time = resp.headers.get("X-Synth-Time", "")

        except aiohttp.ClientConnectorError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Cannot reach OmniVoice server at {OMNIVOICE_URL}: {exc}",
            )

    rtf = ""
    try:
        if audio_duration and synth_time:
            rtf = str(round(float(synth_time) / float(audio_duration), 4))
    except (ValueError, ZeroDivisionError):
        pass

    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Audio-Duration": audio_duration,
            "X-Synth-Time": synth_time,
            "X-RTF": rtf,
            "Content-Disposition": "attachment; filename=synthesis.wav",
        },
    )


@router.post("/ref-audio", status_code=status.HTTP_201_CREATED)
async def upload_ref_audio(
    file: UploadFile = File(...),
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> dict:
    """
    Upload a reference audio file to MinIO for reuse in voice cloning.
    Returns a key that can be passed as ref_audio_key to /synthesize.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Empty file")

    org_id = current_user.get("org_id", "default")
    key = f"{uuid.uuid4().hex}_{file.filename}"
    object_name = _minio_object_name(org_id, key)

    _ensure_bucket()

    import asyncio

    await asyncio.to_thread(
        _minio.client.put_object,
        REF_AUDIO_BUCKET,
        object_name,
        io.BytesIO(content),
        length=len(content),
        content_type=file.content_type or "audio/wav",
    )

    return {
        "key": key,
        "filename": file.filename,
        "size_bytes": len(content),
        "org_id": org_id,
    }


@router.get("/ref-audio")
async def list_ref_audios(
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> List[dict]:
    """List all reference audios stored for the current organization."""
    import asyncio

    org_id = current_user.get("org_id", "default")
    prefix = f"{org_id}/"

    try:
        objects = await asyncio.to_thread(
            _minio.client.list_objects,
            REF_AUDIO_BUCKET,
            prefix=prefix,
        )
        results = []
        for obj in objects:
            key = obj.object_name.removeprefix(prefix)
            filename = key.split("_", 1)[-1] if "_" in key else key
            results.append({
                "key": key,
                "filename": filename,
                "size_bytes": obj.size,
                "last_modified": obj.last_modified.isoformat() if obj.last_modified else None,
            })
        return results
    except Exception:
        return []


@router.get("/ref-audio/{key:path}")
async def get_ref_audio(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> StreamingResponse:
    """Download a stored reference audio file."""
    org_id = current_user.get("org_id", "default")
    object_name = _minio_object_name(org_id, key)

    try:
        obj = await _minio.get_object(REF_AUDIO_BUCKET, object_name)
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Not found: {exc}")

    filename = key.split("_", 1)[-1] if "_" in key else key
    return StreamingResponse(
        obj,
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.delete("/ref-audio/{key:path}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ref_audio(
    key: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
) -> None:
    """Delete a stored reference audio file."""
    import asyncio

    org_id = current_user.get("org_id", "default")
    object_name = _minio_object_name(org_id, key)

    try:
        await asyncio.to_thread(
            _minio.client.remove_object,
            REF_AUDIO_BUCKET,
            object_name,
        )
    except Exception as exc:
        raise HTTPException(status_code=404, detail=f"Not found: {exc}")
