"""Mid-call language switching via OpenAI tool calling for AI4Bharat STT/TTS."""

from __future__ import annotations

import json
from typing import Any, Optional

from loguru import logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.llm_service import FunctionCallParams, LLMService

from config.stt_mappings import STT_LANGUAGE_MAP
from config.tts_mappings import TTS_LANGUAGE_MAP

_LANGUAGE_MAP = TTS_LANGUAGE_MAP["AI4Bharat"]
_STT_LANGUAGE_MAP = STT_LANGUAGE_MAP["AI4Bharat"]

_SUPPORTED_CODES: set[str] = set(_LANGUAGE_MAP.values()) | set(_STT_LANGUAGE_MAP.values())

_LANGUAGE_LOOKUP: dict[str, str] = {}
for display_name, code in {**_LANGUAGE_MAP, **_STT_LANGUAGE_MAP}.items():
    _LANGUAGE_LOOKUP[display_name.lower()] = code
    _LANGUAGE_LOOKUP[code.lower()] = code


def build_language_lookup() -> dict[str, str]:
    return dict(_LANGUAGE_LOOKUP)


def normalize_language(value: str | None) -> str | None:
    if not value or not str(value).strip():
        return None
    key = str(value).strip().lower()
    code = _LANGUAGE_LOOKUP.get(key)
    if code and code in _SUPPORTED_CODES:
        return code
    return None


def _resolve_default_language(default_language: str | None) -> str:
    if not default_language:
        return "hi"
    return normalize_language(default_language) or _LANGUAGE_MAP.get(
        default_language, default_language
    )


LANGUAGE_SWITCH_SYSTEM_PROMPT = """

## Language switching (voice call)
You can switch the spoken and listening language during this call using the `switch_language` tool.

Rules:
- When the user mentions a language by name (e.g. Hindi, Tamil, Telugu, Marathi) or asks to speak/switch language (e.g. "in Tamil", "Tamil la sollunga", "हिंदी में बताइए"), call `switch_language` BEFORE you generate any spoken reply in that language.
- Never stream text in a new language before calling the tool for that language.
- If you code-switch within one reply (e.g. Hindi then English), call `switch_language` before each language block.
- Pass the ISO language code (e.g. hi, ta, mr, te) to the tool.
- After switching, respond naturally in that language. Do not mention the tool or that you switched languages.
- Supported codes: as, bn, brx, bhb, doi, gu, hi, kn, kok, ks, mai, ml, mni, mr, ne, or, pa, sa, sat, sd, ta, te, ur.
"""


def create_switch_language_tool_schema() -> FunctionSchema:
    codes = sorted(_SUPPORTED_CODES)
    return FunctionSchema(
        name="switch_language",
        description=(
            "Switch the voice call STT and TTS language. Call this before speaking "
            "in a new language when the user requests a language change or mentions "
            "a language name. Supported codes: "
            + ", ".join(codes)
        ),
        properties={
            "language": {
                "type": "string",
                "description": (
                    "Target language ISO code (e.g. hi for Hindi, ta for Tamil, mr for Marathi)."
                ),
                "enum": codes,
            }
        },
        required=["language"],
    )


def setup_language_switching(
    *,
    llm: LLMService,
    stt: Any,
    tts: Any,
    context: OpenAILLMContext,
    default_language: str | None,
) -> None:
    current_language = _resolve_default_language(default_language)
    tool_schema = create_switch_language_tool_schema()
    context.set_tools(ToolsSchema([tool_schema]))

    async def switch_language_handler(params: FunctionCallParams) -> None:
        nonlocal current_language
        raw = params.arguments.get("language")
        code = normalize_language(str(raw) if raw is not None else None)

        if not code:
            await params.result_callback(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unsupported language: {raw!r}",
                        "supported": sorted(_SUPPORTED_CODES),
                    }
                )
            )
            return

        if code == current_language:
            await params.result_callback(
                json.dumps({"success": True, "language": code, "unchanged": True})
            )
            return

        if hasattr(stt, "set_language"):
            await stt.set_language(code)
        else:
            logger.warning("STT service does not support set_language; skipping STT switch")

        if hasattr(tts, "set_language"):
            tts.set_language(code)
        else:
            logger.warning("TTS service does not support set_language; skipping TTS switch")

        current_language = code
        logger.info(f"Conversation language switched to: {code}")
        await params.result_callback(
            json.dumps({"success": True, "language": code})
        )

    llm.register_function(
        "switch_language",
        switch_language_handler,
        cancel_on_interruption=False,
    )
