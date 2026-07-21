"""Mid-call language switching via OpenAI tool calling for AI4Bharat STT/TTS."""

from __future__ import annotations

import json
from typing import Any

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


def _dedupe_language_codes(raw_values: list[Any]) -> list[str]:
    codes: list[str] = []
    seen: set[str] = set()
    for raw in raw_values:
        if raw is None:
            continue
        code = normalize_language(str(raw))
        if code and code not in seen:
            seen.add(code)
            codes.append(code)
    return codes


def resolve_agent_language_codes(
    agent_config: dict[str, Any] | None,
) -> tuple[str, list[str]]:
    """Return (primary_code, allowed_codes) from agent language fields.

    Reads ``language`` (primary) and optional ``secondary_languages`` / legacy
    ``secondary_language``. Falls back to an ordered ``languages`` list when
    present. Single-language agents return a one-item ``allowed_codes`` list.
    """
    if not agent_config:
        return "hi", ["hi"]

    raw_values: list[Any] = []

    languages_list = agent_config.get("languages")
    if isinstance(languages_list, list) and languages_list:
        raw_values.extend(languages_list)
    else:
        primary = agent_config.get("language")
        if primary:
            raw_values.append(primary)

        secondary_languages = agent_config.get("secondary_languages")
        if isinstance(secondary_languages, list) and secondary_languages:
            raw_values.extend(secondary_languages)
        else:
            secondary = agent_config.get("secondary_language")
            if secondary:
                raw_values.append(secondary)

    codes = _dedupe_language_codes(raw_values)
    if not codes:
        return "hi", ["hi"]

    return codes[0], codes


def is_bilingual_agent(agent_config: dict[str, Any] | None) -> bool:
    _, allowed_codes = resolve_agent_language_codes(agent_config)
    return len(allowed_codes) >= 2


def build_language_switch_system_prompt(allowed_codes: list[str]) -> str:
    codes = sorted(set(allowed_codes))
    codes_csv = ", ".join(codes)
    return f"""

## Language switching (voice call)
You can switch the spoken and listening language during this call using the `switch_language` tool.

Rules:
- When the user mentions a language by name or asks to speak/switch language, call `switch_language` BEFORE you generate any spoken reply in that language.
- Never stream text in a new language before calling the tool for that language.
- If you code-switch within one reply, call `switch_language` before each language block.
- Pass the ISO language code (e.g. hi, ta, mr, te) to the tool.
- After switching, respond naturally in that language. Do not mention the tool or that you switched languages.
- You may switch ONLY between these configured languages: {codes_csv}.
"""


def create_switch_language_tool_schema(allowed_codes: list[str]) -> FunctionSchema:
    codes = sorted(set(allowed_codes))
    if len(codes) < 2:
        codes = sorted(_SUPPORTED_CODES)
    return FunctionSchema(
        name="switch_language",
        description=(
            "Switch the voice call STT and TTS language. Call this before speaking "
            "in a new language when the user requests a language change or mentions "
            "a language name. Allowed codes for this agent: "
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
    allowed_languages: list[str],
) -> None:
    allowed_codes = _dedupe_language_codes(allowed_languages)
    if len(allowed_codes) < 2:
        logger.warning(
            "Language switching requires at least two languages; got "
            f"{allowed_codes!r}"
        )

    current_language = _resolve_default_language(default_language)
    if allowed_codes and current_language not in allowed_codes:
        current_language = allowed_codes[0]

    permitted_codes = set(allowed_codes)
    tool_schema = create_switch_language_tool_schema(allowed_codes)
    context.set_tools(ToolsSchema([tool_schema]))

    async def switch_language_handler(params: FunctionCallParams) -> None:
        nonlocal current_language
        raw = params.arguments.get("language")
        code = normalize_language(str(raw) if raw is not None else None)

        if not code or code not in permitted_codes:
            await params.result_callback(
                json.dumps(
                    {
                        "success": False,
                        "error": f"Unsupported or disallowed language: {raw!r}",
                        "allowed": sorted(permitted_codes),
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


# Backward-compatible alias for imports expecting a static prompt template.
LANGUAGE_SWITCH_SYSTEM_PROMPT = build_language_switch_system_prompt(
    sorted(_SUPPORTED_CODES)
)
