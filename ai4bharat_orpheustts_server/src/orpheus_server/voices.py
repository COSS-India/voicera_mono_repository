"""The voice roster: languages, speakers, styles, and how a request resolves to them.

``voices.json`` is data, not configuration - it describes what the loaded
checkpoint can actually say. The important property it provides is that speaker
names are unique across languages, which is what lets an OpenAI client select a
language using nothing but the standard ``voice`` field.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .prompt import TEMPLATE_INDIC, TEMPLATES


class VoiceRosterError(ValueError):
    """Raised when voices.json is malformed."""


class Roster:
    """Indexed, validated view of voices.json."""

    def __init__(self, data: dict) -> None:
        self.template: str = data.get("prompt_template", TEMPLATE_INDIC)
        if self.template not in TEMPLATES:
            raise VoiceRosterError(f"prompt_template must be one of {TEMPLATES}, got {self.template!r}")

        self.styles: list[str] = list(data.get("styles") or [])
        self.default_style: Optional[str] = data.get("default_style")
        if self.default_style and self.styles and self.default_style not in self.styles:
            raise VoiceRosterError(f"default_style {self.default_style!r} is not in styles")

        self.languages: list[dict] = list(data.get("languages") or [])
        if not self.languages:
            raise VoiceRosterError("voices.json lists no languages")

        self.by_code: dict[str, dict] = {}
        for lang in self.languages:
            for field in ("code", "name", "voices"):
                if field not in lang:
                    raise VoiceRosterError(f"language entry missing {field!r}: {lang}")
            if lang["code"] in self.by_code:
                raise VoiceRosterError(f"duplicate language code {lang['code']!r}")
            self.by_code[lang["code"]] = lang

        # voice -> [language codes]. A speaker present in exactly one language can
        # be selected without naming the language; an ambiguous one cannot.
        self.voice_languages: dict[str, list[str]] = {}
        for lang in self.languages:
            for voice in lang["voices"]:
                self.voice_languages.setdefault(voice, []).append(lang["code"])

    @property
    def ambiguous_voices(self) -> dict[str, list[str]]:
        """Speakers that appear in more than one language, so ``voice`` alone won't do."""
        return {v: codes for v, codes in self.voice_languages.items() if len(codes) > 1}

    @property
    def all_voices(self) -> list[str]:
        return sorted(self.voice_languages)

    def resolve(
        self,
        voice: str,
        language: Optional[str] = None,
        style: Optional[str] = None,
    ) -> tuple[str, str, str]:
        """Validate a request and return ``(language_code, voice, style)``.

        ``language`` is optional whenever the speaker name is unambiguous.
        Raises ``LookupError`` with a message meant for the client.
        """
        if voice not in self.voice_languages:
            raise LookupError(
                f"unknown voice {voice!r}. See GET /v1/voices for the {len(self.voice_languages)} available speakers."
            )
        candidates = self.voice_languages[voice]

        if language is None:
            if len(candidates) > 1:
                raise LookupError(
                    f"voice {voice!r} exists in several languages {candidates}; "
                    f"pass 'language' to disambiguate."
                )
            language = candidates[0]
        else:
            if language not in self.by_code:
                raise LookupError(f"unknown language {language!r}. See GET /v1/languages.")
            if language not in candidates:
                raise LookupError(
                    f"voice {voice!r} is not available for language {language!r}. "
                    f"Options: {self.by_code[language]['voices']}"
                )

        if style is None:
            style = self.default_style
        elif self.styles and style not in self.styles:
            raise LookupError(f"unknown style {style!r}. Options: {self.styles}")

        return language, voice, style

    def sample_text(self, language_code: str) -> str:
        return self.by_code.get(language_code, {}).get("sample") or "test"

    def catalog(self) -> list[dict]:
        return [
            {
                "code": lang["code"],
                "name": lang["name"],
                "n_voices": len(lang["voices"]),
                "voices": lang["voices"],
                "sample": lang.get("sample", ""),
            }
            for lang in self.languages
        ]


def load_roster(path: Path) -> Roster:
    if not path.is_file():
        raise VoiceRosterError(f"voice roster not found: {path}")
    return Roster(json.loads(path.read_text(encoding="utf-8")))
