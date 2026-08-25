# NMT (AI4Bharat IndicTrans2) Language Code Mappings
#
# Maps agent display names to the language codes the hosted IndicTrans2 Triton
# model accepts as INPUT_LANGUAGE_ID / OUTPUT_LANGUAGE_ID.
#
# This map is DELIBERATELY separate from TTS_LANGUAGE_MAP / STT_LANGUAGE_MAP:
# the codes differ per backend. Notably this deployment expects Konkani as
# ``gom`` (Goan Konkani) and REJECTS ``kok`` with "Language-pair not supported"
# — copying the TTS map (which uses ``kok``) would fail every batch containing
# Konkani. ``en -> en`` also returns garbage, so same-language pairs must never
# be sent (guarded in the client, not here).
#
# Codes verified directly against the hosted model (probe: en -> <code>).

NMT_LANGUAGE_MAP = {
    # English (source side; IndicTrans2 supports en<->Indic and Indic<->Indic)
    "English": "en",
    "English (India)": "en",
    "English (United States)": "en",
    # Indic
    "Assamese": "as",
    "Bengali": "bn",
    "Bodo": "brx",
    "Dogri": "doi",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Kashmiri": "ks",
    "Konkani": "gom",  # NOT "kok" — this deployment only accepts "gom"
    "Maithili": "mai",
    "Malayalam": "ml",
    "Manipuri": "mni",
    "Marathi": "mr",
    "Nepali": "ne",
    "Odia": "or",
    "Punjabi": "pa",
    "Sanskrit": "sa",
    "Santali": "sat",
    "Sindhi": "sd",
    "Tamil": "ta",
    "Telugu": "te",
    "Urdu": "ur",
}

# Set of accepted codes, for quick membership checks.
NMT_SUPPORTED_CODES = set(NMT_LANGUAGE_MAP.values())


def to_nmt_code(display_language: str) -> str | None:
    """Map an agent display language (e.g. "Hindi") to an NMT code, or None.

    Falls back to treating an already-valid code (e.g. "hi") as itself, so a
    config that stores codes directly still resolves.
    """
    if not display_language:
        return None
    name = str(display_language).strip()
    code = NMT_LANGUAGE_MAP.get(name)
    if code:
        return code
    if name.lower() in NMT_SUPPORTED_CODES:
        return name.lower()
    return None
