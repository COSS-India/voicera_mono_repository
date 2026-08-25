// Languages the on-prem NMT engine (AI4Bharat IndicTrans2) can translate.
//
// Single source of truth for the UI. Mirrors config/nmt_mappings.py on the
// voice server — keep the two in sync. Display names must match the app's
// language display names. Verified directly against the hosted model.
//
// Notable: Konkani IS supported (the server maps it internally to "gom"), but
// Bhili is NOT — IndicTrans2 has no Bhili. The LLM engine has no such limits.

export const NMT_SUPPORTED_LANGUAGES: readonly string[] = [
  "English",
  "English (India)",
  "English (United States)",
  "Assamese",
  "Bengali",
  "Bodo",
  "Dogri",
  "Gujarati",
  "Hindi",
  "Kannada",
  "Kashmiri",
  "Konkani",
  "Maithili",
  "Malayalam",
  "Manipuri",
  "Marathi",
  "Nepali",
  "Odia",
  "Punjabi",
  "Sanskrit",
  "Santali",
  "Sindhi",
  "Tamil",
  "Telugu",
  "Urdu",
]

const NMT_SUPPORTED_SET = new Set(NMT_SUPPORTED_LANGUAGES.map((l) => l.toLowerCase()))

export function isNmtSupportedLanguage(displayLanguage: string): boolean {
  return NMT_SUPPORTED_SET.has(String(displayLanguage || "").trim().toLowerCase())
}
