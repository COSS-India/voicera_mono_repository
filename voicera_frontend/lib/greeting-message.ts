import { isTranslationPairAvailable } from "@/lib/chrome-translation"

/** Map agent language display names to BCP-47 codes. */
export const AGENT_LANG_TO_BCP47: Record<string, string> = {
  hindi: "hi",
  marathi: "mr",
  tamil: "ta",
  telugu: "te",
  kannada: "kn",
  bengali: "bn",
  gujarati: "gu",
  malayalam: "ml",
  punjabi: "pa",
  odia: "or",
  assamese: "as",
  urdu: "ur",
  english: "en",
  bhili: "bhb",
}

const ENGLISH_AGENT_KEYS = new Set([
  "english",
  "english (india)",
  "english (united states)",
  "english (us)",
  "english (uk)",
])

const MIN_GREETING_DETECT_LENGTH = 1
const MIN_DETECTION_CONFIDENCE = 0.5
const SHORT_GREETING_MAX_LENGTH = 20
const SHORT_GREETING_MIN_CONFIDENCE = 0.1

export function normalizeAgentLanguageName(name: string): string {
  return name.trim().toLowerCase()
}

/** Normalize BCP-47 tags for comparison (e.g. en-US → en). */
export function normalizeBcp47(code: string): string {
  const trimmed = code.trim().toLowerCase()
  if (!trimmed) return trimmed
  return trimmed.split("-")[0] ?? trimmed
}

const BCP47_TO_DISPLAY: Record<string, string> = {
  en: "English",
  hi: "Hindi",
  mr: "Marathi",
  ta: "Tamil",
  te: "Telugu",
  kn: "Kannada",
  bn: "Bengali",
  gu: "Gujarati",
  ml: "Malayalam",
  pa: "Punjabi",
  or: "Odia",
  as: "Assamese",
  ur: "Urdu",
  bhb: "Bhili",
}

/** Human-readable language name from a BCP-47 code. */
export function bcp47ToDisplayLanguage(code: string): string {
  const base = normalizeBcp47(code)
  return BCP47_TO_DISPLAY[base] ?? base.toUpperCase()
}

export function agentLanguageToBcp47(name: string): string | null {
  const normalized = normalizeAgentLanguageName(name)
  if (!normalized) return null

  if (AGENT_LANG_TO_BCP47[normalized]) {
    return AGENT_LANG_TO_BCP47[normalized]
  }

  if (ENGLISH_AGENT_KEYS.has(normalized)) {
    return "en"
  }

  for (const [key, code] of Object.entries(AGENT_LANG_TO_BCP47)) {
    if (normalized.includes(key) || key.includes(normalized)) {
      return code
    }
  }

  return null
}

export function isEnglishAgentLanguage(name: string): boolean {
  const normalized = normalizeAgentLanguageName(name)
  if (!normalized) return false
  if (ENGLISH_AGENT_KEYS.has(normalized)) return true
  return normalized.startsWith("english")
}

export function greetingContainsComma(text: string): boolean {
  return /,/.test(text)
}

export function canDetectGreetingLanguage(text: string): boolean {
  return text.trim().length >= MIN_GREETING_DETECT_LENGTH
}

export function minDetectionConfidenceForGreeting(text: string): number {
  return text.trim().length <= SHORT_GREETING_MAX_LENGTH
    ? SHORT_GREETING_MIN_CONFIDENCE
    : MIN_DETECTION_CONFIDENCE
}

/** True if text is mostly Latin script (likely English/European input). */
export function isLikelyLatinScript(text: string): boolean {
  const trimmed = text.trim()
  if (!trimmed) return false
  return /^[\p{Script=Latin}\p{N}\p{P}\s{}_]+$/u.test(trimmed)
}

/** True when detected greeting language matches the agent primary language. */
export function greetingLanguageMatchesPrimary(
  detectedLang: string,
  primaryLanguage: string
): boolean {
  const targetBcp47 = agentLanguageToBcp47(primaryLanguage)
  if (!targetBcp47) return true

  return normalizeBcp47(detectedLang) === normalizeBcp47(targetBcp47)
}

/** True when greeting language differs from agent primary language. */
export function shouldOfferGreetingTranslation(
  detectedLang: string,
  primaryLanguage: string,
  confidence: number,
  greetingText = ""
): boolean {
  const minConfidence = minDetectionConfidenceForGreeting(greetingText)
  if (confidence < minConfidence) return false
  if (!primaryLanguage.trim()) return false

  const targetBcp47 = agentLanguageToBcp47(primaryLanguage)
  if (!targetBcp47) return false

  return !greetingLanguageMatchesPrimary(detectedLang, primaryLanguage)
}

export async function shouldOfferGreetingTranslationAsync(
  detectedLang: string,
  primaryLanguage: string,
  confidence: number,
  greetingText = ""
): Promise<boolean> {
  if (
    !shouldOfferGreetingTranslation(
      detectedLang,
      primaryLanguage,
      confidence,
      greetingText
    )
  ) {
    return false
  }

  const targetBcp47 = agentLanguageToBcp47(primaryLanguage)
  if (!targetBcp47) return false

  const sourceBcp47 = normalizeBcp47(detectedLang)
  if (sourceBcp47 === normalizeBcp47(targetBcp47)) return false

  return isTranslationPairAvailable(sourceBcp47, targetBcp47)
}

/** When LanguageDetector fails on short text, infer a source language for translation. */
export async function inferFallbackSourceLanguage(
  greetingText: string,
  primaryLanguage: string
): Promise<string | null> {
  if (!greetingText.trim()) return null

  const targetBcp47 = agentLanguageToBcp47(primaryLanguage)
  if (!targetBcp47 || normalizeBcp47(targetBcp47) === "en") return null

  if (!isLikelyLatinScript(greetingText)) return null

  const available = await isTranslationPairAvailable("en", targetBcp47)
  return available ? "en" : null
}

export { MIN_GREETING_DETECT_LENGTH, MIN_DETECTION_CONFIDENCE }
