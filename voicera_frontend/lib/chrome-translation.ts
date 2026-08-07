/** Chrome built-in Translator / Language Detector (not in all TS libs yet). */

export type ChromeAvailability =
  | "available"
  | "downloadable"
  | "downloading"
  | "unavailable"

export type ChromeTranslator = {
  translate: (input: string) => Promise<string>
  destroy?: () => void
}

export type ChromeLanguageDetector = {
  detect: (
    input: string
  ) => Promise<Array<{ detectedLanguage: string; confidence: number }>>
  destroy?: () => void
}

export type ChromeTranslatorCtor = {
  availability: (options: {
    sourceLanguage: string
    targetLanguage: string
  }) => Promise<ChromeAvailability>
  create: (options: {
    sourceLanguage: string
    targetLanguage: string
    monitor?: (m: {
      addEventListener: (
        type: "downloadprogress",
        listener: (e: { loaded: number }) => void
      ) => void
    }) => void
  }) => Promise<ChromeTranslator>
}

export type ChromeLanguageDetectorCtor = {
  availability: () => Promise<ChromeAvailability>
  create: (options?: {
    monitor?: (m: {
      addEventListener: (
        type: "downloadprogress",
        listener: (e: { loaded: number }) => void
      ) => void
    }) => void
  }) => Promise<ChromeLanguageDetector>
}

export function getChromeTranslatorAPI(): ChromeTranslatorCtor | null {
  const api = (globalThis as { Translator?: ChromeTranslatorCtor }).Translator
  return api ?? null
}

export function getChromeLanguageDetectorAPI(): ChromeLanguageDetectorCtor | null {
  const api = (globalThis as { LanguageDetector?: ChromeLanguageDetectorCtor })
    .LanguageDetector
  return api ?? null
}

export function isChromeTranslationAvailable(): boolean {
  return getChromeTranslatorAPI() != null
}

export type DetectedLanguage = {
  language: string
  confidence: number
}

/** Detect the primary language of a text sample via Chrome LanguageDetector. */
export async function detectTextLanguage(
  text: string
): Promise<DetectedLanguage | null> {
  const sample = text.trim().slice(0, 2000)
  if (!sample) return null

  const LanguageDetectorAPI = getChromeLanguageDetectorAPI()
  if (!LanguageDetectorAPI) return null

  const detectorAvailability = await LanguageDetectorAPI.availability()
  if (detectorAvailability === "unavailable") return null

  const detector = await LanguageDetectorAPI.create({
    monitor(m) {
      m.addEventListener("downloadprogress", () => {})
    },
  })

  try {
    const detections = await detector.detect(sample)
    const top = detections?.[0]
    if (!top?.detectedLanguage || top.detectedLanguage === "und") {
      return null
    }
    return {
      language: top.detectedLanguage,
      confidence: top.confidence ?? 0,
    }
  } finally {
    detector.destroy?.()
  }
}

export type TranslateTextOptions = {
  text: string
  sourceLanguage: string
  targetLanguage: string
}

/** Translate text via Chrome Translator API. Throws if unavailable or unsupported. */
export async function translateText({
  text,
  sourceLanguage,
  targetLanguage,
}: TranslateTextOptions): Promise<string> {
  const trimmed = text.trim()
  if (!trimmed) return text

  const TranslatorAPI = getChromeTranslatorAPI()
  if (!TranslatorAPI) {
    throw new Error("Translation is not available in this browser.")
  }

  const availability = await TranslatorAPI.availability({
    sourceLanguage,
    targetLanguage,
  })
  if (availability === "unavailable") {
    throw new Error(
      `Translation from ${sourceLanguage} to ${targetLanguage} is not supported.`
    )
  }

  const translator = await TranslatorAPI.create({
    sourceLanguage,
    targetLanguage,
    monitor(m) {
      m.addEventListener("downloadprogress", () => {})
    },
  })

  try {
    return await translator.translate(trimmed)
  } finally {
    translator.destroy?.()
  }
}

/** Check whether a translation pair is supported without creating a translator. */
export async function isTranslationPairAvailable(
  sourceLanguage: string,
  targetLanguage: string
): Promise<boolean> {
  const TranslatorAPI = getChromeTranslatorAPI()
  if (!TranslatorAPI) return false
  const availability = await TranslatorAPI.availability({
    sourceLanguage,
    targetLanguage,
  })
  return availability !== "unavailable"
}
