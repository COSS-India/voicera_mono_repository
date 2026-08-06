import sttData from "@/stt.json"
import ttsData from "@/tts.json"

function intersectSets<T>(sets: Set<T>[]): Set<T> {
  if (sets.length === 0) return new Set()
  const [first, ...rest] = sets
  return rest.reduce(
    (acc, set) => new Set([...acc].filter((item) => set.has(item))),
    new Set(first)
  )
}

/** Dedupe language names while preserving order. */
export function dedupeLanguages(languages: string[]): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const lang of languages) {
    const trimmed = lang.trim()
    if (!trimmed) continue
    const key = trimmed.toLowerCase()
    if (seen.has(key)) continue
    seen.add(key)
    result.push(trimmed)
  }
  return result
}

/** Languages used for provider/model intersection. */
export function getActiveLanguages(languages: string[]): string[] {
  return dedupeLanguages(languages)
}

export function hasMultipleLanguages(languages: string[]): boolean {
  return getActiveLanguages(languages).length >= 2
}

/** @deprecated Use hasMultipleLanguages(getActiveLanguages([primary, secondary])) */
export function isBilingual(primary: string, secondary: string): boolean {
  return hasMultipleLanguages([primary, secondary])
}

/** Split ordered selection into one primary and all remaining secondaries. */
export function splitPrimaryAndSecondary(languages: string[]): {
  primary: string
  secondaryLanguages: string[]
  allLanguages: string[]
} {
  const allLanguages = dedupeLanguages(languages)
  return {
    primary: allLanguages[0] || "",
    secondaryLanguages: allLanguages.slice(1),
    allLanguages,
  }
}

/** Reconstruct ordered selection from stored agent language fields. */
export function loadSelectedLanguagesFromConfig(config: {
  language?: string
  languages?: string[]
  secondary_languages?: string[]
  secondary_language?: string
}): string[] {
  if (Array.isArray(config.languages) && config.languages.length > 0) {
    return dedupeLanguages(config.languages.map((lang) => String(lang)))
  }

  const primary = String(config.language || "").trim()
  const secondaryLanguages = Array.isArray(config.secondary_languages)
    ? config.secondary_languages
        .map((lang) => String(lang).trim())
        .filter(Boolean)
    : []

  if (primary || secondaryLanguages.length > 0) {
    return dedupeLanguages([primary, ...secondaryLanguages].filter(Boolean))
  }

  const legacySecondary = String(config.secondary_language || "").trim()
  if (primary && legacySecondary) {
    return dedupeLanguages([primary, legacySecondary])
  }

  return primary ? [primary] : []
}

/** Build agent_config language fields from an ordered selection. */
export function buildLanguageConfigFields(selectedLanguages: string[]): {
  language: string
  secondary_languages?: string[]
  languages?: string[]
  secondary_language?: string
} {
  const { primary, secondaryLanguages, allLanguages } =
    splitPrimaryAndSecondary(selectedLanguages)

  const fields: {
    language: string
    secondary_languages?: string[]
    languages?: string[]
    secondary_language?: string
  } = { language: primary }

  if (secondaryLanguages.length > 0) {
    fields.secondary_languages = secondaryLanguages
    fields.languages = allLanguages
    fields.secondary_language = secondaryLanguages[0]
  }

  return fields
}

export function languageSwitchingStackEligible(
  llmProvider: string,
  sttProvider: string,
  sttModel: string,
  ttsProvider: string,
  ttsModel: string
): boolean {
  const ttsEligible =
    (ttsProvider === "ai4bharat" && ttsModel === "indic-parler-tts") ||
    (ttsProvider === "springlab" && ttsModel === "indic-mio")
  return (
    llmProvider === "openai" &&
    sttProvider === "ai4bharat" &&
    sttModel === "indic-conformer-stt" &&
    ttsEligible
  )
}

export function getIntersectedSTTProviders(languages: string[]): Set<string> {
  if (languages.length === 0) return new Set()

  const perLang = languages.map((lang) => {
    const langData =
      sttData.stt.languages[lang as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()
    return new Set(
      Object.entries(langData.models)
        .filter(([, models]) => Array.isArray(models) && models.length > 0)
        .map(([provider]) => provider)
    )
  })

  return intersectSets(perLang)
}

export function getIntersectedSTTModels(
  languages: string[],
  provider: string
): Set<string> {
  if (languages.length === 0 || !provider) return new Set()

  const perLang = languages.map((lang) => {
    const langData =
      sttData.stt.languages[lang as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()
    const models =
      langData.models[provider as keyof typeof langData.models]
    return new Set(Array.isArray(models) ? models : [])
  })

  return intersectSets(perLang)
}

export function getIntersectedTTSProviders(languages: string[]): Set<string> {
  if (languages.length === 0) return new Set()

  const perLang = languages.map((lang) => {
    const langData =
      ttsData.tts.languages[lang as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()
    return new Set(
      Object.entries(langData.models)
        .filter(([, data]) => {
          const modelData = data as { available?: boolean }
          return modelData.available === true
        })
        .map(([provider]) => provider)
    )
  })

  return intersectSets(perLang)
}

export function getIntersectedTTSModels(
  languages: string[],
  provider: string
): Set<string> {
  if (languages.length === 0 || !provider) return new Set()

  const perLang = languages.map((lang) => {
    const langData =
      ttsData.tts.languages[lang as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()

    const providerData = langData.models[
      provider as keyof typeof langData.models
    ] as { model?: string; models?: string[]; available?: boolean } | undefined

    if (!providerData || !providerData.available) return new Set<string>()

    const models: string[] = []
    if (providerData.models && Array.isArray(providerData.models)) {
      models.push(...providerData.models)
    }
    if (providerData.model) {
      models.push(providerData.model)
    }
    return new Set(models)
  })

  return intersectSets(perLang)
}
