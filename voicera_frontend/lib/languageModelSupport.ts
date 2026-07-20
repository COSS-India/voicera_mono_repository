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

/** Languages used for provider/model intersection (primary + optional secondary). */
export function getActiveLanguages(primary: string, secondary: string): string[] {
  const langs: string[] = []
  if (primary) langs.push(primary)
  if (secondary && secondary !== primary) langs.push(secondary)
  return langs
}

export function isBilingual(primary: string, secondary: string): boolean {
  return Boolean(primary && secondary && secondary !== primary)
}

export function languageSwitchingStackEligible(
  llmProvider: string,
  sttProvider: string,
  sttModel: string,
  ttsProvider: string,
  ttsModel: string
): boolean {
  return (
    llmProvider === "openai" &&
    sttProvider === "ai4bharat" &&
    sttModel === "indic-conformer-stt" &&
    ttsProvider === "ai4bharat" &&
    ttsModel === "indic-parler-tts"
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
