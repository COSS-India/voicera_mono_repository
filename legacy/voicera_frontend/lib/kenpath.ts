export type KenpathVariant = "prod" | "dev" | "bharatvistaar" | "bharatvistaar_dev"

const BHARAT_VISTAAR_PROD_LANGUAGES = new Set([
  "English (United States)",
  "English (India)",
  "Hindi",
])

const BHARAT_VISTAAR_DEV_LANGUAGES = new Set([
  ...BHARAT_VISTAAR_PROD_LANGUAGES,
  "Bengali",
  "Telugu",
  "Marathi",
  "Tamil",
  "Gujarati",
  "Kannada",
  "Malayalam",
  "Assamese",
])

export function isBharatVistaarLanguageSupported(
  variant: KenpathVariant,
  languageCode: string
): boolean {
  if (variant !== "bharatvistaar" && variant !== "bharatvistaar_dev") {
    return true
  }
  const supported =
    variant === "bharatvistaar_dev"
      ? BHARAT_VISTAAR_DEV_LANGUAGES
      : BHARAT_VISTAAR_PROD_LANGUAGES
  return supported.has(languageCode)
}

export function kenpathVariantFromLlmModel(model?: {
  kenpath_backend?: string
  vistaar_environment?: string
}): KenpathVariant {
  if (model?.kenpath_backend === "bharatvistaar") {
    return model?.vistaar_environment === "dev" ? "bharatvistaar_dev" : "bharatvistaar"
  }
  return model?.vistaar_environment === "dev" ? "dev" : "prod"
}

export function kenpathLlmFieldsFromVariant(variant: KenpathVariant): {
  kenpath_backend: "vistaar" | "bharatvistaar"
  vistaar_environment: "prod" | "dev"
} {
  if (variant === "bharatvistaar") {
    return { kenpath_backend: "bharatvistaar", vistaar_environment: "prod" }
  }
  if (variant === "bharatvistaar_dev") {
    return { kenpath_backend: "bharatvistaar", vistaar_environment: "dev" }
  }
  return { kenpath_backend: "vistaar", vistaar_environment: variant }
}

export function kenpathVariantLabel(variant: KenpathVariant): string {
  switch (variant) {
    case "prod":
      return "Production"
    case "dev":
      return "Development"
    case "bharatvistaar":
      return "Bharat Vistaar"
    case "bharatvistaar_dev":
      return "Bharat Vistaar Dev API"
  }
}

export function kenpathVariantHelpText(variant: KenpathVariant): string {
  if (variant === "bharatvistaar_dev") {
    return "Bharat Vistaar development streaming API (dev-vistaar.da.gov.in). Supports English, Hindi, Bengali, Telugu, Marathi, Tamil, Gujarati, Kannada, Malayalam, and Assamese."
  }
  if (variant === "bharatvistaar") {
    return "Bharat Vistaar production API for English/Hindi agricultural schemes (chat-vistaar.da.gov.in)."
  }
  if (variant === "dev") {
    return "Vistaar development API for Hindi, Marathi, and Bhili."
  }
  return "Vistaar production API for Hindi, Marathi, and Bhili."
}
