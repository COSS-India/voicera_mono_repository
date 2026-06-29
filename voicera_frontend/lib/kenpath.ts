export type KenpathVariant = "prod" | "dev" | "bharatvistaar"

export function kenpathVariantFromLlmModel(model?: {
  kenpath_backend?: string
  vistaar_environment?: string
}): KenpathVariant {
  if (model?.kenpath_backend === "bharatvistaar") return "bharatvistaar"
  return model?.vistaar_environment === "dev" ? "dev" : "prod"
}

export function kenpathLlmFieldsFromVariant(variant: KenpathVariant): {
  kenpath_backend: "vistaar" | "bharatvistaar"
  vistaar_environment: "prod" | "dev"
} {
  if (variant === "bharatvistaar") {
    return { kenpath_backend: "bharatvistaar", vistaar_environment: "prod" }
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
  }
}

export function kenpathVariantHelpText(variant: KenpathVariant): string {
  if (variant === "bharatvistaar") {
    return "Bharat Vistaar production API for English/Hindi agricultural schemes (chat-vistaar.da.gov.in)."
  }
  if (variant === "dev") {
    return "Vistaar development API for Hindi, Marathi, and Bhili."
  }
  return "Vistaar production API for Hindi, Marathi, and Bhili."
}
