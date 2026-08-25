/**
 * Decides which providers the agent forms may offer.
 *
 * A provider is offered only when it can actually serve a call:
 * - self-hosted models (AI4Bharat STT/TTS, Qwen) must be reachable, as
 *   reported by the voice server's /providers/available probe
 * - API providers must have a key saved in the org's Integrations
 *
 * Saved agents keep their current provider listed even when it is no longer
 * available, so opening the edit form does not silently blank a working
 * agent's config.
 */

export type ProviderKind = "llm" | "stt" | "tts"

/** Providers served by our own model servers rather than a third-party API. */
const SELF_HOSTED: Record<ProviderKind, string[]> = {
  llm: ["qwen"],
  stt: ["ai4bharat"],
  tts: ["ai4bharat"],
}

/**
 * Telephony credentials, stored per-org in Integrations as two separate
 * entries. Both halves are required: an auth id without its token cannot
 * place a call.
 */
const TELEPHONY_CREDENTIALS: Record<string, [string, string]> = {
  Vobiz: ["vobizauthid", "vobizauthtoken"],
  Plivo: ["plivoauthid", "plivoauthtoken"],
}

export function isTelephonyProviderAvailable(
  provider: string,
  integratedProviders: Set<string>,
): boolean {
  const credentials = TELEPHONY_CREDENTIALS[provider]
  if (!credentials) return false
  return credentials.every((model) => integratedProviders.has(model))
}

export function isProviderAvailable(
  kind: ProviderKind,
  providerId: string,
  providerName: string,
  integratedProviders: Set<string>,
  selfHostedProviders: Set<string>,
): boolean {
  if (SELF_HOSTED[kind].includes(providerId)) {
    return selfHostedProviders.has(`${kind}:${providerId}`)
  }
  return (
    integratedProviders.has(providerId) ||
    integratedProviders.has(providerName.toLowerCase())
  )
}
