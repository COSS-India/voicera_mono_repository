export function requireJohnaicServerUrl(): string {
  const url = process.env.NEXT_PUBLIC_JOHNAIC_SERVER_URL?.trim()
  if (!url) {
    throw new Error(
      "NEXT_PUBLIC_JOHNAIC_SERVER_URL is not set. Add it to voicera_frontend/.env.local",
    )
  }
  return url.replace(/\/$/, "")
}

export function johnaicServerUrlToWebSocket(serverUrl: string): string {
  return serverUrl.replace(/^http:\/\//i, "ws://").replace(/^https:\/\//i, "wss://")
}

export function getBrowserAgentWebSocketUrl(agentId: string): string {
  const wsBase = johnaicServerUrlToWebSocket(requireJohnaicServerUrl())
  return `${wsBase}/browser/agent/${encodeURIComponent(agentId)}`
}

/** Presenter (host) leg of a live-translation room; requires a host token. */
export function getTranslationPublishWebSocketUrl(agentId: string, token: string): string {
  const wsBase = johnaicServerUrlToWebSocket(requireJohnaicServerUrl())
  return `${wsBase}/translate/publish/${encodeURIComponent(agentId)}?token=${encodeURIComponent(token)}`
}

/** Public listener leg; pick a target language from the agent's allowed set. */
export function getTranslationListenWebSocketUrl(shareToken: string, language: string): string {
  const wsBase = johnaicServerUrlToWebSocket(requireJohnaicServerUrl())
  return `${wsBase}/translate/listen/${encodeURIComponent(shareToken)}?lang=${encodeURIComponent(language)}`
}
