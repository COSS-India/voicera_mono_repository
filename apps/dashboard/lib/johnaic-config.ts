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
