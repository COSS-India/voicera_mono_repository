import type { Agent } from "@/lib/api"

export const WEBSOCKET_PROVIDER = "WebSocket"
export const AGENT_CATEGORY_WEBSOCKET = "voicera_websocket"
export const AGENT_CATEGORY_TELEPHONY = "voicera_telephony"

export type AgentDeliveryMode = "telephony" | "websocket"

export function isWebSocketAgent(
  agent: Pick<Agent, "agent_category" | "telephony_provider">
): boolean {
  if (agent.agent_category === AGENT_CATEGORY_WEBSOCKET) return true
  return agent.telephony_provider === WEBSOCKET_PROVIDER
}

export function isTelephonyAgent(
  agent: Pick<Agent, "agent_category" | "telephony_provider">
): boolean {
  return !isWebSocketAgent(agent)
}

export function deliveryModeLabel(mode: AgentDeliveryMode): string {
  return mode === "websocket" ? "WebSocket" : "Telephony"
}

export function agentCategoryForProvider(provider: string): string {
  return provider === WEBSOCKET_PROVIDER ? AGENT_CATEGORY_WEBSOCKET : AGENT_CATEGORY_TELEPHONY
}
