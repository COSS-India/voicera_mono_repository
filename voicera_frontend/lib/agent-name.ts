export const AGENT_NAME_PATTERN = /^[a-zA-Z0-9_-]+$/

export const AGENT_NAME_ERROR =
  "Agent name may only contain letters, numbers, underscores, and hyphens (no spaces)"

export function isValidAgentName(name: string): boolean {
  const trimmed = name.trim()
  return trimmed.length > 0 && AGENT_NAME_PATTERN.test(trimmed)
}

export function validateAgentName(name: string): string | null {
  const trimmed = name.trim()
  if (!trimmed) {
    return "Agent name cannot be empty"
  }
  if (!AGENT_NAME_PATTERN.test(trimmed)) {
    return AGENT_NAME_ERROR
  }
  return null
}

/** Strip characters that are not allowed in agent names as the user types. */
export function sanitizeAgentNameInput(value: string): string {
  return value.replace(/[^a-zA-Z0-9_-]/g, "")
}

export function slugifyAgentId(name: string): string {
  return name.trim().toLowerCase()
}
