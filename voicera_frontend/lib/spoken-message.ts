/** Allowed in TTS spoken text: letters, numbers, and whitespace. */
const SPOKEN_MESSAGE_ALLOWED = /[\p{L}\p{N}\s]/gu

export function sanitizeSpokenMessageInput(text: string): string {
  return [...text.matchAll(SPOKEN_MESSAGE_ALLOWED)].map((m) => m[0]).join("")
}

export function isAllowedSpokenMessageChar(char: string): boolean {
  if (!char) return true
  return /^[\p{L}\p{N}\s]$/u.test(char)
}

export function spokenMessageContainsSpecialChars(text: string): boolean {
  return sanitizeSpokenMessageInput(text) !== text
}
