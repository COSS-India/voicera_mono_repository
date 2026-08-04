/** Letters, numbers, and whitespace — used only to detect special characters for warnings. */
const SPOKEN_MESSAGE_ALLOWED = /[\p{L}\p{N}\s]/gu

export function spokenMessageContainsSpecialChars(text: string): boolean {
  const stripped = [...text.matchAll(SPOKEN_MESSAGE_ALLOWED)].map((m) => m[0]).join("")
  return stripped !== text
}
