/** Map stored agent language codes to UI labels (e.g. bhb → Bhili). */
export const LANGUAGE_DISPLAY: Record<string, string> = {
  bhb: "Bhili",
}

export function displayLanguageName(code: string): string {
  if (!code) return code
  return LANGUAGE_DISPLAY[code] ?? code
}
