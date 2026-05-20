/**
 * Masks the last N numeric digits in a phone string for display (PII).
 * Preserves non-digit characters (+, spaces, dashes, etc.).
 * If there are fewer than N digits total, all digits are masked.
 */
export function maskPhoneLastDigits(
  phone: string | null | undefined,
  digitCount: number = 4
): string {
  if (phone == null || phone === "") return "-"

  const chars = [...phone]
  const digitIndicesFromEnd: number[] = []
  for (let i = chars.length - 1; i >= 0; i--) {
    if (/\d/.test(chars[i]!)) digitIndicesFromEnd.push(i)
  }
  if (digitIndicesFromEnd.length === 0) return phone

  const n =
    digitIndicesFromEnd.length <= digitCount
      ? digitIndicesFromEnd.length
      : digitCount
  for (let k = 0; k < n; k++) {
    chars[digitIndicesFromEnd[k]!] = "*"
  }
  return chars.join("")
}
