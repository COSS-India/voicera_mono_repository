import type { Meeting } from "@/lib/api"
import { resolveCallType } from "@/lib/call-type"

/** Whether this from/to field should be masked for the given call direction. */
export function shouldMaskCallPhone(
  inbound: boolean,
  field: "from" | "to"
): boolean {
  return inbound ? field === "from" : field === "to"
}

/** Display a from/to number with PII masking based on call direction. */
export function displayCallPhoneNumber(
  phone: string | null | undefined,
  inbound: boolean,
  field: "from" | "to"
): string {
  if (phone == null || phone === "") return "-"
  return shouldMaskCallPhone(inbound, field)
    ? maskPhoneLastDigits(phone)
    : phone
}

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

/** Mask from/to for CSV/PDF export — same rules as the history table. */
export function exportMeetingPhoneNumber(
  meeting: Pick<Meeting, "call_type" | "inbound" | "meeting_id" | "from_number" | "to_number">,
  field: "from" | "to"
): string {
  const callType = resolveCallType(meeting)
  const phone = field === "from" ? meeting.from_number : meeting.to_number
  const isInbound = callType === "inbound"
  return displayCallPhoneNumber(phone, isInbound, field)
}
