import type { Meeting } from "@/lib/api"

export type CallType = "inbound" | "outbound" | "web"

export const CALL_TYPE_LABELS: Record<CallType, string> = {
  inbound: "Inbound",
  outbound: "Outbound",
  web: "Web",
}

/** Tailwind classes for call-type badges (soft background + readable text). */
export const CALL_TYPE_BADGE_CLASS: Record<CallType, string> = {
  inbound: "bg-emerald-50 text-emerald-700",
  outbound: "bg-blue-50 text-blue-700",
  web: "bg-orange-50 text-orange-700",
}

export function isBrowserMeetingId(meetingId?: string | null): boolean {
  return typeof meetingId === "string" && meetingId.startsWith("browser-")
}

/** Resolve call type from stored field or legacy meeting shape. */
export function resolveCallType(meeting: Pick<Meeting, "call_type" | "inbound" | "meeting_id">): CallType {
  const stored = meeting.call_type?.trim().toLowerCase()
  if (stored === "inbound" || stored === "outbound" || stored === "web") {
    return stored
  }
  if (isBrowserMeetingId(meeting.meeting_id)) {
    return "web"
  }
  return meeting.inbound === true ? "inbound" : "outbound"
}

export function getCallTypeLabel(meeting: Pick<Meeting, "call_type" | "inbound" | "meeting_id">): string {
  return CALL_TYPE_LABELS[resolveCallType(meeting)]
}
