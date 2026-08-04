import { Globe, PhoneIncoming, PhoneOutgoing } from "lucide-react"
import type { Meeting } from "@/lib/api"
import {
  CALL_TYPE_BADGE_CLASS,
  getCallTypeLabel,
  resolveCallType,
  type CallType,
} from "@/lib/call-type"
import { cn } from "@/lib/utils"

interface CallTypeBadgeProps {
  meeting: Pick<Meeting, "call_type" | "inbound" | "meeting_id">
  className?: string
  showIcon?: boolean
}

function CallTypeIcon({ type }: { type: CallType }) {
  if (type === "web") {
    return <Globe className="h-3 w-3" aria-hidden="true" />
  }
  if (type === "inbound") {
    return <PhoneIncoming className="h-3 w-3" aria-hidden="true" />
  }
  return <PhoneOutgoing className="h-3 w-3" aria-hidden="true" />
}

export function CallTypeBadge({ meeting, className, showIcon = true }: CallTypeBadgeProps) {
  const type = resolveCallType(meeting)
  const label = getCallTypeLabel(meeting)

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold",
        CALL_TYPE_BADGE_CLASS[type],
        className
      )}
      aria-label={`${label} call`}
      title={`${label} call`}
    >
      {showIcon ? <CallTypeIcon type={type} /> : null}
      {label}
    </span>
  )
}
