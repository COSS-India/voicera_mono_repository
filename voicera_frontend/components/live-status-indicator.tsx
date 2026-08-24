import { cn } from "@/lib/utils"

export type LiveStatusState = "live" | "connecting" | "waiting" | "idle" | "error"

const DOT_COLOR: Record<LiveStatusState, string> = {
  live: "bg-green-500",
  connecting: "bg-amber-400",
  waiting: "bg-amber-400",
  idle: "bg-slate-300",
  error: "bg-red-500",
}

interface LiveStatusIndicatorProps {
  state: LiveStatusState
  label: string
  className?: string
}

// Standard "on air" pattern: a dot (pulsing only while actually live) paired
// with a text label, so the state never depends on color alone.
export function LiveStatusIndicator({ state, label, className }: LiveStatusIndicatorProps) {
  const dot = DOT_COLOR[state]
  return (
    <span
      role="status"
      aria-live="polite"
      className={cn("flex items-center gap-2 text-sm text-slate-600", className)}
    >
      <span className="relative inline-flex h-2.5 w-2.5">
        {state === "live" && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full opacity-75 motion-reduce:animate-none",
              dot,
            )}
          />
        )}
        <span className={cn("relative inline-flex h-2.5 w-2.5 rounded-full", dot)} />
      </span>
      {label}
    </span>
  )
}
