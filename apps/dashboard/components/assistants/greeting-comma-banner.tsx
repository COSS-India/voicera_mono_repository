"use client"

import { AlertCircle } from "lucide-react"
import { cn } from "@/lib/utils"

type GreetingCommaBannerProps = {
  context?: "greeting" | "hold"
  className?: string
}

const TIP_COPY = {
  greeting:
    "Avoid commas and special characters in greeting messages when possible — they can cause audio stutter or breaks during playback.",
  hold: "Avoid commas and special characters in hold messages when possible — they can cause audio stutter or breaks during playback.",
} as const

export function GreetingCommaBanner({
  context = "greeting",
  className,
}: GreetingCommaBannerProps) {
  return (
    <div
      className={cn(
        "flex items-start gap-2.5 rounded-lg border border-stone-200 bg-stone-50 px-3.5 py-2.5",
        className
      )}
    >
      <AlertCircle
        className="h-4 w-4 shrink-0 text-stone-500 mt-0.5"
        aria-hidden="true"
      />
      <p className="text-[13px] leading-relaxed text-stone-600">
        {TIP_COPY[context]}
      </p>
    </div>
  )
}
