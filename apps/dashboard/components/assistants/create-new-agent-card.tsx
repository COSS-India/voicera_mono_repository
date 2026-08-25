"use client"

import { Plus } from "lucide-react"

interface CreateNewAgentCardProps {
  onCreateNew: () => void
}

export function CreateNewAgentCard({ onCreateNew }: CreateNewAgentCardProps) {
  return (
    <div
      onClick={onCreateNew}
      className="group relative flex min-h-[190px] cursor-pointer items-center justify-center overflow-hidden rounded-[14px] border border-[#e0ddd6] bg-white p-4 text-center transition-all duration-150 hover:-translate-y-[2px]"
      tabIndex={0}
      role="button"
      aria-label="Create New Agent"
      onKeyDown={e => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault()
          onCreateNew()
        }
      }}
    >
      <div
        className="absolute inset-0"
        style={{
          backgroundColor: "#FFFFFF",
          backgroundImage: "radial-gradient(circle, #e7e4de 1.2px, transparent 1.2px)",
          backgroundSize: "28px 28px",
        }}
      />

      <div className="relative z-[1] flex flex-col items-center">
        <div className="inline-flex h-12 w-12 items-center justify-center rounded-full border border-[#d6d2cb] bg-white">
          <Plus className="h-6 w-6 text-[#5a5650]" />
        </div>
        <div className="mt-3">
          <h3 className="text-[18px] font-semibold text-[#2e2b27]">Create New Agent</h3>
          <p className="mt-1.5 text-[13px] leading-[1.5] text-[#8a8680]">
            Generate a new AI agent for your team
          </p>
        </div>
      </div>
    </div>
  )
}
