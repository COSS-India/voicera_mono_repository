"use client"

import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { 
  PhoneCall,
  Monitor,
  Clock,
  MoreVertical,
  Settings,
  Trash2,
} from "lucide-react"
import type { Agent } from "@/lib/api"
import { cn } from "@/lib/utils"

interface AgentCardProps {
  agent: Agent
  getAgentDisplayName: (agent: Agent) => string
  getAgentDescription: (agent: Agent) => string
  onViewConfig: (agent: Agent) => void
  onTestCall: (agent: Agent) => void
  onTestBrowser: (agent: Agent) => void
  onViewHistory: (agent: Agent) => void
  onDelete?: (agent: Agent) => void
  callCount?: number
}

export function AgentCard({
  agent,
  getAgentDisplayName,
  getAgentDescription,
  onViewConfig,
  onTestCall,
  onTestBrowser,
  onViewHistory,
  onDelete,
  callCount = 0,
}: AgentCardProps) {
  const displayName = getAgentDisplayName(agent)
  const description = getAgentDescription(agent)

  const isConnected = Boolean(agent?.phone_number)

  const handleDeleteClick = async (e: React.MouseEvent) => {
    e.stopPropagation()
    if (onDelete) {
      await onDelete(agent)
    }
  }

  const handleCardClick = (e: React.MouseEvent) => {
    // Don't navigate if clicking on interactive elements
    const target = e.target as HTMLElement
    if (
      target.closest('button') ||
      target.closest('[role="menuitem"]') ||
      target.closest('[role="dialog"]')
    ) {
      return
    }
    onViewConfig(agent)
  }

  return (
    <div 
      onClick={handleCardClick}
      className="group cursor-pointer rounded-xl border-[0.5px] border-slate-200 bg-white p-[18px] transition-all duration-150 hover:-translate-y-[2px] hover:border-slate-300"
    >
      <div className="flex items-center justify-between">
        <div
          className="flex h-10 w-10 items-center justify-center rounded-[10px] border-0 bg-[#F1EFE8] text-[#5F5E5A]"
          aria-label="Agent phone icon"
        >
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#5F5E5A" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M15.05 5A5 5 0 0 1 19 8.95M15.05 1A9 9 0 0 1 23 8.94"/>
            <path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/>
          </svg>
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            className="inline-flex h-8 items-center gap-1.5 rounded-full border-[0.5px] border-slate-300 bg-transparent px-2.5 text-[13px] font-medium text-slate-600 transition-colors hover:bg-slate-50"
            onClick={(e) => {
              e.stopPropagation()
              onViewHistory(agent)
            }}
            title="View call history"
          >
            <Clock className="h-3 w-3" />
            History
          </button>

          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                type="button"
                onClick={(e) => e.stopPropagation()}
                className="inline-flex h-8 w-8 items-center justify-center rounded-full border-[0.5px] border-slate-300 bg-transparent text-slate-600 transition-colors hover:bg-slate-50"
                title="More options"
              >
                <MoreVertical className="h-4 w-4" />
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem
                onClick={(e) => {
                  e.stopPropagation()
                  onViewConfig(agent)
                }}
                className="cursor-pointer"
              >
                <Settings className="mr-2 h-4 w-4 text-slate-500" />
                Configure Agent
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={handleDeleteClick}
                className="cursor-pointer text-red-600 focus:bg-red-50 focus:text-red-600"
              >
                <Trash2 className="mr-2 h-4 w-4" />
                Delete Agent
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      </div>

      <div className="mt-4 space-y-4">
        <div>
          <h3 className="line-clamp-1 text-[17px] font-medium text-slate-900">
            {displayName}
          </h3>
          <p className="mt-1 line-clamp-2 text-[13px] leading-[1.5] text-slate-500">
            {description}
          </p>
        </div>

        <div>
          <button
            type="button"
            className="inline-flex h-8 items-center gap-1.5 rounded-full px-2.5 text-[13px] font-medium"
            style={{
              backgroundColor: isConnected ? "#EAF3DE" : "#FCEBEB",
              color: isConnected ? "#3B6D11" : "#A32D2D",
            }}
            onClick={(e) => {
              e.stopPropagation()
              window.location.assign("/numbers")
            }}
            title="Manage numbers"
          >
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ backgroundColor: isConnected ? "#3B6D11" : "#A32D2D" }}
            />
            {isConnected ? agent.phone_number : "Not linked"}
          </button>
          {callCount > 0 && (
            <span className="ml-2 inline-flex h-8 items-center rounded-full bg-slate-100 px-2.5 text-[13px] font-medium text-slate-700">
              {callCount.toLocaleString()} Calls
            </span>
          )}
        </div>
      </div>

      <div className="my-4 border-t-[0.5px] border-slate-200" />

      <div className="grid grid-cols-2 gap-2">
        <Button
          onClick={(e) => {
            e.stopPropagation()
            if (isConnected) {
              onTestCall(agent)
            }
          }}
          variant="outline"
          disabled={!isConnected}
          className={cn(
            "h-9 rounded-[7px] border-[0.5px] bg-transparent px-2 text-[13px] font-medium shadow-none",
            "hover:bg-[var(--color-background-secondary)] hover:border-[var(--color-border-secondary)] hover:text-[var(--color-text-primary)]",
            isConnected
              ? "border-slate-300 text-slate-700"
              : "border-slate-200 text-slate-400"
          )}
          style={{ transition: "background 0.15s ease, border-color 0.15s ease, color 0.15s ease" }}
          title={!isConnected ? "Please attach a phone number to this agent first" : "Make a test call"}
        >
          <PhoneCall className="mr-1.5 h-3.5 w-3.5" />
          Test Call
        </Button>

        <Button
          onClick={(e) => {
            e.stopPropagation()
            onTestBrowser(agent)
          }}
          variant="outline"
          className="h-9 rounded-[7px] border-[0.5px] border-slate-300 bg-transparent px-2 text-[13px] font-medium text-slate-700 shadow-none hover:bg-[var(--color-background-secondary)] hover:border-[var(--color-border-secondary)] hover:text-[var(--color-text-primary)]"
          style={{ transition: "background 0.15s ease, border-color 0.15s ease, color 0.15s ease" }}
          title="Test this agent directly in your browser"
        >
          <Monitor className="mr-1.5 h-3.5 w-3.5" />
          Test on Browser
        </Button>
      </div>
    </div>
  )
}
