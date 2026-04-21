"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  MoreVertical,
  Trash2,
  Settings,
  PhoneCall,
  Monitor,
  Clock,
  Unplug
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
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const [isDeleting, setIsDeleting] = useState(false)

  const isConnected = Boolean(agent?.phone_number)

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowDeleteDialog(true)
  }

  const handleConfirmDelete = async () => {
    if (!onDelete) {
      console.error("onDelete callback is not provided")
      return
    }

    try {
      setIsDeleting(true)
      setShowDeleteDialog(false)
      await onDelete(agent)
    } catch (error) {
      console.error("Error deleting agent:", error)
      alert(error instanceof Error ? error.message : "Failed to delete agent")
    } finally {
      setIsDeleting(false)
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
      className="shadow-md group relative bg-white rounded-2xl border border-slate-200 overflow-hidden hover:shadow-lg hover:border-slate-300 transition-all duration-300 cursor-pointer"
    >
      {/* Top Bar: Icon + Menu */}
      <div className="absolute top-4 left-4 right-4 z-10 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            className={cn(
              "inline-flex h-7 min-w-[108px] items-center justify-center gap-1 rounded-full px-2.5 text-xs font-semibold cursor-pointer",
              agent && (agent as any).phone_number
                ? "bg-green-50 text-green-700 border border-green-200"
                : "bg-red-50 text-red-500 border border-red-200"
            )}
            onClick={e => {
              e.stopPropagation()
              window.location.assign("/numbers")
            }}
            title="Manage numbers"
            type="button"
          >
            <Unplug className={`h-3 w-3 ${agent && (agent as any).phone_number ? "text-green-600" : "text-red-400"}`} />
            {(agent && (agent as any).phone_number)
              ? (agent as any).phone_number
              : "Not linked"}
          </button>

          <button
            type="button"
            className="inline-flex h-7 min-w-[108px] items-center justify-center gap-1 rounded-full px-2.5 text-xs font-semibold border border-slate-200 bg-white/90 text-slate-700 hover:bg-slate-50 cursor-pointer"
            onClick={(e) => {
              e.stopPropagation()
              onViewHistory(agent)
            }}
            title="View call history"
          >
            <Clock className="h-3 w-3" />
            History
          </button>
        </div>

        <DropdownMenu>
          <DropdownMenuTrigger asChild>
            <button
              onClick={(e) => e.stopPropagation()}
              className="h-9 w-9 rounded-lg bg-white/80 backdrop-blur-sm border border-slate-200/50 flex items-center justify-center shadow-sm opacity-0 group-hover:opacity-100 transition-opacity"
            >
              <MoreVertical className="h-4 w-4 text-slate-500" />
            </button>
          </DropdownMenuTrigger>
          <DropdownMenuContent align="end" className="w-48">
            <DropdownMenuItem
              onClick={(e) => {
                e.stopPropagation()
                onViewConfig(agent)
              }}
              className="cursor-pointer"
            >
              <Settings className="h-4 w-4 mr-2 text-slate-500" />
              Configure
            </DropdownMenuItem>

            <DropdownMenuSeparator />
            <DropdownMenuItem
              onClick={handleDeleteClick}
              className="cursor-pointer text-red-600 focus:text-red-600 focus:bg-red-50"
            >
              <Trash2 className="h-4 w-4 mr-2" />
              Delete Agent
            </DropdownMenuItem>
          </DropdownMenuContent>
        </DropdownMenu>
      </div>

      {/* Wave Hero Section - Enhanced with prominent waves */}
      <div className="relative h-40 bg-gradient-to-br from-slate-100 via-slate-50 to-white overflow-hidden">
        {/* Animated Wave SVG - More prominent with wider lines */}
        <svg
          className="absolute inset-0 w-full h-full"
          viewBox="0 0 400 160"
          preserveAspectRatio="xMidYMid slice"
        >
          <defs>
            {/* Primary gradient for main waves - higher contrast */}
            <linearGradient id={`waveGradient-primary-${agent.agent_type}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(100, 116, 139, 0.85)">
                <animate attributeName="stop-color" values="rgba(100, 116, 139, 0.85);rgba(71, 85, 105, 0.95);rgba(100, 116, 139, 0.85)" dur="3s" repeatCount="indefinite" />
              </stop>
              <stop offset="50%" stopColor="rgba(148, 163, 184, 0.7)">
                <animate attributeName="stop-color" values="rgba(148, 163, 184, 0.7);rgba(100, 116, 139, 0.85);rgba(148, 163, 184, 0.7)" dur="3s" begin="0.3s" repeatCount="indefinite" />
              </stop>
              <stop offset="100%" stopColor="rgba(100, 116, 139, 0.85)">
                <animate attributeName="stop-color" values="rgba(100, 116, 139, 0.85);rgba(71, 85, 105, 0.95);rgba(100, 116, 139, 0.85)" dur="3s" begin="0.6s" repeatCount="indefinite" />
              </stop>
            </linearGradient>
            {/* Secondary gradient for background waves */}
            <linearGradient id={`waveGradient-secondary-${agent.agent_type}`} x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="rgba(71, 85, 105, 0.5)" />
              <stop offset="50%" stopColor="rgba(100, 116, 139, 0.4)" />
              <stop offset="100%" stopColor="rgba(71, 85, 105, 0.5)" />
            </linearGradient>
          </defs>

          {/* Background subtle waves - wider amplitude for depth (Law of Prägnanz - clear grouping) */}
          {[...Array(4)].map((_, i) => (
            <path
              key={`bg-${i}`}
              fill="none"
              stroke={`url(#waveGradient-secondary-${agent.agent_type})`}
              strokeWidth={3.5 - i * 0.3}
              opacity={0.4 - i * 0.06}
              strokeLinecap="round"
            >
              <animate
                attributeName="d"
                values={`
                  M-50,${90 + i * 12} C50,${50 + i * 10} 150,${130 + i * 10} 250,${90 + i * 12} S450,${50 + i * 10} 550,${90 + i * 12};
                  M-50,${90 + i * 12} C50,${130 + i * 10} 150,${50 + i * 10} 250,${90 + i * 12} S450,${130 + i * 10} 550,${90 + i * 12};
                  M-50,${90 + i * 12} C50,${50 + i * 10} 150,${130 + i * 10} 250,${90 + i * 12} S450,${50 + i * 10} 550,${90 + i * 12}
                `}
                dur={`${10 + i * 2}s`}
                repeatCount="indefinite"
                calcMode="spline"
                keySplines="0.4 0 0.6 1; 0.4 0 0.6 1"
              />
            </path>
          ))}

          {/* Primary prominent waves - main visual focus (Focal Point principle) */}
          {[...Array(6)].map((_, i) => (
            <path
              key={`primary-${i}`}
              fill="none"
              stroke={`url(#waveGradient-primary-${agent.agent_type})`}
              strokeWidth={5 - i * 0.5}
              opacity={0.95 - i * 0.12}
              strokeLinecap="round"
            >
              <animate
                attributeName="d"
                values={`
                  M-80,${75 + i * 8} C20,${40 + i * 6} 120,${110 + i * 6} 220,${75 + i * 8} S420,${40 + i * 6} 520,${75 + i * 8};
                  M-80,${75 + i * 8} C20,${110 + i * 6} 120,${40 + i * 6} 220,${75 + i * 8} S420,${110 + i * 6} 520,${75 + i * 8};
                  M-80,${75 + i * 8} C20,${40 + i * 6} 120,${110 + i * 6} 220,${75 + i * 8} S420,${40 + i * 6} 520,${75 + i * 8}
                `}
                dur={`${5 + i * 0.6}s`}
                repeatCount="indefinite"
                calcMode="spline"
                keySplines="0.45 0.05 0.55 0.95; 0.45 0.05 0.55 0.95"
              />
            </path>
          ))}

          {/* Accent highlight waves - thinner, adds visual rhythm (Law of Continuity) */}
          {[...Array(3)].map((_, i) => (
            <path
              key={`accent-${i}`}
              fill="none"
              stroke="rgba(71, 85, 105, 0.75)"
              strokeWidth={3 - i * 0.5}
              opacity={0.7 - i * 0.15}
              strokeLinecap="round"
            >
              <animate
                attributeName="d"
                values={`
                  M-60,${85 + i * 15} Q70,${50 + i * 12} 200,${85 + i * 15} T460,${85 + i * 15};
                  M-60,${85 + i * 15} Q70,${120 + i * 12} 200,${85 + i * 15} T460,${85 + i * 15};
                  M-60,${85 + i * 15} Q70,${50 + i * 12} 200,${85 + i * 15} T460,${85 + i * 15}
                `}
                dur={`${4 + i * 0.8}s`}
                repeatCount="indefinite"
                calcMode="spline"
                keySplines="0.42 0 0.58 1; 0.42 0 0.58 1"
              />
            </path>
          ))}
        </svg>

        {/* Soft gradient overlay - lighter to preserve wave visibility */}
        <div className="absolute inset-0 bg-gradient-to-t from-white/70 via-transparent to-transparent" />
      </div>

      {/* Content Section */}
      <div className="px-5 pb-5 -mt-2 py-4">
        {/* Agent Name & Description */}
        <div className="mb-4">
          <h3 className="text-lg font-semibold text-slate-900 mb-0.5 line-clamp-1">
            {getAgentDisplayName(agent)}
          </h3>
          <p className="text-sm text-slate-500">
            {getAgentDescription(agent)}
          </p>
        </div>

        {/* Status Row */}
        <div className="flex items-center justify-between mb-5">

          {callCount > 0 && (
            <span className="text-sm font-medium text-slate-700 px-2.5 py-1 bg-slate-100 rounded-full">
              {callCount.toLocaleString()} Calls
            </span>
          )}
        </div>

        {/* Action Buttons */}
        <div className="flex items-center gap-2">
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
              "flex-1 basis-0 min-w-0 h-11 rounded-xl text-sm font-medium transition-all cursor-pointer",
              isConnected
                ? "border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300"
                : "border-slate-200 text-slate-400 cursor-not-allowed"
            )}
            title={!isConnected ? "Please attach a phone number to this agent first" : "Make a test call"}
          >
            <PhoneCall className="h-4 w-4 mr-2" />
            Test Call
          </Button>

          <Button
            onClick={(e) => {
              e.stopPropagation()
              onTestBrowser(agent)
            }}
            variant="outline"
            className="flex-1 basis-0 min-w-0 h-11 rounded-xl border-slate-200 text-slate-700 hover:bg-slate-50 hover:border-slate-300 text-sm font-medium transition-all"
            title="Test this agent directly in your browser"
          >
            <Monitor className="h-4 w-4 mr-2" />
            Test on Browser
          </Button>
        </div>
      </div>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Delete Agent</DialogTitle>
            <DialogDescription className="pt-2">
              Are you sure you want to delete <span className="font-medium text-slate-700">&quot;{getAgentDisplayName(agent)}&quot;</span>?
              This will remove all configurations and cannot be undone.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2 sm:gap-1 w-full">
            <Button
              variant="outline"
              onClick={() => setShowDeleteDialog(false)}
              disabled={isDeleting}
              className="flex-1 sm:flex-none"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleConfirmDelete}
              disabled={isDeleting}
              className="flex-1 sm:flex-none"
            >
              {isDeleting ? "Deleting..." : "Delete"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
