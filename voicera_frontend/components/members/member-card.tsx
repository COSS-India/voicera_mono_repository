"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
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
  Loader2,
  Trash2,
  MoreVertical,
} from "lucide-react"
import { Member } from "@/lib/api"

const formatDate = (dateString: string) => {
  return new Date(dateString).toLocaleDateString("en-US", {
    year: "numeric",
    month: "short",
    day: "numeric",
  })
}

interface MemberCardProps {
  member: Member
  isCurrentUser: boolean
  isDeleting: boolean
  onDelete: (member: Member) => void
}

export function MemberCard({ member, isCurrentUser, isDeleting, onDelete }: MemberCardProps) {
  const [showDeleteDialog, setShowDeleteDialog] = useState(false)
  const isOwner = member.is_owner === true
  const canDelete = !isCurrentUser && !isOwner

  const handleDeleteClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    setShowDeleteDialog(true)
  }

  const handleConfirmDelete = () => {
    setShowDeleteDialog(false)
    onDelete(member)
  }

  return (
    <div
      className={`group relative flex flex-col items-center gap-3 rounded-[12px] border-[1.5px] bg-white p-[18px] text-center ${
        isOwner ? "border-[#2C2C2A]" : "border-[#e5e5e5]"
      }`}
    >

      {/* Dropdown menu */}
      {canDelete && (
        <div className="absolute top-2.5 right-2.5">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <button
                onClick={(e) => e.stopPropagation()}
                className="flex h-8 w-8 items-center justify-center rounded-md border border-[#e5e5e5] bg-white"
              >
                {isDeleting ? (
                  <Loader2 className="h-[18px] w-[18px] text-slate-500 animate-spin" />
                ) : (
                  <MoreVertical className="h-[18px] w-[18px] text-slate-500" />
                )}
              </button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-44">
              <DropdownMenuItem
                onClick={handleDeleteClick}
                className="cursor-pointer text-red-600 focus:text-red-600 focus:bg-red-50"
                disabled={isDeleting}
              >
                <Trash2 className="h-4 w-4 mr-2" />
                Remove member
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}

      {/* Avatar */}
      <div className="flex h-14 w-14 items-center justify-center rounded-full border-[0.5px] border-[#e5e5e5] bg-[#f5f5f5]">
        <svg width="30" height="30" viewBox="0 0 24 24" fill="none" aria-hidden="true">
          <circle cx="12" cy="8" r="4" fill="#aaaaaa" />
          <path d="M4 20c0-4 3.6-7 8-7s8 3 8 7" fill="#aaaaaa" />
        </svg>
      </div>

      {/* Badges row - owner card only */}
      {isOwner ? (
        <div className="flex items-center gap-1.5">
          <span className="rounded-full bg-[#2C2C2A] px-2.5 py-[2px] text-[11px] font-medium text-[#F1EFE8]">
            Owner
          </span>
          {isCurrentUser ? (
            <span className="rounded-full bg-[#f0f0f0] px-2.5 py-[2px] text-[11px] text-[#888]">
              You
            </span>
          ) : null}
        </div>
      ) : null}

      <p className="text-[15px] font-semibold text-[#1f1f1f] leading-none">{member.name}</p>
      <p className="text-[13px] text-[#8b8b8b] leading-none">{member.company_name || "-"}</p>

      <div className="my-0.5 w-full border-t border-[#ececec]" />
      <p className="w-full truncate text-[13px] text-[#8d8d8d]">{member.email}</p>
      <p className="text-[12px] text-[#8d8d8d]">Joined {formatDate(member.created_at)}</p>

      {/* Delete Confirmation Dialog */}
      <Dialog open={showDeleteDialog} onOpenChange={setShowDeleteDialog}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>Remove Member</DialogTitle>
            <DialogDescription className="pt-2">
              Are you sure you want to remove <span className="font-medium text-slate-700">"{member.name}"</span>?
              They will lose access to this organization.
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
              {isDeleting ? "Removing..." : "Remove"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}