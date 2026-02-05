"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Separator } from "@/components/ui/separator"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogFooter,
} from "@/components/ui/dialog"
import {
  UserPlus,
  Users,
  Copy,
  Check,
  Loader2,
  Search,
} from "lucide-react"
import { getOrgId, getOrgMembers, deleteMember, getCurrentUser, getStoredEmail, Member, User } from "@/lib/api"
import { MemberCard } from "@/components/members/member-card"

/** Generate a UUID v4; uses crypto.randomUUID when available, else a fallback. */
function generateUUID(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID()
  }
  const bytes = new Uint8Array(16)
  if (typeof crypto !== "undefined" && typeof crypto.getRandomValues === "function") {
    crypto.getRandomValues(bytes)
  } else {
    for (let i = 0; i < 16; i++) bytes[i] = Math.floor(Math.random() * 256)
  }
  bytes[6] = (bytes[6]! & 0x0f) | 0x40
  bytes[8] = (bytes[8]! & 0x3f) | 0x80
  const hex = [...bytes].map((b) => b.toString(16).padStart(2, "0")).join("")
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export default function MembersPage() {
  const [members, setMembers] = useState<Member[]>([])
  const [currentUser, setCurrentUser] = useState<User | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [inviteLink, setInviteLink] = useState("")
  const [copied, setCopied] = useState(false)
  const [deletingMember, setDeletingMember] = useState<string | null>(null)
  const [deleteModalOpen, setDeleteModalOpen] = useState(false)
  const [memberToDelete, setMemberToDelete] = useState<Member | null>(null)
  const [searchQuery, setSearchQuery] = useState("")

  // Fetch members and current user on mount
  useEffect(() => {
    fetchData()
  }, [])

  const fetchData = async () => {
    try {
      setIsLoading(true)
      
      // Fetch current user first
      const user = await getCurrentUser()
      setCurrentUser(user)
      
      // Then fetch members
      await fetchMembers()
    } catch (error) {
      console.error("Error fetching data:", error)
    } finally {
      setIsLoading(false)
    }
  }

  const fetchMembers = async () => {
    try {
      const orgId = getOrgId()
      
      if (!orgId) {
        console.error("No org_id found")
        return
      }

      const fetchedMembers = await getOrgMembers(orgId)
      setMembers(fetchedMembers)
    } catch (error) {
      console.error("Error fetching members:", error)
    }
  }

  // Check if a member is the current user
  const isCurrentUser = (email: string) => {
    return currentUser?.email === email || getStoredEmail() === email
  }

  // All members are now returned from the backend (including owner)
  // Sort to put owner first
  const allMembers = [...members].sort((a, b) => {
    if (a.is_owner && !b.is_owner) return -1
    if (!a.is_owner && b.is_owner) return 1
    return 0
  })

  // Filter members based on search query
  const filteredMembers = allMembers.filter((member) => {
    if (!searchQuery.trim()) return true
    const query = searchQuery.toLowerCase()
    return (
      member.name.toLowerCase().includes(query) ||
      member.email.toLowerCase().includes(query) ||
      member.company_name?.toLowerCase().includes(query)
    )
  })

  const generateInviteLink = () => {
    const orgId = getOrgId()
    const uuid = generateUUID()
    const link = `${window.location.origin}/add-member/${orgId}-${uuid}`
    setInviteLink(link)
    setIsModalOpen(true)
    setCopied(false)
    
    // Log the invite link generation
    console.log("Generated invite link:", {
      org_id: orgId,
      uuid: uuid,
      full_link: link,
    })
  }

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(inviteLink)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error("Failed to copy:", error)
    }
  }

  const openDeleteModal = (member: Member) => {
    setMemberToDelete(member)
    setDeleteModalOpen(true)
  }

  const handleDeleteMember = async () => {
    if (!memberToDelete) return
    
    const orgId = getOrgId()
    if (!orgId) return

    try {
      setDeletingMember(memberToDelete.email)
      await deleteMember(memberToDelete.email, orgId)
      // Refresh the members list
      await fetchMembers()
      setDeleteModalOpen(false)
      setMemberToDelete(null)
    } catch (error) {
      console.error("Error deleting member:", error)
      alert(error instanceof Error ? error.message : "Failed to delete member")
    } finally {
      setDeletingMember(null)
    }
  }

  // Helper for delete modal initials
  const getInitials = (name: string) => {
    return name
      .split(" ")
      .map((n) => n[0])
      .join("")
      .toUpperCase()
      .slice(0, 2)
  }

  return (
    <div className="flex flex-1 flex-col h-full overflow-hidden">
      {/* Fixed Header */}
      <div className="flex-shrink-0 p-6 pb-0 space-y-4">
        {/* Page Title */}
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Members</h1>
          <p className="text-muted-foreground">
            Manage your organization&apos;s team members
          </p>
        </div>

        {/* Action Bar: Search, Count, Add Button */}
        {!isLoading && (
          <div className="flex items-center gap-4">
            {/* Search Input - Left aligned, primary action */}
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search by name or email..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9"
              />
            </div>

            {/* Members Count - Center, informational */}
            <div className="flex items-center gap-2 px-3 py-1.5 bg-muted/50 rounded-full">
              <Users className="h-4 w-4 text-muted-foreground" />
              <span className="text-sm font-medium text-muted-foreground">
                {filteredMembers.length === allMembers.length 
                  ? `${allMembers.length} ${allMembers.length === 1 ? "member" : "members"}`
                  : `${filteredMembers.length} of ${allMembers.length}`
                }
              </span>
            </div>

            {/* Spacer */}
            <div className="flex-1" />

            {/* Add Member Button - Right aligned, secondary action */}
            <Button onClick={generateInviteLink} className="gap-2">
              <UserPlus className="h-4 w-4" />
              Add New Member
            </Button>
          </div>
        )}
      </div>

      {/* Members Grid */}
      <ScrollArea className="flex-1 px-6 pb-6">
        <div className="pt-4">
          {isLoading ? (
            <div className="flex items-center justify-center py-12">
              <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
              <span className="ml-2 text-muted-foreground">Loading members...</span>
            </div>
          ) : filteredMembers.length > 0 ? (
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4">
              {filteredMembers.map((member) => (
                <MemberCard
                  key={member.email}
                  member={member}
                  isCurrentUser={isCurrentUser(member.email)}
                  isDeleting={deletingMember === member.email}
                  onDelete={openDeleteModal}
                />
              ))}
            </div>
          ) : searchQuery ? (
            <div className="flex flex-col items-center justify-center py-16 text-center border rounded-xl border-dashed bg-muted/20">
              <div className="rounded-full bg-muted p-4 mb-4">
                <Search className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="font-medium text-lg mb-1">No members found</h3>
              <p className="text-sm text-muted-foreground mb-4 max-w-sm">
                No members match your search for &quot;{searchQuery}&quot;. Try a different search term.
              </p>
              <Button variant="outline" onClick={() => setSearchQuery("")} className="gap-2">
                Clear search
              </Button>
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-16 text-center border rounded-xl border-dashed bg-muted/20">
              <div className="rounded-full bg-muted p-4 mb-4">
                <Users className="h-8 w-8 text-muted-foreground" />
              </div>
              <h3 className="font-medium text-lg mb-1">No team members yet</h3>
              <p className="text-sm text-muted-foreground mb-4 max-w-sm">
                Invite your colleagues to collaborate and manage your organization together.
              </p>
              <Button onClick={generateInviteLink} className="gap-2">
                <UserPlus className="h-4 w-4" />
                Invite your first member
              </Button>
            </div>
          )}
        </div>
      </ScrollArea>

      {/* Invite Modal */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="sm:max-w-lg">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              <UserPlus className="h-5 w-5" />
              Invite Team Member
            </DialogTitle>
            <DialogDescription>
              Share this link with your team member so they can join your organization.
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium">Invite Link</label>
              <div className="flex gap-2">
                <Input
                  value={inviteLink}
                  readOnly
                  className="font-mono text-sm"
                />
                <Button
                  variant="outline"
                  size="icon"
                  onClick={copyToClipboard}
                  className="shrink-0"
                >
                  {copied ? (
                    <Check className="h-4 w-4 text-green-600" />
                  ) : (
                    <Copy className="h-4 w-4" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Anyone with this link can join your organization.
              </p>
            </div>
          </div>

          <div className="flex justify-end">
            <Button onClick={() => setIsModalOpen(false)}>Done</Button>
          </div>
        </DialogContent>
      </Dialog>

      {/* Delete Confirmation Modal */}
      <Dialog open={deleteModalOpen} onOpenChange={setDeleteModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2 text-slate-900">
              Delete user account?
            </DialogTitle>
            <DialogDescription className="text-base text-slate-600 pt-2">
              This action will permanently remove this user.
            </DialogDescription>
          </DialogHeader>

          <Separator className="my-2" />

          <div className="py-3">
            <p className="text-sm text-muted-foreground">
              This user belongs to only one organization, so their entire account will be deleted.
            </p>
            {memberToDelete && (
              <div className="mt-4 p-3 bg-slate-50 rounded-lg border">
                <div className="flex items-center gap-3">
                  <div className="flex items-center justify-center h-10 w-10 rounded-full bg-red-100 text-red-600 font-semibold text-sm">
                    {memberToDelete.name
                      .split(" ")
                      .map((n) => n[0])
                      .join("")
                      .toUpperCase()
                      .slice(0, 2)}
                  </div>
                  <div>
                    <p className="font-medium text-slate-900">{memberToDelete.name}</p>
                    <p className="text-sm text-muted-foreground">{memberToDelete.email}</p>
                  </div>
                </div>
              </div>
            )}
          </div>

          <DialogFooter className="gap-2 sm:gap-0">
            <Button
              variant="outline"
              onClick={() => {
                setDeleteModalOpen(false)
                setMemberToDelete(null)
              }}
              disabled={deletingMember !== null}
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={handleDeleteMember}
              disabled={deletingMember !== null}
              className="bg-red-600 hover:bg-red-700"
            >
              {deletingMember ? (
                <>
                  <Loader2 className="h-4 w-4 animate-spin mr-2" />
                  Deleting...
                </>
              ) : (
                "Delete user"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
