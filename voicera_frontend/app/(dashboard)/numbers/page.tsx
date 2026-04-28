"use client"

import { useState, useEffect } from "react"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { 
  ChevronRight, 
  Copy, 
  Phone,
  ShoppingCart,
  LayoutGrid,
  Tags,
  Unplug,
  Loader2,
  Plug2,
  Check,
  ChevronsUpDown,
  AlertCircle,
  XCircle,
  CheckCircle2,
  AlertTriangleIcon,
} from "lucide-react"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { fetchApiRoute, getOrgId, getAgents, getVobizNumbers, getPlivoNumbers, linkVobizNumber, linkPlivoNumber, unlinkVobizNumber, unlinkPlivoNumber, type Agent } from "@/lib/api"
import { format } from "date-fns"
import { cn } from "@/lib/utils"

interface PhoneNumber {
  id?: string
  _id?: string
  phone_number: string
  provider: string
  agent_type?: string
  org_id: string
  created_at?: string
  updated_at?: string
}

interface PhoneNumberDisplay extends PhoneNumber {
  id: string
  number: string
  addedOn: string
  usedBy: string | null
  agentName?: string
}
type AgentWithTelephony = Agent & {
  telephony_provider?: string
  agent_category?: string
  plivo_app_id?: string
}

export default function NumbersPage() {
  const [phoneNumbers, setPhoneNumbers] = useState<PhoneNumberDisplay[]>([])
  const [agents, setAgents] = useState<Agent[]>([])
  const [loading, setLoading] = useState(true)
  const [copiedId, setCopiedId] = useState<string | null>(null)
  const [attachDialogOpen, setAttachDialogOpen] = useState(false)
  const [selectedPhoneNumber, setSelectedPhoneNumber] = useState<PhoneNumberDisplay | null>(null)
  const [selectedAgentType, setSelectedAgentType] = useState<string>("")
  const [isAttaching, setIsAttaching] = useState(false)
  const [showSuccessToast, setShowSuccessToast] = useState(false)
  const [successMessage, setSuccessMessage] = useState("")
  const [searchQuery, setSearchQuery] = useState("")
  const [errorDialogOpen, setErrorDialogOpen] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")
  const [detachDialogOpen, setDetachDialogOpen] = useState(false)
  const [phoneToDetach, setPhoneToDetach] = useState<PhoneNumberDisplay | null>(null)
  const [isDetaching, setIsDetaching] = useState(false)
  const [agentPopoverOpen, setAgentPopoverOpen] = useState(false)
  const [addNumberDialogOpen, setAddNumberDialogOpen] = useState(false)
  const [newPhoneNumber, setNewPhoneNumber] = useState("+91")
  const [selectedProvider, setSelectedProvider] = useState<string>("")
  const [isAddingNumber, setIsAddingNumber] = useState(false)
  const [vobizNumbers, setVobizNumbers] = useState<string[]>([])
  const [isLoadingVobizNumbers, setIsLoadingVobizNumbers] = useState(false)
  const [selectedVobizNumber, setSelectedVobizNumber] = useState<string>("")
  const orgId = getOrgId()

  // Format date from ISO string to readable format
  const formatDate = (dateString?: string): string => {
    if (!dateString) return "-"
    try {
      const date = new Date(dateString)
      return format(date, "dd MMM yyyy, h:mm a")
    } catch {
      return "-"
    }
  }

  // Get agent display name
  const getAgentDisplayName = (agentType?: string): string | null => {
    if (!agentType) return null
    const agent = agents.find(a => a.agent_type === agentType)
    if (!agent) return agentType
    const telephonyAgent = agent as AgentWithTelephony
    return telephonyAgent.agent_category || agent.agent_type || agentType
  }

  // Fetch phone numbers and agents
  useEffect(() => {
    async function fetchData() {
      if (!orgId) {
        setLoading(false)
        return
      }

      try {
        setLoading(true)
        
        // Fetch phone numbers
        const phoneResponse = await fetchApiRoute(`/api/phone-numbers?org_id=${encodeURIComponent(orgId)}`)
        if (!phoneResponse.ok) {
          throw new Error("Failed to fetch phone numbers")
        }
        const phoneData: PhoneNumber[] = await phoneResponse.json()

        // Fetch agents
        const agentsData = await getAgents(orgId)
        setAgents(agentsData)

        // Transform phone numbers data
        const transformed: PhoneNumberDisplay[] = phoneData.map((phone) => {
          const agentName = getAgentDisplayName(phone.agent_type)
          return {
            ...phone,
            id: phone.id || phone._id || phone.phone_number,
            number: phone.phone_number,
            addedOn: formatDate(phone.created_at),
            usedBy: phone.agent_type ? (agentName || phone.agent_type) : null,
            agentName: agentName || undefined,
          }
        })

        setPhoneNumbers(transformed)
      } catch (error) {
        console.error("Error fetching data:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchData()
  }, [orgId])

  // Update phone numbers when agents change
  useEffect(() => {
    if (agents.length > 0 && phoneNumbers.length > 0) {
      const updated = phoneNumbers.map(phone => ({
        ...phone,
        usedBy: phone.agent_type ? (getAgentDisplayName(phone.agent_type) || phone.agent_type) : null,
        agentName: phone.agent_type ? (getAgentDisplayName(phone.agent_type) || undefined) : undefined,
      }))
      setPhoneNumbers(updated)
    }
  }, [agents])

  const copyToClipboard = (number: string, id: string) => {
    navigator.clipboard.writeText(number)
    setCopiedId(id)
    setTimeout(() => setCopiedId(null), 2000)
  }

  const handleAttachClick = (phone: PhoneNumberDisplay) => {
    setSelectedPhoneNumber(phone)
    setSelectedAgentType(phone.agent_type || "")
    setAgentPopoverOpen(false)
    setAttachDialogOpen(true)
  }

  const handleAttachSubmit = async () => {
    if (!selectedPhoneNumber || !selectedAgentType) return

    try {
      setIsAttaching(true)
      
      // Find the selected agent to get vobiz_app_id
      const selectedAgent = agents.find(a => a.agent_type === selectedAgentType)
      
      // If provider is Vobiz, first link to Vobiz application, then update database
      if (selectedPhoneNumber.provider === "Vobiz" && selectedAgent?.vobiz_app_id) {
        await linkVobizNumber(selectedPhoneNumber.number, selectedAgent.vobiz_app_id)
        
        // Step 2: Update database with agent_type
        const response = await fetchApiRoute("/api/phone-numbers/attach", {
          method: "POST",
          body: JSON.stringify({
            phone_number: selectedPhoneNumber.number,
            provider: selectedPhoneNumber.provider,
            agent_type: selectedAgentType,
          }),
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || error.error || "Failed to attach phone number")
        }
      } else if (selectedPhoneNumber.provider === "Plivo" && (selectedAgent as AgentWithTelephony)?.plivo_app_id) {
        await linkPlivoNumber(selectedPhoneNumber.number, (selectedAgent as AgentWithTelephony).plivo_app_id as string)
        
        // Step 2: Update database with agent_type
        const response = await fetchApiRoute("/api/phone-numbers/attach", {
          method: "POST",
          body: JSON.stringify({
            phone_number: selectedPhoneNumber.number,
            provider: selectedPhoneNumber.provider,
            agent_type: selectedAgentType,
          }),
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || error.error || "Failed to attach phone number")
        }
      } else {
        // Use regular attach API for Plivo or if no vobiz_app_id
        const response = await fetchApiRoute("/api/phone-numbers/attach", {
          method: "POST",
          body: JSON.stringify({
            phone_number: selectedPhoneNumber.number,
            provider: selectedPhoneNumber.provider,
            agent_type: selectedAgentType,
          }),
        })

        if (!response.ok) {
          const error = await response.json()
          throw new Error(error.detail || error.error || "Failed to attach phone number")
        }
      }

      // Refresh phone numbers
      if (orgId) {
        const phoneResponse = await fetchApiRoute(`/api/phone-numbers?org_id=${encodeURIComponent(orgId)}`)
        if (phoneResponse.ok) {
          const phoneData: PhoneNumber[] = await phoneResponse.json()
          const transformed: PhoneNumberDisplay[] = phoneData.map((phone) => {
            const agentName = getAgentDisplayName(phone.agent_type)
            return {
              ...phone,
              id: phone.id || phone._id || phone.phone_number,
              number: phone.phone_number,
              addedOn: formatDate(phone.created_at),
              usedBy: phone.agent_type ? (agentName || phone.agent_type) : null,
              agentName: agentName || undefined,
            }
          })
          setPhoneNumbers(transformed)
        }
      }

      setAttachDialogOpen(false)
      setSuccessMessage("Phone number attached successfully")
      setShowSuccessToast(true)
      setTimeout(() => {
        setShowSuccessToast(false)
        setSuccessMessage("")
      }, 3000)
    } catch (error) {
      console.error("Error attaching phone number:", error)
      setErrorMessage(error instanceof Error ? error.message : "Failed to attach phone number")
      setErrorDialogOpen(true)
    } finally {
      setIsAttaching(false)
    }
  }

  const handleDetachClick = (phone: PhoneNumberDisplay) => {
    setPhoneToDetach(phone)
    setDetachDialogOpen(true)
  }

  const handleDetachConfirm = async () => {
    if (!phoneToDetach) return

    try {
      setIsDetaching(true)
      
      // If provider is Vobiz, first unlink from Vobiz application, then update database
      if (phoneToDetach.provider === "Vobiz") {
        await unlinkVobizNumber(phoneToDetach.number)
      } else if (phoneToDetach.provider === "Plivo") {
        await unlinkPlivoNumber(phoneToDetach.number)
      }
      
      // Step 2: Update database to remove agent_type (for both Vobiz and Plivo)
      const response = await fetchApiRoute("/api/phone-numbers/detach", {
        method: "DELETE",
        body: JSON.stringify({
          phone_number: phoneToDetach.number,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || error.error || "Failed to detach phone number")
      }

      // Refresh phone numbers
      if (orgId) {
        const phoneResponse = await fetchApiRoute(`/api/phone-numbers?org_id=${encodeURIComponent(orgId)}`)
        if (phoneResponse.ok) {
          const phoneData: PhoneNumber[] = await phoneResponse.json()
          const transformed: PhoneNumberDisplay[] = phoneData.map((phone) => {
            const agentName = getAgentDisplayName(phone.agent_type)
            return {
              ...phone,
              id: phone.id || phone._id || phone.phone_number,
              number: phone.phone_number,
              addedOn: formatDate(phone.created_at),
              usedBy: phone.agent_type ? (agentName || phone.agent_type) : null,
              agentName: agentName || undefined,
            }
          })
          setPhoneNumbers(transformed)
        }
      }

      setDetachDialogOpen(false)
      setPhoneToDetach(null)
      setSuccessMessage("Phone number detached successfully")
      setShowSuccessToast(true)
      setTimeout(() => {
        setShowSuccessToast(false)
        setSuccessMessage("")
      }, 3000)
    } catch (error) {
      console.error("Error detaching phone number:", error)
      setErrorMessage(error instanceof Error ? error.message : "Failed to detach phone number")
      setErrorDialogOpen(true)
    } finally {
      setIsDetaching(false)
    }
  }

  const handlePhoneNumberChange = (value: string) => {
    // Ensure it starts with +91
    if (!value.startsWith("+91")) {
      setNewPhoneNumber("+91")
      return
    }
    // Only allow digits after +91, max 10 digits
    const digits = value.slice(3).replace(/\D/g, "").slice(0, 10)
    setNewPhoneNumber("+91" + digits)
  }

  // Fetch Vobiz numbers when provider changes to Vobiz
  useEffect(() => {
    const fetchVobizNumbersData = async () => {
      if (selectedProvider === "Vobiz" && addNumberDialogOpen) {
        try {
          setIsLoadingVobizNumbers(true)
          const data = await getVobizNumbers()
          setVobizNumbers(data.numbers || [])
        } catch (error) {
          console.error("Error fetching Vobiz numbers:", error)
          setErrorMessage(error instanceof Error ? error.message : "Failed to fetch Vobiz numbers")
          setErrorDialogOpen(true)
        } finally {
          setIsLoadingVobizNumbers(false)
        }
      } else if (selectedProvider === "Plivo" && addNumberDialogOpen) {
        try {
          setIsLoadingVobizNumbers(true)
          const data = await getPlivoNumbers()
          setVobizNumbers(data.numbers || [])
        } catch (error) {
          console.error("Error fetching Plivo numbers:", error)
          setErrorMessage(error instanceof Error ? error.message : "Failed to fetch Plivo numbers")
          setErrorDialogOpen(true)
        } finally {
          setIsLoadingVobizNumbers(false)
        }
      } else if (selectedProvider !== "Vobiz") {
        // Clear Vobiz numbers when switching away from Vobiz
        setVobizNumbers([])
        setSelectedVobizNumber("")
      }
    }

    fetchVobizNumbersData()
  }, [selectedProvider, addNumberDialogOpen])

  const handleAddNewNumber = async () => {
    if (!selectedProvider) return

    // For telephony providers using provider inventory, validate a number is selected
    if (selectedProvider === "Vobiz" || selectedProvider === "Plivo") {
      if (!selectedVobizNumber) {
        setErrorMessage("Please select a phone number")
        setErrorDialogOpen(true)
        return
      }
    } else {
      // For Plivo, validate phone number is exactly +91 followed by 10 digits
      const digits = newPhoneNumber.slice(3)
      if (digits.length !== 10) {
        setErrorMessage("Phone number must be exactly 10 digits after +91")
        setErrorDialogOpen(true)
        return
      }
    }

    // Check if phone number already exists
    const phoneNumberToAdd = (selectedProvider === "Vobiz" || selectedProvider === "Plivo")
      ? selectedVobizNumber
      : newPhoneNumber
    const numberExists = phoneNumbers.some(phone => phone.number === phoneNumberToAdd)
    
    if (numberExists) {
      setErrorMessage("This phone number is already in use. Please try another number.")
      setErrorDialogOpen(true)
      return
    }

    try {
      setIsAddingNumber(true)
      
      const response = await fetchApiRoute("/api/phone-numbers/attach", {
        method: "POST",
        body: JSON.stringify({
          phone_number: phoneNumberToAdd,
          provider: selectedProvider,
        }),
      })

      if (!response.ok) {
        const error = await response.json()
        throw new Error(error.detail || error.error || "Failed to add phone number")
      }

      // Refresh phone numbers
      if (orgId) {
        const phoneResponse = await fetchApiRoute(`/api/phone-numbers?org_id=${encodeURIComponent(orgId)}`)
        if (phoneResponse.ok) {
          const phoneData: PhoneNumber[] = await phoneResponse.json()
          const transformed: PhoneNumberDisplay[] = phoneData.map((phone) => {
            const agentName = getAgentDisplayName(phone.agent_type)
            return {
              ...phone,
              id: phone.id || phone._id || phone.phone_number,
              number: phone.phone_number,
              addedOn: formatDate(phone.created_at),
              usedBy: phone.agent_type ? (agentName || phone.agent_type) : null,
              agentName: agentName || undefined,
            }
          })
          setPhoneNumbers(transformed)
        }
      }

      setAddNumberDialogOpen(false)
      setNewPhoneNumber("+91")
      setSelectedProvider("")
      setSelectedVobizNumber("")
      setVobizNumbers([])
      setSuccessMessage("Phone number added successfully")
      setShowSuccessToast(true)
      setTimeout(() => {
        setShowSuccessToast(false)
        setSuccessMessage("")
      }, 3000)
    } catch (error) {
      console.error("Error adding phone number:", error)
      setErrorMessage(error instanceof Error ? error.message : "Failed to add phone number")
      setErrorDialogOpen(true)
    } finally {
      setIsAddingNumber(false)
    }
  }

  // Filter phone numbers based on search
  const filteredPhoneNumbers = phoneNumbers.filter((phone) => {
    if (!searchQuery) return true
    const query = searchQuery.toLowerCase()
    return (
      phone.number.toLowerCase().includes(query) ||
      (phone.usedBy && phone.usedBy.toLowerCase().includes(query))
    )
  })

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen bg-neutral-50/40">
        {/* Header */}
        <header className="flex h-16 items-center justify-between gap-4 border-b bg-background/80 backdrop-blur-sm px-5 lg:px-8 sticky top-0 z-10">
          <div className="flex items-center gap-4">
            <nav className="flex items-center gap-1.5 text-base">
              <span className="text-muted-foreground hover:text-foreground cursor-pointer transition-colors">Dashboard</span>
              <ChevronRight className="h-4 w-4 text-muted-foreground/60" />
              <span className="text-foreground font-medium">Numbers</span>
            </nav>
          </div>
          
          
        </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto p-6 lg:p-10">
          {/* Telephony Section */}
          <div className="flex items-center justify-between mb-8">
            <div className="flex flex-col gap-3 w-full max-w-xs">
              <Input
                type="text"
                placeholder="Search numbers"
                className="h-12 rounded-xl border border-neutral-200 bg-white px-4 py-2 w-full font-semibold"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
              />
            </div>
            <Button
              onClick={() => setAddNumberDialogOpen(true)}
              className="h-12 rounded-xl bg-neutral-900 text-white hover:bg-neutral-800 gap-2 font-medium px-6"
            >
              <Phone className="h-4 w-4" />
              Add New Number
            </Button>
          </div>

          {/* Numbers Table */}
          <div className="bg-white rounded-2xl border border-neutral-200 overflow-hidden">
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Loader2 className="h-6 w-6 animate-spin text-neutral-400" />
              </div>
            ) : filteredPhoneNumbers.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-16">
                <Phone className="h-12 w-12 text-neutral-300 mb-4" />
                <p className="text-neutral-600 font-medium mb-1">No phone numbers found</p>
                <p className="text-sm text-neutral-400">
                  {searchQuery ? "Try adjusting your search query" : "Get started by adding a phone number"}
                </p>
              </div>
            ) : (
              <Table>
                <TableHeader>
                  <TableRow className="hover:bg-transparent border-b border-neutral-100">
                    <TableHead className="h-14 pl-6 text-neutral-500 font-medium text-sm">Number</TableHead>
                    <TableHead className="h-14 text-neutral-500 font-medium text-sm">Added On</TableHead>
                    <TableHead className="h-14 text-neutral-500 font-medium text-sm">Provider</TableHead>
                    <TableHead className="h-14 text-neutral-500 font-medium text-sm">Used By</TableHead>
                    <TableHead className="h-14 text-neutral-500 font-medium text-sm">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {filteredPhoneNumbers.map((phone) => (
                    <TableRow 
                      key={phone.id} 
                      className="hover:bg-neutral-50/50 border-b border-neutral-100 last:border-0"
                    >
                      {/* Number Cell */}
                      <TableCell className="pl-6 py-5">
                        <div className="flex items-center gap-3">
                          <Tooltip>
                            <TooltipTrigger asChild>
                              <button
                                onClick={() => copyToClipboard(phone.number, phone.id)}
                                className="h-8 w-8 rounded-lg border border-neutral-200 flex items-center justify-center hover:bg-neutral-100 transition-colors"
                              >
                                <Copy className="h-4 w-4 text-neutral-500" />
                              </button>
                            </TooltipTrigger>
                            <TooltipContent>
                              <p>{copiedId === phone.id ? "Copied!" : "Copy number"}</p>
                            </TooltipContent>
                          </Tooltip>
                          <span className="font-medium text-neutral-900">{phone.number}</span>
                        </div>
                      </TableCell>

                      {/* Added On Cell */}
                      <TableCell className="py-5">
                        <span className="text-neutral-600">{phone.addedOn}</span>
                      </TableCell>

                      {/* Provider Cell */}
                      <TableCell className="py-5">
                        {phone.provider ? (
                          <Badge
                            className={
                              phone.provider === "Vobiz"
                                ? "bg-blue-100 text-blue-700 border-blue-200 hover:bg-blue-100"
                                : phone.provider === "Plivo"
                                ? "bg-purple-100 text-purple-700 border-purple-200 hover:bg-purple-100"
                                : "bg-neutral-100 text-neutral-700 border-neutral-200 hover:bg-neutral-100"
                            }
                          >
                            {phone.provider}
                          </Badge>
                        ) : (
                          <span className="text-neutral-400">-</span>
                        )}
                      </TableCell>

                      {/* Used By Cell */}
                      <TableCell className="py-5">
                        {phone.usedBy ? (
                          <span className="text-neutral-900">{phone.agent_type}</span>
                        ) : (
                          <span className="text-neutral-400">-</span>
                        )}
                      </TableCell>

                      {/* Actions Cell */}
                      <TableCell className="py-5">
                        <div className="flex items-center gap-2">
                          {phone.usedBy ? (
                            <Button 
                              variant="outline" 
                              size="sm" 
                              className="h-8 rounded-lg border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300 hover:text-red-700 gap-1.5 text-sm font-medium transition-all duration-200 focus-visible:ring-2 focus-visible:ring-red-500 focus-visible:ring-offset-2"
                              onClick={() => handleDetachClick(phone)}
                              aria-label={`Detach ${phone.number} from ${phone.usedBy}`}
                            >
                              <Unplug className="h-3.5 w-3.5" />
                              Detach
                            </Button>
                          ) : (
                            <Button 
                              size="sm" 
                              className="h-8 rounded-lg bg-neutral-900 text-white hover:bg-neutral-800 hover:scale-[1.02] active:scale-[0.98] gap-1.5 text-sm font-medium transition-all duration-200 focus-visible:ring-2 focus-visible:ring-neutral-900 focus-visible:ring-offset-2"
                              onClick={() => handleAttachClick(phone)}
                              aria-label={`Attach ${phone.number} to an agent`}
                            >
                              <Plug2 className="h-3.5 w-3.5" />
                              Attach
                            </Button>
                          )}
                        </div>
                      </TableCell>
                    </TableRow>
                  ))}
                </TableBody>
              </Table>
            )}
          </div>
        </main>

        {/* Attach Dialog */}
        <Dialog open={attachDialogOpen} onOpenChange={setAttachDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Attach Phone Number</DialogTitle>
              <DialogDescription>
                Select an agent to attach {selectedPhoneNumber?.number} to.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="agent">Agent</Label>
                <Popover open={agentPopoverOpen} onOpenChange={setAgentPopoverOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={agentPopoverOpen}
                      className="w-full justify-between h-10"
                      disabled={isAttaching}
                    >
                      {selectedAgentType
                        ? agents.find((agent) => agent.agent_type === selectedAgentType)?.agent_type
                        : "Select an agent..."}
                      <ChevronsUpDown className="ml-2 h-4 w-4 shrink-0 opacity-50" />
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-full p-0" align="start">
                    <Command>
                      <CommandInput placeholder="Search agents..." />
                      <CommandList>
                        <CommandEmpty>
                          {selectedPhoneNumber
                            ? `No agents found with ${selectedPhoneNumber.provider} provider.`
                            : "No agents found."}
                        </CommandEmpty>
                        <CommandGroup>
                          {agents
                            .filter((agent) => {
                              // Only show agents with telephony_provider matching the phone number's provider
                              if (!selectedPhoneNumber) return false
                              const agentProvider = (agent as AgentWithTelephony).telephony_provider
                              if (agentProvider !== selectedPhoneNumber.provider) return false
                              
                              // Exclude agents that are already attached to a phone number
                              const isAlreadyAttached = phoneNumbers.some(
                                (phone) => phone.agent_type === agent.agent_type && phone.id !== selectedPhoneNumber.id
                              )
                              return !isAlreadyAttached
                            })
                            .map((agent) => (
                              <CommandItem
                                key={agent.agent_type}
                                value={agent.agent_type}
                                onSelect={() => {
                                  setSelectedAgentType(agent.agent_type)
                                  setAgentPopoverOpen(false)
                                }}
                              >
                                <Check
                                  className={cn(
                                    "mr-2 h-4 w-4",
                                    selectedAgentType === agent.agent_type
                                      ? "opacity-100"
                                      : "opacity-0"
                                  )}
                                />
                                {agent.agent_type}
                              </CommandItem>
                            ))}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                  </PopoverContent>
                </Popover>
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setAttachDialogOpen(false)
                  setSelectedAgentType("")
                  setAgentPopoverOpen(false)
                }}
                disabled={isAttaching}
              >
                Cancel
              </Button>
              <Button
                onClick={handleAttachSubmit}
                disabled={!selectedAgentType || isAttaching}
                className="bg-neutral-900 hover:bg-neutral-800"
              >
                {isAttaching ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Attaching...
                  </>
                ) : (
                  "Attach"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Detach Confirmation Dialog */}
        <Dialog open={detachDialogOpen} onOpenChange={setDetachDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-orange-100">
                  <AlertCircle className="h-5 w-5 text-orange-600" />
                </div>
                <div>
                  <DialogTitle>Detach Phone Number?</DialogTitle>
                  <DialogDescription className="mt-1">
                    This action cannot be undone.
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>
            <div className="py-4">
              <p className="text-sm text-neutral-600">
                Are you sure you want to detach <span className="font-semibold text-neutral-900">{phoneToDetach?.number}</span> from <span className="font-semibold text-neutral-900">{phoneToDetach?.usedBy}</span>?
              </p>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setDetachDialogOpen(false)
                  setPhoneToDetach(null)
                }}
                disabled={isDetaching}
              >
                Cancel
              </Button>
              <Button
                variant="destructive"
                onClick={handleDetachConfirm}
                disabled={isDetaching}
              >
                {isDetaching ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Detaching...
                  </>
                ) : (
                  "Detach"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Error Dialog */}
        <Dialog open={errorDialogOpen} onOpenChange={setErrorDialogOpen}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <div className="flex items-center gap-3">
                <div className="flex h-10 w-10 items-center justify-center rounded-full bg-yellow-100">
                  <AlertTriangleIcon className="h-5 w-5 text-yellow-600" />
                </div>
                <div>
                  <DialogTitle>Try Another Number</DialogTitle>
                  <DialogDescription className="mt-1">
                    We couldn&apos;t process your request. Please try another phone number.
                  </DialogDescription>
                </div>
              </div>
            </DialogHeader>
            <div className="py-4">
              <p className="text-sm text-neutral-600">{errorMessage}</p>
            </div>
            <DialogFooter>
              <Button
                onClick={() => {
                  setErrorDialogOpen(false)
                  setErrorMessage("")
                }}
                className="bg-neutral-900 hover:bg-neutral-800"
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Add New Number Dialog */}
        <Dialog open={addNumberDialogOpen} onOpenChange={(open) => {
          setAddNumberDialogOpen(open)
          if (!open) {
            // Reset state when dialog closes
            setNewPhoneNumber("+91")
            setSelectedProvider("")
            setSelectedVobizNumber("")
            setVobizNumbers([])
          }
        }}>
          <DialogContent className="sm:max-w-[425px]">
            <DialogHeader>
              <DialogTitle>Add New Number</DialogTitle>
              <DialogDescription>
                Add a new phone number to your organization.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Label htmlFor="provider">Telephony Provider</Label>
                <Select
                  value={selectedProvider}
                  onValueChange={setSelectedProvider}
                  disabled={isAddingNumber}
                >
                  <SelectTrigger className="w-full h-10">
                    <SelectValue placeholder="Select provider" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="Vobiz">Vobiz</SelectItem>
                    <SelectItem value="Plivo">
                      Plivo
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="phone">Phone Number</Label>
                {(selectedProvider === "Vobiz" || selectedProvider === "Plivo") ? (
                  <>
                    {isLoadingVobizNumbers ? (
                      <div className="flex items-center justify-center h-10 border border-neutral-200 rounded-md">
                        <Loader2 className="h-4 w-4 animate-spin text-neutral-400" />
                        <span className="ml-2 text-sm text-neutral-500">Loading numbers...</span>
                      </div>
                    ) : (
                      <Select
                        value={selectedVobizNumber}
                        onValueChange={setSelectedVobizNumber}
                        disabled={isAddingNumber || isLoadingVobizNumbers}
                      >
                        <SelectTrigger className="w-full h-10">
                          <SelectValue placeholder="Select a phone number" />
                        </SelectTrigger>
                        <SelectContent>
                          {vobizNumbers.length === 0 ? (
                            <div className="px-2 py-1.5 text-sm text-neutral-500">No numbers available</div>
                          ) : (
                            vobizNumbers.map((number) => (
                              <SelectItem key={number} value={number}>
                                {number}
                              </SelectItem>
                            ))
                          )}
                        </SelectContent>
                      </Select>
                    )}
                    <p className="text-xs text-neutral-500">
                      Select a phone number from your {selectedProvider} account
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-neutral-500 py-2">
                    Please select a provider to continue
                  </p>
                )}
              </div>
            </div>
            <DialogFooter>
              <Button
                variant="outline"
                onClick={() => {
                  setAddNumberDialogOpen(false)
                  setNewPhoneNumber("+91")
                  setSelectedProvider("")
                  setSelectedVobizNumber("")
                  setVobizNumbers([])
                }}
                disabled={isAddingNumber}
              >
                Cancel
              </Button>
              <Button
                onClick={handleAddNewNumber}
                disabled={
                  !selectedProvider || 
                  isAddingNumber ||
                  ((selectedProvider === "Vobiz" || selectedProvider === "Plivo") ? !selectedVobizNumber : newPhoneNumber.length !== 13) ||
                  isLoadingVobizNumbers
                }
                className="bg-neutral-900 hover:bg-neutral-800"
              >
                {isAddingNumber ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Adding...
                  </>
                ) : (
                  "Add New Number"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>

        {/* Success Toast */}
        {showSuccessToast && (
          <div className="fixed top-20 right-6 z-50 animate-in slide-in-from-top-5 fade-in-0 bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 min-w-[300px]">
            <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
            <p className="font-medium">{successMessage}</p>
          </div>
        )}
      </div>
    </TooltipProvider>
  )
}

