"use client"

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
  SheetDescription,
  SheetFooter,
} from "@/components/ui/sheet"
import type { Agent } from "@/lib/api"
import { Phone, Loader2, CheckCircle2, XCircle, Info, FileText, AudioLines } from "lucide-react"

interface TestCallSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agent: Agent | null
  getAgentDisplayName: (agent: Agent) => string
}

// E.164 format validation regex
const E164_REGEX = /^\+[1-9]\d{1,14}$/

export function TestCallSheet({
  open,
  onOpenChange,
  agent,
  getAgentDisplayName,
}: TestCallSheetProps) {
  const [customerNumber, setCustomerNumber] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [touched, setTouched] = useState(false)
  const isCallInProgress = useRef(false)

  // Reset state when sheet closes
  useEffect(() => {
    if (!open) {
      setCustomerNumber("")
      setError("")
      setSuccess(false)
      setTouched(false)
      setIsLoading(false)
      isCallInProgress.current = false
    }
  }, [open])

  // Validate phone number format
  const isValidPhone = customerNumber.trim() === "" || E164_REGEX.test(customerNumber.trim())
  const showError = touched && !isValidPhone

  const handleClose = () => {
    if (!isLoading) {
      setCustomerNumber("")
      setError("")
      setSuccess(false)
      setTouched(false)
      onOpenChange(false)
    }
  }

  const handlePhoneChange = (value: string) => {
    setCustomerNumber(value)
    setError("")
    setSuccess(false)
    if (!touched) setTouched(true)
  }

  const handleMakeTestCall = async () => {
    // Prevent double calls
    if (isCallInProgress.current || isLoading || success || !agent) {
      return
    }

    const trimmedNumber = customerNumber.trim()
    
    // Validation
    if (!trimmedNumber) {
      setError("Phone number is required")
      setTouched(true)
      return
    }

    if (!E164_REGEX.test(trimmedNumber)) {
      setError("Please enter a valid phone number in E.164 format (e.g., +1234567890)")
      setTouched(true)
      return
    }

    // Set flag to prevent concurrent calls
    isCallInProgress.current = true
    setIsLoading(true)
    setError("")
    setSuccess(false)

    if (!agent.phone_number) {
      setError("Agent does not have a phone number configured. Please attach a phone number to this agent first.")
      return
    }

    const payload = {
      customer_number: trimmedNumber,
      agent_id: agent.agent_id,
      caller_id: agent.phone_number,
    }

    try {
      const res = await fetch("/api/outbound-call", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}))
        const msg =
          errorData.error ||
          errorData.detail ||
          errorData.message ||
          `Request failed with status ${res.status}`
        throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg))
      }

      setSuccess(true)
    } catch (error) {
      console.error("Failed to initiate test call:", error)
      setError(error instanceof Error ? error.message : "An error occurred while making the test call. Please try again.")
    } finally {
      setIsLoading(false)
      isCallInProgress.current = false
    }
  }

  return (
    <Sheet open={open} onOpenChange={handleClose}>
      <SheetContent side="right" className="w-full sm:w-3/4 p-6 flex flex-col">
        <SheetHeader className="space-y-3 pb-6 border-b">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10">
              <Phone className="h-5 w-5 text-primary" />
            </div>
            <div className="flex-1">
              <SheetTitle className="text-xl">
                {agent ? getAgentDisplayName(agent) : "Test Call"}
              </SheetTitle>
              <SheetDescription className="mt-1">
                Initiate a test call to verify your agent configuration
              </SheetDescription>
            </div>
          </div>
        </SheetHeader>

        <div className="flex-1 overflow-y-auto py-6">
          <div className="flex flex-col gap-6">
            {/* Success Message */}
            {success && (
              <div className="p-5 rounded-lg bg-green-50 border border-green-200 space-y-4">
                <div className="flex items-start gap-3">
                  <CheckCircle2 className="h-5 w-5 text-green-600 shrink-0 mt-0.5" />
                  <div className="flex-1">
                    <p className="text-sm font-semibold text-green-900">Test call initiated successfully!</p>
                    
                  </div>
                </div>
                
                <div className="pt-3 border-t border-green-200">
                  <p className="text-xs font-medium text-green-900 mb-2.5">After the call completes:</p>
                  <div className="space-y-2.5">
                    <div className="flex items-start gap-2.5">
                      <FileText className="h-4 w-4 text-green-700 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-xs font-medium text-green-900">Transcript</p>
                        <p className="text-xs text-green-700 mt-0.5">
                          The full conversation transcript will be available for review after the call ends.
                        </p>
                      </div>
                    </div>
                    <div className="flex items-start gap-2.5">
                      <AudioLines className="h-4 w-4 text-green-700 shrink-0 mt-0.5" />
                      <div className="flex-1">
                        <p className="text-xs font-medium text-green-900">Audio Recording</p>
                        <p className="text-xs text-green-700 mt-0.5">
                          The complete audio file of the call will be accessible for playback and analysis.
                        </p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}

            {/* Error Message */}
            {error && (
              <div className="p-4 rounded-lg bg-red-50 border border-red-200 flex items-start gap-3">
                <XCircle className="h-5 w-5 text-red-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  <p className="text-sm font-medium text-red-900">Failed to initiate call</p>
                  <p className="text-xs text-red-700 mt-1">{error}</p>
                </div>
              </div>
            )}

            {/* Customer Number Input */}
            <div className="space-y-3">
              <Label htmlFor="customer-number" className="text-sm font-medium text-slate-900">
                Your Test Phone Number
                <span className="text-red-500 ml-1">*</span>
              </Label>
              <div className="relative">
                <Phone className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none" />
                <Input
                  id="customer-number"
                  type="tel"
                  placeholder="+919999999999"
                  value={customerNumber}
                  onChange={(e) => {
                    // If input is empty, reset to +91
                    if (e.target.value === "") {
                      handlePhoneChange("+91");
                    } else if (!e.target.value.startsWith("+91")) {
                      handlePhoneChange("+91" + e.target.value.replace(/^\+*/, ""));
                    } else {
                      handlePhoneChange(e.target.value);
                    }
                  }}
                  onFocus={(e) => {
                    // On focus, if input is empty, set default to "+91"
                    if (!e.target.value || e.target.value === "") {
                      handlePhoneChange("+91");
                    }
                  }}
                  onBlur={() => setTouched(true)}
                  className={`w-full pl-10 ${showError ? "border-red-500 focus-visible:ring-red-500/20" : ""}`}
                  aria-invalid={showError}
                  aria-describedby={showError ? "phone-error" : "phone-help"}
                  disabled={isLoading || success}
                />
              </div>
              {showError ? (
                <p id="phone-error" className="text-xs text-red-600 flex items-center gap-1">
                  <XCircle className="h-3 w-3" />
                  Please enter a valid phone number in E.164 format (e.g., +1234567890)
                </p>
              ) : (
                <p id="phone-help" className="text-xs text-slate-500 flex items-start gap-1.5">
                  <Info className="h-3 w-3 mt-0.5 shrink-0" />
                  <span>Enter your phone number in E.164 format. Include country code (e.g., +1 for US, +91 for India)</span>
                </p>
              )}
            </div>

            {/* Caller ID (Read-only, from agent config) */}
            <div className="space-y-3">
              <Label htmlFor="caller-id" className="text-sm font-medium text-slate-900 flex items-center gap-2">
                Caller ID
              </Label>
              <div className="relative">
                <Input
                  id="caller-id"
                  type="text"
                  value={agent?.phone_number || "No phone number configured"}
                  readOnly
                  disabled
                  tabIndex={-1}
                  className={`w-full bg-slate-100 border-slate-200 text-slate-500 cursor-not-allowed pointer-events-none ${
                    !agent?.phone_number ? "text-slate-400 italic" : ""
                  }`}
                  aria-readonly
                  aria-disabled
                />
                <span className="absolute right-3 top-1/2 -translate-y-1/2 text-xs text-slate-400 select-none">
                  {agent?.phone_number ? "Not editable" : "Not configured"}
                </span>
              </div>
              <p className="text-xs text-slate-500 flex items-start gap-1.5">
                <Info className="h-3 w-3 mt-0.5 shrink-0" />
                <span>
                  {agent?.phone_number ? (
                    <>
                      This is the phone number that will appear on your caller ID when you receive the test call.&nbsp;
                      <span className="text-slate-400 font-medium">
                        (This cannot be changed)
                      </span>
                    </>
                  ) : (
                    <>
                      This agent does not have a phone number configured. Please attach a phone number to this agent in the Numbers section before making a test call.
                    </>
                  )}
                </span>
              </p>
            </div>

            
          </div>
        </div>

        <SheetFooter className="gap-3 pt-6 border-t mt-auto">
          <Button
            variant="outline"
            onClick={handleClose}
            disabled={isLoading}
            className="flex-1"
          >
            {success ? "Close" : "Cancel"}
          </Button>
          <Button
            type="button"
            onClick={handleMakeTestCall}
            disabled={!customerNumber.trim() || !isValidPhone || !agent?.phone_number || isLoading || success}
            className="flex-1"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                Initiating...
              </>
            ) : success ? (
              <>
                <CheckCircle2 className="h-4 w-4 mr-2" />
                Call Initiated
              </>
            ) : (
              <>
                <Phone className="h-4 w-4 mr-2" />
                Make Test Call
              </>
            )}
          </Button>
        </SheetFooter>
      </SheetContent>
    </Sheet>
  )
}
