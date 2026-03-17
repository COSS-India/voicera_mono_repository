"use client"

import { useState, useEffect, use } from "react"
import Link from "next/link"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Eye, EyeOff, Mail, Lock, User, Building2, Loader2, Check, Users, AlertCircle } from "lucide-react"
import { joinOrganization, checkUserExists } from "@/lib/api"
import { AnimatePresence, motion } from "framer-motion"

interface PageProps {
  params: Promise<{ slug: string }>
}

export default function AddMemberPage({ params }: PageProps) {
  const { slug } = use(params)
  const router = useRouter()
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [orgId, setOrgId] = useState("")
  const [inviteUuid, setInviteUuid] = useState("")
  const [formData, setFormData] = useState({
    name: "",
    email: "",
    password: "",
    company_name: "",
  })
  const [emailError, setEmailError] = useState("")
  const [isCheckingEmail, setIsCheckingEmail] = useState(false)

  // Parse the slug to extract org_id and uuid
  // Format: add-member-{orgId}-{uuid}
  useEffect(() => {
    if (slug) {
      // The slug format is: {orgId}-{uuid}
      // orgId is typically 6 characters, uuid is 36 characters (with dashes)
      // We need to split carefully since UUID contains dashes
      const parts = slug.split("-")
      
      if (parts.length >= 6) {
        // First part is orgId (6 chars), rest is UUID
        const extractedOrgId = parts[0]
        const extractedUuid = parts.slice(1).join("-")
        
        setOrgId(extractedOrgId)
        setInviteUuid(extractedUuid)
        
        console.log("Parsed invite link:", {
          org_id: extractedOrgId,
          uuid: extractedUuid,
        })
      } else {
        setError("Invalid invite link format")
      }
    }
  }, [slug])

  const passwordRequirements = [
    { met: formData.password.length >= 8, text: "At least 8 characters" },
    { met: /[A-Z]/.test(formData.password), text: "One uppercase letter" },
    { met: /[0-9]/.test(formData.password), text: "One number" },
  ]

  // Check if email already exists when user finishes typing
  const handleEmailBlur = async () => {
    if (!formData.email || !formData.email.includes("@")) return
    
    setIsCheckingEmail(true)
    setEmailError("")
    
    try {
      const existingUser = await checkUserExists(formData.email)
      if (existingUser) {
        setEmailError("This email is already part of an organization. Please try a different email.")
      }
    } catch {
      // If the check fails, we'll let the form submission handle it
    } finally {
      setIsCheckingEmail(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")
    setEmailError("")

    if (!orgId) {
      setError("Invalid invite link - organization ID not found")
      setIsLoading(false)
      return
    }

    // Check if email already exists before submitting
    try {
      const existingUser = await checkUserExists(formData.email)
      if (existingUser) {
        setEmailError("This email is already part of an organization. Please try a different email.")
        setIsLoading(false)
        return
      }
    } catch {
      // If the check fails, continue with the form submission
    }

    // Prepare the payload with org_id from URL
    const payload = {
      email: formData.email,
      password: formData.password,
      name: formData.name,
      company_name: formData.company_name,
      org_id: orgId,
    }

    try {
      await joinOrganization(payload)
      
      // Redirect to login on success
      router.push("/?joined=true")
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-50 flex">
      {/* Left Panel - Branding */}
      <AnimatePresence>
        <div className="hidden lg:flex lg:w-1/2 bg-[#8B5A2B] p-12 flex-col justify-between">
          <div>
            {/* Animated logo section */}
            <motion.div
              className="mb-8 bg-white rounded-lg p-2 inline-block"
              initial={{ opacity: 0, y: -35 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ type: "spring", stiffness: 60, damping: 15, delay: 0.1 }}
            >
              <img src="/ekstep.svg" alt="Ekstep" className="h-14 w-16" />
            </motion.div>
            
            <div className="max-w-lg">
              <motion.h1
                className="text-5xl font-bold text-white mb-6 leading-tight"
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 60, damping: 15, delay: 0.24 }}
              >
                Join Your Team
              </motion.h1>
              <motion.p
                className="text-white/90 text-lg mb-12"
                initial={{ opacity: 0, y: 32 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ type: "spring", stiffness: 70, damping: 14, delay: 0.35 }}
              >
                You&apos;ve been invited to collaborate on India's open Voice AI infrastructure
              </motion.p>
              
              <motion.div
                className="space-y-6"
                initial="hidden"
                animate="visible"
                variants={{
                  hidden: {},
                  visible: {
                    transition: {
                      staggerChildren: 0.15,
                      delayChildren: 0.48,
                    },
                  },
                }}
              >
                <motion.div
                  className="flex items-start gap-3"
                  variants={{
                    hidden: { opacity: 0, x: -25 },
                    visible: { opacity: 1, x: 0 },
                  }}
                >
                  <svg className="w-5 h-5 text-white mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <div>
                    <p className="text-white font-semibold">Built for Bharat</p>
                    <p className="text-white/70">Multilingual, inclusive, voice-first</p>
                  </div>
                </motion.div>
                
                <motion.div
                  className="flex items-start gap-3"
                  variants={{
                    hidden: { opacity: 0, x: -25 },
                    visible: { opacity: 1, x: 0 },
                  }}
                >
                  <svg className="w-5 h-5 text-white mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <div>
                    <p className="text-white font-semibold">Open & interoperable</p>
                    <p className="text-white/70">Modular, DPI-aligned voice stack</p>
                  </div>
                </motion.div>
                
                <motion.div
                  className="flex items-start gap-3"
                  variants={{
                    hidden: { opacity: 0, x: -25 },
                    visible: { opacity: 1, x: 0 },
                  }}
                >
                  <svg className="w-5 h-5 text-white mt-1 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                  </svg>
                  <div>
                    <p className="text-white font-semibold">Proven at scale</p>
                    <p className="text-white/70">Powering real public deployments</p>
                  </div>
                </motion.div>
              </motion.div>
            </div>
          </div>
          
          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 60, damping: 16, delay: 0.67 }}
          >
            <div className="flex items-center gap-4 text-white/60 text-sm mb-2">
              <span>Digital India BHASHINI Division © 2026</span>
              <span>·</span>
              <Link href="#" className="hover:text-white transition-colors">Privacy</Link>
              <span>·</span>
              <Link href="#" className="hover:text-white transition-colors">Terms</Link>
            </div>
            <p className="hover:text-white transition-colors text-white/40 font-light">Built as Digital Public Infrastructure</p>
          </motion.div>
        </div>
      </AnimatePresence>

      {/* Right Panel - Signup Form */}
      <div className="flex-1 flex items-center justify-center p-8 overflow-auto" style={{ backgroundColor: "#f9f9f5" }}>
        <motion.div
          initial={{ opacity: 0, x: 60 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ type: "spring", stiffness: 70, damping: 18, delay: 0.24 }}
          className="w-full max-w-md bg-white rounded-lg p-8 shadow-lg"
        >
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
            <div className="h-10 w-10 rounded-lg bg-slate-900 flex items-center justify-center">
              <svg
                viewBox="0 0 24 24"
                fill="none"
                className="w-6 h-6 text-white"
                stroke="currentColor"
                strokeWidth="2"
              >
                <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z" />
                <path d="M19 10v2a7 7 0 0 1-14 0v-2" />
                <line x1="12" x2="12" y1="19" y2="22" />
              </svg>
            </div>
            <span className="text-xl font-semibold text-slate-900">EkStep</span>
          </div>

          <div className="mb-8">
            <div className="flex items-center gap-2 mb-2">
              <Users className="h-5 w-5 text-[#8B5A2B]" />
              <span className="text-sm font-medium text-[#8B5A2B]">Team Invitation</span>
            </div>
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Join your team on VOICERA</h2>
            <p className="text-slate-500">Complete your profile to join the organization</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* Organization ID (Read-only indicator) */}
            {orgId && (
              <div className="p-3 rounded-lg bg-[#8B5A2B]/10 border border-[#8B5A2B]/20">
                <div className="flex items-center gap-2">
                  <Users className="h-4 w-4 text-[#8B5A2B]" />
                  <span className="text-sm text-[#8B5A2B]">
                    Joining organization: <span className="font-mono font-medium">{orgId}</span>
                  </span>
                </div>
              </div>
            )}

            {/* Name */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Full Name</label>
              <div className="relative">
                <User className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type="text"
                  placeholder="John Doe"
                  value={formData.name}
                  onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                  className="h-12 pl-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                  required
                />
              </div>
            </div>

            {/* Email */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Work Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type="email"
                  placeholder="you@company.com"
                  value={formData.email}
                  onChange={(e) => {
                    setFormData({ ...formData, email: e.target.value })
                    setEmailError("") // Clear error when user starts typing
                  }}
                  onBlur={handleEmailBlur}
                  className={`h-12 pl-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200 ${
                    emailError ? "border-red-300 focus:border-red-400 focus:ring-red-200" : ""
                  }`}
                  required
                />
                {isCheckingEmail && (
                  <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 animate-spin" />
                )}
              </div>
              {emailError && (
                <div className="flex items-start gap-2 text-sm text-red-600">
                  <AlertCircle className="h-4 w-4 mt-0.5 flex-shrink-0" />
                  <span>{emailError}</span>
                </div>
              )}
            </div>

            {/* Company */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Company Name</label>
              <div className="relative">
                <Building2 className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type="text"
                  placeholder="Acme Inc."
                  value={formData.company_name}
                  onChange={(e) => setFormData({ ...formData, company_name: e.target.value })}
                  className="h-12 pl-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type={showPassword ? "text" : "password"}
                  placeholder="••••••••"
                  value={formData.password}
                  onChange={(e) => setFormData({ ...formData, password: e.target.value })}
                  className="h-12 pl-11 pr-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowPassword(!showPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                >
                  {showPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              
              {/* Password requirements */}
              <div className="flex flex-wrap gap-3 mt-3">
                {passwordRequirements.map((req, i) => (
                  <div
                    key={i}
                    className={`flex items-center gap-1.5 text-xs ${
                      req.met ? "text-emerald-600" : "text-slate-500"
                    }`}
                  >
                    <Check className={`h-3.5 w-3.5 ${req.met ? "opacity-100" : "opacity-40"}`} />
                    {req.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Submit */}
            <Button
              type="submit"
              disabled={isLoading || !orgId || !!emailError || isCheckingEmail}
              className="w-full h-12 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg transition-colors"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                "Join Organization"
              )}
            </Button>
          </form>

          {/* Sign in link */}
          <p className="text-center text-slate-500 mt-6">
            Already have an account?{" "}
            <Link
              href="/"
              className="text-slate-900 hover:underline font-medium"
            >
              Sign in
            </Link>
          </p>
        </motion.div>
      </div>
    </div>
  )
}
