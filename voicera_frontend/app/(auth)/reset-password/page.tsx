"use client"

import { useState, Suspense } from "react"
import Link from "next/link"
import { useRouter, useSearchParams } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Eye, EyeOff, Lock, Loader2, Check, CheckCircle2, ArrowLeft } from "lucide-react"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

function ResetPasswordForm() {
  const router = useRouter()
  const searchParams = useSearchParams()
  const token = searchParams.get("token") || ""
  
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [formData, setFormData] = useState({
    password: "",
    confirmPassword: "",
  })

  const passwordRequirements = [
    { met: formData.password.length >= 8, text: "At least 8 characters" },
    { met: /[A-Z]/.test(formData.password), text: "One uppercase letter" },
    { met: /[0-9]/.test(formData.password), text: "One number" },
  ]

  const passwordsMatch = formData.password === formData.confirmPassword && formData.confirmPassword.length > 0

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    if (formData.password !== formData.confirmPassword) {
      setError("Passwords do not match")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/users/reset-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          token,
          new_password: formData.password,
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "Failed to reset password")
      }

      setSuccess(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
  }

  if (!token) {
    return (
      <div className="fixed inset-0 w-full h-full min-h-screen bg-slate-50 flex">
        {/* Left Panel - Branding */}
        <div className="hidden lg:flex lg:w-1/2 bg-slate-900 p-12 flex-col justify-between">
          <div>
            <div className="mb-16">
              <img
                src="/voicera-logo-black-source.png"
                alt="VoiceRA"
                className="h-12 w-auto max-w-[220px]"
                style={{ filter: "brightness(0) invert(1)" }}
              />
            </div>
            
            <div className="max-w-md">
              <h1 className="text-4xl font-bold text-white mb-4 leading-tight">
                Build intelligent voice assistants with ease
              </h1>
              <p className="text-slate-400 text-lg">
                Create, deploy, and manage AI-powered voice agents for your business in minutes.
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-8 text-slate-500 text-sm">
            <Link href="#" className="hover:text-slate-300 transition-colors">Privacy</Link>
            <Link href="#" className="hover:text-slate-300 transition-colors">Terms</Link>
          </div>
        </div>

        {/* Right Panel - Invalid Link Message */}
        <div className="flex-1 flex flex-col justify-center p-8">
          <div className="w-full max-w-md mx-auto">
            {/* Mobile Logo */}
            <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
              <img
                src="/voicera-logo-black-source.png"
                alt="VoiceRA"
                className="h-10 w-auto max-w-[180px]"
                style={{ filter: "brightness(0) saturate(100%)" }}
              />
            </div>

            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-red-100 mb-4">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  className="w-8 h-8 text-red-600"
                  stroke="currentColor"
                  strokeWidth="2"
                >
                  <circle cx="12" cy="12" r="10" />
                  <line x1="12" y1="8" x2="12" y2="12" />
                  <line x1="12" y1="16" x2="12.01" y2="16" />
                </svg>
              </div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Invalid link</h2>
              <p className="text-slate-500">This password reset link is invalid or has expired</p>
            </div>

            <div className="space-y-4">
              <Link href="/forgot-password">
                <Button className="w-full h-12 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg transition-colors">
                  Request new link
                </Button>
              </Link>

              <Link href="/" className="block">
                <Button
                  variant="ghost"
                  className="w-full h-12 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg"
                >
                  <ArrowLeft className="mr-2 h-5 w-5" />
                  Back to sign in
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  if (success) {
    return (
      <div className="fixed inset-0 w-full h-full min-h-screen bg-slate-50 flex">
        {/* Left Panel - Branding */}
        <div className="hidden lg:flex lg:w-1/2 bg-slate-900 p-12 flex-col justify-between">
          <div>
            <div className="mb-16">
              <img
                src="/voicera-logo-black-source.png"
                alt="VoiceRA"
                className="h-12 w-auto max-w-[220px]"
                style={{ filter: "brightness(0) invert(1)" }}
              />
            </div>
            
            <div className="max-w-md">
              <h1 className="text-4xl font-bold text-white mb-4 leading-tight">
                Build intelligent voice assistants with ease
              </h1>
              <p className="text-slate-400 text-lg">
                Create, deploy, and manage AI-powered voice agents for your business in minutes.
              </p>
            </div>
          </div>
          
          <div className="flex items-center gap-8 text-slate-500 text-sm">
            <Link href="#" className="hover:text-slate-300 transition-colors">Privacy</Link>
            <Link href="#" className="hover:text-slate-300 transition-colors">Terms</Link>
          </div>
        </div>

        {/* Right Panel - Success Message */}
        <div className="flex-1 flex flex-col justify-center p-8">
          <div className="w-full max-w-md mx-auto">
            {/* Mobile Logo */}
            <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
              <img
                src="/voicera-logo-black-source.png"
                alt="VoiceRA"
                className="h-10 w-auto max-w-[180px]"
                style={{ filter: "brightness(0) saturate(100%)" }}
              />
            </div>

            <div className="text-center mb-8">
              <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-emerald-100 mb-4">
                <CheckCircle2 className="w-8 h-8 text-emerald-600" />
              </div>
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Password reset!</h2>
              <p className="text-slate-500">Your password has been successfully reset</p>
            </div>

            <div className="space-y-4">
              <Link href="/">
                <Button className="w-full h-12 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg transition-colors">
                  Sign in to your account
                </Button>
              </Link>
            </div>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="fixed inset-0 w-full h-full min-h-screen bg-slate-50 flex">
      {/* Left Panel - Branding */}
      <div className="hidden lg:flex lg:w-1/2 bg-slate-900 p-12 flex-col justify-between">
        <div>
          <div className="mb-16">
            <img
              src="/voicera-logo-black-source.png"
              alt="VoiceRA"
              className="h-12 w-auto max-w-[220px]"
              style={{ filter: "brightness(0) invert(1)" }}
            />
          </div>
          
          <div className="max-w-md">
            <h1 className="text-4xl font-bold text-white mb-4 leading-tight">
              Build intelligent voice assistants with ease
            </h1>
            <p className="text-slate-400 text-lg">
              Create, deploy, and manage AI-powered voice agents for your business in minutes.
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-8 text-slate-500 text-sm">
          <Link href="#" className="hover:text-slate-300 transition-colors">Privacy</Link>
          <Link href="#" className="hover:text-slate-300 transition-colors">Terms</Link>
        </div>
      </div>

      {/* Right Panel - Reset Password Form */}
      <div className="flex-1 flex flex-col justify-center p-8">
        <div className="w-full max-w-md mx-auto">
          {/* Mobile Logo */}
          <div className="lg:hidden flex items-center gap-3 mb-8 justify-center">
            <img
              src="/voicera-logo-black-source.png"
              alt="VoiceRA"
              className="h-10 w-auto max-w-[180px]"
              style={{ filter: "brightness(0) saturate(100%)" }}
            />
          </div>

          <div className="mb-8">
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Set new password</h2>
            <p className="text-slate-500">Your new password must be different from previous passwords</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* New Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">New Password</label>
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
                    <Check className={`h-3.5 w-3.5 ${req.met ? "opacity-100 text-emerald-600" : "opacity-40 text-slate-400"}`} />
                    {req.text}
                  </div>
                ))}
              </div>
            </div>

            {/* Confirm Password */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Confirm Password</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type={showConfirmPassword ? "text" : "password"}
                  placeholder="••••••••"
                  value={formData.confirmPassword}
                  onChange={(e) => setFormData({ ...formData, confirmPassword: e.target.value })}
                  className="h-12 pl-11 pr-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                  required
                />
                <button
                  type="button"
                  onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-slate-400 hover:text-slate-600 transition-colors"
                >
                  {showConfirmPassword ? <EyeOff className="h-5 w-5" /> : <Eye className="h-5 w-5" />}
                </button>
              </div>
              {formData.confirmPassword && (
                <div className={`flex items-center gap-1.5 text-xs ${passwordsMatch ? "text-emerald-600" : "text-red-600"}`}>
                  <Check className={`h-3.5 w-3.5 ${passwordsMatch ? "opacity-100 text-emerald-600" : "opacity-40 text-red-400"}`} />
                  {passwordsMatch ? "Passwords match" : "Passwords do not match"}
                </div>
              )}
            </div>

            {/* Submit */}
            <Button
              type="submit"
              disabled={isLoading || !passwordsMatch}
              className="w-full h-12 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg transition-colors disabled:opacity-50"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                "Reset password"
              )}
            </Button>

            {/* Back to login */}
            <Link href="/" className="block">
              <Button
                type="button"
                variant="ghost"
                className="w-full h-12 text-slate-600 hover:text-slate-900 hover:bg-slate-100 rounded-lg"
              >
                <ArrowLeft className="mr-2 h-5 w-5" />
                Back to sign in
              </Button>
            </Link>
          </form>
        </div>
      </div>
    </div>
  )
}

export default function ResetPasswordPage() {
  return (
    <div className="fixed inset-0 w-full h-full min-h-screen bg-slate-50">
      <Suspense fallback={
        <div className="w-full h-full flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
        </div>
      }>
        <ResetPasswordForm />
      </Suspense>
    </div>
  )
}

