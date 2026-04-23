"use client"

import { useState } from "react"
import Link from "next/link"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Mail, Loader2, ArrowLeft, CheckCircle2 } from "lucide-react"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

export default function ForgotPasswordPage() {
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [email, setEmail] = useState("")

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError("")

    try {
      const response = await fetch(`${API_BASE_URL}/api/v1/users/forgot-password`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.detail || "Failed to send reset email")
      }

      setSuccess(true)
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred")
    } finally {
      setIsLoading(false)
    }
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
              <h2 className="text-2xl font-bold text-slate-900 mb-2">Check your email</h2>
              <p className="text-slate-500">
                We&apos;ve sent a password reset link to{" "}
                <span className="text-slate-900 font-medium">{email}</span>
              </p>
            </div>

            <div className="space-y-4">
              <p className="text-slate-500 text-sm text-center">
                Didn&apos;t receive the email? Check your spam folder or try again.
              </p>
              
              <Button
                onClick={() => setSuccess(false)}
                variant="outline"
                className="w-full h-12 bg-white border-slate-200 text-slate-700 hover:bg-slate-50 rounded-lg"
              >
                Try another email
              </Button>

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

      {/* Right Panel - Forgot Password Form */}
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
            <h2 className="text-2xl font-bold text-slate-900 mb-2">Forgot password?</h2>
            <p className="text-slate-500">No worries, we&apos;ll send you reset instructions</p>
          </div>

          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className="p-4 rounded-lg bg-red-50 border border-red-200 text-red-600 text-sm">
                {error}
              </div>
            )}

            {/* Email */}
            <div className="space-y-2">
              <label className="text-sm font-medium text-slate-700">Email</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-400" />
                <Input
                  type="email"
                  placeholder="you@company.com"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="h-12 pl-11 bg-white border-slate-200 text-slate-900 placeholder:text-slate-400 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                  required
                />
              </div>
            </div>

            {/* Submit */}
            <Button
              type="submit"
              disabled={isLoading}
              className="w-full h-12 bg-slate-900 hover:bg-slate-800 text-white font-medium rounded-lg transition-colors"
            >
              {isLoading ? (
                <Loader2 className="h-5 w-5 animate-spin" />
              ) : (
                "Send reset link"
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
