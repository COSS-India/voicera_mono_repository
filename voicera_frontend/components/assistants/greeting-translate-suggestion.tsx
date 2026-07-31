"use client"

import { useEffect, useRef, useState } from "react"
import { AnimatePresence, motion, useReducedMotion } from "framer-motion"
import { CheckCircle2, Globe2, Loader2, RotateCcw } from "lucide-react"
import { Button } from "@/components/ui/button"
import {
  detectTextLanguage,
  isChromeTranslationAvailable,
  translateText,
} from "@/lib/chrome-translation"
import { sanitizeSpokenMessageInput } from "@/lib/spoken-message"
import {
  agentLanguageToBcp47,
  bcp47ToDisplayLanguage,
  canDetectGreetingLanguage,
  inferFallbackSourceLanguage,
  normalizeBcp47,
  shouldOfferGreetingTranslationAsync,
} from "@/lib/greeting-message"
import { displayLanguageName } from "@/lib/languageLabels"
import { cn } from "@/lib/utils"

type GreetingTranslateSuggestionProps = {
  greeting: string
  primaryLanguage: string
  onTranslated: (text: string) => void
  className?: string
}

type SuggestionPhase =
  | "idle"
  | "checking"
  | "previewing"
  | "ready"
  | "done"

export function GreetingTranslateSuggestion({
  greeting,
  primaryLanguage,
  onTranslated,
  className,
}: GreetingTranslateSuggestionProps) {
  const reducedMotion = useReducedMotion()
  const [phase, setPhase] = useState<SuggestionPhase>("idle")
  const [dismissedForText, setDismissedForText] = useState<string | null>(null)
  const [previousText, setPreviousText] = useState<string | null>(null)
  const [successGreeting, setSuccessGreeting] = useState<string | null>(null)
  const [previewTranslation, setPreviewTranslation] = useState<string | null>(null)
  const [detectedSourceLang, setDetectedSourceLang] = useState<string | null>(null)
  const detectionRequestIdRef = useRef(0)
  const phaseRef = useRef<SuggestionPhase>("idle")
  const [recheckNonce, setRecheckNonce] = useState(0)
  phaseRef.current = phase

  const trimmedGreeting = greeting.trim()
  const agentLangLabel =
    displayLanguageName(primaryLanguage) || primaryLanguage || "your language"
  const sourceLangLabel = detectedSourceLang
    ? bcp47ToDisplayLanguage(detectedSourceLang)
    : "Original"

  useEffect(() => {
    if (phase === "done" && successGreeting && trimmedGreeting !== successGreeting) {
      setPhase("idle")
      setSuccessGreeting(null)
      setPreviousText(null)
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
    }
  }, [trimmedGreeting, phase, successGreeting])

  useEffect(() => {
    if (phaseRef.current === "done") {
      return
    }

    if (!trimmedGreeting || !primaryLanguage.trim()) {
      setPhase("idle")
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
      return
    }

    if (!canDetectGreetingLanguage(trimmedGreeting)) {
      setPhase("idle")
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
      return
    }

    if (dismissedForText === trimmedGreeting) {
      setPhase("idle")
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
      return
    }

    if (!isChromeTranslationAvailable()) {
      setPhase("idle")
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
      return
    }

    const targetBcp47 = agentLanguageToBcp47(primaryLanguage)
    if (!targetBcp47) {
      setPhase("idle")
      setPreviewTranslation(null)
      setDetectedSourceLang(null)
      return
    }

    const requestId = ++detectionRequestIdRef.current
    setPhase("checking")
    setPreviewTranslation(null)
    setDetectedSourceLang(null)

    const timer = window.setTimeout(() => {
      void (async () => {
        try {
          let sourceBcp47: string | null = null

          const detected = await detectTextLanguage(trimmedGreeting)
          if (requestId !== detectionRequestIdRef.current) return

          if (detected) {
            const shouldOffer = await shouldOfferGreetingTranslationAsync(
              detected.language,
              primaryLanguage,
              detected.confidence,
              trimmedGreeting
            )
            if (requestId !== detectionRequestIdRef.current) return
            if (shouldOffer) {
              sourceBcp47 = normalizeBcp47(detected.language)
            }
          } else {
            sourceBcp47 = await inferFallbackSourceLanguage(
              trimmedGreeting,
              primaryLanguage
            )
            if (requestId !== detectionRequestIdRef.current) return
          }

          if (!sourceBcp47) {
            setPhase("idle")
            setPreviewTranslation(null)
            setDetectedSourceLang(null)
            return
          }

          setDetectedSourceLang(sourceBcp47)
          setPhase("previewing")

          const translated = await translateText({
            text: trimmedGreeting,
            sourceLanguage: sourceBcp47,
            targetLanguage: targetBcp47,
          })

          if (requestId !== detectionRequestIdRef.current) return

          setPreviewTranslation(sanitizeSpokenMessageInput(translated))
          setPhase("ready")
        } catch (error) {
          console.error("Greeting translation preview failed:", error)
          if (requestId !== detectionRequestIdRef.current) return
          setPreviewTranslation(null)
          setDetectedSourceLang(null)
          setPhase("idle")
        }
      })()
    }, 500)

    return () => {
      window.clearTimeout(timer)
      detectionRequestIdRef.current += 1
    }
  }, [trimmedGreeting, primaryLanguage, dismissedForText, recheckNonce])

  useEffect(() => {
    if (dismissedForText && dismissedForText !== trimmedGreeting) {
      setDismissedForText(null)
    }
  }, [trimmedGreeting, dismissedForText])

  const handleApply = () => {
    if (!previewTranslation?.trim()) return
    setPreviousText(greeting)
    onTranslated(sanitizeSpokenMessageInput(previewTranslation))
    setSuccessGreeting(sanitizeSpokenMessageInput(previewTranslation))
    setPhase("done")
  }

  const handleKeepOriginal = () => {
    setDismissedForText(trimmedGreeting)
    setPreviewTranslation(null)
    setDetectedSourceLang(null)
    setPhase("idle")
  }

  const handleUndo = () => {
    if (previousText != null) {
      onTranslated(sanitizeSpokenMessageInput(previousText))
    }
    setPreviousText(null)
    setSuccessGreeting(null)
    setPreviewTranslation(null)
    setDetectedSourceLang(null)
    setRecheckNonce((n) => n + 1)
    setPhase("idle")
  }

  const showPreview = phase === "ready" && previewTranslation
  const showSuccess = phase === "done"
  const showChecking = phase === "checking"
  const showPreviewing = phase === "previewing"

  if (
    phase === "idle" &&
    !showChecking &&
    !showPreviewing &&
    !showPreview &&
    !showSuccess
  ) {
    return null
  }

  const motionProps = reducedMotion
    ? {
        initial: { opacity: 0 },
        animate: { opacity: 1 },
        exit: { opacity: 0 },
        transition: { duration: 0.15 },
      }
    : {
        initial: { opacity: 0, y: 8, height: 0 },
        animate: { opacity: 1, y: 0, height: "auto" },
        exit: { opacity: 0, y: -4, height: 0 },
        transition: { duration: 0.28, ease: [0.25, 0.1, 0.25, 1] as const },
      }

  return (
    <div className={cn("overflow-hidden", className)} role="region" aria-live="polite">
      <AnimatePresence mode="wait">
        {(showChecking || showPreviewing) && (
          <motion.div
            key="loading"
            {...motionProps}
            className="flex items-center gap-3 rounded-lg border border-slate-200 bg-slate-50 px-4 py-3"
          >
            <Loader2 className="h-4 w-4 animate-spin shrink-0 text-slate-500" aria-hidden="true" />
            <p className="text-sm text-slate-600">
              {showChecking
                ? "Reviewing your welcome message…"
                : `Preparing a ${agentLangLabel} translation…`}
            </p>
          </motion.div>
        )}

        {showPreview && (
          <motion.div
            key="suggestion"
            {...motionProps}
            className="rounded-lg border border-slate-200 bg-white p-4 space-y-3"
          >
            <div className="flex items-start gap-3">
              <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-slate-100 border border-slate-200">
                <Globe2 className="h-4 w-4 text-slate-600" aria-hidden="true" />
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-semibold text-slate-900">
                  Language mismatch
                </h3>
                <p className="text-sm text-slate-600 mt-0.5">
                  Message is in {sourceLangLabel}; agent speaks {agentLangLabel}.
                </p>
              </div>
            </div>

            <div className="rounded-lg border border-slate-200 bg-slate-50 px-3 py-2.5">
              <p className="text-[11px] font-semibold uppercase tracking-wide text-slate-500 mb-1">
                {agentLangLabel} (Suggested)
              </p>
              <p className="text-sm leading-relaxed text-slate-900">{previewTranslation}</p>
            </div>

            <div className="flex flex-col-reverse gap-2 sm:flex-row sm:items-center sm:justify-end pt-1">
              <Button
                type="button"
                variant="outline"
                onClick={handleKeepOriginal}
                className="h-11 min-h-[44px] w-full sm:w-auto border-slate-200 text-slate-700 hover:bg-slate-50"
              >
                Keep Original
              </Button>
              <Button
                type="button"
                onClick={handleApply}
                className="h-11 min-h-[44px] w-full sm:w-auto bg-slate-900 text-white hover:bg-slate-800"
              >
                Apply Translation
              </Button>
            </div>
          </motion.div>
        )}

        {showSuccess && (
          <motion.div
            key="success"
            {...motionProps}
            className="flex flex-col gap-3 rounded-lg border border-emerald-200 bg-emerald-50 px-4 py-3 sm:flex-row sm:items-center sm:justify-between"
            role="status"
          >
            <div className="flex items-start gap-3 min-w-0">
              <CheckCircle2
                className="h-5 w-5 shrink-0 text-emerald-600 mt-0.5"
                aria-hidden="true"
              />
              <div>
                <p className="text-sm font-medium text-emerald-900">
                  Translation applied
                </p>
                <p className="text-sm text-emerald-800/90 mt-0.5">
                  Your welcome message is now in {agentLangLabel}.
                </p>
              </div>
            </div>
            <Button
              type="button"
              variant="outline"
              onClick={handleUndo}
              className="h-11 min-h-[44px] w-full sm:w-auto shrink-0 border-emerald-300 bg-white text-emerald-800 hover:bg-emerald-100"
            >
              <RotateCcw className="h-4 w-4 mr-2" aria-hidden="true" />
              Undo
            </Button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
