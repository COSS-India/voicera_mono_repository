"use client"

import { Check } from "lucide-react"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

interface LanguageOption {
  code: string
  name: string
}

interface TranslationLanguagesSectionProps {
  /** Candidate listener languages (any language some TTS provider can speak). */
  languageOptions: LanguageOption[]
  /** Currently selected listener languages. */
  targetLanguages: string[]
  /** The presenter's language — cannot also be a listener language. */
  sourceLanguage: string
  /** Selected listener languages the chosen TTS provider cannot speak. */
  unsupportedLanguages: string[]
  /** Per-target voice map (voices are language-specific for AI4Bharat/Bhashini). */
  targetVoices: Record<string, string>
  ttsProvider: string
  /** Fixed voice list for a language, or [] when the provider takes a free-form ID. */
  getVoicesForLanguage: (language: string) => string[]
  displayLanguageName: (code: string) => string
  onToggleLanguage: (code: string) => void
  onVoiceChange: (language: string, voice: string) => void
}

/**
 * Listener-language and per-language voice configuration for a live-translation
 * agent. Shared by the create wizard and the agent edit page so both stay in sync.
 */
export function TranslationLanguagesSection({
  languageOptions,
  targetLanguages,
  sourceLanguage,
  unsupportedLanguages,
  targetVoices,
  ttsProvider,
  getVoicesForLanguage,
  displayLanguageName,
  onToggleLanguage,
  onVoiceChange,
}: TranslationLanguagesSectionProps) {
  return (
    <>
      <div className="space-y-4 pt-6 border-t border-slate-100">
        <div>
          <label className="text-base font-bold text-slate-900">Listener Languages</label>
          <p className="text-sm text-slate-500">
            Listeners choose one of these on the shared link. Each active language costs one
            translation stream, no matter how many people listen to it.
          </p>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-2">
          {languageOptions.map((lang) => {
            const isSelected = targetLanguages.includes(lang.code)
            const isSource = lang.code === sourceLanguage
            return (
              <button
                key={lang.code}
                type="button"
                disabled={isSource}
                onClick={() => onToggleLanguage(lang.code)}
                className={`flex items-center gap-2 rounded-lg border px-3 py-2 text-sm text-left transition-all ${
                  isSelected
                    ? "border-slate-900 bg-slate-900 text-white"
                    : "border-slate-200 bg-white text-slate-700 hover:border-slate-300"
                } ${isSource ? "opacity-40 cursor-not-allowed" : ""}`}
                title={isSource ? "This is the presenter's language" : undefined}
              >
                {isSelected && <Check className="h-3.5 w-3.5 shrink-0" />}
                <span className="truncate">{lang.name}</span>
              </button>
            )
          })}
        </div>
        {targetLanguages.length === 0 && (
          <p className="text-sm text-amber-700">Select at least one listener language.</p>
        )}
        {unsupportedLanguages.length > 0 && (
          <p className="text-sm text-red-600">
            {ttsProvider || "The selected TTS provider"} does not support:{" "}
            {unsupportedLanguages.join(", ")}. Remove them or pick another provider below.
          </p>
        )}
      </div>

      {targetLanguages.length > 0 && (
        <div className="space-y-3 pt-6 border-t border-slate-100">
          <label className="text-sm font-semibold text-slate-700">
            Voice per listener language
          </label>
          <div className="space-y-2">
            {targetLanguages.map((lang) => {
              const voices = getVoicesForLanguage(lang)
              return (
                <div key={lang} className="flex items-center gap-3">
                  <span className="w-40 shrink-0 truncate text-sm text-slate-700">
                    {displayLanguageName(lang)}
                  </span>
                  {voices.length === 0 ? (
                    <Input
                      value={targetVoices[lang] || ""}
                      onChange={(e) => onVoiceChange(lang, e.target.value)}
                      placeholder="Voice ID (optional)"
                      className="h-11 flex-1 rounded-lg border-slate-200 bg-white"
                    />
                  ) : (
                    <Select
                      value={targetVoices[lang] || ""}
                      onValueChange={(v) => onVoiceChange(lang, v)}
                    >
                      <SelectTrigger className="h-11 flex-1 rounded-lg border-slate-200 bg-white font-medium">
                        <SelectValue placeholder="Select voice" />
                      </SelectTrigger>
                      <SelectContent className="rounded-lg max-h-[200px]">
                        {voices.map((voice) => (
                          <SelectItem key={voice} value={voice} className="py-2.5">
                            <span className="font-medium">{voice}</span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}
    </>
  )
}
