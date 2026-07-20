"use client"

import { useMemo } from "react"
import { Button } from "@/components/ui/button"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { displayLanguageName } from "@/lib/languageLabels"
import {
  getActiveLanguages,
  isBilingual,
  languageSwitchingStackEligible,
} from "@/lib/languageModelSupport"
import {
  type KenpathVariant,
  isBharatVistaarLanguageSupported,
} from "@/lib/kenpath"
import { cn } from "@/lib/utils"
import {
  ArrowLeftRight,
  Check,
  Languages,
  Star,
  X,
} from "lucide-react"

const MAX_LANGUAGES = 2

type LanguageOption = { code: string; name: string }

type LanguageSelectionSectionProps = {
  primaryLanguage: string
  secondaryLanguage: string
  allLanguages: LanguageOption[]
  llmProvider: string
  kenpathVariant: KenpathVariant
  sttProvider: string
  sttModel: string
  ttsProvider: string
  ttsModel: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onPrimaryChange: (language: string) => void
  onSecondaryChange: (language: string) => void
  onSwapRoles: () => void
}

export function LanguageSelectionSection({
  primaryLanguage,
  secondaryLanguage,
  allLanguages,
  llmProvider,
  kenpathVariant,
  sttProvider,
  sttModel,
  ttsProvider,
  ttsModel,
  open,
  onOpenChange,
  onPrimaryChange,
  onSecondaryChange,
  onSwapRoles,
}: LanguageSelectionSectionProps) {
  const selectedSet = useMemo(() => {
    const set = new Set<string>()
    if (primaryLanguage) set.add(primaryLanguage)
    if (secondaryLanguage) set.add(secondaryLanguage)
    return set
  }, [primaryLanguage, secondaryLanguage])

  const bilingual = isBilingual(primaryLanguage, secondaryLanguage)
  const activeLanguages = getActiveLanguages(primaryLanguage, secondaryLanguage)
  const switchingEligible = languageSwitchingStackEligible(
    llmProvider,
    sttProvider,
    sttModel,
    ttsProvider,
    ttsModel
  )

  const triggerLabel = useMemo(() => {
    if (!primaryLanguage) return "Select languages..."
    if (bilingual) {
      return `${displayLanguageName(primaryLanguage)} + ${displayLanguageName(secondaryLanguage)}`
    }
    return displayLanguageName(primaryLanguage)
  }, [primaryLanguage, secondaryLanguage, bilingual])

  const toggleLanguage = (code: string) => {
    const isSelected = selectedSet.has(code)

    if (isSelected) {
      if (code === primaryLanguage) {
        if (secondaryLanguage) {
          onPrimaryChange(secondaryLanguage)
          onSecondaryChange("")
        } else {
          onPrimaryChange("")
          onSecondaryChange("")
        }
      } else {
        onSecondaryChange("")
      }
      return
    }

    if (selectedSet.size >= MAX_LANGUAGES) return

    if (!primaryLanguage) {
      onPrimaryChange(code)
      return
    }

    onSecondaryChange(code)
  }

  const removeLanguage = (code: string) => {
    if (code === primaryLanguage) {
      if (secondaryLanguage) {
        onPrimaryChange(secondaryLanguage)
        onSecondaryChange("")
      } else {
        onPrimaryChange("")
      }
    } else {
      onSecondaryChange("")
    }
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <label className="text-base font-bold text-slate-900">Languages</label>
        <p className="text-sm text-slate-500">
          Choose one language for single-language agents, or two to allow mid-call
          switching. STT and TTS options reflect what works for every selected
          language.
        </p>
      </div>

      <Popover open={open} onOpenChange={onOpenChange}>
        <PopoverTrigger asChild>
          <Button
            variant="outline"
            role="combobox"
            aria-expanded={open}
            className="w-full max-w-md h-12 justify-between rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
          >
            <div className="flex items-center gap-2 truncate">
              <Languages className="h-4 w-4 shrink-0 text-blue-500" />
              <span className="truncate">{triggerLabel}</span>
            </div>
            {selectedSet.size > 0 && (
              <span className="ml-2 shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600">
                {selectedSet.size}/{MAX_LANGUAGES}
              </span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[400px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search languages..." />
            <CommandList>
              <CommandEmpty>No language found.</CommandEmpty>
              <CommandGroup heading={`Languages (max ${MAX_LANGUAGES})`}>
                {allLanguages.map((lang) => {
                  const bharatBlocked =
                    llmProvider === "kenpath" &&
                    !isBharatVistaarLanguageSupported(kenpathVariant, lang.code)
                  const isSelected = selectedSet.has(lang.code)
                  const atCapacity =
                    selectedSet.size >= MAX_LANGUAGES && !isSelected

                  return (
                    <CommandItem
                      key={lang.code}
                      value={`${lang.code} ${lang.name}`}
                      disabled={bharatBlocked || atCapacity}
                      onSelect={() => {
                        if (bharatBlocked || atCapacity) return
                        toggleLanguage(lang.code)
                      }}
                      className="py-2.5"
                    >
                      <span
                        className={cn(
                          "mr-2 flex h-4 w-4 items-center justify-center rounded-sm border",
                          isSelected
                            ? "border-blue-500 bg-blue-500 text-white"
                            : "border-slate-300"
                        )}
                      >
                        {isSelected && <Check className="h-3 w-3" />}
                      </span>
                      <span
                        className={cn(
                          "font-medium",
                          bharatBlocked || atCapacity ? "text-slate-400" : ""
                        )}
                      >
                        {lang.name}
                      </span>
                      {bharatBlocked && (
                        <span className="ml-2 text-xs text-slate-400">
                          (not supported)
                        </span>
                      )}
                      {atCapacity && (
                        <span className="ml-2 text-xs text-slate-400">
                          (max reached)
                        </span>
                      )}
                    </CommandItem>
                  )
                })}
              </CommandGroup>
            </CommandList>
          </Command>
        </PopoverContent>
      </Popover>

      {selectedSet.size > 0 && (
        <div className="flex flex-wrap gap-2">
          {[primaryLanguage, secondaryLanguage].filter(Boolean).map((code) => (
            <span
              key={code}
              className="inline-flex items-center gap-1.5 rounded-full border border-slate-200 bg-white px-3 py-1 text-sm font-medium text-slate-700"
            >
              {displayLanguageName(code)}
              <button
                type="button"
                onClick={() => removeLanguage(code)}
                className="rounded-full p-0.5 text-slate-400 hover:bg-slate-100 hover:text-slate-600"
                aria-label={`Remove ${displayLanguageName(code)}`}
              >
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
        </div>
      )}

      {bilingual && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4 space-y-3">
          <div className="flex items-center justify-between gap-3">
            <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
              Assign roles
            </p>
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onSwapRoles}
              className="h-8 gap-1.5 rounded-lg border-slate-200 bg-white text-xs font-semibold text-slate-700 hover:bg-slate-100"
            >
              <ArrowLeftRight className="h-3.5 w-3.5" />
              Swap
            </Button>
          </div>

          <div className="grid gap-3 sm:grid-cols-2">
            <RoleCard
              role="Primary"
              language={primaryLanguage}
              hint="Call starts in this language"
              accent="blue"
            />
            <RoleCard
              role="Secondary"
              language={secondaryLanguage}
              hint="Switch target during the call"
              accent="emerald"
            />
          </div>

          <p className="text-sm text-slate-600">
            {switchingEligible ? (
              <>
                <span className="font-medium text-emerald-700">
                  Language switching enabled
                </span>{" "}
                between {displayLanguageName(primaryLanguage)} and{" "}
                {displayLanguageName(secondaryLanguage)}.
              </>
            ) : (
              <>
                Bilingual selection saved. Mid-call switching activates with
                OpenAI + AI4Bharat Indic STT/TTS models.
              </>
            )}
          </p>
        </div>
      )}

      {primaryLanguage && !bilingual && (
        <p className="text-sm text-slate-500">
          Single language selected — no mid-call language switching.
        </p>
      )}

      {activeLanguages.length > 0 && (
        <p className="text-xs text-slate-400">
          Model options below are filtered to providers and models supported by{" "}
          {activeLanguages.map((l) => displayLanguageName(l)).join(" and ")}.
        </p>
      )}
    </div>
  )
}

function RoleCard({
  role,
  language,
  hint,
  accent,
}: {
  role: string
  language: string
  hint: string
  accent: "blue" | "emerald"
}) {
  const styles =
    accent === "blue"
      ? {
          dot: "bg-blue-500",
          badge: "bg-blue-50 border-blue-200 text-blue-700",
        }
      : {
          dot: "bg-emerald-500",
          badge: "bg-emerald-50 border-emerald-200 text-emerald-700",
        }

  return (
    <div className="rounded-lg border border-slate-200 bg-white p-3">
      <div className="flex items-center gap-1.5 mb-2">
        <div className={cn("size-1.5 rounded-full", styles.dot)} />
        <span
          className={cn(
            "text-[10px] font-semibold uppercase tracking-widest",
            accent === "blue" ? "text-blue-600" : "text-emerald-600"
          )}
        >
          {role}
        </span>
        {role === "Primary" && (
          <Star className="h-3 w-3 text-amber-500 fill-amber-400" />
        )}
      </div>
      <p className="text-sm font-semibold text-slate-900">
        {displayLanguageName(language)}
      </p>
      <p className="text-xs text-slate-500 mt-1">{hint}</p>
    </div>
  )
}
