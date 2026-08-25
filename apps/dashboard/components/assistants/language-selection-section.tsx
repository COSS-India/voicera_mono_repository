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
  dedupeLanguages,
  getActiveLanguages,
  hasMultipleLanguages,
  languageSwitchingStackEligible,
  splitPrimaryAndSecondary,
} from "@/lib/languageModelSupport"
import {
  type KenpathVariant,
  isBharatVistaarLanguageSupported,
} from "@/lib/kenpath"
import { cn } from "@/lib/utils"
import {
  Check,
  Languages,
  Star,
  X,
} from "lucide-react"

type LanguageOption = { code: string; name: string }

type LanguageSelectionSectionProps = {
  selectedLanguages: string[]
  allLanguages: LanguageOption[]
  llmProvider: string
  kenpathVariant: KenpathVariant
  sttProvider: string
  sttModel: string
  ttsProvider: string
  ttsModel: string
  open: boolean
  onOpenChange: (open: boolean) => void
  onLanguagesChange: (languages: string[]) => void
}

export function LanguageSelectionSection({
  selectedLanguages,
  allLanguages,
  llmProvider,
  kenpathVariant,
  sttProvider,
  sttModel,
  ttsProvider,
  ttsModel,
  open,
  onOpenChange,
  onLanguagesChange,
}: LanguageSelectionSectionProps) {
  const activeLanguages = getActiveLanguages(selectedLanguages)
  const { primary, secondaryLanguages } = splitPrimaryAndSecondary(activeLanguages)
  const selectedSet = useMemo(
    () => new Set(activeLanguages.map((l) => l.toLowerCase())),
    [activeLanguages]
  )
  const multiLanguage = hasMultipleLanguages(activeLanguages)
  const switchingEligible = languageSwitchingStackEligible(
    llmProvider,
    sttProvider,
    sttModel,
    ttsProvider,
    ttsModel
  )

  const triggerLabel = useMemo(() => {
    if (activeLanguages.length === 0) return "Select languages..."
    if (activeLanguages.length === 1) {
      return displayLanguageName(activeLanguages[0])
    }
    if (activeLanguages.length === 2) {
      return `${displayLanguageName(activeLanguages[0])} + ${displayLanguageName(activeLanguages[1])}`
    }
    return `${displayLanguageName(activeLanguages[0])} + ${activeLanguages.length - 1} more`
  }, [activeLanguages])

  const toggleLanguage = (code: string) => {
    const isSelected = selectedSet.has(code.toLowerCase())

    if (isSelected) {
      onLanguagesChange(
        activeLanguages.filter((l) => l.toLowerCase() !== code.toLowerCase())
      )
      return
    }

    onLanguagesChange(dedupeLanguages([...activeLanguages, code]))
  }

  const removeLanguage = (code: string) => {
    onLanguagesChange(
      activeLanguages.filter((l) => l.toLowerCase() !== code.toLowerCase())
    )
  }

  const makePrimary = (code: string) => {
    const rest = activeLanguages.filter(
      (l) => l.toLowerCase() !== code.toLowerCase()
    )
    onLanguagesChange(dedupeLanguages([code, ...rest]))
  }

  return (
    <div className="space-y-4">
      <div className="space-y-1.5">
        <label className="text-base font-bold text-slate-900">Languages</label>
        <p className="text-sm text-slate-500">
          Pick one primary language (call start language). Any additional languages
          are secondary and available for mid-call switching. STT and TTS options
          reflect what works for every selected language.
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
            {activeLanguages.length > 0 && (
              <span className="ml-2 shrink-0 rounded-full bg-slate-100 px-2 py-0.5 text-xs font-semibold text-slate-600">
                {activeLanguages.length}
              </span>
            )}
          </Button>
        </PopoverTrigger>
        <PopoverContent className="w-[400px] p-0" align="start">
          <Command>
            <CommandInput placeholder="Search languages..." />
            <CommandList>
              <CommandEmpty>No language found.</CommandEmpty>
              <CommandGroup heading="Languages">
                {allLanguages.map((lang) => {
                  const bharatBlocked =
                    llmProvider === "kenpath" &&
                    !isBharatVistaarLanguageSupported(kenpathVariant, lang.code)
                  const isSelected = selectedSet.has(lang.code.toLowerCase())

                  return (
                    <CommandItem
                      key={lang.code}
                      value={`${lang.code} ${lang.name}`}
                      disabled={bharatBlocked}
                      onSelect={() => {
                        if (bharatBlocked) return
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
                          bharatBlocked ? "text-slate-400" : ""
                        )}
                      >
                        {lang.name}
                      </span>
                      {bharatBlocked && (
                        <span className="ml-2 text-xs text-slate-400">
                          (not supported)
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

      {activeLanguages.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {primary && (
            <LanguageChip
              code={primary}
              role="primary"
              onRemove={() => removeLanguage(primary)}
            />
          )}
          {secondaryLanguages.map((code) => (
            <LanguageChip
              key={code}
              code={code}
              role="secondary"
              onMakePrimary={() => makePrimary(code)}
              onRemove={() => removeLanguage(code)}
            />
          ))}
        </div>
      )}

      {multiLanguage && (
        <div className="rounded-xl border border-slate-200 bg-slate-50/60 p-4 space-y-3">
          <p className="text-[10px] font-semibold uppercase tracking-widest text-slate-500">
            Language roles
          </p>
          <div className="grid gap-3 sm:grid-cols-2">
            <RoleCard
              role="Primary"
              language={displayLanguageName(primary)}
              hint="Call starts in this language"
              accent="blue"
            />
            <RoleCard
              role="Secondary"
              language={
                secondaryLanguages.length === 1
                  ? displayLanguageName(secondaryLanguages[0])
                  : `${secondaryLanguages.length} languages`
              }
              hint={
                secondaryLanguages.length === 1
                  ? "Switch target during the call"
                  : secondaryLanguages.map((l) => displayLanguageName(l)).join(", ")
              }
              accent="emerald"
            />
          </div>
          <p className="text-sm text-slate-600">
            {switchingEligible ? (
              <>
                <span className="font-medium text-emerald-700">
                  Language switching enabled
                </span>{" "}
                between {displayLanguageName(primary)} and{" "}
                {secondaryLanguages.map((l) => displayLanguageName(l)).join(", ")}.
              </>
            ) : (
              <>
                Multi-language selection saved. Mid-call switching activates
                with OpenAI + AI4Bharat Indic STT/TTS models.
              </>
            )}
          </p>
        </div>
      )}

      {activeLanguages.length === 1 && (
        <p className="text-sm text-slate-500">
          Single language selected — no mid-call language switching.
        </p>
      )}

      {activeLanguages.length > 0 && (
        <p className="text-xs text-slate-400">
          Model options below are filtered to providers and models supported by{" "}
          {activeLanguages.map((l) => displayLanguageName(l)).join(", ")}.
        </p>
      )}
    </div>
  )
}

function LanguageChip({
  code,
  role,
  onMakePrimary,
  onRemove,
}: {
  code: string
  role: "primary" | "secondary"
  onMakePrimary?: () => void
  onRemove: () => void
}) {
  const isPrimary = role === "primary"

  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded-full border px-3 py-1 text-sm font-medium",
        isPrimary
          ? "border-blue-200 bg-blue-50 text-blue-800"
          : "border-emerald-200 bg-emerald-50 text-emerald-800"
      )}
    >
      {isPrimary ? (
        <Star className="h-3 w-3 text-amber-500 fill-amber-400" />
      ) : null}
      <span>{displayLanguageName(code)}</span>
      <span
        className={cn(
          "rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide",
          isPrimary
            ? "bg-blue-100 text-blue-700"
            : "bg-emerald-100 text-emerald-700"
        )}
      >
        {isPrimary ? "Primary" : "Secondary"}
      </span>
      {!isPrimary && onMakePrimary && (
        <button
          type="button"
          onClick={onMakePrimary}
          className="rounded-full px-1.5 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-slate-500 hover:bg-white/80 hover:text-slate-700"
          aria-label={`Make ${displayLanguageName(code)} primary`}
        >
          Set primary
        </button>
      )}
      <button
        type="button"
        onClick={onRemove}
        className="rounded-full p-0.5 text-slate-400 hover:bg-white/80 hover:text-slate-600"
        aria-label={`Remove ${displayLanguageName(code)}`}
      >
        <X className="h-3.5 w-3.5" />
      </button>
    </span>
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
        }
      : {
          dot: "bg-emerald-500",
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
      <p className="text-sm font-semibold text-slate-900">{language}</p>
      <p className="text-xs text-slate-500 mt-1">{hint}</p>
    </div>
  )
}
