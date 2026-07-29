"use client"

import * as React from "react"
import { useEffect, useRef, useState, useCallback } from "react"
import {
  Mic,
  Upload,
  Trash2,
  Play,
  Pause,
  Download,
  Loader2,
  Volume2,
  Clock,
  ChevronDown,
  X,
  Check,
  Wand2,
  AudioLines,
} from "lucide-react"
import { useWavesurfer } from "@wavesurfer/react"

import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { Badge } from "@/components/ui/badge"

import {
  getTTSLanguages,
  getVoiceDesignPresets,
  listRefAudios,
  uploadRefAudio,
  deleteRefAudio,
  synthesizeTTS,
  type TTSLanguage,
  type VoiceDesignPreset,
  type RefAudio,
  type TTSSynthesizeResult,
} from "@/lib/api"

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type Mode = "clone" | "design"

interface HistoryEntry {
  id: string
  text: string
  language: string
  mode: Mode
  audioUrl: string
  audioDuration: number | null
  synthTime: number | null
  rtf: number | null
  timestamp: number
}

const MAX_HISTORY = 5
const HISTORY_KEY = "tts_studio_history"

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function loadHistory(): HistoryEntry[] {
  try {
    return JSON.parse(localStorage.getItem(HISTORY_KEY) ?? "[]")
  } catch {
    return []
  }
}

function saveHistory(h: HistoryEntry[]) {
  localStorage.setItem(HISTORY_KEY, JSON.stringify(h.slice(0, MAX_HISTORY)))
}

function formatDuration(s: number | null): string {
  if (s === null) return "—"
  return s < 60 ? `${s.toFixed(2)}s` : `${Math.floor(s / 60)}m ${(s % 60).toFixed(0)}s`
}

// ---------------------------------------------------------------------------
// WaveSurfer player sub-component
// ---------------------------------------------------------------------------

function AudioPlayer({ url, label }: { url: string; label?: string }) {
  const containerRef = useRef<HTMLDivElement>(null)
  const { wavesurfer, isPlaying } = useWavesurfer({
    container: containerRef,
    url,
    waveColor: "#94a3b8",
    progressColor: "#0f172a",
    height: 48,
    barWidth: 2,
    barGap: 1,
    barRadius: 2,
  })

  return (
    <div className="flex flex-col gap-2">
      {label && <p className="text-xs text-slate-500">{label}</p>}
      <div ref={containerRef} className="w-full rounded-md bg-slate-50 px-2 py-1" />
      <div className="flex items-center gap-2">
        <Button
          size="sm"
          variant="outline"
          onClick={() => wavesurfer?.playPause()}
          className="h-8 w-8 p-0"
        >
          {isPlaying ? <Pause className="size-4" /> : <Play className="size-4" />}
        </Button>
        <a href={url} download="synthesis.wav">
          <Button size="sm" variant="outline" className="h-8 w-8 p-0">
            <Download className="size-4" />
          </Button>
        </a>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Ref-audio picker
// ---------------------------------------------------------------------------

function RefAudioPicker({
  stored,
  selected,
  onSelect,
  onUpload,
  onDelete,
}: {
  stored: RefAudio[]
  selected: string | null
  onSelect: (key: string | null) => void
  onUpload: (file: File) => Promise<void>
  onDelete: (key: string) => Promise<void>
}) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [uploading, setUploading] = useState(false)

  async function handleFile(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file) return
    setUploading(true)
    try {
      await onUpload(file)
    } finally {
      setUploading(false)
      if (inputRef.current) inputRef.current.value = ""
    }
  }

  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between">
        <p className="text-sm font-medium text-slate-700">Reference Audio</p>
        <Button
          size="sm"
          variant="outline"
          onClick={() => inputRef.current?.click()}
          disabled={uploading}
          className="h-7 gap-1 text-xs"
        >
          {uploading ? (
            <Loader2 className="size-3 animate-spin" />
          ) : (
            <Upload className="size-3" />
          )}
          Upload
        </Button>
        <input
          ref={inputRef}
          type="file"
          accept="audio/*"
          className="hidden"
          onChange={handleFile}
        />
      </div>

      {stored.length === 0 ? (
        <p className="rounded-md border border-dashed border-slate-200 py-4 text-center text-xs text-slate-400">
          No reference audios yet. Upload one above.
        </p>
      ) : (
        <div className="max-h-48 space-y-1 overflow-y-auto">
          {stored.map((r) => (
            <div
              key={r.key}
              onClick={() => onSelect(selected === r.key ? null : r.key)}
              className={`flex cursor-pointer items-center justify-between rounded-md border px-3 py-2 text-sm transition-colors ${
                selected === r.key
                  ? "border-slate-900 bg-slate-900 text-white"
                  : "border-slate-200 bg-white hover:border-slate-300 hover:bg-slate-50 text-slate-700"
              }`}
            >
              <div className="flex items-center gap-2 min-w-0">
                {selected === r.key ? (
                  <Check className="size-3 shrink-0" />
                ) : (
                  <AudioLines className="size-3 shrink-0 text-slate-400" />
                )}
                <span className="truncate text-xs">{r.filename}</span>
              </div>
              <Button
                size="sm"
                variant="ghost"
                className="h-6 w-6 p-0 hover:text-red-500"
                onClick={(e) => {
                  e.stopPropagation()
                  onDelete(r.key)
                }}
              >
                <Trash2 className="size-3" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function TTSStudioPage() {
  // ---- data from API ----
  const [languages, setLanguages] = useState<TTSLanguage[]>([])
  const [presets, setPresets] = useState<VoiceDesignPreset[]>([])
  const [storedRefs, setStoredRefs] = useState<RefAudio[]>([])

  // ---- form state ----
  const [text, setText] = useState("")
  const [language, setLanguage] = useState("English")
  const [mode, setMode] = useState<Mode>("clone")
  const [selectedRefKey, setSelectedRefKey] = useState<string | null>(null)
  const [refText, setRefText] = useState("")
  const [selectedPreset, setSelectedPreset] = useState<string>("")
  const [customInstruct, setCustomInstruct] = useState("")
  const [speed, setSpeed] = useState(1.0)

  // ---- output state ----
  const [generating, setGenerating] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [currentResult, setCurrentResult] = useState<TTSSynthesizeResult | null>(null)
  const [currentAudioUrl, setCurrentAudioUrl] = useState<string | null>(null)
  const [history, setHistory] = useState<HistoryEntry[]>([])

  // ---- load initial data ----
  useEffect(() => {
    getTTSLanguages().then(setLanguages).catch(() => {})
    getVoiceDesignPresets().then(setPresets).catch(() => {})
    listRefAudios().then(setStoredRefs).catch(() => {})
    setHistory(loadHistory())
  }, [])

  // ---- derive instruct from preset or custom ----
  const effectiveInstruct =
    mode === "design"
      ? customInstruct ||
        presets.find((p) => p.id === selectedPreset)?.instruct ||
        ""
      : undefined

  // ---- handlers ----
  const handleUploadRef = useCallback(async (file: File) => {
    const uploaded = await uploadRefAudio(file)
    setStoredRefs((prev) => [
      { key: uploaded.key, filename: uploaded.filename, size_bytes: uploaded.size_bytes, last_modified: null },
      ...prev,
    ])
    setSelectedRefKey(uploaded.key)
  }, [])

  const handleDeleteRef = useCallback(async (key: string) => {
    await deleteRefAudio(key)
    setStoredRefs((prev) => prev.filter((r) => r.key !== key))
    if (selectedRefKey === key) setSelectedRefKey(null)
  }, [selectedRefKey])

  const handleGenerate = useCallback(async () => {
    if (!text.trim()) return
    setError(null)
    setGenerating(true)

    try {
      const result = await synthesizeTTS({
        text: text.trim(),
        language,
        refAudioKey: mode === "clone" ? selectedRefKey ?? undefined : undefined,
        refText: mode === "clone" && refText ? refText : undefined,
        instruct: effectiveInstruct || undefined,
        speed: speed !== 1.0 ? speed : undefined,
      })

      const url = URL.createObjectURL(result.audioBlob)
      setCurrentResult(result)
      setCurrentAudioUrl(url)

      // Prepend to history
      const entry: HistoryEntry = {
        id: Date.now().toString(),
        text: text.trim().slice(0, 120),
        language,
        mode,
        audioUrl: url,
        audioDuration: result.audioDuration,
        synthTime: result.synthTime,
        rtf: result.rtf,
        timestamp: Date.now(),
      }
      setHistory((prev) => {
        const next = [entry, ...prev].slice(0, MAX_HISTORY)
        saveHistory(next)
        return next
      })
    } catch (err) {
      setError(err instanceof Error ? err.message : "Synthesis failed")
    } finally {
      setGenerating(false)
    }
  }, [text, language, mode, selectedRefKey, refText, effectiveInstruct, speed])

  // ---- preset change ----
  function handlePresetChange(presetId: string) {
    setSelectedPreset(presetId)
    const p = presets.find((x) => x.id === presetId)
    if (p) setCustomInstruct(p.instruct)
  }

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex h-full flex-col overflow-hidden">
      {/* Page header */}
      <div className="flex items-center justify-between border-b border-slate-200 bg-white px-6 py-4">
        <div>
          <h1 className="text-lg font-semibold text-slate-900">TTS Studio</h1>
          <p className="text-sm text-slate-500">
            Synthesize speech with OmniVoice — clone a voice or design one from scratch
          </p>
        </div>
      </div>

      {/* Body */}
      <div className="flex flex-1 overflow-hidden">
        {/* ---- LEFT PANEL ---- */}
        <div className="flex w-[420px] shrink-0 flex-col gap-5 overflow-y-auto border-r border-slate-200 bg-[#F9F8F5] p-5">

          {/* Text */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-slate-700">Text to synthesize</label>
            <Textarea
              placeholder="Enter text here…"
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={5}
              className="resize-none bg-white text-sm"
            />
            <p className="text-right text-xs text-slate-400">{text.length} chars</p>
          </div>

          {/* Language */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-slate-700">Language</label>
            <Select value={language} onValueChange={setLanguage}>
              <SelectTrigger className="bg-white text-sm">
                <SelectValue placeholder="Select language" />
              </SelectTrigger>
              <SelectContent className="max-h-64">
                {languages.length === 0 ? (
                  // Fallback list while loading
                  ["English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam",
                   "Bengali", "Chinese", "Japanese", "French", "German", "Spanish"].map((l) => (
                    <SelectItem key={l} value={l}>{l}</SelectItem>
                  ))
                ) : (
                  languages.map((l) => (
                    <SelectItem key={l.code + l.name} value={l.name}>{l.name}</SelectItem>
                  ))
                )}
              </SelectContent>
            </Select>
          </div>

          {/* Mode toggle */}
          <div className="space-y-1.5">
            <label className="text-sm font-medium text-slate-700">Mode</label>
            <div className="flex rounded-md border border-slate-200 bg-white p-0.5">
              {(["clone", "design"] as Mode[]).map((m) => (
                <button
                  key={m}
                  onClick={() => setMode(m)}
                  className={`flex-1 rounded py-1.5 text-sm font-medium transition-colors ${
                    mode === m
                      ? "bg-slate-900 text-white"
                      : "text-slate-600 hover:text-slate-900"
                  }`}
                >
                  {m === "clone" ? "Voice Clone" : "Voice Design"}
                </button>
              ))}
            </div>
          </div>

          {/* Voice Clone fields */}
          {mode === "clone" && (
            <>
              <RefAudioPicker
                stored={storedRefs}
                selected={selectedRefKey}
                onSelect={setSelectedRefKey}
                onUpload={handleUploadRef}
                onDelete={handleDeleteRef}
              />
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-slate-700">
                  Reference Text{" "}
                  <span className="font-normal text-slate-400">(optional)</span>
                </label>
                <Textarea
                  placeholder="Transcript of the reference audio (improves quality)"
                  value={refText}
                  onChange={(e) => setRefText(e.target.value)}
                  rows={2}
                  className="resize-none bg-white text-sm"
                />
              </div>
            </>
          )}

          {/* Voice Design fields */}
          {mode === "design" && (
            <div className="space-y-3">
              <div className="space-y-1.5">
                <label className="text-sm font-medium text-slate-700">Voice Preset</label>
                <Select value={selectedPreset} onValueChange={handlePresetChange}>
                  <SelectTrigger className="bg-white text-sm">
                    <SelectValue placeholder="Choose a preset…" />
                  </SelectTrigger>
                  <SelectContent>
                    {presets.length === 0 ? (
                      [
                        { id: "male_neutral", label: "Male · Neutral", instruct: "male, young adult, moderate pitch" },
                        { id: "female_neutral", label: "Female · Neutral", instruct: "female, young adult, moderate pitch" },
                        { id: "male_deep", label: "Male · Deep", instruct: "male, middle-aged, low pitch" },
                        { id: "female_soft", label: "Female · Soft", instruct: "female, young adult, high pitch" },
                        { id: "whisper", label: "Whisper", instruct: "whisper" },
                        { id: "child", label: "Child", instruct: "child, high pitch" },
                        { id: "male_indian", label: "Male · Indian Accent", instruct: "male, young adult, indian accent" },
                        { id: "female_british", label: "Female · British", instruct: "female, young adult, british accent" },
                      ].map((p) => (
                        <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>
                      ))
                    ) : (
                      presets.map((p) => (
                        <SelectItem key={p.id} value={p.id}>{p.label}</SelectItem>
                      ))
                    )}
                  </SelectContent>
                </Select>
              </div>

              <div className="space-y-1.5">
                <label className="text-sm font-medium text-slate-700">
                  Custom Instruct Tags
                  <span className="ml-1 font-normal text-slate-400 text-xs">
                    (overrides preset)
                  </span>
                </label>
                <Textarea
                  placeholder="e.g. male, young adult, indian accent"
                  value={customInstruct}
                  onChange={(e) => setCustomInstruct(e.target.value)}
                  rows={2}
                  className="resize-none bg-white font-mono text-sm"
                />
                <p className="text-xs text-slate-400">
                  Comma-separated tags: gender, age, pitch, accent, whisper
                </p>
              </div>
            </div>
          )}

          {/* Speed */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-sm font-medium text-slate-700">Speed</label>
              <span className="rounded bg-slate-100 px-1.5 py-0.5 text-xs font-mono text-slate-600">
                {speed.toFixed(1)}×
              </span>
            </div>
            <Slider
              min={0.5}
              max={2.0}
              step={0.1}
              value={[speed]}
              onValueChange={([v]) => setSpeed(v)}
              className="w-full"
            />
            <div className="flex justify-between text-xs text-slate-400">
              <span>0.5×</span>
              <span>1.0×</span>
              <span>2.0×</span>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="flex items-start gap-2 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
              <X className="mt-0.5 size-4 shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {/* Generate button */}
          <Button
            onClick={handleGenerate}
            disabled={generating || !text.trim()}
            className="w-full gap-2 bg-slate-900 text-white hover:bg-slate-700"
            size="lg"
          >
            {generating ? (
              <>
                <Loader2 className="size-4 animate-spin" />
                Generating…
              </>
            ) : (
              <>
                <Wand2 className="size-4" />
                Generate
              </>
            )}
          </Button>
        </div>

        {/* ---- RIGHT PANEL ---- */}
        <div className="flex flex-1 flex-col overflow-y-auto p-6 gap-6">

          {/* Current result */}
          {currentAudioUrl && currentResult ? (
            <div className="rounded-xl border border-slate-200 bg-white p-5 shadow-sm">
              <div className="mb-3 flex items-center justify-between">
                <h2 className="font-semibold text-slate-900">Latest Result</h2>
                <div className="flex gap-2">
                  {currentResult.audioDuration !== null && (
                    <Badge variant="secondary" className="gap-1 text-xs">
                      <Volume2 className="size-3" />
                      {formatDuration(currentResult.audioDuration)}
                    </Badge>
                  )}
                  {currentResult.rtf !== null && (
                    <Badge variant="outline" className="text-xs">
                      RTF {currentResult.rtf.toFixed(3)}
                    </Badge>
                  )}
                  {currentResult.synthTime !== null && (
                    <Badge variant="outline" className="gap-1 text-xs">
                      <Clock className="size-3" />
                      {currentResult.synthTime.toFixed(2)}s
                    </Badge>
                  )}
                </div>
              </div>
              <AudioPlayer url={currentAudioUrl} />
            </div>
          ) : (
            <div className="flex flex-1 flex-col items-center justify-center rounded-xl border border-dashed border-slate-200 bg-slate-50 py-16 text-center">
              <Mic className="mb-3 size-8 text-slate-300" />
              <p className="text-sm font-medium text-slate-500">No audio generated yet</p>
              <p className="mt-1 text-xs text-slate-400">
                Enter text on the left and click Generate
              </p>
            </div>
          )}

          {/* History */}
          {history.length > 0 && (
            <div className="space-y-3">
              <h2 className="font-semibold text-slate-900">History</h2>
              <div className="space-y-2">
                {history.map((entry) => (
                  <div
                    key={entry.id}
                    className="rounded-lg border border-slate-200 bg-white p-4"
                  >
                    <div className="mb-2 flex items-start justify-between gap-3">
                      <p className="line-clamp-2 text-sm text-slate-700">{entry.text}</p>
                      <div className="flex shrink-0 gap-1">
                        <Badge variant="outline" className="text-xs">
                          {entry.language}
                        </Badge>
                        <Badge
                          variant={entry.mode === "clone" ? "secondary" : "outline"}
                          className="text-xs capitalize"
                        >
                          {entry.mode}
                        </Badge>
                      </div>
                    </div>
                    <div className="flex items-center gap-3">
                      <AudioPlayer url={entry.audioUrl} />
                      <div className="ml-auto flex gap-2 text-xs text-slate-400">
                        {entry.audioDuration !== null && (
                          <span>{formatDuration(entry.audioDuration)}</span>
                        )}
                        {entry.rtf !== null && (
                          <span>RTF {entry.rtf.toFixed(3)}</span>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
