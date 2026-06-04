"use client"

import { useCallback, useEffect, useMemo, useState } from "react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import {
  Activity,
  Clock,
  Mic,
  MessageSquare,
  Search,
  Volume2,
  X,
  FileAudio,
  Download,
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { MeetingDetailSheet } from "@/components/history/meeting-detail-sheet"
import {
  CallLatencyDetailSkeleton,
  CallLatencyListSkeleton,
} from "@/components/telemetry/telemetry-skeletons"
import { cn } from "@/lib/utils"
import {
  getMeeting,
  getMeetingDetails,
  type CallLatencyMetrics,
  type Meeting,
  type MeetingDetails,
} from "@/lib/api"
import { normalizeLatencyMetrics } from "@/lib/dedupe-latency-turns"
import { buildMeetingsParams } from "@/lib/meetings-params"
import { useMeetingsQuery } from "@/lib/queries/meetings"
import { format } from "date-fns"
import { toZonedTime } from "date-fns-tz"

const IST = "Asia/Kolkata"

function formatMs(value: number | null | undefined): string {
  if (value == null || Number.isNaN(value)) return "—"
  return `${Math.round(value)} ms`
}

function formatCallDate(meeting: Meeting): string {
  const raw = meeting.start_time_utc || meeting.created_at
  if (!raw) return "—"
  try {
    return format(toZonedTime(new Date(raw), IST), "MMM d, yyyy · HH:mm")
  } catch {
    return raw
  }
}

function formatDuration(seconds: number | undefined): string {
  if (seconds == null) return "—"
  const m = Math.floor(seconds / 60)
  const s = Math.round(seconds % 60)
  return m > 0 ? `${m}m ${s}s` : `${s}s`
}

function escapeCsvCell(value: string | number | null | undefined): string {
  if (value == null) return ""
  const s = String(value)
  if (s.includes(",") || s.includes('"') || s.includes("\n")) {
    return `"${s.replace(/"/g, '""')}"`
  }
  return s
}

function downloadCallLatencyCsv(meeting: Meeting, metrics: CallLatencyMetrics) {
  const headers = [
    "meeting_id",
    "agent_type",
    "call_type",
    "call_date",
    "duration_sec",
    "turn_index",
    "user_text",
    "stt_ms",
    "llm_ttfb_ms",
    "tts_first_chunk_ms",
    "avg_stt_ms",
    "avg_llm_ttfb_ms",
    "avg_tts_first_chunk_ms",
    "turn_count",
  ]
  const summary = metrics.summary
  const callDate = formatCallDate(meeting)
  const callType = meeting.inbound === true ? "Inbound" : "Outbound"
  const dataRows = metrics.turns.map((turn) =>
    [
      meeting.meeting_id,
      meeting.agent_type ?? "",
      callType,
      callDate,
      meeting.duration ?? "",
      turn.turn_index,
      turn.user_text_preview ?? "",
      turn.stt_ms ?? "",
      turn.llm_ttfb_ms ?? "",
      turn.tts_first_chunk_ms ?? "",
      summary?.avg_stt_ms ?? "",
      summary?.avg_llm_ttfb_ms ?? "",
      summary?.avg_tts_first_chunk_ms ?? "",
      summary?.turn_count ?? metrics.turns.length,
    ]
      .map(escapeCsvCell)
      .join(",")
  )

  const csvContent = [headers.join(","), ...dataRows].join("\n")
  const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
  const link = document.createElement("a")
  const url = URL.createObjectURL(blob)
  link.setAttribute("href", url)
  link.setAttribute(
    "download",
    `call_latency_${meeting.meeting_id}_${format(new Date(), "yyyy-MM-dd")}.csv`
  )
  link.style.visibility = "hidden"
  document.body.appendChild(link)
  link.click()
  document.body.removeChild(link)
  URL.revokeObjectURL(url)
}

const LIST_ROW_LAYOUT =
  "flex items-center gap-4 px-5"
const LIST_COL_CALL_TYPE = "w-[88px] shrink-0"
const LIST_COL_ASSISTANT = "min-w-0 flex-1 basis-0"
const LIST_COL_DATETIME = "min-w-0 flex-1 basis-0"
const LIST_COL_DURATION = "w-14 shrink-0 text-right tabular-nums"
const LIST_COL_TURNS = "w-11 shrink-0 text-right tabular-nums"

function CallTypeBadge({ inbound }: { inbound?: boolean }) {
  const isInbound = inbound === true
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold min-w-[84px]",
        isInbound ? "bg-emerald-50 text-emerald-700" : "bg-blue-50 text-blue-700"
      )}
    >
      {isInbound ? "Inbound" : "Outbound"}
    </span>
  )
}

/** Figma-style motion: ~400ms spatial ease-out; label fades in slightly after width starts */
const COMPACT_ACTION_BUTTON_MOTION =
  "transition-[padding,background-color,border-color] duration-[400ms] ease-[cubic-bezier(0.16,1,0.3,1)] group-hover:px-3"

const BUTTON_HOVER_LABEL_CLASS = cn(
  "inline-block max-w-0 overflow-hidden opacity-0 whitespace-nowrap",
  "transition-[max-width,margin-left] duration-[400ms] ease-[cubic-bezier(0.16,1,0.3,1)]",
  "transition-opacity duration-[300ms] ease-out delay-0",
  "group-hover:max-w-[11rem] group-hover:opacity-100 group-hover:ml-1.5 group-hover:delay-100",
  "motion-reduce:transition-none motion-reduce:max-w-[11rem] motion-reduce:opacity-100 motion-reduce:ml-1.5"
)

function ViewCallButton({
  onClick,
  className,
  compact = false,
}: {
  onClick: () => void
  className?: string
  /** Icon only until hover; label expands on hover */
  compact?: boolean
}) {
  return (
    <Button
      type="button"
      size="sm"
      className={cn(
        "bg-slate-900 hover:bg-slate-800 text-white shadow-sm font-semibold shrink-0",
        compact ? cn("group gap-0 px-2.5", COMPACT_ACTION_BUTTON_MOTION) : "gap-1.5",
        className
      )}
      onClick={onClick}
      aria-label="View transcript and recording"
    >
      <FileAudio className="h-4 w-4 shrink-0" />
      {compact ? (
        <span className={BUTTON_HOVER_LABEL_CLASS}>View transcript</span>
      ) : (
        "View transcript"
      )}
    </Button>
  )
}

function DownloadLatencyCsvButton({
  meeting,
  metrics,
  className,
}: {
  meeting: Meeting
  metrics: CallLatencyMetrics
  className?: string
}) {
  return (
    <Button
      type="button"
      variant="outline"
      size="sm"
      className={cn("group gap-0 px-2.5 shrink-0", COMPACT_ACTION_BUTTON_MOTION, className)}
      onClick={() => downloadCallLatencyCsv(meeting, metrics)}
      aria-label="Download call latency data as CSV"
    >
      <Download className="h-4 w-4 shrink-0" />
      <span className={BUTTON_HOVER_LABEL_CLASS}>Download CSV</span>
    </Button>
  )
}

function CallMetricsDetail({
  meeting,
  metrics,
  loading,
  onOpenCallDetails,
}: {
  meeting: Meeting
  metrics: CallLatencyMetrics | undefined
  loading: boolean
  onOpenCallDetails: () => void
}) {
  const chartData = useMemo(() => {
    if (!metrics?.turns?.length) return []
    return metrics.turns.map((t) => ({
      name: `T${t.turn_index}`,
      stt: t.stt_ms ?? 0,
      llm: t.llm_ttfb_ms ?? 0,
      tts: t.tts_first_chunk_ms ?? 0,
    }))
  }, [metrics])

  const summary = metrics?.summary
  const summaryItems = [
    { label: "Avg STT", value: summary?.avg_stt_ms, icon: Mic },
    { label: "Avg LLM TTFB", value: summary?.avg_llm_ttfb_ms, icon: MessageSquare },
    {
      label: "Avg TTS 1st chunk",
      labelTitle: "Avg TTS first chunk",
      value: summary?.avg_tts_first_chunk_ms,
      icon: Volume2,
    },
  ]

  if (loading) {
    return <CallLatencyDetailSkeleton />
  }

  if (!metrics?.turns?.length) {
    return (
      <div className="flex flex-1 flex-col items-center justify-center gap-5 py-16 px-6 text-center">
        <p className="text-sm text-slate-500">No latency metrics stored for this call.</p>
        <ViewCallButton
          onClick={onOpenCallDetails}
          className="h-10 px-5 text-sm"
        />
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-5 lg:p-6 space-y-5 bg-slate-50/50">
      <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-semibold text-slate-900 truncate">
            {meeting.agent_type || "Call"}
          </h2>
        </div>
        <div className="flex flex-wrap items-center gap-2 shrink-0">
          <ViewCallButton onClick={onOpenCallDetails} compact />
          <DownloadLatencyCsvButton meeting={meeting} metrics={metrics} />
        </div>
      </div>
     

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {summaryItems.map(({ label, labelTitle, value, icon: Icon }) => (
          <div
            key={label}
            className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm flex flex-col min-h-[5.5rem] min-w-0"
          >
            <p
              className="text-xs font-medium text-slate-500 flex items-center gap-1.5 mb-2 min-h-5 whitespace-nowrap"
              title={labelTitle}
            >
              <Icon className="h-3.5 w-3.5 shrink-0" aria-hidden />
              <span className="truncate">{label}</span>
            </p>
            <p className="text-2xl font-semibold text-slate-900 tabular-nums mt-auto">
              {formatMs(value)}
            </p>
          </div>
        ))}
      </div>

      <Card className="border-slate-200 shadow-sm">
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Latency by turn</CardTitle>
          <CardDescription>STT, LLM, and TTS per user utterance</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="h-52">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 8, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                <XAxis dataKey="name" tick={{ fontSize: 11 }} />
                <YAxis tick={{ fontSize: 11 }} />
                <Tooltip formatter={(v) => `${Math.round(Number(v ?? 0))} ms`} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="stt" name="STT" fill="#64748b" radius={[2, 2, 0, 0]} />
                <Bar dataKey="llm" name="LLM TTFB" fill="#111827" radius={[2, 2, 0, 0]} />
                <Bar dataKey="tts" name="TTS 1st chunk" fill="#94a3b8" radius={[2, 2, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </CardContent>
      </Card>

      <div className="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm">
        <div className="px-4 py-3 border-b border-slate-200 bg-slate-50 flex items-center gap-2">
          <Activity className="h-4 w-4 text-slate-500" />
          <span className="text-sm font-medium text-slate-900">Turn-by-turn breakdown</span>
        </div>
        <div className="overflow-auto max-h-[min(400px,50vh)]">
          <table className="w-full text-sm">
            <thead className="bg-slate-50 sticky top-0 text-xs font-medium text-slate-600">
              <tr className="border-b border-slate-200">
                <th className="px-4 py-3 text-left">Turn</th>
                <th className="px-4 py-3 text-left">User</th>
                <th className="px-4 py-3 text-right">STT</th>
                <th className="px-4 py-3 text-right">LLM TTFB</th>
                <th className="px-4 py-3 text-right">TTS 1st chunk</th>
              </tr>
            </thead>
            <tbody>
              {metrics.turns.map((turn) => (
                <tr
                  key={turn.turn_index}
                  className="border-b border-slate-100 last:border-0 hover:bg-slate-50"
                >
                  <td className="px-4 py-3 font-medium tabular-nums text-slate-900">
                    {turn.turn_index}
                  </td>
                  <td
                    className="px-4 py-3 text-slate-600 max-w-[220px] truncate"
                    title={turn.user_text_preview ?? undefined}
                  >
                    {turn.user_text_preview || "—"}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">{formatMs(turn.stt_ms)}</td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {formatMs(turn.llm_ttfb_ms)}
                  </td>
                  <td className="px-4 py-3 text-right tabular-nums">
                    {formatMs(turn.tts_first_chunk_ms)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="px-4 py-3 text-[10px] text-slate-400 border-t border-slate-100">
          Pipeline-internal timings only (not telephony round-trip). Recorded after each call
          ends.
        </p>
      </div>
    </div>
  )
}

function matchesSearch(meeting: Meeting, query: string): boolean {
  const q = query.trim().toLowerCase()
  if (!q) return true
  const parts = [
    meeting.meeting_id,
    meeting.agent_type,
    meeting.from_number,
    meeting.to_number,
    meeting.inbound ? "inbound" : "outbound",
  ]
  return parts.some((p) => p && String(p).toLowerCase().includes(q))
}

const CALLS_WITH_METRICS_LIMIT = 10000

export function CallLatencySection() {
  const [searchQuery, setSearchQuery] = useState("")
  const [selectedId, setSelectedId] = useState<string | null>(null)
  const [detailLoading, setDetailLoading] = useState(false)
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null)
  const [sheetOpen, setSheetOpen] = useState(false)
  const [sheetMeeting, setSheetMeeting] = useState<Meeting | null>(null)
  const [meetingDetails, setMeetingDetails] = useState<MeetingDetails | null>(null)
  const [sheetLoading, setSheetLoading] = useState(false)

  const meetingsParams = useMemo(
    () =>
      buildMeetingsParams(
        1,
        [],
        { from: undefined, to: undefined },
        "latest",
        null,
        { limit: CALLS_WITH_METRICS_LIMIT, forExport: true }
      ),
    []
  )

  const { data: meetingsPage, isPending, isError, isFetching } =
    useMeetingsQuery(meetingsParams)

  const callsWithMetrics = useMemo(() => {
    const withMetrics = (meetingsPage?.items ?? []).filter(
      (m) => (m.latency_metrics?.turns?.length ?? 0) > 0
    )
    return withMetrics.filter((m) => matchesSearch(m, searchQuery))
  }, [meetingsPage?.items, searchQuery])

  const loadCallDetail = useCallback(async (meeting: Meeting) => {
    setSelectedId(meeting.meeting_id)
    setSelectedMeeting(meeting)
    setDetailLoading(true)
    try {
      const full = await getMeeting(meeting.meeting_id)
      setSelectedMeeting(full)
    } catch {
      setSelectedMeeting(meeting)
    } finally {
      setDetailLoading(false)
    }
  }, [])

  const handleCallClick = (meeting: Meeting) => {
    void loadCallDetail(meeting)
  }

  const openCallSheet = useCallback(async (meeting: Meeting) => {
    setSheetMeeting(meeting)
    setSheetOpen(true)
    setSheetLoading(true)
    setMeetingDetails(null)
    try {
      const details = await getMeetingDetails(meeting.meeting_id)
      setMeetingDetails(details)
    } catch {
      setMeetingDetails(null)
    } finally {
      setSheetLoading(false)
    }
  }, [])

  const isLoadingList = isPending || isFetching

  useEffect(() => {
    if (isLoadingList || isError) return

    if (callsWithMetrics.length === 0) {
      setSelectedId(null)
      setSelectedMeeting(null)
      return
    }

    const selectionStillVisible =
      selectedId != null &&
      callsWithMetrics.some((m) => m.meeting_id === selectedId)

    if (!selectionStillVisible) {
      void loadCallDetail(callsWithMetrics[0])
    }
  }, [
    isLoadingList,
    isError,
    callsWithMetrics,
    selectedId,
    loadCallDetail,
  ])

  const latencyMetrics = useMemo(
    () => normalizeLatencyMetrics(selectedMeeting?.latency_metrics),
    [selectedMeeting?.latency_metrics]
  )

  return (
    <div className="flex flex-col lg:flex-row gap-4 min-h-[calc(100vh-10rem)]">
      {/* Call list — History table pattern */}
      <div className="lg:w-[min(100%,600px)] lg:shrink-0 flex flex-col gap-3">
        <div className="bg-white rounded-xl border border-slate-200 overflow-hidden flex flex-col flex-1 min-h-[320px]">
          <div className="px-5 py-3 border-b border-slate-200 bg-slate-50 space-y-3">
            <div>
              <h2 className="text-sm font-semibold text-slate-900">Calls with latency data</h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Select a row for latency metrics; open View transcript in the detail panel
              </p>
            </div>
            <div className="relative">
              <Search
                className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400 pointer-events-none"
                aria-hidden
              />
              <Input
                type="search"
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                placeholder="Search calls…"
                className="h-9 pl-9 pr-9 bg-white border-slate-200"
                aria-label="Search calls with latency data"
              />
              {searchQuery.length > 0 && (
                <button
                  type="button"
                  onClick={() => setSearchQuery("")}
                  className="absolute right-2 top-1/2 -translate-y-1/2 p-1 rounded-md text-slate-400 hover:text-slate-600 hover:bg-slate-100"
                  aria-label="Clear search"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              )}
            </div>
            {searchQuery.trim() && !isLoadingList && (
              <p className="text-xs text-slate-500">
                {callsWithMetrics.length === 0
                  ? "No matching calls"
                  : `${callsWithMetrics.length} match${callsWithMetrics.length === 1 ? "" : "es"}`}
              </p>
            )}
          </div>

          <div
            className={cn(
              LIST_ROW_LAYOUT,
              "py-3 bg-slate-50 border-b border-slate-200 text-xs font-medium text-slate-600"
            )}
          >
            <div className={LIST_COL_CALL_TYPE}>Call Type</div>
            <div className={LIST_COL_ASSISTANT}>Assistant</div>
            <div className={LIST_COL_DATETIME}>Date & Time</div>
            <div className={LIST_COL_DURATION}>Duration</div>
            <div className={LIST_COL_TURNS}>Turns</div>
          </div>

          <div className="flex-1 overflow-y-auto">
            {isLoadingList && <CallLatencyListSkeleton />}
            {isError && (
              <div className="px-5 py-12 text-center text-sm text-red-600">
                Failed to load calls. Please try again.
              </div>
            )}
            {!isLoadingList && !isError && callsWithMetrics.length === 0 && (
              <div className="px-5 py-12 text-center text-sm text-slate-500">
                {searchQuery.trim()
                  ? "No calls match your search. Try a different term or clear the search."
                  : "No calls with latency metrics yet. Place a new call to collect data."}
              </div>
            )}
            {!isLoadingList &&
              !isError &&
              callsWithMetrics.map((meeting) => {
                const isSelected = selectedId === meeting.meeting_id
                const turnCount = meeting.latency_metrics?.turns?.length ?? 0
                const avgLlm = meeting.latency_metrics?.summary?.avg_llm_ttfb_ms
                return (
                  <button
                    key={meeting.meeting_id}
                    type="button"
                    onClick={() => handleCallClick(meeting)}
                    className={cn(
                      "w-full",
                      LIST_ROW_LAYOUT,
                      "py-4 border-b border-slate-100 text-left transition-colors hover:bg-slate-50",
                      isSelected && "bg-slate-100 hover:bg-slate-100"
                    )}
                  >
                    <div className={LIST_COL_CALL_TYPE}>
                      <CallTypeBadge inbound={meeting.inbound} />
                    </div>
                    <div className={LIST_COL_ASSISTANT}>
                      <p className="text-sm font-medium text-slate-900 truncate">
                        {meeting.agent_type || "—"}
                      </p>
                      {avgLlm != null && (
                        <p className="text-[10px] text-slate-400 mt-0.5 tabular-nums truncate">
                          avg LLM {formatMs(avgLlm)}
                        </p>
                      )}
                    </div>
                    <div className={cn(LIST_COL_DATETIME, "text-xs text-slate-600 truncate")}>
                      {formatCallDate(meeting)}
                    </div>
                    <div className={cn(LIST_COL_DURATION, "text-xs text-slate-600")}>
                      {formatDuration(meeting.duration)}
                    </div>
                    <div className={cn(LIST_COL_TURNS, "text-xs font-medium text-slate-900")}>
                      {turnCount}
                    </div>
                  </button>
                )
              })}
          </div>
        </div>
      </div>

      {/* Detail panel */}
      <div className="flex-1 min-w-0 bg-white rounded-xl border border-slate-200 overflow-hidden flex flex-col min-h-[320px]">
        {!selectedId ? (
          <div className="flex flex-1 items-center justify-center p-8 text-center text-sm text-slate-500">
            Select a call from the list to view STT, LLM, and TTS latency metrics.
          </div>
        ) : detailLoading ? (
          <CallLatencyDetailSkeleton />
        ) : selectedMeeting ? (
          <CallMetricsDetail
            meeting={selectedMeeting}
            metrics={latencyMetrics}
            loading={false}
            onOpenCallDetails={() => void openCallSheet(selectedMeeting)}
          />
        ) : null}
      </div>

      <MeetingDetailSheet
        open={sheetOpen}
        onOpenChange={setSheetOpen}
        meeting={sheetMeeting}
        meetingDetails={meetingDetails}
        isLoading={sheetLoading}
      />
    </div>
  )
}
