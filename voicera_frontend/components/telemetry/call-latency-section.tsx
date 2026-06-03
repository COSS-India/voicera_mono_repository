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
} from "lucide-react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import {
  CallLatencyDetailSkeleton,
  CallLatencyListSkeleton,
} from "@/components/telemetry/telemetry-skeletons"
import { cn } from "@/lib/utils"
import { getMeeting, type CallLatencyMetrics, type Meeting } from "@/lib/api"
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

function CallMetricsDetail({
  meeting,
  metrics,
  loading,
}: {
  meeting: Meeting
  metrics: CallLatencyMetrics | undefined
  loading: boolean
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
    { label: "Avg TTS first chunk", value: summary?.avg_tts_first_chunk_ms, icon: Volume2 },
  ]

  if (loading) {
    return <CallLatencyDetailSkeleton />
  }

  if (!metrics?.turns?.length) {
    return (
      <div className="flex flex-1 items-center justify-center py-16 text-sm text-slate-500 px-6 text-center">
        No latency metrics stored for this call.
      </div>
    )
  }

  return (
    <div className="flex-1 overflow-y-auto p-5 lg:p-6 space-y-5 bg-slate-50/50">
      <div>
        <h2 className="text-base font-semibold text-slate-900 truncate">
          {meeting.agent_type || "Call"}
        </h2>
        <p className="text-xs text-slate-500 font-mono mt-1 break-all">{meeting.meeting_id}</p>
        <div className="flex flex-wrap gap-3 mt-2 text-xs text-slate-600">
          <span className="flex items-center gap-1">
            <Clock className="h-3.5 w-3.5" />
            {formatDuration(meeting.duration)}
          </span>
          {summary?.turn_count != null && (
            <span>
              {summary.turn_count} turn{summary.turn_count === 1 ? "" : "s"}
            </span>
          )}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {summaryItems.map(({ label, value, icon: Icon }) => (
          <div
            key={label}
            className="bg-white rounded-xl border border-slate-200 p-4 shadow-sm"
          >
            <p className="text-xs font-medium text-slate-500 flex items-center gap-1.5 mb-2">
              <Icon className="h-3.5 w-3.5" />
              {label}
            </p>
            <p className="text-2xl font-semibold text-slate-900 tabular-nums">
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
                Search by assistant, call ID, or phone number
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

          <div className="grid grid-cols-[88px_minmax(0,1fr)_minmax(0,112px)_100px_64px_48px] gap-2 px-5 py-3 bg-slate-50 border-b border-slate-200 text-xs font-medium text-slate-600">
            <div>Call Type</div>
            <div>Assistant</div>
            <div>Call ID</div>
            <div>Date & Time</div>
            <div className="text-right">Duration</div>
            <div className="text-right">Turns</div>
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
                      "w-full grid grid-cols-[88px_minmax(0,1fr)_minmax(0,112px)_100px_64px_48px] gap-2 px-5 py-4 border-b border-slate-100 items-center text-left transition-colors hover:bg-slate-50",
                      isSelected && "bg-slate-100 hover:bg-slate-100"
                    )}
                  >
                    <div>
                      <CallTypeBadge inbound={meeting.inbound} />
                    </div>
                    <div className="min-w-0">
                      <p className="text-sm font-medium text-slate-900 truncate">
                        {meeting.agent_type || "—"}
                      </p>
                      {avgLlm != null && (
                        <p className="text-[10px] text-slate-400 mt-0.5 tabular-nums">
                          avg LLM {formatMs(avgLlm)}
                        </p>
                      )}
                    </div>
                    <div
                      className="min-w-0 text-[11px] text-slate-600 font-mono truncate"
                      title={meeting.meeting_id}
                    >
                      {meeting.meeting_id || "—"}
                    </div>
                    <div className="text-xs text-slate-600 truncate">
                      {formatCallDate(meeting)}
                    </div>
                    <div className="text-xs text-slate-600 text-right tabular-nums">
                      {formatDuration(meeting.duration)}
                    </div>
                    <div className="text-xs font-medium text-slate-900 text-right tabular-nums">
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
          />
        ) : null}
      </div>
    </div>
  )
}
