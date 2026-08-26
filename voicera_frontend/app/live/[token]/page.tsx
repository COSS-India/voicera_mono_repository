"use client"

import { use, useCallback, useEffect, useRef, useState } from "react"
import { getTranslationListenWebSocketUrl } from "@/lib/johnaic-config"
import type { PublicAgent } from "@/lib/api"
import { LiveStatusIndicator, type LiveStatusState } from "@/components/live-status-indicator"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"

const DEFAULT_SAMPLE_RATE = 16000
// Schedule audio a little ahead of the clock so network jitter between 20 ms
// frames doesn't cause audible gaps. Kept small to minimise added latency; raise
// it if listeners on jittery links hear dropouts.
const JITTER_BUFFER_SECS = 0.15
// Translated speech is routinely longer than the source it came from, so a
// listener's backlog grows for as long as the presenter keeps talking. Two
// mechanisms drain it, in order of how audible they are:
//
// 1. Above SOFT_LEAD_SECS, play slightly fast. At 5 % the pitch shift is barely
//    perceptible on speech and it costs nothing — no audio is discarded.
// 2. Above MAX_LEAD_SECS, skip forward. Speeding up alone cannot beat an accrual
//    rate of tens of percent, so a real skip is eventually unavoidable; what
//    matters is *where* it lands. We cut at a sentence boundary when one is
//    close, else at the end of the 20 ms frame that is sounding right now, and we
//    never stop a buffer that is already playing — stopping mid-buffer is exactly
//    the mid-word "chop" this used to produce.
const SOFT_LEAD_SECS = 2.5
const SOFT_DRAIN_RATE = 1.05
const MAX_LEAD_SECS = 6
// How long a skip will wait for the next sentence end before cutting at the
// current frame instead. Beyond this the wait costs more than the cleaner cut.
const BOUNDARY_CUT_WAIT_SECS = 1.5
// Frames arrive 20 ms at a time but are scheduled in blocks of about this long.
// A block is resampled as one window, so the catch-up playback rate above costs
// two interpolated edges per block instead of two per 20 ms frame. Only applies
// once there is audio in hand to play: while the buffer is starved, every frame
// is scheduled the moment it arrives, so this never delays speech.
const COALESCE_SECS = 0.2

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i)
  return new Int16Array(bytes.buffer)
}

function int16ToFloat32(input: Int16Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) out[i] = input[i] / 0x8000
  return out
}

function getListenerStatus({
  sessionEnded,
  isConnected,
  isConnecting,
  isReconnecting,
  presenterOnline,
}: {
  sessionEnded: boolean
  isConnected: boolean
  isConnecting: boolean
  isReconnecting: boolean
  presenterOnline: boolean
}): { state: LiveStatusState; label: string } {
  if (sessionEnded) return { state: "idle", label: "Broadcast ended" }
  if (isReconnecting) return { state: "connecting", label: "Reconnecting…" }
  if (isConnecting) return { state: "connecting", label: "Connecting…" }
  if (isConnected) {
    return presenterOnline
      ? { state: "live", label: "Listening live" }
      : { state: "waiting", label: "Waiting for presenter…" }
  }
  return { state: "idle", label: "Not connected" }
}

interface ScheduledFrame {
  source: AudioBufferSourceNode
  startAt: number
  endAt: number
}

interface LiveTranslationPageProps {
  params: Promise<{ token: string }>
}

export default function LiveTranslationPage({ params }: LiveTranslationPageProps) {
  const { token } = use(params)

  const [agent, setAgent] = useState<PublicAgent | null>(null)
  const [loadError, setLoadError] = useState("")
  const [language, setLanguage] = useState("")
  const [isConnecting, setIsConnecting] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isReconnecting, setIsReconnecting] = useState(false)
  const [presenterOnline, setPresenterOnline] = useState(false)
  const [sessionEnded, setSessionEnded] = useState(false)
  const [error, setError] = useState("")
  const [notice, setNotice] = useState("")
  const [transcripts, setTranscripts] = useState<
    Array<{ id: string; source?: string; content: string }>
  >([])

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  // Every audio frame we schedule, with the window it occupies, so a skip can
  // drop the ones still queued without touching the one that is sounding and
  // without layering new audio on top of either.
  const scheduledSourcesRef = useRef<Set<ScheduledFrame>>(new Set())
  // Playout times at which a translated sentence finishes, ascending — the safe
  // places to skip forward from.
  const boundariesRef = useRef<number[]>([])
  // Frames received but not yet scheduled (see COALESCE_SECS).
  const pendingChunksRef = useRef<Float32Array[]>([])
  const pendingLenRef = useRef(0)
  const pendingRateRef = useRef(DEFAULT_SAMPLE_RATE)
  const playbackTimeRef = useRef(0)
  const transcriptViewportRef = useRef<HTMLDivElement | null>(null)
  const wantConnectedRef = useRef(false)
  // Synchronous connect guard: React state (isConnecting) lags a render behind,
  // so two fast clicks would both pass an isConnecting check and open two sockets.
  const connectingRef = useRef(false)
  const keepAliveRef = useRef<number | null>(null)
  const reconnectRef = useRef<number | null>(null)
  const reconnectAttemptsRef = useRef(0)
  const languageRef = useRef("")
  const openSocketRef = useRef<() => void>(() => {})

  useEffect(() => {
    let cancelled = false
    fetch(`/api/public/agents/${encodeURIComponent(token)}`)
      .then(async (res) => {
        if (!res.ok) throw new Error((await res.json())?.detail || "Link not found")
        return res.json()
      })
      .then((data: PublicAgent) => {
        if (cancelled) return
        setAgent(data)
        if (data.target_languages?.length) setLanguage(data.target_languages[0])
      })
      .catch((e) => {
        if (!cancelled) setLoadError(e instanceof Error ? e.message : "Link not found")
      })
    return () => {
      cancelled = true
    }
  }, [token])

  const clearTimers = useCallback(() => {
    if (keepAliveRef.current !== null) {
      window.clearInterval(keepAliveRef.current)
      keepAliveRef.current = null
    }
    if (reconnectRef.current !== null) {
      window.clearTimeout(reconnectRef.current)
      reconnectRef.current = null
    }
  }, [])

  const stopScheduledSources = useCallback(() => {
    for (const frame of scheduledSourcesRef.current) {
      frame.source.onended = null
      try {
        frame.source.stop()
      } catch {
        // already stopped/ended
      }
    }
    scheduledSourcesRef.current.clear()
    boundariesRef.current = []
    pendingChunksRef.current = []
    pendingLenRef.current = 0
  }, [])

  // Drop only what is still queued at or after `from`, leaving anything already
  // sounding to finish. Frames are contiguous, so cutting at a frame end or a
  // sentence end means nothing kept can overlap what we schedule next.
  const dropQueuedFrom = useCallback((from: number) => {
    for (const frame of Array.from(scheduledSourcesRef.current)) {
      if (frame.startAt >= from - 1e-6) {
        frame.source.onended = null
        try {
          frame.source.stop()
        } catch {
          // already stopped/ended
        }
        scheduledSourcesRef.current.delete(frame)
      }
    }
    boundariesRef.current = boundariesRef.current.filter((b) => b <= from)
  }, [])

  // Where a catch-up skip should land: the next sentence end if one is close,
  // otherwise the end of the frame currently being played.
  const chooseCutTime = useCallback((now: number) => {
    const boundary = boundariesRef.current.find((b) => b > now + 0.02)
    if (boundary !== undefined && boundary - now <= BOUNDARY_CUT_WAIT_SECS) return boundary
    let end = now
    for (const frame of scheduledSourcesRef.current) {
      if (frame.startAt <= now && frame.endAt > end) end = frame.endAt
    }
    return end
  }, [])

  const teardown = useCallback(async () => {
    // User-initiated stop (or unmount): stop wanting a connection so the socket's
    // own close handler doesn't try to reconnect us.
    wantConnectedRef.current = false
    connectingRef.current = false
    clearTimers()
    const ws = wsRef.current
    wsRef.current = null
    if (ws) {
      ws.onclose = null
      ws.close()
    }
    stopScheduledSources()
    const ctx = audioContextRef.current
    audioContextRef.current = null
    if (ctx && ctx.state !== "closed") await ctx.close()
    playbackTimeRef.current = 0
    setIsConnected(false)
    setIsConnecting(false)
    setIsReconnecting(false)
    setPresenterOnline(false)
  }, [clearTimers, stopScheduledSources])

  useEffect(() => {
    return () => {
      void teardown()
    }
  }, [teardown])

  useEffect(() => {
    const viewport = transcriptViewportRef.current
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [transcripts])

  const scheduleBuffer = useCallback((float32: Float32Array<ArrayBuffer>, sampleRate: number) => {
    const ctx = audioContextRef.current
    if (!ctx) return
    const buffer = ctx.createBuffer(1, float32.length, sampleRate)
    buffer.copyToChannel(float32, 0)
    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)
    const now = ctx.currentTime
    // Boundaries already in the past can never be a cut point again.
    if (boundariesRef.current.length) {
      boundariesRef.current = boundariesRef.current.filter((b) => b > now - 1)
    }
    // Drifted too far behind live speech → skip forward. Cut at a sentence end
    // if one is close, else at the end of the frame that is sounding now, and
    // drop everything queued past that point (otherwise the new frames would
    // play *on top* of it and the listener hears a doubled voice).
    if (playbackTimeRef.current - now > MAX_LEAD_SECS) {
      const lead = playbackTimeRef.current - now
      const cutAt = chooseCutTime(now)
      const droppedSecs = playbackTimeRef.current - cutAt
      const droppedFrames = Array.from(scheduledSourcesRef.current).filter(
        (f) => f.startAt >= cutAt - 1e-6,
      ).length
      // TEMP DIAGNOSTIC: confirm/measure the catch-up cut believed responsible
      // for "last sentence never plays" reports. Remove once confirmed.
      console.warn(
        `[translate] catch-up cut fired: lead=${lead.toFixed(2)}s now=${now.toFixed(2)} ` +
          `playbackTimeRef=${playbackTimeRef.current.toFixed(2)} cutAt=${cutAt.toFixed(2)} ` +
          `dropping ~${droppedSecs.toFixed(2)}s across ${droppedFrames} scheduled frame(s)`,
      )
      dropQueuedFrom(cutAt)
      playbackTimeRef.current = cutAt
    }
    const startAt = Math.max(now + JITTER_BUFFER_SECS, playbackTimeRef.current)
    // Still behind, but not enough to justify discarding speech: shave the lead
    // down by playing a little fast.
    const rate = startAt - now > SOFT_LEAD_SECS ? SOFT_DRAIN_RATE : 1
    source.playbackRate.value = rate
    const endAt = startAt + buffer.duration / rate
    source.start(startAt)
    const frame: ScheduledFrame = { source, startAt, endAt }
    scheduledSourcesRef.current.add(frame)
    source.onended = () => scheduledSourcesRef.current.delete(frame)
    playbackTimeRef.current = endAt
  }, [chooseCutTime, dropQueuedFrom])

  const flushPendingAudio = useCallback(() => {
    const chunks = pendingChunksRef.current
    if (!chunks.length) return
    const merged = new Float32Array(pendingLenRef.current)
    let offset = 0
    for (const chunk of chunks) {
      merged.set(chunk, offset)
      offset += chunk.length
    }
    pendingChunksRef.current = []
    pendingLenRef.current = 0
    scheduleBuffer(merged, pendingRateRef.current)
  }, [scheduleBuffer])

  const handleIncomingAudio = useCallback((payloadB64: string, sampleRate: number) => {
    const ctx = audioContextRef.current
    if (!ctx) return
    const int16 = base64ToInt16(payloadB64)
    if (!int16.length) return
    if (sampleRate !== pendingRateRef.current) {
      flushPendingAudio()
      pendingRateRef.current = sampleRate
    }
    pendingChunksRef.current.push(int16ToFloat32(int16))
    pendingLenRef.current += int16.length
    // Nothing (or nearly nothing) left to play → schedule at once; a gap now is
    // worse than the extra nodes. Otherwise let the block fill first.
    const starved =
      playbackTimeRef.current - ctx.currentTime < JITTER_BUFFER_SECS + 0.05
    if (starved || pendingLenRef.current >= COALESCE_SECS * sampleRate) {
      flushPendingAudio()
    }
  }, [flushPendingAudio])

  const scheduleReconnect = useCallback(() => {
    if (!wantConnectedRef.current) return
    const attempt = reconnectAttemptsRef.current
    reconnectAttemptsRef.current = attempt + 1
    const delay = Math.min(15000, 1000 * 2 ** attempt) // capped exponential backoff
    setIsConnected(false)
    setIsReconnecting(true)
    // The new socket re-reports presence on open; don't carry the old value over
    // or we'd flash "Listening live" before the fresh status frame arrives.
    setPresenterOnline(false)
    reconnectRef.current = window.setTimeout(() => openSocketRef.current(), delay)
  }, [])

  const openSocket = useCallback(() => {
    if (!audioContextRef.current) return
    // Defensive: never leave a previous socket open when spinning up a new one.
    // A second live socket would feed every segment into the same context twice.
    const prev = wsRef.current
    if (prev) {
      prev.onclose = null
      prev.onmessage = null
      try {
        prev.close()
      } catch {
        // already closed
      }
    }
    const ws = new WebSocket(getTranslationListenWebSocketUrl(token, languageRef.current))
    wsRef.current = ws

    ws.onopen = () => {
      reconnectAttemptsRef.current = 0
      setIsConnecting(false)
      setIsReconnecting(false)
      setIsConnected(true)
      setError("")
      // Clear transient close notices (e.g. "at capacity") now that we're back on.
      setNotice("")
      // Fresh stream: drop any audio still scheduled from a prior socket and
      // reset the playout cursor so a reconnect can't overlap the old tail.
      stopScheduledSources()
      playbackTimeRef.current = 0
      if (keepAliveRef.current !== null) window.clearInterval(keepAliveRef.current)
      // Keep the socket warm: an idle proxy (a Cloudflare quick tunnel drops
      // idle connections ~100s) would otherwise cut a waiting listener before
      // the presenter starts, or during a long pause.
      keepAliveRef.current = window.setInterval(() => {
        if (ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ event: "ping" }))
      }, 25000)
    }

    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data)
        if (msg?.event === "playAudio" && msg?.media?.payload) {
          handleIncomingAudio(
            msg.media.payload,
            Number(msg?.media?.sampleRate) || DEFAULT_SAMPLE_RATE,
          )
        } else if (msg?.event === "audio_boundary") {
          // End of one translated sentence, in playout time. Recorded so a
          // catch-up skip can land between sentences instead of mid-word. Flush
          // first so the boundary accounts for all of that sentence's audio.
          flushPendingAudio()
          const boundary = playbackTimeRef.current
          const boundaries = boundariesRef.current
          if (!boundaries.length || boundary > boundaries[boundaries.length - 1]) {
            boundaries.push(boundary)
          }
        } else if (msg?.event === "status") {
          setPresenterOnline(Boolean(msg.presenter_online))
        } else if (msg?.event === "presenter_live") {
          setPresenterOnline(true)
        } else if (msg?.event === "session_ended") {
          // Nothing more is coming, so play out whatever is still held back.
          flushPendingAudio()
          // Presenter ended on purpose — don't fight it with reconnects.
          wantConnectedRef.current = false
          setSessionEnded(true)
          setNotice("The presenter ended the broadcast.")
        } else if (msg?.event === "transcript" && msg?.content) {
          setTranscripts((prev) => {
            const next = {
              id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
              source: msg.source ? String(msg.source) : undefined,
              content: String(msg.content),
            }
            const merged = [...prev, next]
            return merged.length > 120 ? merged.slice(merged.length - 120) : merged
          })
        }
      } catch {
        // ignore malformed frames
      }
    }

    ws.onerror = () => setError("Connection error")
    ws.onclose = (ev) => {
      if (keepAliveRef.current !== null) {
        window.clearInterval(keepAliveRef.current)
        keepAliveRef.current = null
      }
      // Policy closes (unknown token / invalid language) are permanent; retrying
      // would just loop. Everything else is treated as a transient drop.
      if (ev.code === 4404 || ev.code === 4400) {
        wantConnectedRef.current = false
        setError("This live translation link is no longer available.")
      } else if (ev.code === 4503) {
        // Server misconfigured (multi-worker): retrying can't help.
        wantConnectedRef.current = false
        setError("Live translation is temporarily unavailable. Please try again later.")
      } else if (ev.code === 1013) {
        // At capacity: keep retrying (a slot may free up) but say so, instead of
        // showing a bare "reconnecting…" that looks like a fault.
        setNotice("The session is at capacity — retrying shortly…")
      }
      if (wantConnectedRef.current) {
        scheduleReconnect()
      } else {
        // Terminal close (presenter ended / bad link): let already-scheduled
        // audio finish, then release the context so a later re-Listen is fresh.
        const ctx = audioContextRef.current
        audioContextRef.current = null
        wsRef.current = null
        if (ctx && ctx.state !== "closed") {
          const remaining = Math.max(0, playbackTimeRef.current - ctx.currentTime)
          window.setTimeout(() => {
            if (ctx.state !== "closed") void ctx.close()
          }, remaining * 1000 + 200)
        }
        playbackTimeRef.current = 0
        // Release the connect re-entrancy guard: this socket is done and no
        // reconnect is pending, so a later "Listen" click must be allowed through.
        connectingRef.current = false
        setIsConnected(false)
        setIsConnecting(false)
        setIsReconnecting(false)
      }
    }
  }, [token, handleIncomingAudio, flushPendingAudio, scheduleReconnect, stopScheduledSources])

  useEffect(() => {
    openSocketRef.current = openSocket
  }, [openSocket])

  const connect = async () => {
    if (!agent || !language || isConnecting || isConnected || connectingRef.current) return
    connectingRef.current = true
    setError("")
    setNotice("")
    setSessionEnded(false)
    setPresenterOnline(false)
    setIsConnecting(true)
    setTranscripts([])
    try {
      // Ask for a context at the stream's own rate. Otherwise every 20 ms buffer
      // is resampled to the device rate in isolation, with no filter state carried
      // across buffers — a discontinuity at each frame edge, i.e. a 50 Hz buzz on
      // top of the speech. Not all browsers honour the hint, so fall back.
      let ctx: AudioContext
      try {
        ctx = new AudioContext({ sampleRate: DEFAULT_SAMPLE_RATE })
      } catch {
        ctx = new AudioContext()
      }
      audioContextRef.current = ctx
      await ctx.resume()
      languageRef.current = language
      wantConnectedRef.current = true
      reconnectAttemptsRef.current = 0
      openSocket()
    } catch (e) {
      await teardown()
      setError(e instanceof Error ? e.message : "Failed to connect")
    }
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md rounded-xl border border-slate-200 bg-white p-8 text-center">
          <img
            src="/voicera-wordmark-trimmed.png"
            alt="Voicera"
            className="mx-auto mb-4 h-6 w-auto"
            style={{ filter: "brightness(0) saturate(100%)" }}
          />
          <h1 className="mb-2 text-lg font-semibold text-slate-900">Link unavailable</h1>
          <p className="text-sm text-slate-600">{loadError}</p>
        </div>
      </div>
    )
  }

  if (!agent) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50">
        <p className="text-sm text-slate-500">Loading…</p>
      </div>
    )
  }

  if (agent.interaction_mode !== "translation") {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md rounded-xl border border-slate-200 bg-white p-8 text-center">
          <img
            src="/voicera-wordmark-trimmed.png"
            alt="Voicera"
            className="mx-auto mb-4 h-6 w-auto"
            style={{ filter: "brightness(0) saturate(100%)" }}
          />
          <h1 className="mb-2 text-lg font-semibold text-slate-900">{agent.display_name}</h1>
          <p className="text-sm text-slate-600">This link is not a live translation session.</p>
        </div>
      </div>
    )
  }

  const listenerStatus = getListenerStatus({
    sessionEnded,
    isConnected,
    isConnecting,
    isReconnecting,
    presenterOnline,
  })

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="mb-6 flex items-center justify-center gap-2.5">
          <img
            src="/voicera-wordmark-trimmed.png"
            alt="Voicera"
            className="block h-2.5 w-auto"
            style={{ filter: "brightness(0) saturate(100%)" }}
          />
          <span className="h-3 w-px bg-slate-300" />
          <span className="text-sm font-normal leading-none text-slate-400">Live translation</span>
        </div>
        <div className="mb-6 text-center">
          <h1 className="text-xl font-semibold text-slate-900">{agent.display_name}</h1>
          {agent.source_language && (
            <p className="mt-1 text-sm text-slate-500">Presenter speaks {agent.source_language}</p>
          )}
        </div>

        <label className="mb-2 block text-sm font-medium text-slate-700">Your language</label>
        <Select
          value={language}
          onValueChange={setLanguage}
          disabled={isConnected || isConnecting || isReconnecting}
        >
          <SelectTrigger className="mb-4 h-11 w-full rounded-lg border-slate-300 bg-white text-sm text-slate-900">
            <SelectValue placeholder="Select a language" />
          </SelectTrigger>
          <SelectContent className="rounded-lg">
            {agent.target_languages.map((lang) => (
              <SelectItem key={lang} value={lang} className="text-sm">
                {lang}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>

        <div className="mb-4 flex items-center justify-between">
          <LiveStatusIndicator state={listenerStatus.state} label={listenerStatus.label} />
          {!isConnected && !isReconnecting ? (
            <button
              type="button"
              onClick={connect}
              disabled={isConnecting || !language}
              className="rounded-lg bg-slate-900 px-4 py-2 text-sm font-medium text-white hover:bg-slate-800 disabled:opacity-60"
            >
              {isConnecting ? "Connecting…" : "Listen"}
            </button>
          ) : (
            <button
              type="button"
              onClick={() => void teardown()}
              className="rounded-lg border border-slate-300 px-4 py-2 text-sm font-medium text-slate-700 hover:bg-slate-50"
            >
              Stop
            </button>
          )}
        </div>

        {error && (
          <p role="alert" className="mb-3 text-xs text-red-600">
            {error}
          </p>
        )}
        {notice && (
          <p role="status" aria-live="polite" className="mb-3 text-xs text-slate-600">
            {notice}
          </p>
        )}

        <div
          ref={transcriptViewportRef}
          className="max-h-72 overflow-y-auto rounded-lg border border-slate-200 bg-slate-50 p-3"
        >
          {transcripts.length === 0 ? (
            <p className="py-8 text-center text-sm text-slate-400">
              Translated speech will appear here.
            </p>
          ) : (
            <div className="space-y-2">
              {transcripts.map((t) => (
                <div key={t.id} className="rounded-lg bg-white px-3 py-2 text-sm text-slate-900 shadow-sm">
                  {t.source && <p className="mb-1 text-xs text-slate-400">{t.source}</p>}
                  <p className="whitespace-pre-wrap">{t.content}</p>
                </div>
              ))}
            </div>
          )}
        </div>

        <p className="mt-6 text-center text-xs text-slate-400">
          Powered by Voicera
        </p>
      </div>
    </div>
  )
}
