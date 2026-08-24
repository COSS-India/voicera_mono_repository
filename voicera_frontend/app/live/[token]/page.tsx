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
// Safety valve for genuine long-run drift only: if translated speech is
// consistently longer than the source, a listener would fall further and further
// behind, so past this much buffered-but-unplayed audio we drop the backlog and
// resync to now. It must stay well ABOVE one utterance's worth of audio: the
// backend sends a whole sentence's frames as fast as TTS yields them, so normal
// buffering routinely runs several seconds ahead. Setting this too low makes the
// resync fire mid-sentence, discarding translated speech the listener never hears
// (and, before the sources were tracked, layering it into a doubled voice).
const MAX_LEAD_SECS = 8

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
  // Every audio frame we schedule, so a resync can stop the ones already queued
  // instead of layering new audio on top of them.
  const scheduledSourcesRef = useRef<Set<AudioBufferSourceNode>>(new Set())
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
    for (const source of scheduledSourcesRef.current) {
      source.onended = null
      try {
        source.stop()
      } catch {
        // already stopped/ended
      }
    }
    scheduledSourcesRef.current.clear()
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

  const handleIncomingAudio = useCallback((payloadB64: string, sampleRate: number) => {
    const ctx = audioContextRef.current
    if (!ctx) return
    const int16 = base64ToInt16(payloadB64)
    if (!int16.length) return
    const float32 = int16ToFloat32(int16)
    const buffer = ctx.createBuffer(1, float32.length, sampleRate)
    buffer.copyToChannel(float32, 0)
    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)
    const earliest = ctx.currentTime + JITTER_BUFFER_SECS
    let startAt = Math.max(earliest, playbackTimeRef.current)
    // Drifted too far ahead → drop the accumulated lead and resync to now.
    // Stop everything already scheduled out to the old cursor first; otherwise
    // the resync'd frames play *on top* of it and the listener hears the same
    // translation twice for up to MAX_LEAD_SECS.
    if (startAt - ctx.currentTime > MAX_LEAD_SECS) {
      stopScheduledSources()
      startAt = earliest
    }
    source.start(startAt)
    scheduledSourcesRef.current.add(source)
    source.onended = () => scheduledSourcesRef.current.delete(source)
    playbackTimeRef.current = startAt + buffer.duration
  }, [stopScheduledSources])

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
        } else if (msg?.event === "status") {
          setPresenterOnline(Boolean(msg.presenter_online))
        } else if (msg?.event === "presenter_live") {
          setPresenterOnline(true)
        } else if (msg?.event === "session_ended") {
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
  }, [token, handleIncomingAudio, scheduleReconnect, stopScheduledSources])

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
      const ctx = new AudioContext()
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
