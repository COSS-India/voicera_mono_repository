"use client"

import { use, useCallback, useEffect, useRef, useState } from "react"
import { getTranslationListenWebSocketUrl } from "@/lib/johnaic-config"
import type { PublicAgent } from "@/lib/api"

const DEFAULT_SAMPLE_RATE = 16000

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
  const [error, setError] = useState("")
  const [notice, setNotice] = useState("")
  const [transcripts, setTranscripts] = useState<
    Array<{ id: string; source?: string; content: string }>
  >([])

  const wsRef = useRef<WebSocket | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const playbackTimeRef = useRef(0)
  const transcriptViewportRef = useRef<HTMLDivElement | null>(null)

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

  const teardown = useCallback(async () => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws) ws.close()
    const ctx = audioContextRef.current
    audioContextRef.current = null
    if (ctx && ctx.state !== "closed") await ctx.close()
    playbackTimeRef.current = 0
    setIsConnected(false)
    setIsConnecting(false)
  }, [])

  useEffect(() => {
    return () => {
      void teardown()
    }
  }, [teardown])

  useEffect(() => {
    const viewport = transcriptViewportRef.current
    if (viewport) viewport.scrollTop = viewport.scrollHeight
  }, [transcripts])

  const handleIncomingAudio = (payloadB64: string, sampleRate: number) => {
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
    const now = ctx.currentTime + 0.05
    const startAt = Math.max(now, playbackTimeRef.current)
    source.start(startAt)
    playbackTimeRef.current = startAt + buffer.duration
  }

  const connect = async () => {
    if (!agent || !language || isConnecting || isConnected) return
    setError("")
    setNotice("")
    setIsConnecting(true)
    setTranscripts([])
    try {
      const ctx = new AudioContext()
      audioContextRef.current = ctx
      await ctx.resume()

      const ws = new WebSocket(getTranslationListenWebSocketUrl(token, language))
      wsRef.current = ws

      ws.onopen = () => {
        setIsConnecting(false)
        setIsConnected(true)
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
            setNotice(
              msg.presenter_online
                ? ""
                : "Connected — waiting for the presenter to start speaking.",
            )
          } else if (msg?.event === "presenter_live") {
            setNotice("")
          } else if (msg?.event === "session_ended") {
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
      ws.onclose = () => {
        void teardown()
      }
    } catch (e) {
      await teardown()
      setError(e instanceof Error ? e.message : "Failed to connect")
    }
  }

  if (loadError) {
    return (
      <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
        <div className="max-w-md rounded-xl border border-slate-200 bg-white p-8 text-center">
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
          <h1 className="mb-2 text-lg font-semibold text-slate-900">{agent.display_name}</h1>
          <p className="text-sm text-slate-600">This link is not a live translation session.</p>
        </div>
      </div>
    )
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-slate-50 p-6">
      <div className="w-full max-w-lg rounded-2xl border border-slate-200 bg-white p-8 shadow-sm">
        <div className="mb-6 text-center">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-400">
            Live translation
          </p>
          <h1 className="mt-1 text-xl font-semibold text-slate-900">{agent.display_name}</h1>
          {agent.source_language && (
            <p className="mt-1 text-sm text-slate-500">Presenter speaks {agent.source_language}</p>
          )}
        </div>

        <label className="mb-2 block text-sm font-medium text-slate-700">Your language</label>
        <select
          value={language}
          onChange={(e) => setLanguage(e.target.value)}
          disabled={isConnected || isConnecting}
          className="mb-4 w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm text-slate-900 disabled:opacity-60"
        >
          {agent.target_languages.map((lang) => (
            <option key={lang} value={lang}>
              {lang}
            </option>
          ))}
        </select>

        <div className="mb-4 flex items-center justify-between">
          <span className="flex items-center gap-2 text-sm text-slate-600">
            <span
              className={`inline-block h-2.5 w-2.5 rounded-full ${
                isConnected ? "bg-green-500" : isConnecting ? "bg-amber-400" : "bg-slate-300"
              }`}
            />
            {isConnected ? "Listening live" : isConnecting ? "Connecting…" : "Not connected"}
          </span>
          {!isConnected ? (
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

        {error && <p className="mb-3 text-xs text-red-600">{error}</p>}
        {notice && <p className="mb-3 text-xs text-slate-600">{notice}</p>}

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
      </div>
    </div>
  )
}
