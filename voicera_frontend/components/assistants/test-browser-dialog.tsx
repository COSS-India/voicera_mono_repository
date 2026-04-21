"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Orb, type AgentState as OrbAgentState } from "@/components/ui/orb"
import type { Agent } from "@/lib/api"
import { Loader2, Mic, MicOff, PhoneOff, Radio } from "lucide-react"

interface TestBrowserDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agent: Agent | null
  getAgentDisplayName: (agent: Agent) => string
}

const TARGET_SAMPLE_RATE = 16000

function rms(samples: Float32Array): number {
  if (!samples.length) return 0
  let sum = 0
  for (let i = 0; i < samples.length; i++) {
    const s = samples[i]
    sum += s * s
  }
  return Math.sqrt(sum / samples.length)
}

function normalizeVolume(v: number): number {
  return Math.max(0, Math.min(1, v * 5))
}

function floatToInt16Base64(float32: Float32Array): string {
  const int16 = new Int16Array(float32.length)
  for (let i = 0; i < float32.length; i++) {
    const s = Math.max(-1, Math.min(1, float32[i]))
    int16[i] = s < 0 ? s * 0x8000 : s * 0x7fff
  }
  const bytes = new Uint8Array(int16.buffer)
  let binary = ""
  const chunk = 0x8000
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode(...bytes.subarray(i, i + chunk))
  }
  return btoa(binary)
}

function base64ToInt16(base64: string): Int16Array {
  const binary = atob(base64)
  const bytes = new Uint8Array(binary.length)
  for (let i = 0; i < binary.length; i++) {
    bytes[i] = binary.charCodeAt(i)
  }
  return new Int16Array(bytes.buffer)
}

function int16ToFloat32(input: Int16Array): Float32Array {
  const out = new Float32Array(input.length)
  for (let i = 0; i < input.length; i++) {
    out[i] = input[i] / 0x8000
  }
  return out
}

function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate === TARGET_SAMPLE_RATE) return input
  if (inputRate < TARGET_SAMPLE_RATE) return input

  const ratio = inputRate / TARGET_SAMPLE_RATE
  const newLength = Math.round(input.length / ratio)
  const result = new Float32Array(newLength)
  let offsetResult = 0
  let offsetBuffer = 0

  while (offsetResult < result.length) {
    const nextOffsetBuffer = Math.round((offsetResult + 1) * ratio)
    let accum = 0
    let count = 0
    for (let i = offsetBuffer; i < nextOffsetBuffer && i < input.length; i++) {
      accum += input[i]
      count++
    }
    result[offsetResult] = count > 0 ? accum / count : 0
    offsetResult++
    offsetBuffer = nextOffsetBuffer
  }

  return result
}

function getBrowserWsUrl(agentId: string): string {
  const explicitWsBase = process.env.NEXT_PUBLIC_JOHNAIC_WEBSOCKET_URL
  const serverBase = process.env.NEXT_PUBLIC_JOHNAIC_SERVER_URL

  let wsBase = explicitWsBase || ""
  if (!wsBase && serverBase) {
    wsBase = serverBase.replace(/^http:\/\//i, "ws://").replace(/^https:\/\//i, "wss://")
  }
  if (!wsBase) {
    wsBase = "ws://localhost:7860"
  }

  return `${wsBase.replace(/\/$/, "")}/browser/agent/${encodeURIComponent(agentId)}`
}

export function TestBrowserDialog({
  open,
  onOpenChange,
  agent,
  getAgentDisplayName,
}: TestBrowserDialogProps) {
  const [isConnecting, setIsConnecting] = useState(false)
  const [isConnected, setIsConnected] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [error, setError] = useState("")
  const [orbState, setOrbState] = useState<OrbAgentState>(null)
  const [transcripts, setTranscripts] = useState<Array<{ id: string; role: "user" | "assistant"; content: string }>>([])

  const wsRef = useRef<WebSocket | null>(null)
  const isMutedRef = useRef(false)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null)
  const playbackTimeRef = useRef(0)
  const sessionIdRef = useRef("")
  const inputVolumeRef = useRef(0)
  const outputVolumeRef = useRef(0)
  const lastOutputAtRef = useRef(0)
  const transcriptViewportRef = useRef<HTMLDivElement | null>(null)

  const sessionLabel = useMemo(() => {
    if (isConnected) return "Live"
    if (isConnecting) return "Connecting"
    return "Idle"
  }, [isConnected, isConnecting])

  const teardown = useCallback(async () => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.close(1000, "client-end")
    } else if (ws) {
      ws.close()
    }

    if (processorNodeRef.current) {
      processorNodeRef.current.disconnect()
      processorNodeRef.current.onaudioprocess = null
      processorNodeRef.current = null
    }

    if (sourceNodeRef.current) {
      sourceNodeRef.current.disconnect()
      sourceNodeRef.current = null
    }

    if (mediaStreamRef.current) {
      mediaStreamRef.current.getTracks().forEach((t) => t.stop())
      mediaStreamRef.current = null
    }

    const ctx = audioContextRef.current
    audioContextRef.current = null
    if (ctx && ctx.state !== "closed") {
      await ctx.close()
    }

    playbackTimeRef.current = 0
    sessionIdRef.current = ""
    isMutedRef.current = false
    inputVolumeRef.current = 0
    outputVolumeRef.current = 0
    lastOutputAtRef.current = 0
    setIsConnected(false)
    setIsConnecting(false)
    setOrbState(null)
    setTranscripts([])
  }, [])

  const stopSession = useCallback(async () => {
    await teardown()
  }, [teardown])

  const handleIncomingAudio = (payloadB64: string, sampleRate = TARGET_SAMPLE_RATE) => {
    const ctx = audioContextRef.current
    if (!ctx) return

    const int16 = base64ToInt16(payloadB64)
    if (!int16.length) return
    const float32 = int16ToFloat32(int16)
    outputVolumeRef.current = normalizeVolume(rms(float32))
    lastOutputAtRef.current = performance.now()
    const buffer = ctx.createBuffer(1, float32.length, sampleRate)
    buffer.copyToChannel(float32, 0)

    const source = ctx.createBufferSource()
    source.buffer = buffer
    source.connect(ctx.destination)

    const now = ctx.currentTime + 0.02
    const startAt = Math.max(now, playbackTimeRef.current)
    source.start(startAt)
    playbackTimeRef.current = startAt + buffer.duration
  }

  const startSession = async () => {
    if (!agent?.agent_id || isConnecting || isConnected) return

    setError("")
    setIsConnecting(true)
    sessionIdRef.current = `browser-${Date.now()}-${Math.random().toString(36).slice(2, 8)}`
    setTranscripts([])

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          channelCount: 1,
        },
      })
      mediaStreamRef.current = stream

      const ctx = new AudioContext()
      audioContextRef.current = ctx
      await ctx.resume()

      const ws = new WebSocket(getBrowserWsUrl(agent.agent_id))
      wsRef.current = ws

      ws.onopen = () => {
        // Start muted by default; user must click Unmute to send mic audio.
        isMutedRef.current = true
        setIsMuted(true)

        ws.send(
          JSON.stringify({
            event: "start",
            start: {
              callSid: sessionIdRef.current,
              streamSid: sessionIdRef.current,
            },
          }),
        )

        const src = ctx.createMediaStreamSource(stream)
        const processor = ctx.createScriptProcessor(1024, 1, 1)
        sourceNodeRef.current = src
        processorNodeRef.current = processor

        processor.onaudioprocess = (ev) => {
          const socket = wsRef.current
          if (!socket || socket.readyState !== WebSocket.OPEN || isMutedRef.current) return

          const channelData = ev.inputBuffer.getChannelData(0)
          const downsampled = downsampleTo16k(channelData, ctx.sampleRate)
          if (!downsampled.length) return
          inputVolumeRef.current = normalizeVolume(rms(downsampled))

          const payload = floatToInt16Base64(downsampled)
          socket.send(
            JSON.stringify({
              event: "media",
              media: {
                contentType: "audio/x-l16",
                sampleRate: TARGET_SAMPLE_RATE,
                payload,
              },
            }),
          )
        }

        src.connect(processor)
        processor.connect(ctx.destination)
        setIsConnecting(false)
        setIsConnected(true)
        setOrbState("listening")
      }

      ws.onmessage = (ev) => {
        try {
          const msg = JSON.parse(ev.data)
          if (msg?.event === "playAudio" && msg?.media?.payload) {
            handleIncomingAudio(msg.media.payload, Number(msg?.media?.sampleRate) || TARGET_SAMPLE_RATE)
          } else if (msg?.event === "transcript" && msg?.content) {
            const role = msg.role === "assistant" ? "assistant" : "user"
            const content = String(msg.content).trim()
            if (!content) return
            setTranscripts((prev) => {
              const last = prev[prev.length - 1]
              if (last && last.role === role && last.content === content) return prev
              const next = {
                id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
                role,
                content,
              }
              const merged = [...prev, next]
              return merged.length > 120 ? merged.slice(merged.length - 120) : merged
            })
          }
        } catch {
          // Ignore malformed frames.
        }
      }

      ws.onerror = () => {
        setError("WebSocket connection failed")
      }

      ws.onclose = () => {
        void teardown()
      }
    } catch (err) {
      await teardown()
      setError(err instanceof Error ? err.message : "Failed to start browser test")
    }
  }

  useEffect(() => {
    const id = window.setInterval(() => {
      if (!isConnected) {
        setOrbState(null)
        return
      }

      if (performance.now() - lastOutputAtRef.current > 220) {
        outputVolumeRef.current *= 0.68
      }
      inputVolumeRef.current *= 0.9

      if (outputVolumeRef.current > 0.08) {
        setOrbState("talking")
      } else {
        setOrbState("listening")
      }
    }, 60)

    return () => window.clearInterval(id)
  }, [isConnected])

  useEffect(() => {
    return () => {
      void stopSession()
    }
  }, [stopSession])

  useEffect(() => {
    const viewport = transcriptViewportRef.current
    if (!viewport) return
    viewport.scrollTop = viewport.scrollHeight
  }, [transcripts])

  const handleDialogOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      void stopSession()
      setError("")
      setIsMuted(true)
      isMutedRef.current = true
    }
    onOpenChange(nextOpen)
  }

  return (
    <Dialog open={open} onOpenChange={handleDialogOpenChange}>
      <DialogContent className="sm:max-w-lg">
        <DialogHeader>
          <DialogTitle>{agent ? getAgentDisplayName(agent) : "Test Agent on Browser"}</DialogTitle>
          <DialogDescription>
            Talk directly with your agent in real time from this browser.
          </DialogDescription>
        </DialogHeader>

        <div className="rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="mb-3 flex justify-center">
            <div className="h-36 w-36 rounded-full bg-slate-100 p-1 shadow-[inset_0_2px_8px_rgba(0,0,0,0.08)]">
              <div className="h-full w-full overflow-hidden rounded-full bg-white shadow-[inset_0_0_12px_rgba(0,0,0,0.05)]">
                <Orb
                  agentState={orbState}
                  volumeMode="manual"
                  inputVolumeRef={inputVolumeRef}
                  outputVolumeRef={outputVolumeRef}
                  colors={["#CADCFC", "#A0B9D1"]}
                  className="h-full w-full"
                />
              </div>
            </div>
          </div>

          <div
            ref={transcriptViewportRef}
            className="mb-3 max-h-52 overflow-y-auto rounded-lg border border-slate-200 bg-white p-3"
          >
            {transcripts.length === 0 ? (
              <p className="text-xs text-slate-500">Transcript will appear here during the session.</p>
            ) : (
              <div className="space-y-2">
                {transcripts.map((t) => (
                  <div key={t.id} className={`flex ${t.role === "user" ? "justify-end" : "justify-start"}`}>
                    <div
                      className={`max-w-[90%] rounded-2xl px-3 py-2 text-sm ${
                        t.role === "user"
                          ? "bg-slate-900 text-white"
                          : "bg-slate-100 text-slate-900"
                      }`}
                    >
                      <p className="mb-1 text-[10px] font-medium uppercase tracking-wide opacity-70">
                        {t.role === "user" ? "You" : "Agent"}
                      </p>
                      <p className="whitespace-pre-wrap">{t.content}</p>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm text-slate-700">
              <Radio className={`h-4 w-4 ${isConnected ? "text-green-600" : "text-slate-400"}`} />
              <span>Status: {sessionLabel}</span>
            </div>
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={!isConnected}
              onClick={() =>
                setIsMuted((v) => {
                  const next = !v
                  isMutedRef.current = next
                  return next
                })
              }
              className={
                isMuted
                  ? "h-8 border-red-300 bg-red-50 text-red-700 hover:bg-red-100 hover:text-red-800"
                  : "h-8 border-slate-200 bg-white text-slate-900 hover:bg-slate-50"
              }
            >
              {isMuted ? <MicOff className="h-4 w-4 mr-1" /> : <Mic className="h-4 w-4 mr-1" />}
              {isMuted ? "Unmute" : "Mute"}
            </Button>
          </div>
          {error && <p className="mt-3 text-xs text-red-600">{error}</p>}
        </div>

        <DialogFooter className="gap-2 sm:gap-0">
          {!isConnected ? (
            <Button type="button" onClick={startSession} disabled={isConnecting || !agent?.agent_id}>
              {isConnecting ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Mic className="h-4 w-4 mr-2" />}
              {isConnecting ? "Connecting..." : "Start Browser Test"}
            </Button>
          ) : (
            <Button type="button" variant="destructive" onClick={stopSession}>
              <PhoneOff className="h-4 w-4 mr-2" />
              End Session
            </Button>
          )}
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
