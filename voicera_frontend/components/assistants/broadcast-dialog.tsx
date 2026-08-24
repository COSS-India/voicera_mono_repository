"use client"

import { useCallback, useEffect, useRef, useState } from "react"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import type { Agent, BroadcastToken } from "@/lib/api"
import { getAuthToken } from "@/lib/api"
import { getTranslationPublishWebSocketUrl } from "@/lib/johnaic-config"
import { LiveStatusIndicator, type LiveStatusState } from "@/components/live-status-indicator"
import { Copy, Loader2, Mic, MicOff, Radio, Square } from "lucide-react"

function getBroadcasterStatus({
  isConnecting,
  isLive,
  isMuted,
}: {
  isConnecting: boolean
  isLive: boolean
  isMuted: boolean
}): { state: LiveStatusState; label: string } {
  if (isConnecting) return { state: "connecting", label: "Connecting" }
  if (isLive) {
    return isMuted
      ? { state: "waiting", label: "Connected (mic muted)" }
      : { state: "live", label: "Live" }
  }
  return { state: "idle", label: "Idle" }
}

interface BroadcastDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  agent: Agent | null
  getAgentDisplayName: (agent: Agent) => string
}

const TARGET_SAMPLE_RATE = 16000

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

function downsampleTo16k(input: Float32Array, inputRate: number): Float32Array {
  if (inputRate <= TARGET_SAMPLE_RATE) return input
  const ratio = inputRate / TARGET_SAMPLE_RATE
  const result = new Float32Array(Math.round(input.length / ratio))
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

export function BroadcastDialog({
  open,
  onOpenChange,
  agent,
  getAgentDisplayName,
}: BroadcastDialogProps) {
  const [isConnecting, setIsConnecting] = useState(false)
  const [isLive, setIsLive] = useState(false)
  const [isMuted, setIsMuted] = useState(true)
  const [error, setError] = useState("")
  const [copied, setCopied] = useState(false)

  const wsRef = useRef<WebSocket | null>(null)
  const isMutedRef = useRef(true)
  const audioContextRef = useRef<AudioContext | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const sourceNodeRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const processorNodeRef = useRef<ScriptProcessorNode | null>(null)

  const shareUrl =
    agent?.share_token && typeof window !== "undefined"
      ? `${window.location.origin}/live/${agent.share_token}`
      : ""

  const teardown = useCallback(async () => {
    const ws = wsRef.current
    wsRef.current = null
    if (ws) ws.close(1000, "host-end")

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
    if (ctx && ctx.state !== "closed") await ctx.close()

    isMutedRef.current = true
    setIsMuted(true)
    setIsLive(false)
    setIsConnecting(false)
  }, [])

  useEffect(() => {
    return () => {
      void teardown()
    }
  }, [teardown])

  const startBroadcast = async () => {
    if (!agent?.agent_type || isConnecting || isLive) return
    setError("")
    setIsConnecting(true)

    try {
      const authToken = getAuthToken()
      const res = await fetch("/api/agents/broadcast-token", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(authToken ? { Authorization: `Bearer ${authToken}` } : {}),
        },
        body: JSON.stringify({ agent_type: agent.agent_type }),
      })
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        throw new Error(data?.detail || data?.error || "Could not start broadcast")
      }
      const { token, agent_id }: BroadcastToken = await res.json()

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

      const ws = new WebSocket(getTranslationPublishWebSocketUrl(agent_id, token))
      wsRef.current = ws

      ws.onopen = () => {
        // Start muted so the presenter controls exactly when they go live.
        isMutedRef.current = true
        setIsMuted(true)

        const sessionId = `host-${Date.now()}`
        ws.send(
          JSON.stringify({
            event: "start",
            start: { callSid: sessionId, streamSid: sessionId },
          }),
        )

        const src = ctx.createMediaStreamSource(stream)
        const processor = ctx.createScriptProcessor(1024, 1, 1)
        sourceNodeRef.current = src
        processorNodeRef.current = processor

        processor.onaudioprocess = (ev) => {
          const socket = wsRef.current
          if (!socket || socket.readyState !== WebSocket.OPEN || isMutedRef.current) return
          const downsampled = downsampleTo16k(ev.inputBuffer.getChannelData(0), ctx.sampleRate)
          if (!downsampled.length) return
          socket.send(
            JSON.stringify({
              event: "media",
              media: {
                contentType: "audio/x-l16",
                sampleRate: TARGET_SAMPLE_RATE,
                payload: floatToInt16Base64(downsampled),
              },
            }),
          )
        }

        src.connect(processor)
        processor.connect(ctx.destination)
        setIsConnecting(false)
        setIsLive(true)
      }

      ws.onerror = () => setError("Failed to connect to the translation server")
      ws.onclose = (ev) => {
        if (ev.code === 4409) setError("Someone else is already broadcasting this agent")
        else if (ev.code === 4401) setError("Not authorized to broadcast this agent")
        else if (ev.code === 4402)
          setError("Translation isn't configured for this agent (no OpenAI credential)")
        else if (ev.code === 1013) setError("Capacity or usage limit reached — try again later")
        void teardown()
      }
    } catch (e) {
      await teardown()
      setError(e instanceof Error ? e.message : "Failed to start broadcast")
    }
  }

  const handleOpenChange = (nextOpen: boolean) => {
    if (!nextOpen) {
      void teardown()
      setError("")
    }
    onOpenChange(nextOpen)
  }

  const copyShareUrl = async () => {
    if (!shareUrl) return
    await navigator.clipboard.writeText(shareUrl)
    setCopied(true)
    window.setTimeout(() => setCopied(false), 1500)
  }

  return (
    <Dialog open={open} onOpenChange={handleOpenChange}>
      <DialogContent className="flex max-h-[85dvh] flex-col overflow-hidden sm:max-w-lg">
        <DialogHeader className="shrink-0 pr-8">
          <DialogTitle>{agent ? getAgentDisplayName(agent) : "Broadcast"}</DialogTitle>
          <DialogDescription>
            Speak into your microphone. Listeners on the shared link hear a live translation in
            the language they pick.
          </DialogDescription>
        </DialogHeader>

        <div className="min-h-0 flex-1 overflow-y-auto rounded-xl border border-slate-200 bg-slate-50 px-4 py-3">
          <div className="mb-3">
            <p className="mb-1 text-xs font-medium uppercase tracking-wide text-slate-500">
              Listener link
            </p>
            {shareUrl ? (
              <div className="flex items-center gap-2">
                <code className="min-w-0 flex-1 truncate rounded border border-slate-200 bg-white px-2 py-1.5 text-xs text-slate-700">
                  {shareUrl}
                </code>
                <Button
                  type="button"
                  variant="outline"
                  size="sm"
                  className="shrink-0"
                  onClick={copyShareUrl}
                >
                  <Copy className="mr-1 h-3.5 w-3.5" />
                  {copied ? "Copied" : "Copy"}
                </Button>
              </div>
            ) : (
              <p className="text-xs text-amber-700">
                Public sharing is off for this agent. Enable it in the agent settings to get a
                listener link.
              </p>
            )}
          </div>

          <LiveStatusIndicator
            {...getBroadcasterStatus({ isConnecting, isLive, isMuted })}
          />
          {error && (
            <p role="alert" className="mt-3 text-xs text-red-600">
              {error}
            </p>
          )}
        </div>

        <DialogFooter className="shrink-0 flex-row flex-wrap justify-center gap-2 sm:justify-center">
          {!isLive ? (
            <Button type="button" onClick={startBroadcast} disabled={isConnecting || !agent}>
              {isConnecting ? (
                <Loader2 className="mr-2 h-4 w-4 animate-spin" />
              ) : (
                <Radio className="mr-2 h-4 w-4" />
              )}
              {isConnecting ? "Connecting..." : "Start Broadcasting"}
            </Button>
          ) : (
            <Button type="button" variant="destructive" onClick={() => void teardown()}>
              <Square className="mr-2 h-4 w-4" />
              Stop Broadcasting
            </Button>
          )}
          <Button
            type="button"
            variant="outline"
            disabled={!isLive}
            onClick={() =>
              setIsMuted((v) => {
                const next = !v
                isMutedRef.current = next
                return next
              })
            }
            className={`min-w-[7rem] ${
              isMuted
                ? "border-red-300 bg-red-50 text-red-700 hover:bg-red-100 hover:text-red-800"
                : "border-slate-200 bg-white text-slate-900 hover:bg-slate-50"
            }`}
          >
            {isMuted ? <MicOff className="mr-1 h-4 w-4" /> : <Mic className="mr-1 h-4 w-4" />}
            {isMuted ? "Unmute" : "Mute"}
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}
