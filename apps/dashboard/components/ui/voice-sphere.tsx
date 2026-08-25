"use client"

import { useCallback, useEffect, useMemo, useRef, useState } from "react"

type VoiceSphereProps = {
  size?: number
  sensitivity?: number
  isListening?: boolean
  onListeningChange?: (listening: boolean) => void
  onAmplitudeChange?: (amp: number) => void
}

type Dot = {
  x: number
  y: number
  z: number
  phase: number
  size: number
}

const DOT_COUNT = 520
const GOLDEN_ANGLE = Math.PI * (3 - Math.sqrt(5))
const MIN_DOT_RADIUS = 1.0
const MAX_DOT_RADIUS = 1.6

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value))
}

function lerp(from: number, to: number, t: number): number {
  return from + (to - from) * t
}

function makeFibonacciDots(count: number): Dot[] {
  const dots: Dot[] = []
  for (let i = 0; i < count; i++) {
    const y = 1 - (i / (count - 1)) * 2
    const ring = Math.sqrt(Math.max(0, 1 - y * y))
    const theta = GOLDEN_ANGLE * i
    dots.push({
      x: Math.cos(theta) * ring,
      y,
      z: Math.sin(theta) * ring,
      phase: (i * 0.73) % (Math.PI * 2),
      size: MIN_DOT_RADIUS + (i % 7) / 6 * (MAX_DOT_RADIUS - MIN_DOT_RADIUS),
    })
  }
  return dots
}

function layeredNoise3D(x: number, y: number, z: number, time: number, phase: number): number {
  let total = 0
  let amplitude = 1
  let frequency = 1.35
  let norm = 0

  for (let octave = 0; octave < 3; octave++) {
    const n =
      Math.sin((x * frequency + time * (0.8 + octave * 0.2)) + phase) +
      Math.cos((y * frequency - time * (0.9 + octave * 0.2)) - phase * 0.7) +
      Math.sin((z * frequency + time * (0.65 + octave * 0.15)) + phase * 1.1)

    total += (n / 3) * amplitude
    norm += amplitude
    amplitude *= 0.52
    frequency *= 1.95
  }

  return norm > 0 ? total / norm : 0
}

export function VoiceSphere({
  size = 320,
  sensitivity = 5,
  isListening,
  onListeningChange,
  onAmplitudeChange,
}: VoiceSphereProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const rafRef = useRef<number | null>(null)

  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const sourceRef = useRef<MediaStreamAudioSourceNode | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const freqDataRef = useRef<Uint8Array | null>(null)

  const [internalListening, setInternalListening] = useState(false)
  const [localSensitivity, setLocalSensitivity] = useState(clamp(sensitivity, 1, 10))
  const [error, setError] = useState("")

  const smoothAmpRef = useRef(0)
  const rawAmpRef = useRef(0)
  const lastTimeRef = useRef(0)

  const dots = useMemo(() => makeFibonacciDots(DOT_COUNT), [])
  const controlled = typeof isListening === "boolean"
  const activeListening = controlled ? Boolean(isListening) : internalListening

  useEffect(() => {
    setLocalSensitivity(clamp(sensitivity, 1, 10))
  }, [sensitivity])

  const stopListening = useCallback(async () => {
    if (sourceRef.current) {
      sourceRef.current.disconnect()
      sourceRef.current = null
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop())
      streamRef.current = null
    }

    analyserRef.current = null
    freqDataRef.current = null

    const ctx = audioContextRef.current
    audioContextRef.current = null
    if (ctx && ctx.state !== "closed") {
      await ctx.close()
    }

    if (!controlled) {
      setInternalListening(false)
    }
  }, [controlled])

  const startListening = useCallback(async () => {
    if (audioContextRef.current && analyserRef.current && streamRef.current) {
      if (!controlled) {
        setInternalListening(true)
      }
      return
    }

    setError("")
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      })

      const audioContext = new AudioContext()
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.2

      const source = audioContext.createMediaStreamSource(stream)
      source.connect(analyser)

      const freqData = new Uint8Array(analyser.frequencyBinCount)

      streamRef.current = stream
      audioContextRef.current = audioContext
      analyserRef.current = analyser
      sourceRef.current = source
      freqDataRef.current = freqData

      if (!controlled) {
        setInternalListening(true)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Microphone access failed")
      await stopListening()
    }
  }, [controlled, stopListening])

  useEffect(() => {
    if (!controlled) return
    if (isListening) {
      void startListening()
    } else {
      void stopListening()
    }
  }, [controlled, isListening, startListening, stopListening])

  const toggleListening = useCallback(() => {
    if (controlled) {
      onListeningChange?.(!activeListening)
      return
    }
    if (activeListening) {
      void stopListening()
      return
    }
    void startListening()
  }, [activeListening, controlled, onListeningChange, startListening, stopListening])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext("2d")
    if (!ctx) return

    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.round(size * dpr)
    canvas.height = Math.round(size * dpr)
    canvas.style.width = `${size}px`
    canvas.style.height = `${size}px`
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

    const center = size / 2
    const baseRadius = size * 0.34
    const cameraDistance = baseRadius * 3.2

    const frame = (timeMs: number) => {
      const t = timeMs * 0.001
      const dt = lastTimeRef.current > 0 ? (timeMs - lastTimeRef.current) / 1000 : 1 / 60
      lastTimeRef.current = timeMs

      let speechAmp = 0
      if (activeListening && analyserRef.current && freqDataRef.current && audioContextRef.current) {
        const analyser = analyserRef.current
        const freqData = freqDataRef.current
        analyser.getByteFrequencyData(freqData)

        const sampleRate = audioContextRef.current.sampleRate
        const hzPerBin = sampleRate / 2 / freqData.length
        const low = clamp(Math.floor(80 / hzPerBin), 0, freqData.length - 1)
        const high = clamp(Math.ceil(3000 / hzPerBin), low, freqData.length - 1)

        let sum = 0
        let count = 0
        for (let i = low; i <= high; i++) {
          sum += freqData[i]
          count++
        }
        const avg = count > 0 ? sum / (count * 255) : 0
        const boosted = Math.pow(avg, 0.55)
        const sensitivityScale = localSensitivity / 5
        speechAmp = clamp(boosted * sensitivityScale, 0, 1.4)
      }
      rawAmpRef.current = speechAmp

      const smooth = smoothAmpRef.current
      const attack = 0.55
      const decay = 0.06
      smoothAmpRef.current =
        speechAmp > smooth
          ? lerp(smooth, speechAmp, attack)
          : lerp(smooth, speechAmp, decay)

      const voiceAmp = clamp(smoothAmpRef.current, 0, 1)
      onAmplitudeChange?.(voiceAmp)

      const idleBreath = 0.035 + 0.025 * Math.sin(t * 1.15) + 0.01 * Math.sin(t * 2.3)
      const activityAmp = activeListening ? voiceAmp : clamp(idleBreath, 0.01, 0.12)

      const rotY = t * 0.35
      const rotX = t * 0.2
      const sinY = Math.sin(rotY)
      const cosY = Math.cos(rotY)
      const sinX = Math.sin(rotX)
      const cosX = Math.cos(rotX)

      ctx.clearRect(0, 0, size, size)

      const drawDots = new Array<{
        x: number
        y: number
        z: number
        radius: number
        alpha: number
      }>(dots.length)

      for (let i = 0; i < dots.length; i++) {
        const d = dots[i]

        const noise = layeredNoise3D(d.x, d.y, d.z, t, d.phase)
        const outwardNoise = noise * 0.5 + 0.5
        const maxExpansion = 0.4
        const expansion = activeListening
          ? maxExpansion * activityAmp * outwardNoise
          : idleBreath * 0.08 * outwardNoise

        const radiusScale = 1 + expansion
        const px = d.x * baseRadius * radiusScale
        const py = d.y * baseRadius * radiusScale
        const pz = d.z * baseRadius * radiusScale

        const xzX = px * cosY + pz * sinY
        const xzZ = -px * sinY + pz * cosY

        const yzY = py * cosX - xzZ * sinX
        const yzZ = py * sinX + xzZ * cosX

        const perspective = cameraDistance / (cameraDistance - yzZ)
        const screenX = center + xzX * perspective
        const screenY = center + yzY * perspective

        const depthNorm = clamp((yzZ / (baseRadius * 1.45) + 1) * 0.5, 0, 1)
        const alpha = lerp(0.05, 0.88, depthNorm)
        const dotSize = d.size * (1 + activityAmp * 0.22) * perspective

        drawDots[i] = {
          x: screenX,
          y: screenY,
          z: yzZ,
          radius: dotSize,
          alpha,
        }
      }

      drawDots.sort((a, b) => a.z - b.z)

      for (let i = 0; i < drawDots.length; i++) {
        const dot = drawDots[i]
        ctx.beginPath()
        ctx.arc(dot.x, dot.y, dot.radius, 0, Math.PI * 2)
        ctx.fillStyle = `rgba(10,10,10,${dot.alpha.toFixed(4)})`
        ctx.fill()
      }

      rafRef.current = window.requestAnimationFrame(frame)
      if (dt <= 0) {
        rafRef.current = window.requestAnimationFrame(frame)
      }
    }

    rafRef.current = window.requestAnimationFrame(frame)
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current)
      }
    }
  }, [activeListening, dots, localSensitivity, onAmplitudeChange, size])

  useEffect(() => {
    return () => {
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current)
      }
      void stopListening()
    }
  }, [stopListening])

  return (
    <div className="flex flex-col items-center gap-3">
      <canvas ref={canvasRef} className="block" aria-label="Voice reactive dot sphere" />

      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={toggleListening}
          disabled={controlled && !onListeningChange}
          className="rounded-md border border-slate-300 px-3 py-1.5 text-sm font-medium text-slate-900 hover:bg-slate-50"
        >
          {activeListening ? "Stop listening" : "Start listening"}
        </button>
      </div>

      <label className="flex w-full max-w-[280px] items-center gap-3 text-sm text-slate-700">
        <span className="shrink-0">Sensitivity</span>
        <input
          type="range"
          min={1}
          max={10}
          step={1}
          value={localSensitivity}
          onChange={(event) => setLocalSensitivity(Number(event.target.value))}
          className="w-full"
          aria-label="Sensitivity"
        />
        <span className="w-6 text-right">{localSensitivity}</span>
      </label>

      {error ? <p className="text-xs text-red-600">{error}</p> : null}
    </div>
  )
}
