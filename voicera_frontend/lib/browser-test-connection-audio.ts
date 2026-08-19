/**
 * Subtle connection-state chimes for the browser test dialog.
 * Uses a single shared AudioContext and Web Audio API synthesis only.
 */

let sharedContext: AudioContext | null = null

function getSharedAudioContext(): AudioContext | null {
  if (typeof window === "undefined") return null

  try {
    if (!sharedContext || sharedContext.state === "closed") {
      sharedContext = new AudioContext()
    }
    return sharedContext
  } catch {
    return null
  }
}

/**
 * Resume the shared AudioContext after a user gesture (e.g. Start Browser Test).
 * Safe to call repeatedly; never throws.
 */
export async function resumeConnectionChimeContext(): Promise<void> {
  try {
    const ctx = getSharedAudioContext()
    if (ctx && ctx.state === "suspended") {
      await ctx.resume()
    }
  } catch {
    // Audio must never block or break the browser test flow.
  }
}

type ToneOptions = {
  frequency: number
  startTime: number
  duration: number
  peakGain: number
  fadeOut?: boolean
}

function scheduleTone(ctx: AudioContext, options: ToneOptions): void {
  const { frequency, startTime, duration, peakGain, fadeOut = false } = options

  const oscillator = ctx.createOscillator()
  const gain = ctx.createGain()

  oscillator.type = "triangle"
  oscillator.frequency.setValueAtTime(frequency, startTime)

  const attackEnd = startTime + 0.015
  const releaseStart = fadeOut
    ? startTime + Math.max(duration - 0.12, 0.04)
    : startTime + Math.max(duration - 0.06, 0.03)
  const releaseEnd = startTime + duration

  gain.gain.setValueAtTime(0.0001, startTime)
  gain.gain.exponentialRampToValueAtTime(Math.max(peakGain, 0.0001), attackEnd)
  gain.gain.setValueAtTime(peakGain, releaseStart)
  gain.gain.exponentialRampToValueAtTime(0.0001, releaseEnd)

  oscillator.connect(gain)
  gain.connect(ctx.destination)

  oscillator.start(startTime)
  oscillator.stop(releaseEnd + 0.02)
}

/**
 * Pleasant ascending "connected" chime (~650 ms).
 */
export function playConnectedChime(): void {
  try {
    const ctx = getSharedAudioContext()
    if (!ctx) return

    if (ctx.state === "suspended") {
      void ctx.resume().catch(() => {})
    }

    const now = ctx.currentTime
    const peakGain = 0.07

    scheduleTone(ctx, {
      frequency: 392,
      startTime: now,
      duration: 0.26,
      peakGain,
    })
    scheduleTone(ctx, {
      frequency: 523.25,
      startTime: now + 0.3,
      duration: 0.38,
      peakGain: peakGain * 0.92,
      fadeOut: true,
    })
  } catch {
    // Ignore audio failures.
  }
}

/**
 * Subtle descending "disconnected" chime (~680 ms) with fade-out.
 */
export function playDisconnectedChime(): void {
  try {
    const ctx = getSharedAudioContext()
    if (!ctx) return

    if (ctx.state === "suspended") {
      void ctx.resume().catch(() => {})
    }

    const now = ctx.currentTime
    const peakGain = 0.065

    scheduleTone(ctx, {
      frequency: 523.25,
      startTime: now,
      duration: 0.26,
      peakGain,
    })
    scheduleTone(ctx, {
      frequency: 392,
      startTime: now + 0.3,
      duration: 0.4,
      peakGain: peakGain * 0.88,
      fadeOut: true,
    })
  } catch {
    // Ignore audio failures.
  }
}
