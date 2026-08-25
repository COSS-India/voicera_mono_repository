import type { CallLatencyMetrics, CallLatencyTurn } from "@/lib/api"

function normalizeText(text: string): string {
  return text.trim().replace(/\s+/g, " ")
}

function isSameUtterance(a: string, b: string): boolean {
  if (!a || !b) return false
  if (a === b) return true
  return a.startsWith(b) || b.startsWith(a)
}

function pickMetric(
  existing: number | null | undefined,
  incoming: number | null | undefined,
  preferNonzero = false
): number | null | undefined {
  if (incoming == null) return existing
  if (existing == null) return incoming
  if (preferNonzero && existing === 0 && incoming > 0) return incoming
  if (preferNonzero && incoming === 0) return existing
  return Math.max(existing, incoming)
}

function mergeTurn(into: CallLatencyTurn, other: CallLatencyTurn): CallLatencyTurn {
  const a = normalizeText(into.user_text_preview || "")
  const b = normalizeText(other.user_text_preview || "")
  const longer = a.length >= b.length ? a : b
  return {
    turn_index: into.turn_index,
    user_text_preview: longer || into.user_text_preview || other.user_text_preview,
    stt_ms: pickMetric(into.stt_ms, other.stt_ms, true),
    llm_ttfb_ms: pickMetric(into.llm_ttfb_ms, other.llm_ttfb_ms),
    tts_first_chunk_ms: pickMetric(into.tts_first_chunk_ms, other.tts_first_chunk_ms),
  }
}

/** Merge partial + final transcription rows (legacy stored data). */
export function dedupeLatencyTurns(turns: CallLatencyTurn[]): CallLatencyTurn[] {
  if (!turns.length) return []
  const merged: CallLatencyTurn[] = []
  for (const turn of turns) {
    const prev = merged[merged.length - 1]
    if (prev) {
      const prevText = normalizeText(prev.user_text_preview || "")
      const curText = normalizeText(turn.user_text_preview || "")
      if (isSameUtterance(prevText, curText)) {
        merged[merged.length - 1] = mergeTurn(prev, turn)
        continue
      }
    }
    merged.push({ ...turn })
  }
  return merged.map((t, i) => ({ ...t, turn_index: i + 1 }))
}

export function normalizeLatencyMetrics(
  metrics: CallLatencyMetrics | undefined
): CallLatencyMetrics | undefined {
  if (!metrics?.turns?.length) return metrics
  const turns = dedupeLatencyTurns(metrics.turns)
  const summary = { ...metrics.summary, turn_count: turns.length }
  for (const [field, avgKey, maxKey] of [
    ["stt_ms", "avg_stt_ms", "max_stt_ms"] as const,
    ["llm_ttfb_ms", "avg_llm_ttfb_ms", "max_llm_ttfb_ms"] as const,
    ["tts_first_chunk_ms", "avg_tts_first_chunk_ms", "max_tts_first_chunk_ms"] as const,
  ]) {
    const values = turns
      .map((t) => t[field])
      .filter((v): v is number => v != null && !Number.isNaN(v))
    if (values.length) {
      ;(summary as Record<string, number>)[avgKey] =
        Math.round((values.reduce((a, b) => a + b, 0) / values.length) * 10) / 10
      ;(summary as Record<string, number>)[maxKey] = Math.max(...values)
    }
  }
  return { turns, summary }
}
