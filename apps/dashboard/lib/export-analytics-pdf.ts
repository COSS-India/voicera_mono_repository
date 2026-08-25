import jsPDF from "jspdf"
import autoTable from "jspdf-autotable"
import { format } from "date-fns"

import type { Analytics } from "@/lib/api"

export type AnalyticsPdfFilters = {
  dateRangeLabel: string
  agentLabel: string
  phoneLabel: string
}

function formatDurationMinutes(minutes: number): string {
  if (minutes < 1) {
    return `${Math.round(minutes * 60)}s`
  }
  if (minutes < 60) {
    return `${Math.round(minutes)}m`
  }
  const hours = Math.floor(minutes / 60)
  const mins = Math.round(minutes % 60)
  return `${hours}h ${mins}m`
}

function formatNumberLocale(num: number): string {
  return num.toLocaleString()
}

function connectionRatePercent(analytics: Analytics): string {
  if (analytics.calls_attempted <= 0) return "0.0"
  return ((analytics.calls_connected / analytics.calls_attempted) * 100).toFixed(1)
}

/**
 * Builds a PDF report for the current analytics snapshot and applied filters.
 */
export function downloadAnalyticsPdf(
  analytics: Analytics,
  filters: AnalyticsPdfFilters
): void {
  const doc = new jsPDF()
  const margin = 14
  let y = 15

  doc.setFontSize(16)
  doc.setFont("helvetica", "bold")
  doc.text("VoiceRA Analytics", margin, y)
  y += 10

  doc.setFont("helvetica", "normal")
  doc.setFontSize(10)
  doc.text(`Exported on: ${format(new Date(), "dd/MM/yyyy, hh:mm:ss a")}`, margin, y)
  y += 6
  doc.text(
    `Data as of: ${format(new Date(analytics.calculated_at), "PPp")}`,
    margin,
    y
  )
  y += 10

  doc.setFont("helvetica", "bold")
  doc.setFontSize(11)
  doc.text("Filters applied", margin, y)
  y += 6
  doc.setFont("helvetica", "normal")
  doc.setFontSize(10)
  const filterLines = [
    `Date range: ${filters.dateRangeLabel}`,
    `Agent: ${filters.agentLabel}`,
    `Phone number: ${filters.phoneLabel}`,
  ]
  for (const line of filterLines) {
    doc.text(line, margin, y)
    y += 5
  }
  y += 4

  if (analytics.calls_attempted === 0) {
    doc.setFont("helvetica", "normal")
    doc.setFontSize(10)
    doc.text(
      "No call data for the selected filters. Try widening the date range or clearing filters.",
      margin,
      y,
      { maxWidth: 180 }
    )
    doc.save(`analytics_${format(new Date(), "yyyy-MM-dd")}.pdf`)
    return
  }

  const rate = connectionRatePercent(analytics)

  doc.setFont("helvetica", "bold")
  doc.setFontSize(11)
  doc.text("Summary metrics", margin, y)
  y += 6
  doc.setFont("helvetica", "normal")
  doc.setFontSize(10)

  const metricLines = [
    `Connection rate: ${rate}%`,
    `Calls attempted: ${formatNumberLocale(analytics.calls_attempted)}`,
    `Calls connected: ${formatNumberLocale(analytics.calls_connected)}`,
    `Calls failed: ${formatNumberLocale(
      analytics.calls_attempted - analytics.calls_connected
    )}`,
    `Average call duration: ${formatDurationMinutes(analytics.average_call_duration)}`,
    `Total minutes connected: ${formatNumberLocale(
      Math.round(analytics.total_minutes_connected)
    )}`,
  ]
  if (analytics.most_used_agent) {
    metricLines.push(
      `Most used agent: ${analytics.most_used_agent} (${formatNumberLocale(
        analytics.most_used_agent_count
      )} calls)`
    )
  }

  for (const line of metricLines) {
    doc.text(line, margin, y)
    y += 5
  }

  y += 4

  const rows =
    analytics.agent_breakdown?.map((a) => [a.agent_type, String(a.call_count)]) ?? []

  if (rows.length === 0) {
    doc.setFont("helvetica", "italic")
    doc.setFontSize(9)
    doc.text("No per-agent breakdown rows returned for this report.", margin, y)
    y += 6
    doc.setFont("helvetica", "normal")
  }

  autoTable(doc, {
    head: [["Agent", "Calls"]],
    body: rows.length > 0 ? rows : [["—", "—"]],
    startY: y,
    styles: { fontSize: 9 },
    headStyles: { fillColor: [15, 23, 42] },
    alternateRowStyles: { fillColor: [248, 250, 252] },
    margin: { left: margin, right: margin },
  })

  doc.save(`analytics_${format(new Date(), "yyyy-MM-dd")}.pdf`)
}
