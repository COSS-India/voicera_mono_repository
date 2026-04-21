"use client"

import { useEffect, useMemo, useState } from "react"
import { Activity, Cpu, Loader2 } from "lucide-react"
import { PieChart, Pie, Cell, ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip } from "recharts"
import { fetchApiRoute } from "@/lib/api"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"

interface GpuProcess {
  pid: number | null
  name: string
  memory_mb: number
}

interface GpuInfo {
  index: number
  model: string
  utilization_percent: number
  memory_total_mb: number
  memory_used_mb: number
  memory_free_mb: number
  temperature_c: number | null
  power_w: number | null
  power_limit_w: number | null
  processes: GpuProcess[]
}

interface TelemetryPayload {
  status: "ok" | "unavailable" | "error"
  reason?: string
  timestamp_utc?: string
  gpu_count: number
  gpus: GpuInfo[]
}

const POLL_INTERVAL_MS = 4000
const MAX_POINTS = 40

export default function TelemetryPage() {
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [telemetry, setTelemetry] = useState<TelemetryPayload | null>(null)
  const [history, setHistory] = useState<Record<number, Array<{ t: number; utilization: number }>>>({})

  useEffect(() => {
    let timer: ReturnType<typeof setInterval> | null = null
    let cancelled = false

    const fetchTelemetry = async () => {
      try {
        const response = await fetchApiRoute("/api/telemetry/gpu", { method: "GET" })
        const data: TelemetryPayload = await response.json()
        if (!response.ok) throw new Error(data?.reason || data?.status || "Failed to fetch telemetry")
        if (cancelled) return

        setTelemetry(data)
        setError(null)

        if (data.status === "ok" && data.gpus.length > 0) {
          const now = Date.now()
          setHistory((prev) => {
            const next = { ...prev }
            for (const gpu of data.gpus) {
              const points = [...(next[gpu.index] || []), { t: now, utilization: gpu.utilization_percent }]
              next[gpu.index] = points.slice(-MAX_POINTS)
            }
            return next
          })
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "Failed to fetch telemetry")
        }
      } finally {
        if (!cancelled) setLoading(false)
      }
    }

    fetchTelemetry()
    timer = setInterval(fetchTelemetry, POLL_INTERVAL_MS)

    return () => {
      cancelled = true
      if (timer) clearInterval(timer)
    }
  }, [])

  const lastUpdated = useMemo(() => {
    if (!telemetry?.timestamp_utc) return "N/A"
    return new Date(telemetry.timestamp_utc).toLocaleTimeString()
  }, [telemetry?.timestamp_utc])

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      <header className="border-b bg-white">
        <div className="flex h-16 items-center justify-between px-6">
          <div className="flex items-center gap-2">
            <Activity className="h-5 w-5 text-slate-700" />
            <h1 className="text-lg font-semibold text-slate-900">Telemetry</h1>
          </div>
          <Badge variant="outline" className="text-xs">
            Last update: {lastUpdated}
          </Badge>
        </div>
      </header>

      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="mx-auto max-w-7xl space-y-4">
          {loading && (
            <Card>
              <CardContent className="py-10 flex items-center justify-center gap-2 text-slate-600">
                <Loader2 className="h-4 w-4 animate-spin" />
                Loading GPU telemetry...
              </CardContent>
            </Card>
          )}

          {!loading && error && (
            <Card className="border-red-200">
              <CardContent className="py-6 text-sm text-red-700">{error}</CardContent>
            </Card>
          )}

          {!loading && telemetry?.status === "unavailable" && (
            <Card>
              <CardHeader>
                <CardTitle className="text-base">GPU Telemetry Unavailable</CardTitle>
                <CardDescription>{telemetry.reason || "No NVIDIA GPU detected on this host."}</CardDescription>
              </CardHeader>
            </Card>
          )}

          {!loading && telemetry?.status === "ok" && telemetry.gpus.length === 0 && (
            <Card>
              <CardContent className="py-6 text-sm text-slate-600">No GPUs found on this host.</CardContent>
            </Card>
          )}

          {!loading && telemetry?.status === "ok" && telemetry.gpus.length > 0 && (
            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              {telemetry.gpus.map((gpu) => {
                const pieData = [
                  { name: "Used", value: gpu.memory_used_mb },
                  { name: "Free", value: gpu.memory_free_mb },
                ]
                const graphData = (history[gpu.index] || []).map((p) => ({
                  time: p.t,
                  utilization: p.utilization,
                }))

                return (
                  <Card key={gpu.index} className="overflow-hidden">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between gap-3">
                        <div className="min-w-0">
                          <CardTitle className="text-base truncate">GPU {gpu.index}: {gpu.model}</CardTitle>
                          <CardDescription>
                            Utilization {gpu.utilization_percent}% · Memory {gpu.memory_used_mb}/{gpu.memory_total_mb} MB
                          </CardDescription>
                        </div>
                        <Cpu className="h-5 w-5 text-slate-500 shrink-0" />
                      </div>
                    </CardHeader>

                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                        <div className="h-44">
                          <ResponsiveContainer width="100%" height="100%">
                            <PieChart>
                              <Pie data={pieData} dataKey="value" innerRadius={45} outerRadius={70} paddingAngle={3}>
                                <Cell fill="#111827" />
                                <Cell fill="#cbd5e1" />
                              </Pie>
                              <Tooltip formatter={(value: number) => `${value} MB`} />
                            </PieChart>
                          </ResponsiveContainer>
                        </div>

                        <div className="h-44">
                          <ResponsiveContainer width="100%" height="100%">
                            <AreaChart data={graphData}>
                              <XAxis
                                dataKey="time"
                                tickFormatter={(value) => new Date(value).toLocaleTimeString()}
                                hide
                              />
                              <YAxis domain={[0, 100]} />
                              <Tooltip labelFormatter={(value) => new Date(value as number).toLocaleTimeString()} />
                              <Area type="monotone" dataKey="utilization" stroke="#111827" fill="#11182722" />
                            </AreaChart>
                          </ResponsiveContainer>
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-2 text-xs text-slate-700">
                        <div>Temp: {gpu.temperature_c ?? "N/A"} C</div>
                        <div>Power: {gpu.power_w ?? "N/A"} W</div>
                      </div>

                      <div className="rounded-md border">
                        <div className="px-3 py-2 text-xs font-medium border-b bg-slate-50">
                          Running Processes ({gpu.processes.length})
                        </div>
                        {gpu.processes.length === 0 ? (
                          <div className="px-3 py-3 text-xs text-slate-500">No active compute processes.</div>
                        ) : (
                          <div className="max-h-40 overflow-auto">
                            <table className="w-full text-xs">
                              <thead className="bg-slate-50">
                                <tr>
                                  <th className="px-3 py-2 text-left">PID</th>
                                  <th className="px-3 py-2 text-left">Process</th>
                                  <th className="px-3 py-2 text-right">VRAM</th>
                                </tr>
                              </thead>
                              <tbody>
                                {gpu.processes.map((p, i) => (
                                  <tr key={`${p.pid}-${i}`} className="border-t">
                                    <td className="px-3 py-2">{p.pid ?? "-"}</td>
                                    <td className="px-3 py-2 truncate max-w-[220px]" title={p.name}>
                                      {p.name}
                                    </td>
                                    <td className="px-3 py-2 text-right">{p.memory_mb} MB</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
