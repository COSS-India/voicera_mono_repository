"use client"

import { useState, useEffect, useMemo } from "react"
import { Separator } from "@/components/ui/separator"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import { 
  Phone, 
  PhoneCall, 
  Clock, 
  BarChart3,
  Loader2,
  TrendingUp,
  Users,
  Calendar as CalendarIcon,
  List,
  Download,
} from "lucide-react"
import { Calendar } from "@/components/ui/calendar"
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
} from "recharts"
import { getAnalytics, getAgents, fetchApiRoute, type Analytics, type Agent } from "@/lib/api"
import { getOrgId } from "@/lib/api"
import { format } from "date-fns"
import { cn } from "@/lib/utils"
import { downloadAnalyticsPdf } from "@/lib/export-analytics-pdf"

interface PhoneNumber {
  id?: string
  _id?: string
  phone_number: string
  provider: string
  agent_type?: string
  org_id: string
}

export default function AnalyticsPage() {
  const [analytics, setAnalytics] = useState<Analytics | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedAgent, setSelectedAgent] = useState<string>("all")
  const [selectedPhoneNumber, setSelectedPhoneNumber] = useState<string>("all")
  const [agents, setAgents] = useState<Agent[]>([])
  const [phoneNumbers, setPhoneNumbers] = useState<PhoneNumber[]>([])
  const [dateRange, setDateRange] = useState<{ from: Date | undefined; to: Date | undefined }>({
    from: undefined,
    to: undefined,
  })
  const [phoneNumberSearchOpen, setPhoneNumberSearchOpen] = useState(false)
  const [phoneNumberSearch, setPhoneNumberSearch] = useState("")
  const [agentSearchOpen, setAgentSearchOpen] = useState(false)
  const [agentSearch, setAgentSearch] = useState("")
  const [dateRangeOpen, setDateRangeOpen] = useState(false)
  const [agentView, setAgentView] = useState<'chart' | 'list'>('chart')
  const orgId = getOrgId()

  // Fetch agents and phone numbers
  useEffect(() => {
    async function fetchInitialData() {
      if (!orgId) return

      try {
        // Fetch agents
        const agentsData = await getAgents(orgId)
        setAgents(agentsData)

        // Fetch phone numbers
        const phoneResponse = await fetchApiRoute(`/api/phone-numbers?org_id=${encodeURIComponent(orgId)}`)
        if (phoneResponse.ok) {
          const phoneData: PhoneNumber[] = await phoneResponse.json()
          setPhoneNumbers(phoneData)
        }
      } catch (err) {
        console.error("Error fetching initial data:", err)
      }
    }

    fetchInitialData()
  }, [orgId])

  // Fetch analytics data
  useEffect(() => {
    async function fetchAnalytics() {
      if (!orgId) {
        setLoading(false)
        return
      }

      try {
        setLoading(true)
        setError(null)

        const params: any = {}
        if (selectedAgent && selectedAgent !== "all") {
          params.agent_type = selectedAgent
        }
        if (selectedPhoneNumber && selectedPhoneNumber !== "all") {
          params.phone_number = selectedPhoneNumber
        }
        if (dateRange.from) {
          params.start_date = format(dateRange.from, "yyyy-MM-dd")
        }
        if (dateRange.to) {
          params.end_date = format(dateRange.to, "yyyy-MM-dd")
        }

        const analyticsData = await getAnalytics(params)
        setAnalytics(analyticsData)
      } catch (err) {
        console.error("Error fetching analytics:", err)
        setError(err instanceof Error ? err.message : "Failed to load analytics")
      } finally {
        setLoading(false)
      }
    }

    fetchAnalytics()
  }, [orgId, selectedAgent, selectedPhoneNumber, dateRange])

  // Format duration for display
  const formatDuration = (minutes: number): string => {
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

  // Format number with commas
  const formatNumber = (num: number): string => {
    return num.toLocaleString()
  }

  // Calculate connection rate
  const connectionRate = analytics
    ? ((analytics.calls_connected / analytics.calls_attempted) * 100).toFixed(1)
    : "0.0"

  // Prepare chart data
  const agentChartData = useMemo(() => {
    if (!analytics?.agent_breakdown) return []
    return analytics.agent_breakdown.slice(0, 10).map((agent) => ({
      name: agent.agent_type.length > 15 ? `${agent.agent_type.substring(0, 15)}...` : agent.agent_type,
      calls: agent.call_count,
      fullName: agent.agent_type,
    }))
  }, [analytics])

  const filteredPhoneNumbers = useMemo(() => {
    if (!phoneNumberSearch) return phoneNumbers
    return phoneNumbers.filter((phone) =>
      phone.phone_number.toLowerCase().includes(phoneNumberSearch.toLowerCase())
    )
  }, [phoneNumbers, phoneNumberSearch])

  const filteredAgents = useMemo(() => {
    if (!agentSearch) return agents
    return agents.filter((agent) =>
      agent.agent_type.toLowerCase().includes(agentSearch.toLowerCase())
    )
  }, [agents, agentSearch])

  const hasActiveFilters = selectedAgent !== "all" || 
    selectedPhoneNumber !== "all" || 
    dateRange.from || 
    dateRange.to

  const activeFilterCount = useMemo(() => {
    let count = 0
    if (selectedAgent !== "all") count++
    if (selectedPhoneNumber !== "all") count++
    if (dateRange.from) count++
    if (dateRange.to) count++
    return count
  }, [selectedAgent, selectedPhoneNumber, dateRange])

  const clearFilters = () => {
    setSelectedAgent("all")
    setSelectedPhoneNumber("all")
    setDateRange({ from: undefined, to: undefined })
  }

  const handleDownloadPdf = () => {
    if (!analytics) return

    let dateRangeLabel = "All time"
    if (dateRange.from) {
      dateRangeLabel = dateRange.to
        ? `${format(dateRange.from, "MMM d, yyyy")} – ${format(dateRange.to, "MMM d, yyyy")}`
        : format(dateRange.from, "MMM d, yyyy")
    }

    const agentLabel =
      selectedAgent === "all"
        ? "All agents"
        : agents.find((a) => a.agent_type === selectedAgent)?.agent_type ?? selectedAgent

    const phoneLabel =
      selectedPhoneNumber === "all"
        ? "All numbers"
        : phoneNumbers.find((p) => p.phone_number === selectedPhoneNumber)?.phone_number ??
          selectedPhoneNumber

    downloadAnalyticsPdf(analytics, {
      dateRangeLabel,
      agentLabel,
      phoneLabel,
    })
  }

  const canDownloadPdf = Boolean(orgId && analytics && !loading && !error)

  return (
    <div className="flex flex-col h-screen bg-slate-50">
      {/* Header */}
      <header className="border-b bg-white">
        <div className="flex h-16 items-center justify-between gap-4 px-6">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-5 w-5 text-slate-600" />
            <h1 className="text-lg font-semibold text-slate-900">Analytics</h1>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="gap-2"
            disabled={!canDownloadPdf}
            onClick={handleDownloadPdf}
          >
            <Download className="h-4 w-4" />
            Download PDF
          </Button>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6">
        <div className="max-w-7xl mx-auto space-y-4 md:space-y-6">
          {/* Filters Row - inline controls */}
          <div className="flex items-center justify-between gap-2 flex-wrap">
            <div className="flex items-center gap-2 flex-wrap">
              {/* Date Range */}
              <Popover open={dateRangeOpen} onOpenChange={setDateRangeOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="gap-2 h-10">
                    <CalendarIcon className="h-4 w-4" />
                    {dateRange.from ? (
                      dateRange.to ? (
                        `${format(dateRange.from, "MMM d")} - ${format(dateRange.to, "MMM d")}`
                      ) : (
                        format(dateRange.from, "MMM d, yyyy")
                      )
                    ) : (
                      "Date Range"
                    )}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-auto p-0" align="start">
                  <Calendar
                    mode="range"
                    selected={{ from: dateRange.from, to: dateRange.to }}
                    onSelect={(range) => {
                      setDateRange({
                        from: range?.from,
                        to: range?.to,
                      })
                    }}
                    numberOfMonths={2}
                  />
                  {dateRange.from && (
                    <div className="p-3 border-t">
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setDateRange({ from: undefined, to: undefined })
                          setDateRangeOpen(false)
                        }}
                        className="w-full"
                      >
                        Clear date
                      </Button>
                    </div>
                  )}
                </PopoverContent>
              </Popover>

              {/* Agent Type */}
              <Popover open={agentSearchOpen} onOpenChange={setAgentSearchOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="gap-2 h-10">
                    <Users className="h-4 w-4" />
                    {selectedAgent === "all" 
                      ? "All Agents" 
                      : agents.find(a => a.agent_type === selectedAgent)?.agent_type || "All Agents"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[300px] p-0" align="start">
                  <Command>
                    <CommandInput 
                      placeholder="Search agents..." 
                      value={agentSearch}
                      onValueChange={setAgentSearch}
                    />
                    <CommandList>
                      <CommandEmpty>No agents found.</CommandEmpty>
                      <CommandGroup>
                        <CommandItem
                          value="all"
                          onSelect={() => {
                            setSelectedAgent("all")
                            setAgentSearchOpen(false)
                          }}
                        >
                          All Agents
                        </CommandItem>
                        {filteredAgents.map((agent) => (
                          <CommandItem
                            key={agent.agent_type}
                            value={agent.agent_type}
                            onSelect={() => {
                              setSelectedAgent(agent.agent_type)
                              setAgentSearchOpen(false)
                            }}
                          >
                            {agent.agent_type}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>

              {/* Phone Number */}
              <Popover open={phoneNumberSearchOpen} onOpenChange={setPhoneNumberSearchOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="gap-2 h-10">
                    <Phone className="h-4 w-4" />
                    {selectedPhoneNumber === "all" 
                      ? "All Numbers" 
                      : phoneNumbers.find(p => p.phone_number === selectedPhoneNumber)?.phone_number || "All Numbers"}
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-[300px] p-0" align="start">
                  <Command>
                    <CommandInput 
                      placeholder="Search phone numbers..." 
                      value={phoneNumberSearch}
                      onValueChange={setPhoneNumberSearch}
                    />
                    <CommandList>
                      <CommandEmpty>No phone numbers found.</CommandEmpty>
                      <CommandGroup>
                        <CommandItem
                          value="all"
                          onSelect={() => {
                            setSelectedPhoneNumber("all")
                            setPhoneNumberSearchOpen(false)
                          }}
                        >
                          All Numbers
                        </CommandItem>
                        {filteredPhoneNumbers.map((phone) => (
                          <CommandItem
                            key={phone.phone_number}
                            value={phone.phone_number}
                            onSelect={() => {
                              setSelectedPhoneNumber(phone.phone_number)
                              setPhoneNumberSearchOpen(false)
                            }}
                          >
                            {phone.phone_number}
                          </CommandItem>
                        ))}
                      </CommandGroup>
                    </CommandList>
                  </Command>
                </PopoverContent>
              </Popover>

              {/* Clear All Button */}
              {hasActiveFilters && (
                <Button 
                  variant="ghost" 
                  onClick={clearFilters} 
                  className="h-10"
                >
                  Clear all
                </Button>
              )}
            </div>
            
            {analytics && (
              <p className="text-sm text-slate-500">
                Updated {format(new Date(analytics.calculated_at), "PPp")}
              </p>
            )}
          </div>

          {/* Loading State */}
          {loading && (
            <div className="flex items-center justify-center py-24">
              <Loader2 className="h-8 w-8 text-slate-400 animate-pulse" />
            </div>
          )}

          {/* Error State */}
          {error && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="pt-6">
                <p className="text-red-600">{error}</p>
              </CardContent>
            </Card>
          )}

          {/* Analytics Content */}
          {!loading && !error && analytics && (
            <>
              {/* Hero Metric - Typography Focused */}
              {analytics.calls_attempted > 0 && (
                <div className="py-6">
                  <div className="flex items-baseline gap-3">
                    <span className="text-5xl font-bold text-slate-900">{connectionRate}%</span>
                    <span className="text-lg text-slate-500">connection rate</span>
                  </div>
                  <div className="flex items-center gap-4 mt-2">
                    <span className="text-sm text-slate-500">
                      {formatNumber(analytics.calls_connected)} connected · {formatNumber(analytics.calls_attempted - analytics.calls_connected)} failed · {formatDuration(analytics.total_minutes_connected)} total
                    </span>
                    {/* Only show color for trend indicator */}
                    <span className="inline-flex items-center gap-1 text-sm text-green-600">
                      <TrendingUp className="h-3.5 w-3.5" />
                      12% vs last week
                    </span>
                  </div>
                </div>
              )}

              {/* Key Metrics Grid - 2 cols mobile, 4 cols desktop */}
              <div className="grid grid-cols-2 md:grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
                <Card className="border-slate-200 hover:shadow-md transition-shadow">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-slate-600">
                      Calls Attempted
                    </CardTitle>
                    <div className="h-10 w-10 rounded-lg bg-slate-100 flex items-center justify-center">
                      <Phone className="h-5 w-5 text-slate-500" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-slate-900">
                      {formatNumber(analytics.calls_attempted)}
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Total call attempts</p>
                  </CardContent>
                </Card>

                <Card className="border-slate-200 hover:shadow-md transition-shadow">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-slate-600">
                      Calls Connected
                    </CardTitle>
                    <div className="h-10 w-10 rounded-lg bg-slate-100 flex items-center justify-center">
                      <PhoneCall className="h-5 w-5 text-slate-500" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-slate-900">
                      {formatNumber(analytics.calls_connected)}
                    </div>
                    <p className={cn(
                      "text-xs mt-1 flex items-center gap-1",
                      Number(connectionRate) >= 70 
                        ? "text-green-600" 
                        : Number(connectionRate) >= 50 
                        ? "text-amber-600" 
                        : "text-red-600"
                    )}>
                      <span className={cn(
                        "h-1.5 w-1.5 rounded-full",
                        Number(connectionRate) >= 70 
                          ? "bg-green-500" 
                          : Number(connectionRate) >= 50 
                          ? "bg-amber-500" 
                          : "bg-red-500"
                      )} />
                      {connectionRate}% connected
                    </p>
                  </CardContent>
                </Card>

                <Card className="border-slate-200 hover:shadow-md transition-shadow">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-slate-600">
                      Avg Call Duration
                    </CardTitle>
                    <div className="h-10 w-10 rounded-lg bg-slate-100 flex items-center justify-center">
                      <Clock className="h-5 w-5 text-slate-500" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-slate-900">
                      {formatDuration(analytics.average_call_duration)}
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Per connected call</p>
                  </CardContent>
                </Card>

                <Card className="border-slate-200 hover:shadow-md transition-shadow">
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium text-slate-600">
                      Total Minutes
                    </CardTitle>
                    <div className="h-10 w-10 rounded-lg bg-slate-100 flex items-center justify-center">
                      <TrendingUp className="h-5 w-5 text-slate-500" />
                    </div>
                  </CardHeader>
                  <CardContent>
                    <div className="text-3xl font-bold text-slate-900">
                      {formatNumber(Math.round(analytics.total_minutes_connected))}
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Minutes connected</p>
                  </CardContent>
                </Card>
              </div>

              {/* Charts Section */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 md:gap-6">
                {/* Agent Distribution Card with Toggle */}
                <Card className="border-slate-200">
                  <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2 text-base">
                        <Users className="h-5 w-5 text-slate-500" />
                        Agent Performance
                      </CardTitle>
                      <CardDescription>Call distribution by agent</CardDescription>
                    </div>
                    <div className="flex gap-1 bg-slate-100 p-1 rounded-lg">
                      <Button 
                        variant={agentView === 'chart' ? 'default' : 'ghost'} 
                        size="sm"
                        className={cn(
                          "h-8 px-3",
                          agentView === 'chart' 
                            ? "bg-white text-slate-900" 
                            : "text-slate-500 hover:text-slate-700"
                        )}
                        onClick={() => setAgentView('chart')}
                      >
                        <BarChart3 className="h-4 w-4" />
                      </Button>
                      <Button 
                        variant={agentView === 'list' ? 'default' : 'ghost'} 
                        size="sm"
                        className={cn(
                          "h-8 px-3",
                          agentView === 'list' 
                            ? "bg-white text-slate-900" 
                            : "text-slate-500 hover:text-slate-700"
                        )}
                        onClick={() => setAgentView('list')}
                      >
                        <List className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent>
                    {/* Top Agent Highlight */}
                    {analytics.most_used_agent && (
                      <div className="mb-4 p-3 bg-slate-50 rounded-lg flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <div className="h-8 w-8 rounded-full bg-slate-200 flex items-center justify-center">
                            <span className="text-slate-700 text-sm font-bold">1</span>
                          </div>
                          <div>
                            <p className="font-medium text-slate-900">{analytics.most_used_agent}</p>
                            <p className="text-xs text-slate-500">Top performer</p>
                          </div>
                        </div>
                        <span className="text-sm font-semibold text-slate-700">
                          {formatNumber(analytics.most_used_agent_count)} calls
                        </span>
                      </div>
                    )}
                    
                    {/* Chart or List View */}
                    {agentView === 'chart' ? (
                      agentChartData.length > 0 ? (
                        <ResponsiveContainer width="100%" height={250}>
                          <BarChart data={agentChartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
                            <XAxis dataKey="name" tick={{ fontSize: 11 }} angle={-45} textAnchor="end" height={70} />
                            <YAxis tick={{ fontSize: 11 }} />
                            <Tooltip 
                              contentStyle={{ 
                                backgroundColor: "white", 
                                border: "1px solidrgb(13, 96, 204)",
                                borderRadius: "8px",
                                fontSize: "12px"
                              }}
                              formatter={(value: any, name: string | undefined, props: any) => [
                                `${value} calls`,
                                props.payload.fullName
                              ]}
                            />
                            {/* Changed fill to a light blue */}
                            <Bar dataKey="calls" fill="#2563eb" radius={[4, 4, 0, 0]} />
                          </BarChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-[250px] flex items-center justify-center text-slate-500">
                          No agent data available
                        </div>
                      )
                    ) : (
                      analytics.agent_breakdown && analytics.agent_breakdown.length > 0 ? (
                        <div className="space-y-3">
                          {analytics.agent_breakdown.slice(0, 6).map((agent) => (
                            <div key={agent.agent_type} className="space-y-1.5">
                              <div className="flex items-center justify-between text-sm">
                                <span className="font-medium text-slate-700 truncate max-w-[200px]">
                                  {agent.agent_type}
                                </span>
                                <span className="text-slate-900 font-semibold">
                                  {formatNumber(agent.call_count)}
                                </span>
                              </div>
                              <div className="h-1.5 bg-slate-100 rounded-full overflow-hidden">
                                <div 
                                  className="h-full rounded-full transition-all duration-500"
                                  style={{ 
                                    width: `${(agent.call_count / analytics.calls_attempted) * 100}%`,
                                    backgroundColor: "#2563eb" // Light blue bar
                                  }}
                                />
                              </div>
                            </div>
                          ))}
                        </div>
                      ) : (
                        <div className="h-[250px] flex items-center justify-center text-slate-500">
                          No agent data available
                        </div>
                      )
                    )}
                  </CardContent>
                </Card>

                {/* Connection Rate Card (replaces pie chart) */}
                <Card className="border-slate-200">
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-base">
                      <PhoneCall className="h-5 w-5 text-slate-500" />
                      Connection Breakdown
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-6">
                    {/* Main percentage display */}
                    <div className="text-center">
                      <div className="text-5xl font-bold text-slate-900">{connectionRate}%</div>
                      <p className="text-sm text-slate-500 mt-1">of calls connected</p>
                    </div>
                    
                    {/* Progress bar */}
                    <div className="space-y-2">
                      <div className="h-3 bg-slate-100 rounded-full overflow-hidden">
                        <div 
                          className="h-full bg-slate-400 rounded-full transition-all duration-500"
                          style={{ width: `${connectionRate}%` }}
                        />
                      </div>
                    </div>
                    
                    {/* Stats row */}
                    <div className="flex items-center justify-center gap-6 pt-4 text-sm">
                      <div className="text-center">
                        <div className="text-2xl font-semibold text-slate-900">
                          {formatNumber(analytics.calls_connected)}
                        </div>
                        <p className="text-slate-500">connected</p>
                      </div>
                      <div className="h-8 w-px bg-slate-200" />
                      <div className="text-center">
                        <div className="text-2xl font-semibold text-slate-900">
                          {formatNumber(analytics.calls_attempted - analytics.calls_connected)}
                        </div>
                        <p className="text-slate-500">failed</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </div>

            </>
          )}

          {/* Empty State */}
          {!loading && !error && analytics && analytics.calls_attempted === 0 && (
            <Card className="border-slate-200">
              <CardContent className="py-16 text-center">
                <div className="h-16 w-16 rounded-full bg-slate-100 flex items-center justify-center mx-auto mb-4">
                  <BarChart3 className="h-8 w-8 text-slate-500" />
                </div>
                <p className="text-slate-900 font-medium mb-1">No call data yet</p>
                <p className="text-sm text-slate-500">
                  Start making calls to see your analytics here
                </p>
              </CardContent>
            </Card>
          )}
        </div>
      </main>
    </div>
  )
}
