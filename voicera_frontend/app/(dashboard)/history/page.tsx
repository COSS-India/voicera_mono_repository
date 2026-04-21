"use client"

import { useState, useEffect, useMemo, Suspense } from "react"
import { useRouter, useSearchParams } from "next/navigation"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { getMeetings, getMeetingDetails, type Meeting, type MeetingDetails } from "@/lib/api"
import { MeetingDetailSheet } from "@/components/history/meeting-detail-sheet"
import {
  Calendar as CalendarIcon,
  SlidersHorizontal,
  Download,
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Plus,
  X,
  ArrowUpDown,
  Filter,
} from "lucide-react"
import { Calendar } from "@/components/ui/calendar"
import { format } from "date-fns"
import { toZonedTime } from "date-fns-tz"
import jsPDF from "jspdf"
import autoTable from "jspdf-autotable"

function HistoryPageContent() {
  const router = useRouter()
  const searchParams = useSearchParams()

  const [meetings, setMeetings] = useState<Meeting[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [currentPage, setCurrentPage] = useState(1)
  const [itemsPerPage] = useState(100)

  // Sheet state
  const [selectedMeeting, setSelectedMeeting] = useState<Meeting | null>(null)
  const [meetingDetails, setMeetingDetails] = useState<MeetingDetails | null>(null)
  const [isSheetOpen, setIsSheetOpen] = useState(false)
  const [isLoadingDetails, setIsLoadingDetails] = useState(false)

  // Filter state
  const [activeFilters, setActiveFilters] = useState<Array<{ field: string; value: string }>>([])
  const [filterDropdownOpen, setFilterDropdownOpen] = useState(false)
  const [addingFilter, setAddingFilter] = useState<string | null>(null) // Which filter type is being added
  const [filterInputValue, setFilterInputValue] = useState("")

  // Date range state
  const [dateRange, setDateRange] = useState<{ from: Date | undefined; to: Date | undefined }>({ from: undefined, to: undefined })
  const [dateRangeOpen, setDateRangeOpen] = useState(false)

  // Sort state - 'latest' means latest to oldest (default), 'oldest' means oldest to latest
  const [dateSortOrder, setDateSortOrder] = useState<'latest' | 'oldest'>('latest')
  const [durationSortOrder, setDurationSortOrder] = useState<'longest' | 'shortest' | null>(null)

  // Filter popover states
  const [assistantNameFilterOpen, setAssistantNameFilterOpen] = useState(false)
  const [callStatusFilterOpen, setCallStatusFilterOpen] = useState(false)
  const [callTypeFilterOpen, setCallTypeFilterOpen] = useState(false)
  const [fromNumberFilterOpen, setFromNumberFilterOpen] = useState(false)
  const [toNumberFilterOpen, setToNumberFilterOpen] = useState(false)

  // Agent types and phone numbers for filters (extracted from meetings)
  const [uniqueAgentTypes, setUniqueAgentTypes] = useState<string[]>([])
  const [uniqueFromNumbers, setUniqueFromNumbers] = useState<string[]>([])
  const [uniqueToNumbers, setUniqueToNumbers] = useState<string[]>([])

  // Client-side mount state to prevent hydration mismatch
  const [mounted, setMounted] = useState(false)

  // Initialize filters from URL on mount
  useEffect(() => {
    const assistantNameParam = searchParams.get('assistant_name')
    if (assistantNameParam) {
      setActiveFilters([{ field: 'assistant_name', value: assistantNameParam }])
    }
  }, [searchParams])

  // Set mounted to true after component mounts on client
  useEffect(() => {
    setMounted(true)
  }, [])

  // Fetch meetings and extract filter options
  useEffect(() => {
    async function fetchData() {
      try {
        const meetingsData = await getMeetings()
        setMeetings(meetingsData)

        // Extract unique agent types, phone numbers from meetings
        const agentTypes = new Set<string>()
        const fromNumbers = new Set<string>()
        const toNumbers = new Set<string>()

        meetingsData.forEach(meeting => {
          if (meeting.agent_type) agentTypes.add(meeting.agent_type)
          if (meeting.from_number) fromNumbers.add(meeting.from_number)
          if (meeting.to_number) toNumbers.add(meeting.to_number)
        })

        setUniqueAgentTypes(Array.from(agentTypes).sort())
        setUniqueFromNumbers(Array.from(fromNumbers).sort())
        setUniqueToNumbers(Array.from(toNumbers).sort())
      } catch (error) {
        console.error("Failed to fetch meetings:", error)
      } finally {
        setIsLoading(false)
      }
    }

    fetchData()
  }, [])

  // Filter logic with date range and sorting
  const filteredMeetings = useMemo(() => {
    let filtered = meetings

    // Apply date range filter
    if (dateRange.from || dateRange.to) {
      filtered = filtered.filter(meeting => {
        const meetingDate = meeting.start_time_utc || meeting.created_at
        if (!meetingDate) return false

        try {
          const date = new Date(meetingDate)
          if (dateRange.from) {
            const fromDate = new Date(dateRange.from)
            fromDate.setHours(0, 0, 0, 0)
            if (date < fromDate) return false
          }
          if (dateRange.to) {
            const toDate = new Date(dateRange.to)
            toDate.setHours(23, 59, 59, 999)
            if (date > toDate) return false
          }
          return true
        } catch {
          return false
        }
      })
    }

    // Apply other filters
    if (activeFilters.length > 0) {
      filtered = filtered.filter(meeting => {
        return activeFilters.every(filter => {
          switch (filter.field) {
            case 'assistant_name':
              return meeting.agent_type === filter.value
            case 'call_status':
              const status = meeting.call_busy ? "Busy" : meeting.end_time_utc ? "Completed" : "In Progress"
              return status.toLowerCase() === filter.value.toLowerCase()
            case 'call_type':
              const isInbound = filter.value.toLowerCase() === 'inbound'
              return meeting.inbound === isInbound
            case 'from_number':
              return meeting.from_number === filter.value
            case 'to_number':
              return meeting.to_number === filter.value

            default:
              return true
          }
        })
      })
    }

    // Helper function to get duration in seconds
    const getDurationInSeconds = (meeting: Meeting): number => {
      if (meeting.call_busy) return -1 // Busy calls have no duration, put them at the end
      if (meeting.duration !== null && meeting.duration !== undefined) {
        return meeting.duration
      }
      if (meeting.start_time_utc && meeting.end_time_utc) {
        try {
          const start = new Date(meeting.start_time_utc).getTime()
          const end = new Date(meeting.end_time_utc).getTime()
          if (!isNaN(start) && !isNaN(end) && end > start) {
            return (end - start) / 1000
          }
        } catch {
          // If calculation fails, return -1
        }
      }
      return -1 // No duration available
    }

    // Apply sorting by duration if duration sort is active
    if (durationSortOrder) {
      filtered = [...filtered].sort((a, b) => {
        const durationA = getDurationInSeconds(a)
        const durationB = getDurationInSeconds(b)

        // Put calls with no duration (-1) at the end
        if (durationA === -1 && durationB === -1) return 0
        if (durationA === -1) return 1
        if (durationB === -1) return -1

        if (durationSortOrder === 'longest') {
          // Longest to shortest (descending)
          return durationB - durationA
        } else {
          // Shortest to longest (ascending)
          return durationA - durationB
        }
      })
    } else {
      // Apply sorting by date (default when duration sort is not active)
      filtered = [...filtered].sort((a, b) => {
        const dateA = a.start_time_utc || a.created_at
        const dateB = b.start_time_utc || b.created_at

        if (!dateA && !dateB) return 0
        if (!dateA) return 1
        if (!dateB) return -1

        try {
          const timeA = new Date(dateA).getTime()
          const timeB = new Date(dateB).getTime()

          if (dateSortOrder === 'latest') {
            // Latest to oldest (descending)
            return timeB - timeA
          } else {
            // Oldest to latest (ascending)
            return timeA - timeB
          }
        } catch {
          return 0
        }
      })
    }

    return filtered
  }, [meetings, activeFilters, dateRange, dateSortOrder, durationSortOrder])

  // Pagination
  const paginatedMeetings = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage
    const end = start + itemsPerPage
    return filteredMeetings.slice(start, end)
  }, [filteredMeetings, currentPage, itemsPerPage])

  // Reset to page 1 when filters or date range change
  useEffect(() => {
    setCurrentPage(1)
  }, [activeFilters, dateRange])

  // Filter helper functions
  const filterOptions = [
    { id: 'assistant_name', label: 'Agent Name' },
    { id: 'call_status', label: 'Call Status' },
    { id: 'call_type', label: 'Call Type' },
    { id: 'from_number', label: 'From Number' },
    { id: 'to_number', label: 'To Number' },
  ]

  const getFilterLabel = (field: string): string => {
    return filterOptions.find(f => f.id === field)?.label || field
  }

  const addFilter = (field: string, value: string) => {
    if (!value.trim()) return
    setActiveFilters(prev => {
      const existing = prev.findIndex(f => f.field === field)
      if (existing >= 0) {
        const updated = [...prev]
        updated[existing] = { field, value }
        return updated
      }
      return [...prev, { field, value }]
    })
    setAddingFilter(null)
    setFilterInputValue("")
    setFilterDropdownOpen(false)
  }

  const removeFilter = (index: number) => {
    const filterToRemove = activeFilters[index]
    setActiveFilters(prev => prev.filter((_, i) => i !== index))
    if (filterToRemove.field === 'assistant_name') {
      router.push('/history')
    }
  }

  const clearAllFilters = () => {
    setActiveFilters([])
    setDateRange({ from: undefined, to: undefined })
    router.push('/history')
  }

  // Handler functions for table column interactions
  const handleDateSort = () => {
    setDateSortOrder(prev => prev === 'latest' ? 'oldest' : 'latest')
    // Reset duration sort when activating date sort
    setDurationSortOrder(null)
  }

  const handleDurationSort = () => {
    setDurationSortOrder(prev => {
      if (prev === null) {
        // Reset date sort when activating duration sort
        setDateSortOrder('latest')
        return 'longest'
      }
      if (prev === 'longest') return 'shortest'
      return null
    })
  }

  const handleAssistantNameFilter = () => {
    setAssistantNameFilterOpen(true)
  }

  const handleCallStatusFilter = () => {
    setCallStatusFilterOpen(true)
  }

  // Format helper functions - Updated to use IST timezone
  const formatDate = (dateString?: string) => {
    if (!dateString) return "-"
    try {
      // Ensure the timestamp is treated as UTC if no timezone specified
      let utcString = dateString
      if (!dateString.endsWith('Z') && !dateString.includes('+') && !dateString.includes('-', 10)) {
        utcString = dateString + 'Z'
      }
      const date = new Date(utcString)
      // Convert UTC to IST (Asia/Kolkata)
      const istTime = toZonedTime(date, "Asia/Kolkata")
      return format(istTime, "dd/MM/yy, hh:mm:ss a")
    } catch {
      return dateString
    }
  }

  const formatDuration = (duration: number | null | undefined, startTime?: string, endTime?: string) => {
    let totalSeconds: number | null | undefined = duration;

    // If duration is null or undefined, calculate from start_time_utc and end_time_utc
    if ((totalSeconds === null || totalSeconds === undefined) && startTime && endTime) {
      try {
        const start = new Date(startTime).getTime()
        const end = new Date(endTime).getTime()
        if (!isNaN(start) && !isNaN(end) && end > start) {
          totalSeconds = (end - start) / 1000 // Convert milliseconds to seconds
        }
      } catch {
        // If calculation fails, keep totalSeconds as null/undefined
      }
    }

    if (typeof totalSeconds === 'number' && isFinite(totalSeconds) && totalSeconds >= 0) {
      const rounding = Math.round(Number(totalSeconds));
      const mins = Math.floor(rounding / 60);
      const secs = rounding % 60;

      if (mins > 0) {
        // display as "Xm Ys"
        let result = '';
        result += `${mins}m`;
        if (secs > 0) {
          result += ` ${secs}s`;
        }
        return result.trim();
      } else {
        // just seconds
        return `${secs}s`;
      }
    }

    return "N/A";
  }

  // Export functions
  const prepareExportData = () => {
    return filteredMeetings.map(meeting => ({
      "Agent Name": meeting.agent_type || "-",
      "To Number": meeting.to_number || "-",
      "From Number": meeting.from_number || "-",
      "Call Type": meeting.inbound ? "Inbound" : "Outbound",
      "Called On": formatDate(meeting.start_time_utc || meeting.created_at),
      "Call Status": meeting.call_busy
        ? "Busy"
        : meeting.end_time_utc
          ? "Completed"
          : "In Progress",
      "Call Duration": meeting.call_busy ? "N/A" : formatDuration(meeting.duration, meeting.start_time_utc, meeting.end_time_utc),
    }))
  }

  const exportToCSV = () => {
    const data = prepareExportData()
    if (data.length === 0) {
      alert("No data to export")
      return
    }

    const headers = Object.keys(data[0])
    const csvRows = [
      headers.join(","),
      ...data.map(row =>
        headers.map(header => {
          const value = row[header as keyof typeof row]
          // Escape commas and quotes in CSV
          if (typeof value === 'string' && (value.includes(',') || value.includes('"') || value.includes('\n'))) {
            return `"${value.replace(/"/g, '""')}"`
          }
          return value
        }).join(",")
      )
    ]

    const csvContent = csvRows.join("\n")
    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" })
    const link = document.createElement("a")
    const url = URL.createObjectURL(blob)
    link.setAttribute("href", url)
    link.setAttribute("download", `meetings_export_${format(new Date(), "yyyy-MM-dd")}.csv`)
    link.style.visibility = "hidden"
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  const exportToPDF = () => {
    const data = prepareExportData()
    if (data.length === 0) {
      alert("No data to export")
      return
    }

    const doc = new jsPDF()

    // Add title
    doc.setFontSize(16)
    doc.text("Meetings History", 14, 15)

    // Add export date
    doc.setFontSize(10)
    doc.text(`Exported on: ${format(new Date(), "dd/MM/yyyy, hh:mm:ss a")}`, 14, 22)

    // Prepare table data
    const headers = Object.keys(data[0])
    const rows = data.map(row =>
      headers.map(header => String(row[header as keyof typeof row]))
    )

    // Add table
    autoTable(doc, {
      head: [headers],
      body: rows,
      startY: 28,
      styles: { fontSize: 8 },
      headStyles: { fillColor: [15, 23, 42] }, // slate-900
      alternateRowStyles: { fillColor: [248, 250, 252] }, // slate-50
      margin: { top: 28, left: 14, right: 14 },
    })

    // Save PDF
    doc.save(`meetings_export_${format(new Date(), "yyyy-MM-dd")}.pdf`)
  }

  const handleMeetingClick = async (meeting: Meeting) => {
    setSelectedMeeting(meeting)
    setIsSheetOpen(true)
    setIsLoadingDetails(true)

    try {
      // Fetch detailed meeting data including transcript
      const details = await getMeetingDetails(meeting.meeting_id)
      setMeetingDetails(details)
    } catch (error) {
      console.error("Failed to fetch meeting details:", error)
      // Still show sheet with basic meeting data
      setMeetingDetails(null)
    } finally {
      setIsLoadingDetails(false)
    }
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header */}
      <header className="flex h-14 items-center gap-4 border-b border-slate-200 bg-white px-5 lg:px-8 sticky top-0 z-10">
        <h1 className="text-xl font-semibold text-slate-900">History</h1>
      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto">
        {/* Action Bar */}
        <div className="flex items-center justify-between px-5 lg:px-8 py-4 bg-white border-b border-slate-200">
          <div className="flex items-center gap-3">
            {/* Date Range Button */}
            {mounted && (
              <Popover open={dateRangeOpen} onOpenChange={setDateRangeOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="h-10 px-4 gap-2 rounded-lg border-slate-200">
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
                      if (range?.from && range?.to) {
                        setDateRangeOpen(false)
                      }
                    }}
                    numberOfMonths={2}
                    initialFocus
                  />
                  {dateRange.from && (
                    <div className="flex items-center justify-end gap-2 p-3 border-t">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          setDateRange({ from: undefined, to: undefined })
                          setDateRangeOpen(false)
                        }}
                      >
                        Clear
                      </Button>
                    </div>
                  )}
                </PopoverContent>
              </Popover>
            )}
            {!mounted && (
              <Button variant="outline" className="h-10 px-4 gap-2 rounded-lg border-slate-200" disabled>
                <CalendarIcon className="h-4 w-4" />
                Date Range
              </Button>
            )}

            {/* Filter Dropdown */}
            {mounted && (
              <Popover open={filterDropdownOpen} onOpenChange={setFilterDropdownOpen}>
                <PopoverTrigger asChild>
                  <Button variant="outline" className="h-10 px-4 gap-2 rounded-lg border-slate-200">
                    <SlidersHorizontal className="h-4 w-4" />
                    Filter
                  </Button>
                </PopoverTrigger>
                <PopoverContent className="w-56 p-0" align="start">
                  {addingFilter ? (
                    // Show input for the selected filter type
                    <div className="p-3">
                      <div className="flex items-center justify-between mb-2">
                        <span className="text-sm font-medium">{getFilterLabel(addingFilter)}</span>
                        <button onClick={() => setAddingFilter(null)}>
                          <X className="h-4 w-4 text-slate-400" />
                        </button>
                      </div>
                      {addingFilter === 'call_status' ? (
                        <div className="space-y-1">
                          {['Completed', 'Failed', 'In Progress', 'Missed'].map(status => {
                            let bgColor = "";
                            if (status === 'In Progress') {
                              bgColor = "bg-orange-50";
                            } else if (status === 'Missed') {
                              bgColor = "bg-red-50";
                            } else if (status === 'Completed') {
                              bgColor = "bg-green-50";
                            }

                            return (
                              <button
                                key={status}
                                onClick={() => addFilter('call_status', status)}
                                className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded ${bgColor}`}
                              >
                                {status}
                              </button>
                            );
                          })}
                        </div>
                      ) : addingFilter === 'call_type' ? (
                        <div className="space-y-1">
                          {['Inbound', 'Outbound'].map(type => (
                            <button
                              key={type}
                              onClick={() => addFilter('call_type', type)}
                              className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                            >
                              {type}
                            </button>
                          ))}
                        </div>
                      ) : addingFilter === 'assistant_name' ? (
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueAgentTypes.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No agents available</div>
                          ) : (
                            uniqueAgentTypes.map(agentType => (
                              <button
                                key={agentType}
                                onClick={() => {
                                  addFilter('assistant_name', agentType)
                                  setFilterDropdownOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {agentType}
                              </button>
                            ))
                          )}
                        </div>
                      ) : addingFilter === 'from_number' ? (
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueFromNumbers.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No numbers available</div>
                          ) : (
                            uniqueFromNumbers.map(number => (
                              <button
                                key={number}
                                onClick={() => {
                                  addFilter('from_number', number)
                                  setFilterDropdownOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {number}
                              </button>
                            ))
                          )}
                        </div>
                      ) : addingFilter === 'to_number' ? (
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueToNumbers.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No numbers available</div>
                          ) : (
                            uniqueToNumbers.map(number => (
                              <button
                                key={number}
                                onClick={() => {
                                  addFilter('to_number', number)
                                  setFilterDropdownOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {number}
                              </button>
                            ))
                          )}
                        </div>
                      ) : null}
                    </div>
                  ) : (
                    // Show filter options list
                    <div className="py-1">
                      {filterOptions.map(option => (
                        <button
                          key={option.id}
                          onClick={() => setAddingFilter(option.id)}
                          className="w-full flex items-center gap-2 px-3 py-2 text-sm hover:bg-slate-100"
                        >
                          <Plus className="h-4 w-4 text-slate-400" />
                          {option.label}
                        </button>
                      ))}
                    </div>
                  )}
                </PopoverContent>
              </Popover>
            )}
            {!mounted && (
              <Button variant="outline" className="h-10 px-4 gap-2 rounded-lg border-slate-200" disabled>
                <SlidersHorizontal className="h-4 w-4" />
                Filter
              </Button>
            )}
          </div>

          <div className="flex items-center gap-3">
            {/* Export Button */}
            {mounted && (
              <DropdownMenu>
                <DropdownMenuTrigger asChild>
                  <Button className="h-10 px-4 gap-2 rounded-lg bg-slate-900 hover:bg-slate-800">
                    <Download className="h-4 w-4" />
                    Export
                    <ChevronDown className="h-4 w-4" />
                  </Button>
                </DropdownMenuTrigger>
                <DropdownMenuContent align="end">
                  <DropdownMenuItem onClick={exportToCSV}>Export as CSV</DropdownMenuItem>
                  <DropdownMenuItem onClick={exportToPDF}>Export as PDF</DropdownMenuItem>
                </DropdownMenuContent>
              </DropdownMenu>
            )}
            {!mounted && (
              <Button className="h-10 px-4 gap-2 rounded-lg bg-slate-900 hover:bg-slate-800" disabled>
                <Download className="h-4 w-4" />
                Export
                <ChevronDown className="h-4 w-4" />
              </Button>
            )}
          </div>
        </div>

        {/* Active Filters */}
        {(activeFilters.length > 0 || dateRange.from || dateRange.to) && (
          <div className="flex flex-wrap items-center gap-2 px-5 lg:px-8 py-3 bg-white border-b border-slate-200">
            <span className="text-sm text-slate-500">Active filters:</span>
            {(dateRange.from || dateRange.to) && (
              <div className="flex items-center gap-1.5 bg-slate-100 rounded-full px-3 py-1">
                <span className="text-sm text-slate-700">
                  Date Range: <span className="font-medium">
                    {dateRange.from ? format(dateRange.from, "MMM d") : ""}
                    {dateRange.from && dateRange.to ? " - " : ""}
                    {dateRange.to ? format(dateRange.to, "MMM d") : ""}
                  </span>
                </span>
                <button
                  onClick={() => setDateRange({ from: undefined, to: undefined })}
                  className="text-slate-400 hover:text-slate-600"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            )}
            {activeFilters.map((filter, index) => (
              <div
                key={index}
                className="flex items-center gap-1.5 bg-slate-100 rounded-full px-3 py-1"
              >
                <span className="text-sm text-slate-700">
                  {getFilterLabel(filter.field)}: <span className="font-medium">{filter.value}</span>
                </span>
                <button
                  onClick={() => removeFilter(index)}
                  className="text-slate-400 hover:text-slate-600"
                >
                  <X className="h-3.5 w-3.5" />
                </button>
              </div>
            ))}
            <button
              onClick={clearAllFilters}
              className="text-sm text-slate-500 hover:text-slate-700 underline ml-2"
            >
              Clear all
            </button>
          </div>
        )}

        {/* Data Table */}
        <div className="px-5 lg:px-8 py-4">
          <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
            {/* Table Header */}
            <div className="grid grid-cols-8 gap-3 px-5 py-3 bg-slate-50 border-b border-slate-200 text-sm font-medium text-slate-600">
              <div className="flex items-center gap-2">
                Call Type
                {mounted && (
                  <Popover open={callTypeFilterOpen} onOpenChange={setCallTypeFilterOpen}>
                    <PopoverTrigger asChild>
                      <button
                        onClick={() => setCallTypeFilterOpen(true)}
                        className="hover:text-slate-600 transition-colors"
                      >
                        <Filter className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-56 p-0" align="start">
                      <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Call Type</span>
                          <button onClick={() => setCallTypeFilterOpen(false)}>
                            <X className="h-4 w-4 text-slate-400" />
                          </button>
                        </div>
                        <div className="space-y-1">
                          {['Inbound', 'Outbound'].map(type => (
                            <button
                              key={type}
                              onClick={() => {
                                addFilter('call_type', type)
                                setCallTypeFilterOpen(false)
                              }}
                              className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                            >
                              {type}
                            </button>
                          ))}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
                {!mounted && (
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                Agent Name
                {mounted && (
                  <Popover open={assistantNameFilterOpen} onOpenChange={(open) => {
                    setAssistantNameFilterOpen(open)
                    if (!open) setFilterInputValue("")
                  }}>
                    <PopoverTrigger asChild>
                      <button
                        onClick={handleAssistantNameFilter}
                        className="hover:text-slate-600 transition-colors"
                      >
                        <Filter className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-56 p-0" align="start">
                      <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Agent Name</span>
                          <button onClick={() => {
                            setAssistantNameFilterOpen(false)
                          }}>
                            <X className="h-4 w-4 text-slate-400" />
                          </button>
                        </div>
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueAgentTypes.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No agents available</div>
                          ) : (
                            uniqueAgentTypes.map(agentType => (
                              <button
                                key={agentType}
                                onClick={() => {
                                  addFilter('assistant_name', agentType)
                                  setAssistantNameFilterOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {agentType}
                              </button>
                            ))
                          )}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
                {!mounted && (
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                To
                {mounted && (
                  <Popover open={toNumberFilterOpen} onOpenChange={setToNumberFilterOpen}>
                    <PopoverTrigger asChild>
                      <button
                        onClick={() => setToNumberFilterOpen(true)}
                        className="hover:text-slate-600 transition-colors"
                      >
                        <Filter className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-56 p-0" align="start">
                      <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">To Number</span>
                          <button onClick={() => setToNumberFilterOpen(false)}>
                            <X className="h-4 w-4 text-slate-400" />
                          </button>
                        </div>
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueToNumbers.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No numbers available</div>
                          ) : (
                            uniqueToNumbers.map(number => (
                              <button
                                key={number}
                                onClick={() => {
                                  addFilter('to_number', number)
                                  setToNumberFilterOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {number}
                              </button>
                            ))
                          )}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
                {!mounted && (
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                From
                {mounted && (
                  <Popover open={fromNumberFilterOpen} onOpenChange={setFromNumberFilterOpen}>
                    <PopoverTrigger asChild>
                      <button
                        onClick={() => setFromNumberFilterOpen(true)}
                        className="hover:text-slate-600 transition-colors"
                      >
                        <Filter className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-56 p-0" align="start">
                      <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">From Number</span>
                          <button onClick={() => setFromNumberFilterOpen(false)}>
                            <X className="h-4 w-4 text-slate-400" />
                          </button>
                        </div>
                        <div className="space-y-1 max-h-60 overflow-y-auto">
                          {uniqueFromNumbers.length === 0 ? (
                            <div className="px-3 py-2 text-sm text-slate-500">No numbers available</div>
                          ) : (
                            uniqueFromNumbers.map(number => (
                              <button
                                key={number}
                                onClick={() => {
                                  addFilter('from_number', number)
                                  setFromNumberFilterOpen(false)
                                }}
                                className="w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded"
                              >
                                {number}
                              </button>
                            ))
                          )}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
                {!mounted && (
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                Call Status
                {mounted && (
                  <Popover open={callStatusFilterOpen} onOpenChange={setCallStatusFilterOpen}>
                    <PopoverTrigger asChild>
                      <button
                        onClick={handleCallStatusFilter}
                        className="hover:text-slate-600 transition-colors"
                      >
                        <Filter className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                      </button>
                    </PopoverTrigger>
                    <PopoverContent className="w-56 p-0" align="start">
                      <div className="p-3">
                        <div className="flex items-center justify-between mb-2">
                          <span className="text-sm font-medium">Call Status</span>
                          <button onClick={() => setCallStatusFilterOpen(false)}>
                            <X className="h-4 w-4 text-slate-400" />
                          </button>
                        </div>
                        <div className="space-y-1">
                          {['Completed', 'Busy', 'In Progress'].map(status => {
                            let bgColor = "";
                            if (status === 'In Progress') {
                              bgColor = "bg-orange-50";
                            } else if (status === 'Busy') {
                              bgColor = "bg-red-50";
                            } else if (status === 'Completed') {
                              bgColor = "bg-green-50";
                            }

                            return (
                              <button
                                key={status}
                                onClick={() => {
                                  addFilter('call_status', status)
                                  setCallStatusFilterOpen(false)
                                }}
                                className={`w-full text-left px-3 py-2 text-sm hover:bg-slate-100 rounded ${bgColor}`}
                              >
                                {status}
                              </button>
                            );
                          })}
                        </div>
                      </div>
                    </PopoverContent>
                  </Popover>
                )}
                {!mounted && (
                  <Filter className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                Called On
                {mounted && (
                  <button
                    onClick={handleDateSort}
                    className="hover:text-slate-600 transition-colors"
                    title={dateSortOrder === 'latest' ? 'Sort: Latest to Oldest' : 'Sort: Oldest to Latest'}
                  >
                    <ArrowUpDown className="h-3.5 w-3.5 text-slate-400 hover:text-slate-600 cursor-pointer" />
                  </button>
                )}
                {!mounted && (
                  <ArrowUpDown className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center gap-2">
                Call Duration
                {mounted && (
                  <button
                    onClick={handleDurationSort}
                    className="hover:text-slate-600 transition-colors"
                    title={
                      durationSortOrder === null
                        ? 'Sort by duration'
                        : durationSortOrder === 'longest'
                          ? 'Sort: Longest to Shortest'
                          : 'Sort: Shortest to Longest'
                    }
                  >
                    <ArrowUpDown className={`h-3.5 w-3.5 ${durationSortOrder
                        ? "text-slate-600"
                        : "text-slate-400 hover:text-slate-600"
                      } cursor-pointer`} />
                  </button>
                )}
                {!mounted && (
                  <ArrowUpDown className="h-3.5 w-3.5 text-slate-400" />
                )}
              </div>
              <div className="flex items-center justify-center">
                Conversation Data
              </div>
            </div>

            {/* Table Body */}
            {isLoading ? (
              <div className="px-5 py-12 text-center text-slate-500">
                Loading calls...
              </div>
            ) : paginatedMeetings.length === 0 ? (
              <div className="px-5 py-12 text-center text-slate-500">
                {activeFilters.length > 0
                  ? "No calls match your filters"
                  : "No calls found"}
              </div>
            ) : (
              paginatedMeetings.map((meeting) => {
                const callStatus = meeting.call_busy ? "Busy" : meeting.end_time_utc ? "Completed" : "In Progress"
                const statusColors = {
                  "Busy": "bg-red-50 text-red-700",
                  "Completed": "bg-green-50 text-green-700",
                  "In Progress": "bg-orange-50 text-orange-700"
                }

                const isBusy = meeting.call_busy

                const rowContent = (
                  <div
                    onClick={() => {
                      if (!isBusy) {
                        handleMeetingClick(meeting)
                      }
                    }}
                    className={`grid grid-cols-8 gap-3 px-5 py-4 border-b border-slate-100 items-center hover:bg-slate-50 ${isBusy ? "cursor-default" : "cursor-pointer"
                      }`}
                  >
                    <div>
                      <span
                        className={`inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-semibold ${meeting.inbound
                            ? "bg-emerald-50 text-emerald-700"
                            : "bg-blue-50 text-blue-700"
                          }`}
                        style={{
                          minWidth: '84px',
                          justifyContent: 'center',
                          letterSpacing: '0.03em',
                          transition: 'background 0.15s, color 0.15s',
                        }}
                        aria-label={meeting.inbound ? "Inbound Call" : "Outbound Call"}
                        title={meeting.inbound ? "Inbound Call" : "Outbound Call"}
                      >
                        {meeting.inbound ? (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            width="16"
                            height="16"
                            fill="#059669"
                            aria-hidden="true"
                            className="mr-1"
                            style={{ opacity: 0.95, verticalAlign: "middle" }}
                          >
                            <path d="M3.5 2A1.5 1.5 0 0 0 2 3.5V5c0 1.149.15 2.263.43 3.326a13.022 13.022 0 0 0 9.244 9.244c1.063.28 2.177.43 3.326.43h1.5a1.5 1.5 0 0 0 1.5-1.5v-1.148a1.5 1.5 0 0 0-1.175-1.465l-3.223-.716a1.5 1.5 0 0 0-1.767 1.052l-.267.933c-.117.41-.555.643-.95.48a11.542 11.542 0 0 1-6.254-6.254c-.163-.395.07-.833.48-.95l.933-.267a1.5 1.5 0 0 0 1.052-1.767l-.716-3.223A1.5 1.5 0 0 0 4.648 2H3.5Zm13.22.22a.75.75 0 1 1 1.06 1.06L14.56 6.5h2.69a.75.75 0 0 1 0 1.5h-4.5a.75.75 0 0 1-.75-.75v-4.5a.75.75 0 0 1 1.5 0v2.69l3.22-3.22Z" />
                          </svg>
                        ) : (
                          <svg
                            xmlns="http://www.w3.org/2000/svg"
                            viewBox="0 0 20 20"
                            width="16"
                            height="16"
                            fill="#2563eb"
                            aria-hidden="true"
                            className="mr-1"
                            style={{ opacity: 0.95, verticalAlign: "middle" }}
                          >
                            <path d="M3.5 2A1.5 1.5 0 0 0 2 3.5V5c0 1.149.15 2.263.43 3.326a13.022 13.022 0 0 0 9.244 9.244c1.063.28 2.177.43 3.326.43h1.5a1.5 1.5 0 0 0 1.5-1.5v-1.148a1.5 1.5 0 0 0-1.175-1.465l-3.223-.716a1.5 1.5 0 0 0-1.767 1.052l-.267.933c-.117.41-.555.643-.95.48a11.542 11.542 0 0 1-6.254-6.254c-.163-.395.07-.833.48-.95l.933-.267a1.5 1.5 0 0 0 1.052-1.767l-.716-3.223A1.5 1.5 0 0 0 4.648 2H3.5Zm13 2.56l-3.22 3.22a.75.75 0 1 1-1.06-1.06l3.22-3.22h-2.69a.75.75 0 0 1 0-1.5h4.5a.75.75 0 0 1 .75.75v4.5a.75.75 0 0 1-1.5 0V4.56Z" />
                          </svg>
                        )}
                        {meeting.inbound ? "Inbound" : "Outbound"}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span className="text-sm text-slate-900">{meeting.agent_type || "-"}</span>
                    </div>
                    <div className="text-sm">
                      <div className="text-slate-900">{meeting.to_number || "-"}</div>
                    </div>
                    <div className="text-sm">
                      <div className="text-slate-900">{meeting.from_number || "-"}</div>
                    </div>
                    <div>
                      <span className={`inline-flex items-center px-2.5 py-1 rounded-full text-xs font-medium ${statusColors[callStatus as keyof typeof statusColors]}`}>
                        {callStatus}
                      </span>
                    </div>
                    <div className="flex flex-col text-sm">
                      <span className="text-slate-900">
                        {formatDate(meeting.start_time_utc || meeting.created_at)}
                      </span>

                    </div>
                    <div className="text-sm text-slate-900 flex items-center justify-center">
                      {meeting.call_busy ? "N/A" : formatDuration(meeting.duration, meeting.start_time_utc, meeting.end_time_utc)}
                    </div>
                    <div className="flex items-center justify-center">
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation()
                          if (!isBusy) handleMeetingClick(meeting)
                        }}
                        disabled={isBusy}
                        className={`inline-flex flex-col items-center justify-center rounded-lg border px-3 py-1.5 leading-tight ${isBusy
                            ? "border-slate-200 text-slate-300 cursor-not-allowed"
                            : "border-slate-200 text-slate-800 hover:bg-slate-50"
                          }`}
                        title={isBusy ? "No call log data" : "Open recording and transcript"}
                      >
                        <span className="text-xs font-bold tracking-wide">Recording</span>
                        <span className="text-[10px] text-slate-500">transcription etc</span>
                      </button>
                    </div>
                  </div>
                )

                if (isBusy) {
                  return (
                    <Tooltip key={meeting.meeting_id}>
                      <TooltipTrigger asChild>
                        {rowContent}
                      </TooltipTrigger>
                      <TooltipContent>
                        <p>No call log data</p>
                      </TooltipContent>
                    </Tooltip>
                  )
                }

                return (
                  <div key={meeting.meeting_id}>
                    {rowContent}
                  </div>
                )
              })
            )}
          </div>

          {/* Pagination */}
          <div className="flex items-center justify-center gap-4 mt-4">
            <Button
              variant="outline"
              size="icon"
              className="h-9 w-9 rounded-full"
              onClick={() => setCurrentPage(p => Math.max(1, p - 1))}
              disabled={currentPage === 1}
            >
              <ChevronLeft className="h-4 w-4" />
            </Button>
            <span className="text-sm text-slate-600">
              Showing {((currentPage - 1) * itemsPerPage) + 1} to {Math.min(currentPage * itemsPerPage, filteredMeetings.length)} calls
            </span>
            <Button
              variant="outline"
              size="icon"
              className="h-9 w-9 rounded-full"
              onClick={() => setCurrentPage(p => p + 1)}
              disabled={currentPage * itemsPerPage >= filteredMeetings.length}
            >
              <ChevronRight className="h-4 w-4" />
            </Button>
          </div>
        </div>
      </main>



      {/* Meeting Detail Sheet */}
      <MeetingDetailSheet
        open={isSheetOpen}
        onOpenChange={setIsSheetOpen}
        meeting={selectedMeeting}
        meetingDetails={meetingDetails}
        isLoading={isLoadingDetails}
      />
    </div>
  )
}

export default function HistoryPage() {
  return (
    <Suspense fallback={<div>Loading...</div>}>
      <HistoryPageContent />
    </Suspense>
  )
}
