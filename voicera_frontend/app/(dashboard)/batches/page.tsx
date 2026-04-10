"use client"

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ChangeEvent,
} from "react"
import { format } from "date-fns"
import { Calendar } from "@/components/ui/calendar"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Badge } from "@/components/ui/badge"
import {
  cancelBatchSchedule,
  deleteBatch,
  getAgents,
  getBatches,
  getCurrentUser,
  rescheduleBatch,
  runBatch,
  scheduleBatch,
  stopBatch,
  uploadBatchCsv,
  type Agent,
  type Batch,
} from "@/lib/api"
import { AlertCircle, Loader2, Trash2, UploadCloud } from "lucide-react"

const templateCsv = "contact_number,customer_name\n+911234567890,Test User"
const PREVIEW_LIMIT = 30
const MIN_CONCURRENCY = 1
const MAX_CONCURRENCY = 20
const DEFAULT_CONCURRENCY = "5"

type PreviewContact = {
  rowNumber: number
  rawValue: string
  normalizedValue: string
  isValid: boolean
}

const formatCreatedAt = (dateValue?: string | null): string => {
  if (!dateValue) return "—"
  const parsedDate = new Date(dateValue)
  if (Number.isNaN(parsedDate.getTime())) return "—"
  return format(parsedDate, "dd/MM/yyyy, hh:mm a")
}

const getLocalTimezone = (): string => {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || "UTC"
  } catch {
    return "UTC"
  }
}

const toScheduleInputValue = (dateValue?: string | null): string => {
  if (!dateValue) return ""
  const parsedDate = new Date(dateValue)
  if (Number.isNaN(parsedDate.getTime())) return ""
  const localTimestamp = parsedDate.getTime() - parsedDate.getTimezoneOffset() * 60 * 1000
  return new Date(localTimestamp).toISOString().slice(0, 16)
}

const formatScheduledAt = (dateValue?: string | null): string => {
  if (!dateValue) return "—"
  const parsedDate = new Date(dateValue)
  if (Number.isNaN(parsedDate.getTime())) return "—"
  return format(parsedDate, "dd/MM/yyyy, hh:mm a")
}

const parseLocalScheduleDate = (value?: string): Date | undefined => {
  if (!value) return undefined
  const parsed = new Date(value)
  if (Number.isNaN(parsed.getTime())) return undefined
  return parsed
}

const formatStatusLabel = (value: string): string => {
  if (!value) return "—"
  return value
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ")
}

const normalizeContactNumber = (value: string): string => {
  const stripped = (value || "").trim()
  if (!stripped) return ""
  let normalized = stripped.replace(/[ \-\(\)\.]/g, "")
  if (normalized.startsWith("++")) {
    normalized = normalized.replace(/^\++/, "")
  }
  return normalized
}

const isValidContactNumber = (value: string): boolean => {
  return /^\+?\d{8,15}$/.test(value)
}

const parseCsvLine = (line: string): string[] => {
  const cells: string[] = []
  let current = ""
  let inQuotes = false

  for (let index = 0; index < line.length; index += 1) {
    const character = line[index]
    if (character === '"') {
      if (inQuotes && line[index + 1] === '"') {
        current += '"'
        index += 1
      } else {
        inQuotes = !inQuotes
      }
      continue
    }
    if (character === "," && !inQuotes) {
      cells.push(current)
      current = ""
      continue
    }
    current += character
  }

  cells.push(current)
  return cells
}

const parseCsvPreview = (csvText: string): PreviewContact[] => {
  const rows = csvText
    .split(/\r?\n/)
    .map((row) => row.trim())
    .filter((row) => row.length > 0)

  if (rows.length < 2) {
    throw new Error("CSV must contain a header and at least one contact row")
  }

  const headers = parseCsvLine(rows[0]).map((value) => value.trim())
  const contactIndex = headers.findIndex((header) => header === "contact_number")
  if (contactIndex === -1) {
    throw new Error("CSV must include 'contact_number' column")
  }

  const previewRows: PreviewContact[] = []
  for (let rowIndex = 1; rowIndex < rows.length && previewRows.length < PREVIEW_LIMIT; rowIndex += 1) {
    const columns = parseCsvLine(rows[rowIndex])
    const rawValue = (columns[contactIndex] || "").trim()
    const normalizedValue = normalizeContactNumber(rawValue)
    previewRows.push({
      rowNumber: rowIndex + 1,
      rawValue,
      normalizedValue,
      isValid: isValidContactNumber(normalizedValue),
    })
  }

  if (previewRows.length === 0) {
    throw new Error("CSV file has no contact rows")
  }

  return previewRows
}

const parseConcurrency = (value: string): number | null => {
  const parsedValue = Number.parseInt(value, 10)
  if (
    Number.isNaN(parsedValue) ||
    parsedValue < MIN_CONCURRENCY ||
    parsedValue > MAX_CONCURRENCY
  ) {
    return null
  }
  return parsedValue
}

const toIsoWithLocalOffset = (value: string): string | null => {
  const parsedDate = new Date(value)
  if (Number.isNaN(parsedDate.getTime())) return null

  const datePart = value.length === 16 ? `${value}:00` : value
  const offsetMinutes = -parsedDate.getTimezoneOffset()
  const sign = offsetMinutes >= 0 ? "+" : "-"
  const absoluteOffset = Math.abs(offsetMinutes)
  const offsetHours = String(Math.floor(absoluteOffset / 60)).padStart(2, "0")
  const offsetMins = String(absoluteOffset % 60).padStart(2, "0")
  return `${datePart}${sign}${offsetHours}:${offsetMins}`
}

export default function BatchesPage() {
  const [agents, setAgents] = useState<Agent[]>([])
  const [batches, setBatches] = useState<Batch[]>([])
  const [userOrgId, setUserOrgId] = useState("")
  const [isLoadingAgents, setIsLoadingAgents] = useState(true)
  const [isLoadingBatches, setIsLoadingBatches] = useState(true)
  const [isNewBatchDialogOpen, setIsNewBatchDialogOpen] = useState(false)
  const [isBatchControlsOpen, setIsBatchControlsOpen] = useState(false)
  const [selectedBatch, setSelectedBatch] = useState<Batch | null>(null)
  const [selectedBatchAgentType, setSelectedBatchAgentType] = useState("")
  const [selectedBatchConcurrency, setSelectedBatchConcurrency] = useState(DEFAULT_CONCURRENCY)
  const [selectedBatchScheduledAtLocal, setSelectedBatchScheduledAtLocal] = useState("")
  const [newBatchName, setNewBatchName] = useState("")
  const [newBatchAgentType, setNewBatchAgentType] = useState("")
  const [newBatchConcurrency, setNewBatchConcurrency] = useState(DEFAULT_CONCURRENCY)
  const [newBatchScheduledAtLocal, setNewBatchScheduledAtLocal] = useState("")
  const [userTimezone, setUserTimezone] = useState("UTC")
  const [isSchedulePickerOpen, setIsSchedulePickerOpen] = useState(false)
  const [schedulePickerTarget, setSchedulePickerTarget] = useState<"new" | "selected">("new")
  const [schedulePickerDate, setSchedulePickerDate] = useState<Date | undefined>(undefined)
  const [schedulePickerTime, setSchedulePickerTime] = useState("09:00")
  const [newBatchFile, setNewBatchFile] = useState<File | null>(null)
  const [newBatchPreviewRows, setNewBatchPreviewRows] = useState<PreviewContact[]>([])
  const [newBatchId, setNewBatchId] = useState<string | null>(null)
  const [newBatchError, setNewBatchError] = useState<string | null>(null)
  const [isDragActive, setIsDragActive] = useState(false)
  const [isCreatingBatch, setIsCreatingBatch] = useState(false)
  const [deletingBatchId, setDeletingBatchId] = useState<string | null>(null)
  const [runningBatchId, setRunningBatchId] = useState<string | null>(null)
  const [stoppingBatchId, setStoppingBatchId] = useState<string | null>(null)
  const [schedulingBatchId, setSchedulingBatchId] = useState<string | null>(null)
  const [cancelingScheduleBatchId, setCancelingScheduleBatchId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const loadBatches = useCallback(async () => {
    setIsLoadingBatches(true)
    try {
      const batchRows = await getBatches()
      setBatches(batchRows)
    } catch (error) {
      console.error("Failed to load batches:", error)
      setBatches([])
    } finally {
      setIsLoadingBatches(false)
    }
  }, [])

  useEffect(() => {
    async function fetchInitialData() {
      try {
        const user = await getCurrentUser()
        if (!user.org_id) return
        setUserOrgId(user.org_id)

        const agentsData = await getAgents(user.org_id)
        setAgents(agentsData)
        if (agentsData.length > 0) {
          setNewBatchAgentType(agentsData[0].agent_type)
        }
      } catch (error) {
        console.error("Failed to load batches page data:", error)
      } finally {
        setIsLoadingAgents(false)
      }
    }
    fetchInitialData()
  }, [])

  useEffect(() => {
    setUserTimezone(getLocalTimezone())
  }, [])

  useEffect(() => {
    if (isLoadingAgents) return
    loadBatches()
  }, [isLoadingAgents, loadBatches])

  useEffect(() => {
    const hasActiveBatch = batches.some(
      (batch) =>
        batch.execution_status === "running" ||
        batch.execution_status === "stopping"
    )
    if (!hasActiveBatch) return

    const timerId = window.setInterval(() => {
      loadBatches()
    }, 3000)

    return () => window.clearInterval(timerId)
  }, [batches, loadBatches])

  useEffect(() => {
    if (!selectedBatch) return
    const latestBatch = batches.find(
      (batch) => batch.batch_id === selectedBatch.batch_id
    )
    if (latestBatch) {
      setSelectedBatch(latestBatch)
      setSelectedBatchScheduledAtLocal(toScheduleInputValue(latestBatch.scheduled_at_utc))
    }
  }, [batches, selectedBatch])

  const newBatch = useMemo(
    () => (newBatchId ? batches.find((batch) => batch.batch_id === newBatchId) || null : null),
    [batches, newBatchId]
  )

  const templateSheetUrl = useMemo(
    () => `data:text/csv;charset=utf-8,${encodeURIComponent(templateCsv)}`,
    []
  )

  const resetNewBatchState = useCallback(
    (agentTypeOverride?: string) => {
      setNewBatchName("")
      setNewBatchAgentType(agentTypeOverride || agents[0]?.agent_type || "")
      setNewBatchConcurrency(DEFAULT_CONCURRENCY)
      setNewBatchScheduledAtLocal("")
      setNewBatchFile(null)
      setNewBatchPreviewRows([])
      setNewBatchId(null)
      setNewBatchError(null)
      setIsDragActive(false)
      if (fileInputRef.current) fileInputRef.current.value = ""
    },
    [agents]
  )

  const handleNewBatchDialogChange = (open: boolean) => {
    setIsNewBatchDialogOpen(open)
    if (!open) {
      resetNewBatchState()
    }
  }

  const preparePreviewForFile = async (file: File) => {
    const isCsv = file.name.toLowerCase().endsWith(".csv")
    if (!isCsv) {
      setNewBatchError("Only .csv files are supported.")
      setNewBatchFile(null)
      setNewBatchPreviewRows([])
      return
    }

    try {
      const csvText = await file.text()
      const previewRows = parseCsvPreview(csvText)
      setNewBatchError(null)
      setNewBatchFile(file)
      setNewBatchPreviewRows(previewRows)
    } catch (error) {
      setNewBatchError(
        error instanceof Error ? error.message : "Unable to parse CSV preview"
      )
      setNewBatchFile(null)
      setNewBatchPreviewRows([])
    }
  }

  const handleFileSelect = async (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    await preparePreviewForFile(file)
  }

  const handleCreateBatch = async () => {
    const trimmedBatchName = newBatchName.trim()
    if (!trimmedBatchName) {
      setNewBatchError("Batch name is required.")
      return
    }
    if (!newBatchAgentType) {
      setNewBatchError("Please select an agent.")
      return
    }
    if (!newBatchFile) {
      setNewBatchError("Please upload a CSV file.")
      return
    }
    if (!userOrgId) {
      setNewBatchError("Organization not found for current user.")
      return
    }

    setIsCreatingBatch(true)
    setNewBatchError(null)
    try {
      const created = await uploadBatchCsv(
        newBatchFile,
        userOrgId,
        newBatchAgentType,
        trimmedBatchName
      )
      setNewBatchId(created.batch_id)
      setNewBatchConcurrency(String(created.concurrency || 5))
      setNewBatchScheduledAtLocal("")
      await loadBatches()
    } catch (error) {
      setNewBatchError(
        error instanceof Error ? error.message : "Failed to create batch"
      )
    } finally {
      setIsCreatingBatch(false)
    }
  }

  const handleDeleteBatch = async (batchId: string) => {
    if (!window.confirm("Delete this batch? This cannot be undone.")) {
      return
    }
    setDeletingBatchId(batchId)
    try {
      await deleteBatch(batchId)
      await loadBatches()
      if (selectedBatch?.batch_id === batchId) {
        setIsBatchControlsOpen(false)
        setSelectedBatch(null)
      }
      if (newBatchId === batchId) {
        setNewBatchId(null)
      }
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to delete batch")
    } finally {
      setDeletingBatchId(null)
    }
  }

  const handleRunBatch = async (
    batchId: string,
    agentType: string,
    concurrencyInput: string
  ) => {
    if (!agentType) {
      alert("Please select an agent before running this batch.")
      return
    }
    const targetAgent = agents.find((agent) => agent.agent_type === agentType)
    if (!targetAgent?.phone_number) {
      alert("Please attach a number to the selected agent before running batch calls.")
      return
    }
    const parsedConcurrency = parseConcurrency(concurrencyInput)
    if (!parsedConcurrency) {
      alert(`Concurrency must be between ${MIN_CONCURRENCY} and ${MAX_CONCURRENCY}.`)
      return
    }

    setRunningBatchId(batchId)
    try {
      await runBatch(batchId, agentType, parsedConcurrency)
      await loadBatches()
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to run batch")
    } finally {
      setRunningBatchId(null)
    }
  }

  const handleStopBatch = async (batchId: string) => {
    setStoppingBatchId(batchId)
    try {
      await stopBatch(batchId)
      await loadBatches()
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to stop batch")
    } finally {
      setStoppingBatchId(null)
    }
  }

  const handleScheduleBatch = async (
    batchId: string,
    agentType: string,
    concurrencyInput: string,
    scheduledAtLocal: string,
    useReschedule: boolean
  ) => {
    if (!agentType) {
      alert("Please select an agent before scheduling this batch.")
      return
    }
    const targetAgent = agents.find((agent) => agent.agent_type === agentType)
    if (!targetAgent?.phone_number) {
      alert("Please attach a number to the selected agent before scheduling batch calls.")
      return
    }
    const parsedConcurrency = parseConcurrency(concurrencyInput)
    if (!parsedConcurrency) {
      alert(`Concurrency must be between ${MIN_CONCURRENCY} and ${MAX_CONCURRENCY}.`)
      return
    }
    if (!scheduledAtLocal) {
      alert("Please select a date and time for scheduling.")
      return
    }
    const scheduledAtIsoWithOffset = toIsoWithLocalOffset(scheduledAtLocal)
    if (!scheduledAtIsoWithOffset) {
      alert("Invalid date/time value.")
      return
    }
    const selectedTimestamp = new Date(scheduledAtLocal).getTime()
    if (Number.isNaN(selectedTimestamp) || selectedTimestamp <= Date.now()) {
      alert("Scheduled time must be in the future.")
      return
    }

    setSchedulingBatchId(batchId)
    try {
      const payload = {
        scheduled_at_local: scheduledAtIsoWithOffset,
        timezone: userTimezone || "UTC",
        agent_type: agentType,
        concurrency: parsedConcurrency,
      }
      if (useReschedule) {
        await rescheduleBatch(batchId, payload)
      } else {
        await scheduleBatch(batchId, payload)
      }
      await loadBatches()
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to schedule batch")
    } finally {
      setSchedulingBatchId(null)
    }
  }

  const handleCancelSchedule = async (batchId: string) => {
    setCancelingScheduleBatchId(batchId)
    try {
      await cancelBatchSchedule(batchId)
      await loadBatches()
    } catch (error) {
      alert(error instanceof Error ? error.message : "Failed to cancel schedule")
    } finally {
      setCancelingScheduleBatchId(null)
    }
  }

  const openBatchControls = (batch: Batch) => {
    setSelectedBatch(batch)
    setSelectedBatchAgentType(batch.agent_type || agents[0]?.agent_type || "")
    setSelectedBatchConcurrency(String(batch.concurrency || 5))
    setSelectedBatchScheduledAtLocal(toScheduleInputValue(batch.scheduled_at_utc))
    setIsBatchControlsOpen(true)
  }

  const openSchedulePicker = (target: "new" | "selected") => {
    const sourceValue =
      target === "new" ? newBatchScheduledAtLocal : selectedBatchScheduledAtLocal
    const parsedSourceDate = parseLocalScheduleDate(sourceValue)
    if (parsedSourceDate) {
      setSchedulePickerDate(parsedSourceDate)
      setSchedulePickerTime(format(parsedSourceDate, "HH:mm"))
    } else {
      const fallbackDate = new Date()
      fallbackDate.setMinutes(fallbackDate.getMinutes() + 5)
      setSchedulePickerDate(fallbackDate)
      setSchedulePickerTime(format(fallbackDate, "HH:mm"))
    }
    setSchedulePickerTarget(target)
    setIsSchedulePickerOpen(true)
  }

  const applySchedulePicker = () => {
    if (!schedulePickerDate) {
      alert("Please select a date.")
      return
    }
    if (!/^\d{2}:\d{2}$/.test(schedulePickerTime)) {
      alert("Please select a valid time.")
      return
    }
    const [hours, minutes] = schedulePickerTime.split(":").map((value) => Number.parseInt(value, 10))
    if (
      Number.isNaN(hours) ||
      Number.isNaN(minutes) ||
      hours < 0 ||
      hours > 23 ||
      minutes < 0 ||
      minutes > 59
    ) {
      alert("Please select a valid time.")
      return
    }

    const mergedDate = new Date(schedulePickerDate)
    mergedDate.setHours(hours, minutes, 0, 0)
    const localValue = format(mergedDate, "yyyy-MM-dd'T'HH:mm")
    if (schedulePickerTarget === "new") {
      setNewBatchScheduledAtLocal(localValue)
    } else {
      setSelectedBatchScheduledAtLocal(localValue)
    }
    setIsSchedulePickerOpen(false)
  }

  return (
    <div className="flex flex-col h-full min-h-0 bg-slate-50/50">
      <div className="border-b border-slate-200 bg-white px-6 py-5 shrink-0">
        <h1 className="text-2xl font-semibold text-slate-900 tracking-tight">
          Batches
        </h1>
      </div>

      <div className="flex-1 overflow-auto p-6">
        <div className="rounded-xl border border-slate-200 bg-white">
          <div className="flex flex-col gap-3 border-b border-slate-200 px-4 py-4 md:flex-row md:items-center md:justify-between">
            <div className="text-sm text-slate-700">
              Download a template sheet:{" "}
              <a
                href={templateSheetUrl}
                download="batch_template.csv"
                className="font-medium text-blue-600 hover:underline"
              >
                link
              </a>
            </div>

            <Button
              type="button"
              onClick={() => {
                resetNewBatchState(agents[0]?.agent_type || "")
                setIsNewBatchDialogOpen(true)
              }}
              disabled={isLoadingAgents || agents.length === 0}
            >
              New Batch
            </Button>
          </div>

          <div className="p-4">
            <div className="overflow-hidden rounded-lg border border-slate-200">
              <Table>
                <TableHeader>
                  <TableRow className="bg-slate-50 hover:bg-slate-50">
                    <TableHead className="w-[16%] whitespace-normal break-words px-2 py-2">Batch Name</TableHead>
                    <TableHead className="w-[20%] whitespace-normal break-words px-2 py-2">Batch ID</TableHead>
                    <TableHead className="w-[20%] whitespace-normal break-words px-2 py-2">File name</TableHead>
                    <TableHead className="w-[12%] whitespace-normal break-words px-2 py-2">
                      Uploaded contacts (# valid / # total)
                    </TableHead>
                    <TableHead className="w-[10%] whitespace-normal break-words px-2 py-2">Execution Status</TableHead>
                    <TableHead className="w-[8%] whitespace-normal break-words px-2 py-2">Batch Status</TableHead>
                    <TableHead className="w-[10%] whitespace-normal break-words px-2 py-2">Created At</TableHead>
                    <TableHead className="w-[4%] whitespace-normal break-words px-2 py-2 text-center">Delete</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {isLoadingBatches ? (
                    <TableRow>
                      <TableCell
                        colSpan={8}
                        className="h-28 text-center text-base text-slate-600"
                      >
                        Loading batches...
                      </TableCell>
                    </TableRow>
                  ) : batches.length === 0 ? (
                    <TableRow>
                      <TableCell
                        colSpan={8}
                        className="h-28 text-center text-base text-slate-600"
                      >
                        No batches found
                      </TableCell>
                    </TableRow>
                  ) : (
                    batches.map((batch) => (
                      <TableRow
                        key={batch.batch_id}
                        className="cursor-pointer"
                        onClick={() => openBatchControls(batch)}
                      >
                        <TableCell className="whitespace-normal break-all px-2 py-2">
                          {batch.batch_name}
                        </TableCell>
                        <TableCell className="whitespace-normal break-all px-2 py-2">{batch.batch_id}</TableCell>
                        <TableCell className="whitespace-normal break-all px-2 py-2">
                          {batch.original_filename}
                        </TableCell>
                        <TableCell className="whitespace-normal break-words px-2 py-2">
                          {batch.valid_contacts} / {batch.total_contacts}
                        </TableCell>
                        <TableCell className="whitespace-normal break-words px-2 py-2">
                          <p>{formatStatusLabel(batch.execution_status)}</p>
                          {batch.schedule_mode === "scheduled" && batch.scheduled_at_utc ? (
                            <p className="mt-1 text-xs text-slate-500">
                              Scheduled: {formatScheduledAt(batch.scheduled_at_utc)}
                            </p>
                          ) : null}
                        </TableCell>
                        <TableCell className="whitespace-normal break-words px-2 py-2">{formatStatusLabel(batch.status)}</TableCell>
                        <TableCell className="whitespace-normal break-words px-2 py-2">{formatCreatedAt(batch.created_at)}</TableCell>
                        <TableCell className="text-center px-2 py-2">
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            className="text-red-600 hover:text-red-700 hover:bg-red-50 px-1"
                            aria-label={`Delete batch ${batch.batch_id}`}
                            disabled={
                              deletingBatchId === batch.batch_id ||
                              batch.execution_status === "running" ||
                              batch.execution_status === "stopping"
                            }
                            onClick={(event) => {
                              event.stopPropagation()
                              handleDeleteBatch(batch.batch_id)
                            }}
                          >
                            {deletingBatchId === batch.batch_id ? (
                              <Loader2 className="h-4 w-4 animate-spin" />
                            ) : (
                              <Trash2 className="h-4 w-4" />
                            )}
                          </Button>
                        </TableCell>
                      </TableRow>
                    ))
                  )}
                </TableBody>
              </Table>
            </div>
          </div>
        </div>
      </div>

      <Dialog open={isNewBatchDialogOpen} onOpenChange={handleNewBatchDialogChange}>
        <DialogContent
          overlayClassName="left-0 w-screen md:left-64 md:w-[calc(100vw-16rem)]"
          className="left-0 top-6 h-[calc(100vh-3rem)] w-screen max-h-[calc(100vh-3rem)] max-w-none translate-x-0 translate-y-0 rounded-2xl border p-6 data-[state=open]:slide-in-from-left-0 data-[state=open]:slide-in-from-top-0 sm:max-w-none md:left-[calc(16rem+1.25rem)] md:top-6 md:h-[calc(100vh-3rem)] md:w-[calc(100vw-16rem-2.5rem)]"
        >
          <DialogHeader>
            <DialogTitle>New Batch</DialogTitle>
            <DialogDescription>
              Create a batch, review parsed contacts, then control execution.
            </DialogDescription>
          </DialogHeader>

          {newBatch ? (
            <div className="space-y-4">
              <div className="rounded-lg border border-slate-200 bg-slate-50 p-3 text-sm">
                <p className="text-slate-700">
                  <span className="font-medium">Batch:</span>{" "}
                  <span className="break-all">{newBatch.batch_name}</span>
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Batch ID:</span>{" "}
                  <span className="break-all">{newBatch.batch_id}</span>
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Execution Status:</span>{" "}
                  {formatStatusLabel(newBatch.execution_status)}
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Schedule:</span>{" "}
                  {formatStatusLabel(newBatch.scheduled_status || "none")}
                  {newBatch.scheduled_at_utc ? ` · ${formatScheduledAt(newBatch.scheduled_at_utc)}` : ""}
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Calls:</span>{" "}
                  {newBatch.successful_calls || 0} success /{" "}
                  {newBatch.failed_calls || 0} failed /{" "}
                  {newBatch.attempted_calls || 0} attempted
                </p>
                {newBatch.error_message ? (
                  <p className="text-red-600 mt-2">
                    <span className="font-medium">Error:</span>{" "}
                    {newBatch.error_message}
                  </p>
                ) : null}
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">Agent</p>
                  <Select
                    value={newBatchAgentType}
                    onValueChange={setNewBatchAgentType}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder="Select an agent" />
                    </SelectTrigger>
                    <SelectContent>
                      {agents.map((agent) => (
                        <SelectItem
                          key={`${agent.agent_type}:${agent.agent_id}`}
                          value={agent.agent_type}
                        >
                          {agent.agent_id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">
                    Concurrency ({MIN_CONCURRENCY} - {MAX_CONCURRENCY})
                  </p>
                  <Input
                    type="number"
                    min={MIN_CONCURRENCY}
                    max={MAX_CONCURRENCY}
                    value={newBatchConcurrency}
                    onChange={(event) => setNewBatchConcurrency(event.target.value)}
                  />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">Run at (local)</p>
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full justify-start font-normal"
                    onClick={() => openSchedulePicker("new")}
                  >
                    {newBatchScheduledAtLocal
                      ? formatScheduledAt(toIsoWithLocalOffset(newBatchScheduledAtLocal))
                      : "Select date and time"}
                  </Button>
                  <p className="mt-1 text-xs text-slate-500">Timezone: {userTimezone}</p>
                </div>
              </div>

              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() =>
                    handleScheduleBatch(
                      newBatch.batch_id,
                      newBatchAgentType,
                      newBatchConcurrency,
                      newBatchScheduledAtLocal,
                      newBatch.execution_status === "scheduled"
                    )
                  }
                  disabled={
                    schedulingBatchId === newBatch.batch_id ||
                    runningBatchId === newBatch.batch_id ||
                    newBatch.execution_status === "running" ||
                    newBatch.execution_status === "completed"
                  }
                >
                  {schedulingBatchId === newBatch.batch_id
                    ? "Scheduling..."
                    : newBatch.execution_status === "scheduled"
                    ? "Reschedule"
                    : "Set Date & Time"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleCancelSchedule(newBatch.batch_id)}
                  disabled={
                    cancelingScheduleBatchId === newBatch.batch_id ||
                    newBatch.execution_status !== "scheduled"
                  }
                >
                  {cancelingScheduleBatchId === newBatch.batch_id ? "Canceling..." : "Cancel Schedule"}
                </Button>
                <Button
                  type="button"
                  onClick={() =>
                    handleRunBatch(
                      newBatch.batch_id,
                      newBatchAgentType,
                      newBatchConcurrency
                    )
                  }
                  disabled={
                    runningBatchId === newBatch.batch_id ||
                    newBatch.execution_status === "running" ||
                    newBatch.execution_status === "completed"
                  }
                >
                  {runningBatchId === newBatch.batch_id ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    "Run"
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleStopBatch(newBatch.batch_id)}
                  disabled={
                    stoppingBatchId === newBatch.batch_id ||
                    newBatch.execution_status !== "running"
                  }
                >
                  {stoppingBatchId === newBatch.batch_id ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Stopping...
                    </>
                  ) : (
                    "Stop"
                  )}
                </Button>
              </div>
            </div>
          ) : newBatchId ? (
            <div className="flex items-center justify-center py-10 text-slate-600">
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              Preparing batch controls...
            </div>
          ) : (
            <>
              <div className="rounded-lg bg-slate-50 p-4 text-sm text-slate-700">
                <div className="flex items-start gap-2">
                  <AlertCircle className="h-4 w-4 mt-0.5 shrink-0 text-slate-500" />
                  <div>
                    <p className="font-medium text-slate-800">
                      Ensure that your CSV file includes a column{" "}
                      <span className="text-blue-600">contact_number</span> for
                      phone numbers.
                    </p>
                    <p className="mt-1 text-slate-600">
                      Parsed contact preview shows the first {PREVIEW_LIMIT} rows
                      before you proceed.
                    </p>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
                <div className="space-y-4">
                  <div>
                    <p className="text-sm font-medium text-slate-900 mb-2">
                      Batch Name
                    </p>
                    <Input
                      value={newBatchName}
                      onChange={(event) => setNewBatchName(event.target.value)}
                      placeholder="Enter batch name"
                    />
                  </div>

                  <div>
                    <p className="text-sm font-medium text-slate-900 mb-2">Agent</p>
                    <Select
                      value={newBatchAgentType}
                      onValueChange={setNewBatchAgentType}
                    >
                      <SelectTrigger className="h-10">
                        <SelectValue placeholder="Select an agent" />
                      </SelectTrigger>
                      <SelectContent>
                        {agents.length === 0 ? (
                          <SelectItem value="no-agent" disabled>
                            {isLoadingAgents ? "Loading..." : "No agents found"}
                          </SelectItem>
                        ) : (
                          agents.map((agent) => (
                            <SelectItem
                              key={`${agent.agent_type}:${agent.agent_id}`}
                              value={agent.agent_type}
                            >
                              {agent.agent_id}
                            </SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  <div>
                    <p className="mb-2 text-sm font-medium text-slate-900">
                      Upload CSV
                    </p>
                    <input
                      ref={fileInputRef}
                      type="file"
                      accept=".csv,text/csv"
                      className="hidden"
                      onChange={(event) => {
                        void handleFileSelect(event)
                      }}
                    />
                    <div
                      className={`rounded-lg border-2 border-dashed p-8 text-center transition-colors ${
                        isDragActive
                          ? "border-blue-500 bg-blue-50/50"
                          : "border-slate-300 bg-white"
                      }`}
                      onDragOver={(event) => {
                        event.preventDefault()
                        setIsDragActive(true)
                      }}
                      onDragLeave={() => setIsDragActive(false)}
                      onDrop={(event) => {
                        event.preventDefault()
                        setIsDragActive(false)
                        const droppedFile = event.dataTransfer.files?.[0]
                        if (!droppedFile) return
                        void preparePreviewForFile(droppedFile)
                      }}
                    >
                      <UploadCloud className="mx-auto mb-3 h-7 w-7 text-slate-400" />
                      <p className="text-base text-slate-700">
                        Drag and drop your CSV here, or{" "}
                        <button
                          type="button"
                          className="font-medium text-blue-600 hover:underline"
                          onClick={() => fileInputRef.current?.click()}
                        >
                          click to browse
                        </button>
                      </p>
                      <p className="mt-2 text-sm text-slate-500">
                        Only .csv files are supported
                      </p>
                      {newBatchFile && (
                        <p className="mt-3 text-sm font-medium text-slate-700">
                          Selected: {newBatchFile.name}
                        </p>
                      )}
                    </div>
                  </div>
                </div>

                <div className="rounded-lg border border-slate-200">
                  <div className="border-b border-slate-200 px-4 py-3">
                    <p className="text-sm font-medium text-slate-900">
                      Contact Preview (first {PREVIEW_LIMIT})
                    </p>
                  </div>
                  <div className="max-h-[340px] overflow-auto p-3 space-y-2">
                    {newBatchPreviewRows.length === 0 ? (
                      <p className="text-sm text-slate-600">
                        Upload a CSV to preview parsed numbers.
                      </p>
                    ) : (
                      newBatchPreviewRows.map((row) => (
                        <div
                          key={`${row.rowNumber}:${row.normalizedValue}:${row.rawValue}`}
                          className="rounded-md border border-slate-200 bg-white px-3 py-2"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-xs text-slate-500">Row {row.rowNumber}</p>
                            <Badge
                              variant="outline"
                              className={
                                row.isValid
                                  ? "border-emerald-200 text-emerald-700 bg-emerald-50"
                                  : "border-red-200 text-red-700 bg-red-50"
                              }
                            >
                              {row.isValid ? "Valid" : "Invalid"}
                            </Badge>
                          </div>
                          <p className="mt-1 text-sm font-mono break-all text-slate-800">
                            {row.normalizedValue || row.rawValue || "—"}
                          </p>
                        </div>
                      ))
                    )}
                  </div>
                </div>
              </div>

              {newBatchError && (
                <p className="text-sm text-red-600">{newBatchError}</p>
              )}

              <div className="flex justify-end gap-2">
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleNewBatchDialogChange(false)}
                >
                  Cancel
                </Button>
                <Button
                  type="button"
                  onClick={handleCreateBatch}
                  disabled={isCreatingBatch}
                >
                  {isCreatingBatch ? (
                    <>
                      <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    "Proceed"
                  )}
                </Button>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      <Dialog open={isBatchControlsOpen} onOpenChange={setIsBatchControlsOpen}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Batch Controls</DialogTitle>
            <DialogDescription>
              Select an agent, set concurrency, and control execution for this batch.
            </DialogDescription>
          </DialogHeader>

          {selectedBatch ? (
            <div className="space-y-4">
              <div className="rounded-lg bg-slate-50 border border-slate-200 p-3 text-sm">
                <p className="text-slate-700">
                  <span className="font-medium">Batch:</span>{" "}
                  <span className="break-all">{selectedBatch.batch_name}</span>
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Batch ID:</span>{" "}
                  <span className="break-all">{selectedBatch.batch_id}</span>
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Execution Status:</span>{" "}
                  {formatStatusLabel(selectedBatch.execution_status)}
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Batch Status:</span>{" "}
                  {formatStatusLabel(selectedBatch.status)}
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Schedule:</span>{" "}
                  {formatStatusLabel(selectedBatch.scheduled_status || "none")}
                  {selectedBatch.scheduled_at_utc ? ` · ${formatScheduledAt(selectedBatch.scheduled_at_utc)}` : ""}
                </p>
                <p className="text-slate-700 mt-1">
                  <span className="font-medium">Calls:</span>{" "}
                  {selectedBatch.successful_calls || 0} success /{" "}
                  {selectedBatch.failed_calls || 0} failed /{" "}
                  {selectedBatch.attempted_calls || 0} attempted
                </p>
                {selectedBatch.error_message ? (
                  <p className="text-red-600 mt-2">
                    <span className="font-medium">Error:</span>{" "}
                    {selectedBatch.error_message}
                  </p>
                ) : null}
              </div>

              <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">Agent</p>
                  <Select
                    value={selectedBatchAgentType}
                    onValueChange={setSelectedBatchAgentType}
                  >
                    <SelectTrigger className="h-10">
                      <SelectValue placeholder="Select an agent" />
                    </SelectTrigger>
                    <SelectContent>
                      {agents.map((agent) => (
                        <SelectItem
                          key={`${agent.agent_type}:${agent.agent_id}`}
                          value={agent.agent_type}
                        >
                          {agent.agent_id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">
                    Concurrency ({MIN_CONCURRENCY} - {MAX_CONCURRENCY})
                  </p>
                  <Input
                    type="number"
                    min={MIN_CONCURRENCY}
                    max={MAX_CONCURRENCY}
                    value={selectedBatchConcurrency}
                    onChange={(event) => setSelectedBatchConcurrency(event.target.value)}
                  />
                </div>
                <div>
                  <p className="text-sm font-medium text-slate-900 mb-2">Run at (local)</p>
                  <Button
                    type="button"
                    variant="outline"
                    className="w-full justify-start font-normal"
                    onClick={() => openSchedulePicker("selected")}
                  >
                    {selectedBatchScheduledAtLocal
                      ? formatScheduledAt(toIsoWithLocalOffset(selectedBatchScheduledAtLocal))
                      : "Select date and time"}
                  </Button>
                  <p className="mt-1 text-xs text-slate-500">Timezone: {userTimezone}</p>
                </div>
              </div>

              <div className="flex flex-wrap justify-end gap-2">
                <Button
                  type="button"
                  variant="secondary"
                  onClick={() =>
                    handleScheduleBatch(
                      selectedBatch.batch_id,
                      selectedBatchAgentType,
                      selectedBatchConcurrency,
                      selectedBatchScheduledAtLocal,
                      selectedBatch.execution_status === "scheduled"
                    )
                  }
                  disabled={
                    schedulingBatchId === selectedBatch.batch_id ||
                    runningBatchId === selectedBatch.batch_id ||
                    selectedBatch.execution_status === "running" ||
                    selectedBatch.execution_status === "completed"
                  }
                >
                  {schedulingBatchId === selectedBatch.batch_id
                    ? "Scheduling..."
                    : selectedBatch.execution_status === "scheduled"
                    ? "Reschedule"
                    : "Set Date & Time"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleCancelSchedule(selectedBatch.batch_id)}
                  disabled={
                    cancelingScheduleBatchId === selectedBatch.batch_id ||
                    selectedBatch.execution_status !== "scheduled"
                  }
                >
                  {cancelingScheduleBatchId === selectedBatch.batch_id ? "Canceling..." : "Cancel Schedule"}
                </Button>
                <Button
                  type="button"
                  onClick={() =>
                    handleRunBatch(
                      selectedBatch.batch_id,
                      selectedBatchAgentType,
                      selectedBatchConcurrency
                    )
                  }
                  disabled={
                    runningBatchId === selectedBatch.batch_id ||
                    selectedBatch.execution_status === "running" ||
                    selectedBatch.execution_status === "completed"
                  }
                >
                  {runningBatchId === selectedBatch.batch_id ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Running...
                    </>
                  ) : (
                    "Run"
                  )}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => handleStopBatch(selectedBatch.batch_id)}
                  disabled={
                    stoppingBatchId === selectedBatch.batch_id ||
                    selectedBatch.execution_status !== "running"
                  }
                >
                  {stoppingBatchId === selectedBatch.batch_id ? (
                    <>
                      <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                      Stopping...
                    </>
                  ) : (
                    "Stop"
                  )}
                </Button>
              </div>
            </div>
          ) : null}
        </DialogContent>
      </Dialog>

      <Dialog open={isSchedulePickerOpen} onOpenChange={setIsSchedulePickerOpen}>
        <DialogContent className="sm:max-w-md rounded-2xl">
          <DialogHeader>
            <DialogTitle>Select Date & Time</DialogTitle>
            <DialogDescription>
              Pick when you want this batch to start.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="rounded-xl border border-slate-200 bg-slate-50 p-3 flex justify-center">
              <Calendar
                mode="single"
                selected={schedulePickerDate}
                onSelect={setSchedulePickerDate}
                disabled={{ before: new Date() }}
                className="bg-transparent p-0"
              />
            </div>
            <div>
              <p className="text-sm font-medium text-slate-900 mb-2">Time</p>
              <Input
                type="time"
                value={schedulePickerTime}
                onChange={(event) => setSchedulePickerTime(event.target.value)}
              />
              <p className="mt-1 text-xs text-slate-500">Timezone: {userTimezone}</p>
            </div>
            <div className="flex justify-end gap-2">
              <Button
                type="button"
                variant="outline"
                onClick={() => setIsSchedulePickerOpen(false)}
              >
                Cancel
              </Button>
              <Button type="button" onClick={applySchedulePicker}>
                Set
              </Button>
            </div>
          </div>
        </DialogContent>
      </Dialog>
    </div>
  )
}
