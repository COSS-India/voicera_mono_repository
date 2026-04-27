"use client"

import { useState, useRef, useEffect, useMemo } from "react"
import {
  Sheet,
  SheetContent,
  SheetHeader,
  SheetTitle,
} from "@/components/ui/sheet"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { type Meeting, type MeetingDetails } from "@/lib/api"
import {
  ChevronLeft,
  Link2,
  Download,
  Play,
  Pause,
  Clock,
  Braces,
  Bot,
  User,
  Copy,
  PhoneIncoming,
  PhoneOutgoing,
  Loader2,
  Check,
  Hash,
  Phone,
  Zap,
  CalendarClock,
  MessageSquare,
} from "lucide-react"
import { format } from "date-fns"
import { toZonedTime } from "date-fns-tz"
import { getAuthToken } from "@/lib/api"
import { useWavesurfer } from "@wavesurfer/react"
import { motion, AnimatePresence } from "framer-motion"
import { Separator } from "@radix-ui/react-separator"

interface MeetingDetailSheetProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  meeting: Meeting | null
  meetingDetails: MeetingDetails | null
  isLoading: boolean
}

export function MeetingDetailSheet({
  open,
  onOpenChange,
  meeting,
  meetingDetails,
  isLoading,
}: MeetingDetailSheetProps) {
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [copied, setCopied] = useState(false)
  const [copiedLink, setCopiedLink] = useState(false)
  const [audioObjectUrl, setAudioObjectUrl] = useState<string | null>(null)
  const [isLoadingWaveform, setIsLoadingWaveform] = useState(false)
  const waveformRef = useRef<HTMLDivElement>(null)
  const timeUpdateTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Fetch audio as blob and create object URL when recording URL is available
  useEffect(() => {
    const recordingUrl = meetingDetails?.recording_url || meeting?.recording_url
    
    if (!recordingUrl || !open) {
      // Clean up previous object URL
      if (audioObjectUrl) {
        URL.revokeObjectURL(audioObjectUrl)
        setAudioObjectUrl(null)
      }
      return
    }

    setIsLoadingWaveform(true)

    // Fetch audio with authentication
    const fetchAudio = async () => {
      try {
        const token = getAuthToken()
        if (!token) {
          console.error("No auth token available")
          setIsLoadingWaveform(false)
          return
        }

        const response = await fetch(recordingUrl, {
          headers: {
            "Authorization": `Bearer ${token}`,
          },
        })

        if (!response.ok) {
          console.error("Failed to fetch audio:", response.status, response.statusText)
          setIsLoadingWaveform(false)
          return
        }

        const blob = await response.blob()
        const objectUrl = URL.createObjectURL(blob)
        setAudioObjectUrl(objectUrl)
        setIsLoadingWaveform(false)
      } catch (error) {
        console.error("Error fetching audio:", error)
        setIsLoadingWaveform(false)
      }
    }

    fetchAudio()

    // Cleanup function
    return () => {
      // Cleanup will be handled by the next effect run
    }
  }, [meetingDetails?.recording_url, meeting?.recording_url, open])

  // Separate effect to cleanup object URL when component unmounts or meeting changes
  useEffect(() => {
    return () => {
      if (audioObjectUrl) {
        URL.revokeObjectURL(audioObjectUrl)
      }
    }
  }, [audioObjectUrl])


  const formatDuration = (duration: number | null | undefined) => {
    if (!duration) return "N/A"
    const rounded = Math.floor(duration)
    if (rounded < 60) {
      return `${rounded}s`
    }
    return `${Math.floor(rounded / 60)}m ${rounded % 60}s`
  }

  // Initialize WaveSurfer
  const { wavesurfer, isReady } = useWavesurfer({
    container: waveformRef,
    height: 50,
    waveColor: "rgba(255, 255, 255, 0.25)",
    progressColor: "rgba(255, 255, 255, 0.95)",
    cursorColor: "rgba(255, 255, 255, 0.6)",
    barWidth: 2,
    barRadius: 1,
    barGap: 1,
    normalize: true,
    backend: "WebAudio",
    mediaControls: false,
    interact: true,
    dragToSeek: true,
    hideScrollbar: true,
    url: audioObjectUrl || undefined,
    cursorWidth: 1,
    minPxPerSec: 1,
  })

  // Handle WaveSurfer events
  useEffect(() => {
    if (!wavesurfer) return

    const subscriptions = [
      wavesurfer.on("play", () => setIsPlaying(true)),
      wavesurfer.on("pause", () => setIsPlaying(false)),
      wavesurfer.on("timeupdate", (time) => setCurrentTime(time)),
      wavesurfer.on("ready", () => {
        setDuration(wavesurfer.getDuration())
        setIsLoadingWaveform(false)
      }),
      wavesurfer.on("finish", () => {
        setIsPlaying(false)
        setCurrentTime(0)
      }),
    ]

    return () => {
      subscriptions.forEach((unsub) => unsub())
    }
  }, [wavesurfer])

  // Reset audio state when meeting changes
  useEffect(() => {
    if (!open || !meeting) {
      setIsPlaying(false)
      setCurrentTime(0)
      setDuration(0)
      if (wavesurfer) {
        wavesurfer.pause()
        wavesurfer.seekTo(0)
      }
    }
  }, [open, meeting, wavesurfer])

  // Helper functions (defined before useMemo hooks that use them)
  const formatTime = (seconds: number) => {
    if (isNaN(seconds)) return "0:00"
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}:${secs.toString().padStart(2, "0")}`
  }

  const formatDate = (dateString?: string) => {
    if (!dateString) return "N/A"
    try {
      const date = new Date(dateString)
      // Convert UTC to IST (UTC+5:30)
      const istTime = toZonedTime(date, "Asia/Kolkata")
      return format(istTime, "dd MMM yyyy, hh:mm a")
    } catch {
      return dateString
    }
  }

  const formatTranscriptTimestamp = (timestamp?: string) => {
    if (!timestamp) return null
    // Try to parse and format timestamp
    // Timestamps might be in various formats, so we'll try to display them as-is
    // or parse common formats like "HH:MM:SS" or ISO format
    try {
      // If it's a simple time format like "00:12:34"
      if (/^\d{2}:\d{2}:\d{2}/.test(timestamp)) {
        return timestamp
      }
      // If it's an ISO date, extract time portion
      const date = new Date(timestamp)
      if (!isNaN(date.getTime())) {
        return format(date, "HH:mm:ss")
      }
      // Return as-is if we can't parse it
      return timestamp
    } catch {
      return timestamp
    }
  }

  // Derived state for content availability
  const customVars = meetingDetails?.custom_variables || {}
  const transcriptMessages = meetingDetails?.transcript || meeting?.transcript || []
  
  const hasSummaryContent = useMemo(() => {
    return Boolean(
      customVars.summary || 
      customVars.classification || 
      (Array.isArray(customVars.key_points) && customVars.key_points.length > 0) || 
      (Array.isArray(customVars.action_items) && customVars.action_items.length > 0)
    )
  }, [customVars])

  const hasRecording = useMemo(() => {
    return Boolean(meetingDetails?.recording_url || meeting?.recording_url)
  }, [meetingDetails?.recording_url, meeting?.recording_url])

  const hasTranscript = useMemo(() => {
    return transcriptMessages.length > 0
  }, [transcriptMessages.length])

  // Single source of truth for duration
  const displayDuration = useMemo(() => {
    if (meeting?.duration) return meeting.duration
    if (meeting?.start_time_utc && meeting?.end_time_utc) {
      const start = new Date(meeting.start_time_utc)
      const end = new Date(meeting.end_time_utc)
      return (end.getTime() - start.getTime()) / 1000
    }
    return null
  }, [meeting?.duration, meeting?.start_time_utc, meeting?.end_time_utc])

  // Formatted values computed once
  const formattedDate = useMemo(() => {
    return formatDate(meeting?.start_time_utc || meeting?.created_at)
  }, [meeting?.start_time_utc, meeting?.created_at])

  const formattedDuration = useMemo(() => {
    if (!displayDuration) return null
    return formatTime(displayDuration)
  }, [displayDuration])

  // Determine default tab based on available content
  const defaultTab = useMemo(() => {
    if (hasSummaryContent) return 'summary'
    if (hasTranscript) return 'transcript'
    return 'details'
  }, [hasSummaryContent, hasTranscript])

  // Active tab state - initialize with defaultTab
  const [activeTab, setActiveTab] = useState<string>(() => {
    if (hasSummaryContent) return 'summary'
    if (hasTranscript) return 'transcript'
    return 'details'
  })

  // Update activeTab when defaultTab changes
  useEffect(() => {
    setActiveTab(defaultTab)
  }, [defaultTab])

  

  

  const togglePlay = () => {
    if (!wavesurfer) return
    if (isPlaying) {
      wavesurfer.pause()
    } else {
      wavesurfer.play()
    }
  }

  const copyMeetingId = async () => {
    if (!meeting?.meeting_id) return
    try {
      await navigator.clipboard.writeText(meeting.meeting_id)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    } catch (error) {
      console.error("Failed to copy:", error)
    }
  }

  

  const downloadAudio = async () => {
    const recordingUrl = meetingDetails?.recording_url || meeting?.recording_url
    if (!recordingUrl) return
    try {
      const token = getAuthToken()
      if (!token) {
        console.error("No auth token available for recording download")
        return
      }

      const response = await fetch(recordingUrl, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
      })

      if (!response.ok) {
        console.error("Failed to download recording:", response.status, response.statusText)
        return
      }

      const blob = await response.blob()
      const blobUrl = URL.createObjectURL(blob)
      const link = document.createElement("a")
      link.href = blobUrl
      link.download = `${meeting?.meeting_id || "recording"}.wav`
      document.body.appendChild(link)
      link.click()
      document.body.removeChild(link)
      URL.revokeObjectURL(blobUrl)
    } catch (error) {
      console.error("Error downloading recording:", error)
    }
  }

  const downloadTranscript = () => {
    if (!meetingDetails?.transcript_content) return
    const blob = new Blob([meetingDetails.transcript_content], { type: "text/plain" })
    const url = URL.createObjectURL(blob)
    const a = document.createElement("a")
    a.href = url
    a.download = `${meeting?.meeting_id || "transcript"}_transcript.txt`
    a.click()
    URL.revokeObjectURL(url)
  }

  // Get agent name or fallback to from_number
  const agentName = useMemo(() => {
    return meeting?.agent_type || (meeting?.from_number ? `Call with ${meeting.from_number}` : "Call")
  }, [meeting?.agent_type, meeting?.from_number])

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent side="right" className="w-md sm:w-[480px] overflow-y-auto p-0 rounded-xl">
        {/* HEADER - Minimal, action-focused */}
        <SheetHeader className="px-5 py-4 border-b border-slate-200 bg-white sticky top-0 z-10">
          <SheetTitle className="sr-only">Call Details</SheetTitle>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 flex-1 min-w-0">
              <Button
                variant="ghost"
                size="icon"
                onClick={() => onOpenChange(false)}
                className="h-8 w-8 flex-shrink-0"
                aria-label="Close"
              >
                <ChevronLeft className="h-4 w-4" />
              </Button>
              <div className="flex-1 min-w-0">
                <h2 className="text-lg font-semibold text-slate-900 truncate">{agentName}</h2>
                <p className="text-xs text-slate-500 mt-0.5">{formattedDate}</p>
              </div>
            </div>
            <div className="flex items-center gap-2 flex-shrink-0">
            </div>
          </div>
        </SheetHeader>

        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Loader2 className="h-6 w-6 animate-spin text-slate-400" />
            <span className="ml-2 text-slate-500">Loading call details...</span>
          </div>
        ) : meeting ? (
          <div className="flex flex-col">
            {/* HERO SECTION - Audio Player with Waveform (if recording exists) */}
            {hasRecording && (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.3, ease: "easeOut" }}
                className="px-4 py-3 bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 mx-5 mt-4 rounded-xl shadow-xl border border-slate-700/50"
              >
                <div className="flex items-center gap-3">
                  {/* Play/Pause Button */}
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={togglePlay}
                      disabled={!isReady || isLoadingWaveform}
                      className="h-10 w-10 rounded-full bg-white/10 hover:bg-white/20 text-white flex-shrink-0 backdrop-blur-sm border border-white/10 shadow-lg transition-all"
                      aria-label={isPlaying ? "Pause" : "Play"}
                    >
                      <AnimatePresence mode="wait">
                        {isLoadingWaveform ? (
                          <motion.div
                            key="loading"
                            initial={{ opacity: 0, rotate: -90 }}
                            animate={{ opacity: 1, rotate: 0 }}
                            exit={{ opacity: 0, rotate: 90 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Loader2 className="h-4 w-4 animate-spin" />
                          </motion.div>
                        ) : isPlaying ? (
                          <motion.div
                            key="pause"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Pause className="h-4 w-4" />
                          </motion.div>
                        ) : (
                          <motion.div
                            key="play"
                            initial={{ opacity: 0, scale: 0.8 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.8 }}
                            transition={{ duration: 0.2 }}
                          >
                            <Play className="h-4 w-4 ml-0.5" />
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </Button>
                  </motion.div>

                  {/* Waveform Container with Time Display */}
                  <div className="flex-1 min-w-0 flex items-center gap-3">
                    <div className="relative flex-1 min-w-0" style={{ height: "50px" }}>
                      <AnimatePresence mode="wait">
                        {isLoadingWaveform && !isReady ? (
                          <motion.div
                            key="skeleton"
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            className="absolute inset-0 flex items-center justify-center"
                          >
                            <div className="w-full h-full flex items-end justify-center gap-0.5 px-2">
                              {Array.from({ length: 80 }).map((_, i) => (
                                <motion.div
                                  key={i}
                                  className="w-0.5 bg-white/20 rounded-full"
                                  style={{ height: `${20 + Math.random() * 60}%` }}
                                  animate={{
                                    height: [`${20 + Math.random() * 60}%`, `${20 + Math.random() * 60}%`, `${20 + Math.random() * 60}%`],
                                  }}
                                  transition={{
                                    duration: 0.8 + Math.random() * 0.4,
                                    repeat: Infinity,
                                    ease: "easeInOut",
                                    delay: Math.random() * 0.3,
                                  }}
                                />
                              ))}
                            </div>
                          </motion.div>
                        ) : null}
                      </AnimatePresence>
                      <motion.div 
                        ref={waveformRef}
                        className="w-full h-full"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: isReady ? 1 : 0 }}
                        transition={{ duration: 0.3 }}
                      />
                    </div>
                    {/* Time Display - Inline with waveform */}
                    <div className="flex items-center gap-1 flex-shrink-0">
                      <motion.span
                        key={`current-time-${meeting?.meeting_id || 'default'}-${currentTime}`}
                        initial={{ opacity: 0.5 }}
                        animate={{ opacity: 1 }}
                        className="text-xs text-white/90  tabular-nums whitespace-nowrap"
                      >
                        {formatTime(currentTime)}
                      </motion.span>
                      <span className="text-xs text-white/40">/</span>
                      <motion.span
                        key={`duration-${meeting?.meeting_id || 'default'}-${duration}`}
                        initial={{ opacity: 0.5 }}
                        animate={{ opacity: 1 }}
                        className="text-xs text-white/60  tabular-nums whitespace-nowrap"
                      >
                        {formatTime(duration)}
                      </motion.span>
                    </div>
                  </div>

                  {/* Download Button */}
                  <motion.div
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={downloadAudio}
                      className="h-9 w-9 rounded-full bg-white/10 hover:bg-white/20 text-white flex-shrink-0 backdrop-blur-sm border border-white/10 shadow-lg transition-all"
                      aria-label="Download audio"
                    >
                      <Download className="h-3.5 w-3.5" />
                    </Button>
                  </motion.div>
                </div>
              </motion.div>
            )}

            {/* METADATA STRIP - Single row, scannable */}
            <div className="px-5 py-3 bg-white border-b border-slate-200 flex items-center gap-3 flex-wrap">
              <span className="text-xs text-slate-500">{formattedDate}</span>
              <Badge 
                variant="outline" 
                className="bg-slate-100 text-slate-700 border-slate-200"
              >
                {meeting.inbound ? (
                  <>
                    <PhoneIncoming className="h-3 w-3" />
                    <span>Inbound</span>
                  </>
                ) : (
                  <>
                    <PhoneOutgoing className="h-3 w-3" />
                    <span>Outbound</span>
                  </>
                )}
              </Badge>
              <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200">
                {meeting.call_busy ? "Busy" : (meeting.end_time_utc ? "Completed" : "In Progress")}
              </Badge>
              {!hasRecording && meeting.duration && (
                <span className="text-xs text-slate-500 flex items-center gap-1">
                  <Clock className="h-3 w-3" />
                  {meeting.duration}s
                </span>
              )}
            </div>

            {/* TABS - Elevated Design */}
            <Tabs 
              defaultValue={defaultTab} 
              value={activeTab}
              onValueChange={setActiveTab}
              className="flex-1 flex flex-col"
            >
              {/* Tab Bar - Integrated Actions */}
              <div className="flex items-center justify-between px-5 border-b border-slate-200 bg-white sticky top-0 z-10">
                <TabsList className="h-11 bg-transparent p-0 gap-1">
                  {hasTranscript && (
                    <TabsTrigger 
                      value="transcript"
                      className="relative h-11 px-4 rounded-none bg-transparent 
                                 data-[state=active]:bg-transparent
                                 data-[state=active]:shadow-none
                                 data-[state=inactive]:text-slate-500
                                 data-[state=active]:text-slate-900
                                 data-[state=active]:font-semibold
                                 after:absolute after:bottom-0 after:left-0 after:right-0 
                                 after:h-0.5 after:bg-slate-900 
                                 after:scale-x-0 data-[state=active]:after:scale-x-100
                                 after:transition-transform after:duration-200"
                    >
                      Transcript
                    </TabsTrigger>
                  )}
                  <TabsTrigger 
                    value="details"
                    className="relative h-11 px-4 rounded-none bg-transparent 
                               data-[state=active]:bg-transparent
                               data-[state=active]:shadow-none
                               data-[state=inactive]:text-slate-500
                               data-[state=active]:text-slate-900
                               data-[state=active]:font-semibold
                               after:absolute after:bottom-0 after:left-0 after:right-0 
                               after:h-0.5 after:bg-slate-900 
                               after:scale-x-0 data-[state=active]:after:scale-x-100
                               after:transition-transform after:duration-200"
                  >
                    Details
                  </TabsTrigger>
                </TabsList>
                
                {/* Contextual Actions - Show based on active tab */}
                {activeTab === "transcript" && hasTranscript && (
                  <Button 
                    variant="ghost" 
                    size="sm"
                    onClick={downloadTranscript}
                    className="h-8 text-slate-600 hover:text-slate-900"
                    disabled={!meetingDetails?.transcript_content}
                  >
                    <Download className="h-4 w-4 mr-1.5" />
                    <span className="text-xs">Export</span>
                  </Button>
                )}
              </div>

              {/* TRANSCRIPT TAB */}
              {hasTranscript && (
                <TabsContent 
                  value="transcript" 
                  className="flex-1 overflow-y-auto m-0 p-0"
                >
                  {transcriptMessages.length > 0 ? (
                    <div className="px-5 py-6 space-y-5 bg-gradient-to-b from-slate-50 to-white min-h-full">
                      {transcriptMessages.map((message, idx) => {
                        const isUser = message.role === "user"
                        return (
                          <div
                            key={idx}
                            className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}
                          >
                            {/* Avatar */}
                            <div
                              className={`
                                h-8 w-8 rounded-full flex items-center justify-center flex-shrink-0
                                ${isUser 
                                  ? "bg-gradient-to-br from-blue-500 to-blue-600 text-white" 
                                  : "bg-gradient-to-br from-slate-100 to-slate-200 text-slate-600"
                                }
                              `}
                            >
                              {isUser ? <User className="h-4 w-4" /> : <Bot className="h-4 w-4" />}
                            </div>
                            
                            {/* Message Bubble */}
                            <div className={`max-w-[75%] flex flex-col ${isUser ? "items-end" : "items-start"}`}>
                              {/* Timestamp - Above bubble */}
                              {message.timestamp && (
                                <span className={`
                                  text-[10px] text-slate-400 mb-1 block
                                  ${isUser ? "text-right mr-1" : "ml-1"}
                                `}>
                                  {formatTranscriptTimestamp(message.timestamp)}
                                </span>
                              )}
                              
                              {/* Bubble */}
                              <div
                                className={`
                                  rounded-2xl px-4 py-2.5 
                                  ${isUser 
                                    ? "bg-blue-500 text-white rounded-br-md" 
                                    : "bg-white text-slate-900 border border-slate-200 rounded-bl-md shadow-sm"
                                  }
                                `}
                              >
                                <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
                              </div>
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  ) : (
                    <div className="flex flex-col items-center justify-center py-16 px-5 text-center">
                      <div className="h-12 w-12 rounded-full bg-slate-100 flex items-center justify-center mb-4">
                        <MessageSquare className="h-6 w-6 text-slate-400" />
                      </div>
                      <h3 className="text-sm font-medium text-slate-900 mb-1">No transcript available</h3>
                      <p className="text-xs text-slate-500 max-w-[240px]">
                        Transcripts are generated automatically for recorded calls
                      </p>
                    </div>
                  )}
                </TabsContent>
              )}

              {/* DETAILS TAB */}
              <TabsContent 
                value="details" 
                className="flex-1 overflow-y-auto m-0"
              >
                <div className="p-5 space-y-1">
                  
                  {/* Detail Cards - Consistent Pattern */}
                  <div className="bg-white rounded-xl border border-slate-200 divide-y divide-slate-100 overflow-hidden">
                    
                    {/* Meeting ID Row */}
                    {meeting.meeting_id && (
                      <div className="flex items-center justify-between px-4 py-3 hover:bg-slate-50 transition-colors">
                        <div className="flex items-center gap-3 min-w-0">
                          <div className="h-9 w-9 rounded-lg bg-slate-100 flex items-center justify-center flex-shrink-0">
                            <Hash className="h-4 w-4 text-slate-500" />
                          </div>
                          <div className="min-w-0">
                            <div className="text-xs text-slate-500 font-medium">Call ID</div>
                            <div className="text-sm text-slate-900 font-mono truncate max-w-[200px]" title={meeting.meeting_id}>
                              {meeting.meeting_id}
                            </div>
                          </div>
                        </div>
                        <Button
                          variant="ghost"
                          size="sm"
                          onClick={copyMeetingId}
                          className="h-8 w-8 p-0 flex-shrink-0"
                        >
                          {copied ? (
                            <Check className="h-4 w-4 text-emerald-500" />
                          ) : (
                            <Copy className="h-4 w-4 text-slate-400" />
                          )}
                        </Button>
                      </div>
                    )}

                    {/* Caller Number Row */}
                    {meeting.from_number && (
                      <div className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors">
                        <div className="h-9 w-9 rounded-lg bg-blue-50 flex items-center justify-center flex-shrink-0">
                          <Phone className="h-4 w-4 text-blue-600" />
                        </div>
                        <div>
                          <div className="text-xs text-slate-500 font-medium">Caller</div>
                          <div className="text-sm text-slate-900 font-mono">{meeting.from_number}</div>
                        </div>
                      </div>
                    )}

                    {/* Duration Row (if not shown in player) */}
                    { meeting.duration && (
                      <div className="flex items-center gap-3 px-4 py-3 hover:bg-slate-50 transition-colors">
                        <div className="h-9 w-9 rounded-lg bg-amber-50 flex items-center justify-center flex-shrink-0">
                          <Clock className="h-4 w-4 text-amber-600" />
                        </div>
                        <div>
                          <div className="text-xs text-slate-500 font-medium">Duration</div>
                          <div className="text-sm text-slate-900">{formatDuration(meeting.duration)}</div>
                        </div>
                      </div>
                    )}

                    

                  </div>

                 

                </div>
              </TabsContent>
            </Tabs>
          </div>
        ) : (
          <div className="px-5 py-12 text-center">
            <p className="text-slate-500">No meeting data available</p>
          </div>
        )}
      </SheetContent>
    </Sheet>
  )
}
