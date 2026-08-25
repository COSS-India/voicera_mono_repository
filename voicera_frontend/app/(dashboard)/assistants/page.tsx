"use client"

import { useState, useMemo, useEffect } from "react"
import { useRouter } from "next/navigation"
import { useQueryClient } from "@tanstack/react-query"
import { formatDistanceToNow } from "date-fns"
import { isProviderAvailable, isTelephonyProviderAvailable } from "@/lib/provider-availability"
import { getCurrentUser, createAgent, createVobizApplication, createPlivoApplication, deleteVobizApplication, deletePlivoApplication, deleteAgent, unlinkVobizNumber, unlinkPlivoNumber, fetchApiRoute, getIntegrations, getCustomLLMIntegrations, getSelfHostedProviders, getKnowledgeDocuments, type User, type Agent, type CreateAgentRequest, type Integration, type CustomLLMIntegration, type KnowledgeDocument, type InteractionMode } from "@/lib/api"
import { isValidAgentName, sanitizeAgentNameInput, slugifyAgentId, validateAgentName } from "@/lib/agent-name"
import { agentsQueryKey, useAgentsQuery } from "@/lib/queries/agents"
import { requireJohnaicServerUrl } from "@/lib/johnaic-config"
import {
  agentCategoryForProvider,
  isTelephonyAgent,
  isWebSocketAgent,
  WEBSOCKET_PROVIDER,
} from "@/lib/agent-delivery"
import { Separator } from "@/components/ui/separator"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Textarea } from "@/components/ui/textarea"
import { Slider } from "@/components/ui/slider"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { TestCallSheet } from "@/components/assistants/test-call-sheet"
import { TestBrowserDialog } from "@/components/assistants/test-browser-dialog"
import { AgentCard } from "@/components/assistants/agent-card"
import { CreateNewAgentCard } from "@/components/assistants/create-new-agent-card"
import { SidebarTrigger } from "@/components/ui/sidebar"
import { LanguageSelectionSection } from "@/components/assistants/language-selection-section"
import { GreetingCommaBanner } from "@/components/assistants/greeting-comma-banner"
import { GreetingTranslateSuggestion } from "@/components/assistants/greeting-translate-suggestion"
import {
  SpokenMessageInput,
  SpokenMessageTextarea,
} from "@/components/assistants/spoken-message-field"
import {
  ChevronRight,
  ChevronLeft,
  Plus,
  Trash2,
  Search,
  RefreshCw,
  Phone,
  BarChart3,
  FileText,
  Settings,
  Volume2,
  CheckCircle2,
  Mic,
  Loader2,
  Check,
  X,
  Timer,
  MessageSquare,
  Bell,
} from "lucide-react"

// Import JSON data
import sttData from "@/stt.json"
import { displayLanguageName } from "@/lib/languageLabels"
import {
  buildLanguageConfigFields,
  getActiveLanguages,
  getIntersectedSTTModels,
  getIntersectedSTTProviders,
  getIntersectedTTSModels,
  getIntersectedTTSProviders,
  hasMultipleLanguages,
} from "@/lib/languageModelSupport"
import {
  type KenpathVariant,
  isBharatVistaarLanguageSupported,
  kenpathLlmFieldsFromVariant,
  kenpathVariantHelpText,
  kenpathVariantLabel,
} from "@/lib/kenpath"
import ttsData from "@/tts.json"
import descriptionsData from "@/descriptions.json"

// Provider name mappings for official names (used for display and database)
const getProviderOfficialName = (providerId: string): string => {
  const nameMap: Record<string, string> = {
    assembly: "Assembly",
    azure: "Azure",
    anthropic: "Anthropic",
    deepgram: "Deepgram",
    elevenlabs: "Elevenlabs",
    gladia: "Gladia",
    google: "Google",
    gcp: "Google", // GCP is officially called Google
    kenpath: "Kenpath",
    pixa: "Pixa",
    sarvam: "Sarvam",
    smallest: "Smallest",
    ai4bharat: "AI4Bharat",
    bhashini: "Bhashini",
    cartesia: "Cartesia",
    openai: "OpenAI",
    qwen: "Qwen",
    playht: "PlayHT",
    groq: "Groq",
    grok: "Grok",
    custom_llm: "Custom LLM",
  }
  return nameMap[providerId] || providerId.charAt(0).toUpperCase() + providerId.slice(1)
}

// Alias for backward compatibility
const getProviderDisplayName = getProviderOfficialName

// LLM Provider configurations
const llmProviders = {
  azure: {
    name: "Azure",
    models: [
      "gpt-4.1-mini cluster",
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4-turbo",
    ],
  },
  openai: {
    name: "OpenAI",
    models: [
      "gpt-4o",
      "gpt-4o-mini",
      "gpt-4-turbo",
      "gpt-4",
      "gpt-3.5-turbo",
      "o1",
      "o1-mini",
      "o1-preview",
    ],
  },
  qwen: {
    name: "Qwen",
    models: [
      "Qwen/Qwen3-8B",
    ],
  },
  kenpath: {
    name: "Kenpath",
    models: [],
  },
  anthropic: {
    name: "Anthropic",
    models: [
      "claude-sonnet-4-5-20250929",
      "claude-opus-4-6-20250929",
      "claude-sonnet-4-20250514",
      "claude-3-5-sonnet-20241022",
      "claude-3-5-haiku-20241022",
      "claude-3-opus-20240229",
    ],
  },
  google: {
    name: "Google",
    models: [
      "gemini-2.0-flash",
      "gemini-2.0-flash-lite",
      "gemini-1.5-pro",
      "gemini-1.5-flash",
    ],
  },
  groq: {
    name: "Groq",
    models: [
      "llama-3.3-70b-versatile",
      "llama-3.1-8b-instant",
      "mixtral-8x7b-32768",
    ],
  },
  grok: {
    name: "Grok",
    models: [
      "grok-3-beta",
      "grok-2-1212",
      "grok-2-vision-1212",
    ],
  },
  custom_llm: {
    name: "Custom LLM",
    models: [],
  },
}

// Helper function to get agent display name from config
const getAgentDisplayName = (agent: Agent): string => {
  const agentType = agent.agent_type || ""
  if (agentType.length > 0) {
    return agentType
  }
  return `Agent ${agent.id?.slice(0, 8) || "Unknown"}`
}

// Helper function to get agent description from config
const getAgentDescription = (agent: Agent): string => {
  if (agent.agent_config?.interaction_mode === "non_conversational") {
    const alert = agent.agent_config?.greeting_message || ""
    if (alert.length > 0) {
      return alert.slice(0, 50) + (alert.length > 50 ? "..." : "")
    }
    return "One-way alert agent"
  }
  const prompt = agent.agent_config?.system_prompt || ""
  if (prompt.length > 0) {
    return prompt.slice(0, 50) + (prompt.length > 50 ? "..." : "")
  }
  return "Voice Assistant"
}

// Types
type WizardStepKey = "type" | "agent" | "llm" | "audio" | "telephony" | "call_mgmt" | "review"

const WIZARD_STEP_META: Record<
  WizardStepKey,
  { title: string; subtitle: string; icon: typeof FileText }
> = {
  type: { title: "Type", subtitle: "Agent Mode", icon: MessageSquare },
  agent: { title: "Agent", subtitle: "Name & Prompt", icon: FileText },
  llm: { title: "LLM", subtitle: "Model Config", icon: Settings },
  audio: { title: "Audio", subtitle: "STT & TTS", icon: Volume2 },
  telephony: { title: "Delivery", subtitle: "Telephony or WebSocket", icon: Phone },
  call_mgmt: { title: "Call Management", subtitle: "Timeouts & Silence", icon: Timer },
  review: { title: "Review", subtitle: "Confirm", icon: CheckCircle2 },
}

function getWizardStepKeys(mode: InteractionMode | null): WizardStepKey[] {
  if (!mode) return ["type"]
  if (mode === "non_conversational") {
    return ["type", "agent", "audio", "telephony", "review"]
  }
  return ["type", "agent", "llm", "audio", "telephony", "call_mgmt", "review"]
}

interface AgentConfig {
  id: string
  interactionMode: InteractionMode | null
  name?: string
  greetingMessage?: string
  ignoreUserSpeechBeforeGreeting: boolean
  interruptionMinWords: number
  userSilenceHangupSeconds: number
  callTimeoutSeconds: number
  holdMessages: string[]
  holdMessageTimeoutSeconds: number
  userOnlineDetectionEnabled: boolean
  userOnlineDetectionMessage: string
  userOnlineDetectionSeconds: number
  userOnlineDetectionRepeats: number
  userOnlineDetectionClosingMessage: string
  systemPrompt: string
  llmProvider: string
  llmModel: string
  customLlmId: string
  kenpathVariant: KenpathVariant
  knowledgeEnabled: boolean
  knowledgeDocumentIds: string[]
  knowledgeTopK: number
  temperature: number
  maxTokens: number
  selectedLanguages: string[]
  sttProvider: string
  sttModel: string
  keywords: string
  ttsProvider: string
  ttsModel: string
  ttsVoice: string
  ttsDescription: string
  bufferSize: number
  speedRate: number
  similarityBoost: number
  stability: number
  telephonyProvider: string
}
type AgentWithTelephony = Agent & {
  plivo_app_id?: string
}

const defaultConfig: AgentConfig = {
  id: "",
  interactionMode: null,
  name: "",
  greetingMessage: "",
  ignoreUserSpeechBeforeGreeting: true,
  interruptionMinWords: 1,
  userSilenceHangupSeconds: 30,
  callTimeoutSeconds: 600,
  holdMessages: [],
  holdMessageTimeoutSeconds: 0.3,
  userOnlineDetectionEnabled: false,
  userOnlineDetectionMessage: "",
  userOnlineDetectionSeconds: 10,
  userOnlineDetectionRepeats: 1,
  userOnlineDetectionClosingMessage: "",
  systemPrompt: "You are a helpful agent. You will help the customer with their queries and doubts. You will never speak more than 2 sentences. Keep your responses concise",
  llmProvider: "",
  llmModel: "",
  customLlmId: "",
  kenpathVariant: "prod",
  knowledgeEnabled: false,
  knowledgeDocumentIds: [],
  knowledgeTopK: 3,
  temperature: 0.2,
  maxTokens: 450,
  selectedLanguages: ["Hindi"],
  sttProvider: "",
  sttModel: "",
  keywords: "",
  ttsProvider: "",
  ttsModel: "",
  ttsVoice: "",
  ttsDescription: "",
  bufferSize: 50,
  speedRate: 1,
  similarityBoost: 75,
  stability: 50,
  telephonyProvider: "",
}

const formatDurationSeconds = (seconds: number) => {
  if (seconds <= 0) return "Disabled"
  if (seconds >= 60 && seconds % 60 === 0) return `${seconds / 60} min`
  if (seconds >= 60) {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return secs > 0 ? `${mins}m ${secs}s` : `${mins} min`
  }
  return `${seconds}s`
}

export default function AssistantsPage() {
  const router = useRouter()
  const queryClient = useQueryClient()
  const [searchQuery, setSearchQuery] = useState("")
  const [agentSortOrder, setAgentSortOrder] = useState<
    "newest" | "oldest" | "active-first" | "inactive-first"
  >("active-first")
  const [config, setConfig] = useState<AgentConfig>(defaultConfig)
  const [view, setView] = useState<"list" | "create">("list")
  const [createStep, setCreateStep] = useState(1)
  const [interactionModeLocked, setInteractionModeLocked] = useState(false)
  const [user, setUser] = useState<User | null>(null)
  const [isCreatingAgent, setIsCreatingAgent] = useState(false)
  const [isTestCallSheetOpen, setIsTestCallSheetOpen] = useState(false)
  const [isTestBrowserDialogOpen, setIsTestBrowserDialogOpen] = useState(false)
  const [selectedAgentForTest, setSelectedAgentForTest] = useState<Agent | null>(null)
  const [showDeleteSuccessToast, setShowDeleteSuccessToast] = useState(false)
  const [deletingAgentType, setDeletingAgentType] = useState<string | null>(null)
  const [integratedProviders, setIntegratedProviders] = useState<Set<string>>(new Set())
  // Self-hosted models the voice server can reach, keyed "llm:qwen", "stt:ai4bharat", "tts:ai4bharat"
  const [selfHostedProviders, setSelfHostedProviders] = useState<Set<string>>(new Set())
  const [customLLMIntegrations, setCustomLLMIntegrations] = useState<CustomLLMIntegration[]>([])
  const [knowledgeDocs, setKnowledgeDocs] = useState<KnowledgeDocument[]>([])
  const [isKnowledgeLoading, setIsKnowledgeLoading] = useState(false)

  const activeWizardSteps = useMemo(() => {
    const keys = getWizardStepKeys(config.interactionMode)
    return keys.map((key, index) => ({
      id: index + 1,
      key,
      ...WIZARD_STEP_META[key],
      subtitle:
        key === "agent" && config.interactionMode === "non_conversational"
          ? "Name & Alert"
          : key === "audio" && config.interactionMode === "non_conversational"
            ? "TTS"
            : WIZARD_STEP_META[key].subtitle,
    }))
  }, [config.interactionMode])

  const WIZARD_STEP_COUNT = activeWizardSteps.length
  const currentStepKey = activeWizardSteps[createStep - 1]?.key

  const getStepIdByKey = (key: WizardStepKey) =>
    activeWizardSteps.find((step) => step.key === key)?.id ?? 1

  useEffect(() => {
    if (view === "create" && createStep > activeWizardSteps.length) {
      setCreateStep(activeWizardSteps.length)
    }
  }, [view, createStep, activeWizardSteps.length])

  const {
    data: agents = [],
    isPending: isLoadingAgents,
    isError: isAgentsError,
    error: agentsError,
    refetch: refetchAgents,
    isFetching: isFetchingAgents,
    dataUpdatedAt,
  } = useAgentsQuery(user?.org_id)

  // Fetch user, integrations, and knowledge docs on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const userData = await getCurrentUser()
        setUser(userData)

        // Fetch integrations to know which providers have API keys
        try {
          const [integrations, customLlms, selfHosted] = await Promise.all([
            getIntegrations(),
            getCustomLLMIntegrations(),
            getSelfHostedProviders(),
          ])
          setCustomLLMIntegrations(customLlms)
          setSelfHostedProviders(new Set(selfHosted))
          const integrated = new Set<string>()
          integrations.forEach((integration: Integration) => {
            integrated.add(integration.model.toLowerCase())
          })
          if (customLlms.length > 0) {
            integrated.add("custom_llm")
            integrated.add("custom llm")
          }
          setIntegratedProviders(integrated)
        } catch (intError) {
          console.error("Failed to fetch integrations:", intError)
        }
        try {
          setIsKnowledgeLoading(true)
          const docs = await getKnowledgeDocuments()
          setKnowledgeDocs(docs.filter((d) => d.status === "ready"))
        } catch (kbError) {
          console.error("Failed to fetch knowledge docs:", kbError)
          setKnowledgeDocs([])
        } finally {
          setIsKnowledgeLoading(false)
        }
      } catch (error) {
        console.error("Failed to fetch data:", error)
        router.push("/")
      }
    }
    fetchData()
  }, [router])

  useEffect(() => {
    if (isAgentsError) {
      console.error("Failed to fetch agents:", agentsError)
    }
  }, [isAgentsError, agentsError])

  // Filter agents based on search
  const filteredAgents = agents.filter(
    (agent) => {
      const name = getAgentDisplayName(agent).toLowerCase()
      const description = getAgentDescription(agent).toLowerCase()
      const query = searchQuery.toLowerCase()
      return name.includes(query) || description.includes(query)
    }
  )

  const sortedAgents = useMemo(() => {
    const copy = [...filteredAgents]
    const isAgentActive = (agent: Agent) => {
      if (isWebSocketAgent(agent)) return true
      return Boolean(agent.phone_number?.trim())
    }
    /** Prefer created_at; fall back to updated_at for legacy rows without created_at. */
    const sortTime = (a: Agent) =>
      new Date(a.created_at || a.updated_at || 0).getTime()

    copy.sort((a, b) => {
      if (agentSortOrder === "active-first" || agentSortOrder === "inactive-first") {
        const aActive = isAgentActive(a)
        const bActive = isAgentActive(b)
        if (aActive !== bActive) {
          return agentSortOrder === "active-first"
            ? aActive
              ? -1
              : 1
            : aActive
              ? 1
              : -1
        }
        const timeDiff = sortTime(b) - sortTime(a)
        if (timeDiff !== 0) return timeDiff
        return a.agent_type.localeCompare(b.agent_type)
      }

      const diff =
        agentSortOrder === "newest"
          ? sortTime(b) - sortTime(a)
          : sortTime(a) - sortTime(b)
      if (diff !== 0) return diff
      return a.agent_type.localeCompare(b.agent_type)
    })
    return copy
  }, [filteredAgents, agentSortOrder])

  const viewConfig = (agent: Agent) => {
    // agent_type is the stable backend lookup key (unique per org). Composite IDs built from
    // created_at ISO strings break hyphen-based parsing in the detail API route.
    const agentId =
      agent.agent_type ||
      agent.agent_id ||
      agent.id ||
      agent._id
    if (!agentId || agentId === "undefined" || agentId.includes("undefined")) {
      console.error("Agent ID is missing or invalid:", agent)
      return
    }
    router.push(`/assistants/${encodeURIComponent(agentId)}`)
  }

  // Get all unique languages from both STT and TTS
  const allLanguages = useMemo(() => {
    const sttLangs = Object.keys(sttData.stt.languages)
    const ttsLangs = Object.keys(ttsData.tts.languages)
    const merged = new Set([...sttLangs, ...ttsLangs])
    return Array.from(merged)
      .sort()
      .map((code) => ({ code, name: displayLanguageName(code) }))
  }, [])

  // Derive all STT providers from JSON (across all languages)
  const allSTTProviders = useMemo(() => {
    const providerSet = new Set<string>()
    Object.values(sttData.stt.languages).forEach((langData) => {
      Object.keys(langData.models).forEach((provider) => {
        providerSet.add(provider)
      })
    })
    return Array.from(providerSet).map((id) => ({
      id,
      name: getProviderDisplayName(id),
    }))
  }, [])

  // Derive all TTS providers from JSON (across all languages)
  const allTTSProviders = useMemo(() => {
    const providerSet = new Set<string>()
    Object.values(ttsData.tts.languages).forEach((langData) => {
      Object.keys(langData.models).forEach((provider) => {
        providerSet.add(provider)
      })
    })
    return Array.from(providerSet).map((id) => ({
      id,
      name: getProviderDisplayName(id),
    }))
  }, [])

  // Derive all STT models for a provider from JSON (across all languages)
  const getAllSTTModelsForProvider = useMemo(() => {
    const modelMap: Record<string, Set<string>> = {}
    Object.values(sttData.stt.languages).forEach((langData) => {
      Object.entries(langData.models).forEach(([provider, models]) => {
        if (!modelMap[provider]) {
          modelMap[provider] = new Set<string>()
        }
        if (Array.isArray(models)) {
          models.forEach((model) => modelMap[provider].add(model))
        }
      })
    })
    // Convert Sets to Arrays for easier use
    const result: Record<string, string[]> = {}
    Object.entries(modelMap).forEach(([provider, modelSet]) => {
      result[provider] = Array.from(modelSet)
    })
    return result
  }, [])

  // Derive all TTS models for a provider from JSON (across all languages)
  const getAllTTSModelsForProvider = useMemo(() => {
    const modelMap: Record<string, Set<string>> = {}
    Object.values(ttsData.tts.languages).forEach((langData) => {
      Object.entries(langData.models).forEach(([provider, providerData]) => {
        if (!modelMap[provider]) {
          modelMap[provider] = new Set<string>()
        }
        const data = providerData as { model?: string; models?: string[]; available?: boolean }
        if (data.models && Array.isArray(data.models)) {
          data.models.forEach((model) => modelMap[provider].add(model))
        }
        if (data.model) {
          modelMap[provider].add(data.model)
        }
      })
    })
    // Convert Sets to Arrays for easier use
    const result: Record<string, string[]> = {}
    Object.entries(modelMap).forEach(([provider, modelSet]) => {
      result[provider] = Array.from(modelSet)
    })
    return result
  }, [])

  // Languages used for STT/TTS intersection (primary + optional secondary)
  const activeLanguages = useMemo(
    () => getActiveLanguages(config.selectedLanguages),
    [config.selectedLanguages]
  )
  const primaryLanguage = activeLanguages[0] || ""

  // Get supported STT providers for all selected languages (intersection)
  const supportedSTTProviders = useMemo(
    () => getIntersectedSTTProviders(activeLanguages),
    [activeLanguages]
  )

  // Get supported STT models for selected provider across all selected languages
  const supportedSTTModels = useMemo(
    () => getIntersectedSTTModels(activeLanguages, config.sttProvider),
    [activeLanguages, config.sttProvider]
  )

  // Get supported TTS providers for all selected languages (intersection)
  const supportedTTSProviders = useMemo(
    () => getIntersectedTTSProviders(activeLanguages),
    [activeLanguages]
  )

  // Get supported TTS models for selected provider across all selected languages
  const supportedTTSModels = useMemo(
    () => getIntersectedTTSModels(activeLanguages, config.ttsProvider),
    [activeLanguages, config.ttsProvider]
  )

  // Providers offerable right now: self-hosted models that answer their health
  // probe, plus API providers with a key saved in Integrations.
  const availableLLMProviders = useMemo(
    () =>
      Object.entries(llmProviders).filter(([id, provider]) =>
        isProviderAvailable("llm", id, provider.name, integratedProviders, selfHostedProviders)
      ),
    [integratedProviders, selfHostedProviders]
  )

  const availableSTTProviders = useMemo(
    () =>
      allSTTProviders.filter(
        (p) =>
          supportedSTTProviders.has(p.id) &&
          isProviderAvailable("stt", p.id, p.name, integratedProviders, selfHostedProviders)
      ),
    [supportedSTTProviders, integratedProviders, selfHostedProviders]
  )

  const availableTTSProviders = useMemo(
    () =>
      allTTSProviders.filter(
        (p) =>
          supportedTTSProviders.has(p.id) &&
          isProviderAvailable("tts", p.id, p.name, integratedProviders, selfHostedProviders)
      ),
    [supportedTTSProviders, integratedProviders, selfHostedProviders]
  )

  // Telephony providers whose credentials are saved in Integrations. WebSocket
  // is listed separately since it needs no provider at all.
  const availableTelephonyProviders = useMemo(
    () =>
      ["Vobiz", "Plivo"].filter((provider) =>
        isTelephonyProviderAvailable(provider, integratedProviders)
      ),
    [integratedProviders]
  )

  // Get available TTS voices for primary language + selected provider/model
  const availableTTSVoices = useMemo(() => {
    if (!primaryLanguage || !config.ttsProvider) return []
    const langData =
      ttsData.tts.languages[primaryLanguage as keyof typeof ttsData.tts.languages]
    if (!langData) return []

    const providerData = langData.models[config.ttsProvider as keyof typeof langData.models] as {
      voices?: string | string[]
    }
    if (!providerData) return []

    if (Array.isArray(providerData.voices)) {
      return providerData.voices
    }
    return []
  }, [primaryLanguage, config.ttsProvider])

  // Clear STT/TTS selections when they fall outside the intersected language support
  useEffect(() => {
    if (activeLanguages.length === 0) return

    setConfig((prev) => {
      let changed = false
      const next = { ...prev }

      if (prev.sttModel && !supportedSTTModels.has(prev.sttModel)) {
        next.sttModel = ""
        changed = true
      }
      if (prev.sttProvider && !supportedSTTProviders.has(prev.sttProvider)) {
        next.sttProvider = ""
        next.sttModel = ""
        changed = true
      }
      if (prev.ttsModel && !supportedTTSModels.has(prev.ttsModel)) {
        next.ttsModel = ""
        changed = true
      }
      if (prev.ttsProvider && !supportedTTSProviders.has(prev.ttsProvider)) {
        next.ttsProvider = ""
        next.ttsModel = ""
        next.ttsVoice = ""
        changed = true
      }
      if (
        prev.ttsVoice &&
        availableTTSVoices.length > 0 &&
        !availableTTSVoices.includes(prev.ttsVoice)
      ) {
        next.ttsVoice = ""
        changed = true
      }

      return changed ? next : prev
    })
  }, [
    activeLanguages,
    supportedSTTModels,
    supportedSTTProviders,
    supportedTTSModels,
    supportedTTSProviders,
    availableTTSVoices,
  ])

  // Get LLM models for selected provider
  const availableLLMModels = useMemo(() => {
    if (!config.llmProvider) return []
    const provider = llmProviders[config.llmProvider as keyof typeof llmProviders]
    return provider?.models || []
  }, [config.llmProvider])

  // Handle create new agent
  const handleCreateNew = () => {
    setConfig({ ...defaultConfig, id: "new", telephonyProvider: "" })
    setCreateStep(1)
    setInteractionModeLocked(false)
    setView("create")
  }

  // Handle back to list
  const handleBackToList = () => {
    setView("list")
    setCreateStep(1)
    setInteractionModeLocked(false)
    setConfig({ ...defaultConfig, telephonyProvider: "" })
  }


  const handleTestCall = (agent: Agent) => {
    setSelectedAgentForTest(agent)
    setIsTestCallSheetOpen(true)
  }

  const handleTestBrowser = (agent: Agent) => {
    setSelectedAgentForTest(agent)
    setIsTestBrowserDialogOpen(true)
  }

  const handleViewHistory = (agent: Agent) => {
    if (!agent.agent_type) {
      console.error("Agent type is missing:", agent)
      return
    }
    router.push(`/history?assistant_name=${encodeURIComponent(agent.agent_type)}`)
  }

  const handleDelete = async (agent: Agent) => {
    if (!user?.org_id) {
      console.error("No org_id found")
      return
    }

    setDeletingAgentType(agent.agent_type)

    try {
      // Step 1: Detach phone number if agent has one
      if (isTelephonyAgent(agent) && agent.phone_number) {
        try {
          // If provider is Vobiz, unlink from Vobiz application first
          if (agent.telephony_provider === "Vobiz") {
            await unlinkVobizNumber(agent.phone_number)
          } else if (agent.telephony_provider === "Plivo") {
            await unlinkPlivoNumber(agent.phone_number)
          }
          
          // Detach phone number from agent in database
          const detachResponse = await fetchApiRoute("/api/phone-numbers/detach", {
            method: "DELETE",
            body: JSON.stringify({
              phone_number: agent.phone_number,
            }),
          })

          if (!detachResponse.ok) {
            console.error("Failed to detach phone number, but continuing with agent deletion")
            // Continue with agent deletion even if detach fails
          }
        } catch (error) {
          console.error("Error detaching phone number:", error)
          // Continue with agent deletion even if detach fails
        }
      }

      // Step 2: Delete Vobiz application if it exists
      if (isTelephonyAgent(agent) && agent.telephony_provider === "Vobiz" && agent.vobiz_app_id) {
        try {
          await deleteVobizApplication(agent.vobiz_app_id)
        } catch (error) {
          console.error("Failed to delete Vobiz application:", error)
          // Continue with agent deletion even if Vobiz deletion fails
        }
      }
      const telephonyAgent = agent as AgentWithTelephony
      if (isTelephonyAgent(agent) && agent.telephony_provider === "Plivo" && telephonyAgent.plivo_app_id) {
        try {
          await deletePlivoApplication(telephonyAgent.plivo_app_id)
        } catch (error) {
          console.error("Failed to delete Plivo application:", error)
        }
      }

      // Step 3: Delete the agent
      if (!agent.agent_type) {
        throw new Error("Agent type is missing")
      }
      await deleteAgent(agent.agent_type)

      await queryClient.invalidateQueries({
        queryKey: agentsQueryKey(user.org_id),
      })
      
      // Show success toast
      setShowDeleteSuccessToast(true)
      setTimeout(() => {
        setShowDeleteSuccessToast(false)
      }, 3000)
    } catch (error) {
      console.error("Failed to delete agent:", error)
      alert(error instanceof Error ? error.message : "Failed to delete agent")
      throw error
    } finally {
      setDeletingAgentType(null)
    }
  }

  // Reset STT/TTS when language selection changes; auto-pick ai4bharat when applicable
  const applyLanguageAudioDefaults = (
    updated: AgentConfig,
    primaryLanguage: string
  ) => {
    updated.sttProvider = ""
    updated.sttModel = ""
    updated.ttsProvider = ""
    updated.ttsModel = ""
    updated.ttsVoice = ""

    if (
      primaryLanguage &&
      primaryLanguage !== "English (United States)" &&
      primaryLanguage !== "English (India)"
    ) {
      const sttLangData =
        sttData.stt.languages[primaryLanguage as keyof typeof sttData.stt.languages]
      if (
        sttLangData?.models?.ai4bharat &&
        Array.isArray(sttLangData.models.ai4bharat) &&
        sttLangData.models.ai4bharat.length > 0
      ) {
        updated.sttProvider = "ai4bharat"
        updated.sttModel = sttLangData.models.ai4bharat[0]
      }

      const ttsLangData =
        ttsData.tts.languages[primaryLanguage as keyof typeof ttsData.tts.languages]
      const ttsAi4bharatData = ttsLangData?.models?.ai4bharat as
        | { available?: boolean; model?: string; voices?: string[] }
        | undefined
      if (ttsAi4bharatData?.available && ttsAi4bharatData.model) {
        updated.ttsProvider = "ai4bharat"
        updated.ttsModel = ttsAi4bharatData.model
        if (
          ttsAi4bharatData.voices &&
          Array.isArray(ttsAi4bharatData.voices) &&
          ttsAi4bharatData.voices.length > 0
        ) {
          updated.ttsVoice = ttsAi4bharatData.voices[0]
        }
      }
    }
  }

  const setSelectedLanguages = (languages: string[]) => {
    setConfig((prev) => {
      const updated = { ...prev, selectedLanguages: languages }
      if (languages[0]) {
        applyLanguageAudioDefaults(updated, languages[0])
      }
      return updated
    })
  }

  // Update config helper
  const updateConfig = <K extends keyof AgentConfig>(
    key: K,
    value: AgentConfig[K]
  ) => {
    setConfig((prev) => {
      const updated = { ...prev, [key]: value }
      if (key === "selectedLanguages") {
        const languages = value as string[]
        if (languages[0]) {
          applyLanguageAudioDefaults(updated, languages[0])
        }
      }
      if (key === "sttProvider") {
        updated.sttModel = ""
      }
      if (key === "ttsProvider") {
        updated.ttsModel = ""
        updated.ttsVoice = ""
        updated.ttsDescription = ""
      }
      if (key === "llmProvider") {
        updated.llmModel = ""
        updated.customLlmId = ""
        if ((value as string) === "kenpath") {
          updated.kenpathVariant = "prod"
        }
        if ((value as string) !== "openai") {
          updated.knowledgeEnabled = false
          updated.knowledgeDocumentIds = []
        }
      }
      if (key === "kenpathVariant") {
        const variant = value as KenpathVariant
        if (
          updated.llmProvider === "kenpath" &&
          updated.selectedLanguages.some(
            (lang) => !isBharatVistaarLanguageSupported(variant, lang)
          )
        ) {
          updated.selectedLanguages = []
          updated.sttProvider = ""
          updated.sttModel = ""
          updated.ttsProvider = ""
          updated.ttsModel = ""
          updated.ttsVoice = ""
          updated.ttsDescription = ""
        }
      }
      if (key === "customLlmId") {
        const selected = customLLMIntegrations.find((item) => item.id === value)
        updated.customLlmId = value as string
        updated.llmModel = selected?.model || ""
      }
      return updated
    })
  }

  const nameSnakeCase = config.name.toLowerCase().replace(/ /g, "_")
  const selectedKnowledgeDocs = useMemo(
    () =>
      knowledgeDocs.filter((d) => config.knowledgeDocumentIds.includes(d.document_id)),
    [knowledgeDocs, config.knowledgeDocumentIds]
  )

  const toggleKnowledgeDocument = (documentId: string) => {
    setConfig((prev) => {
      const exists = prev.knowledgeDocumentIds.includes(documentId)
      return {
        ...prev,
        knowledgeDocumentIds: exists
          ? prev.knowledgeDocumentIds.filter((id) => id !== documentId)
          : [...prev.knowledgeDocumentIds, documentId],
      }
    })
  }

  // Handle save agent
  const handleSaveAgent = async () => {
    if (!user?.org_id) {
      console.error("No org_id found")
      return
    }

    setIsCreatingAgent(true)

    try {
      const trimmedName = config.name.trim()
      const nameError = validateAgentName(trimmedName)
      if (nameError) {
        alert(nameError)
        return
      }

      const agentId = slugifyAgentId(trimmedName)

      const languageFields = buildLanguageConfigFields(config.selectedLanguages)

      // Build LLM model object with official provider name
      const llmModel: {
        name: string
        model?: string
        custom_llm_id?: string
        vistaar_environment?: "prod" | "dev"
        kenpath_backend?: "vistaar" | "bharatvistaar"
      } = {
        name: getProviderOfficialName(config.llmProvider),
      }
      if (config.llmProvider === "custom_llm") {
        llmModel.custom_llm_id = config.customLlmId
        llmModel.model = config.llmModel
      } else if (config.llmProvider !== "kenpath") {
        llmModel.model = config.llmModel
      } else {
        Object.assign(llmModel, kenpathLlmFieldsFromVariant(config.kenpathVariant))
      }

      // Build STT model object WITHOUT language inside, using official provider name
      const sttModel: { name: string; model?: string; keywords?: string } = {
        name: getProviderOfficialName(config.sttProvider),
        model: config.sttModel,
      }
      if (config.keywords) {
        sttModel.keywords = config.keywords
      }

      // Build TTS model object WITHOUT language, using official provider name
      const usesArgsVoiceProvider =
        config.ttsProvider === "cartesia" ||
        config.ttsProvider === "gcp" ||
        config.ttsProvider === "elevenlabs"

      const ttsModel: Record<string, unknown> = {
        name: getProviderOfficialName(config.ttsProvider),
        ...(usesArgsVoiceProvider && {
          args: {
            ...(config.ttsModel && { model: config.ttsModel }),
            ...(config.ttsVoice && { voice_id: config.ttsVoice }),
          },
        }),
        ...(!usesArgsVoiceProvider && config.ttsModel && { model: config.ttsModel }),
        speaker: usesArgsVoiceProvider ? "" : (config.ttsVoice || ""),
      }
      if ((config.ttsProvider === "ai4bharat" || config.ttsProvider === "bhashini") && config.ttsDescription) {
        ttsModel.description = config.ttsDescription
      }
      if (config.ttsProvider === "gcp" || config.ttsProvider === "cartesia" || config.ttsProvider === "sarvam") {
        ttsModel.speed = config.speedRate
      }
      if (config.ttsProvider === "gcp" || config.ttsProvider === "sarvam") {
        ttsModel.pitch = config.stability
      }
      if (config.ttsProvider === "cartesia") {
        ttsModel.emotion_intensity = config.similarityBoost
      }
      if (config.ttsProvider === "sarvam") {
        ttsModel.loudness = config.similarityBoost
      }

      // If Vobiz provider, create Vobiz application first
      let vobizAppId: string | undefined
      let vobizAnswerUrl: string | undefined
      let plivoAppId: string | undefined
      let plivoAnswerUrl: string | undefined
      if (config.telephonyProvider === "Vobiz") {
        vobizAnswerUrl = `${requireJohnaicServerUrl()}/answer?agent_id=${agentId}`
        console.log(" answer url", vobizAnswerUrl)
        
        // // Create Vobiz application
         const vobizAppResponse = await createVobizApplication(config.name, vobizAnswerUrl)
        console.log("vobizAppResponse", vobizAppResponse)
        if (vobizAppResponse.status === "success" && vobizAppResponse.app_id) {
          vobizAppId = vobizAppResponse.app_id
        } else {
          throw new Error(vobizAppResponse.message || "Failed to create Vobiz application")
        }
      }
      if (config.telephonyProvider === "Plivo") {
        plivoAnswerUrl = `${requireJohnaicServerUrl()}/plivo/answer?agent_id=${agentId}`
        const plivoAppResponse = await createPlivoApplication(config.name, plivoAnswerUrl)
        if (plivoAppResponse.status === "success" && plivoAppResponse.app_id) {
          plivoAppId = plivoAppResponse.app_id
        } else {
          throw new Error(plivoAppResponse.message || "Failed to create Plivo application")
        }
      }

      const agentData: CreateAgentRequest = {
        org_id: user.org_id,
        agent_category: agentCategoryForProvider(config.telephonyProvider),
        agent_type: trimmedName,
        agent_id: agentId,
        agent_config:
          config.interactionMode === "non_conversational"
            ? {
                interaction_mode: "non_conversational",
                greeting_message: config.greetingMessage ?? "",
                ...languageFields,
                tts_model: ttsModel,
              }
            : {
                interaction_mode: "conversational",
                system_prompt: config.systemPrompt,
                greeting_message: config.greetingMessage ?? "",
                ignore_user_speech_before_greeting: config.ignoreUserSpeechBeforeGreeting,
                interruption_min_words: config.interruptionMinWords,
                user_silence_hangup_seconds: config.userSilenceHangupSeconds,
                call_timeout_seconds: config.callTimeoutSeconds,
                hold_messages: config.holdMessages
                  .map((m) => m.trim())
                  .filter(Boolean),
                hold_message_timeout_seconds: config.holdMessageTimeoutSeconds,
                user_online_detection_enabled: config.userOnlineDetectionEnabled,
                user_online_detection_message: config.userOnlineDetectionMessage.trim(),
                user_online_detection_seconds: config.userOnlineDetectionSeconds,
                user_online_detection_repeats: config.userOnlineDetectionRepeats,
                user_online_detection_closing_message:
                  config.userOnlineDetectionClosingMessage.trim(),
                ...languageFields,
                knowledge_base_enabled: config.llmProvider === "openai" ? config.knowledgeEnabled : false,
                knowledge_document_ids:
                  config.llmProvider === "openai" && config.knowledgeEnabled
                    ? config.knowledgeDocumentIds
                    : [],
                knowledge_top_k: config.knowledgeTopK,
                llm_model: llmModel,
                stt_model: sttModel,
                tts_model: ttsModel,
              },
        telephony_provider: config.telephonyProvider,
        ...(config.telephonyProvider === "Vobiz" && {
          vobiz_app_id: vobizAppId,
          vobiz_answer_url: vobizAnswerUrl,
        }),
        ...(config.telephonyProvider === "Plivo" && {
          plivo_app_id: plivoAppId,
          plivo_answer_url: plivoAnswerUrl,
        }),
      }

      await createAgent(agentData)

      if (user.org_id) {
        await queryClient.invalidateQueries({
          queryKey: agentsQueryKey(user.org_id),
        })
      }
      
      // Reset and go back to list
      handleBackToList()
    } catch (error) {
      console.error("Failed to create agent:", error)
      alert(error instanceof Error ? error.message : "Failed to create agent")
    } finally {
      setIsCreatingAgent(false)
    }
  }

  // Navigate to next step
  const handleNextStep = () => {
    if (createStep < WIZARD_STEP_COUNT) {
      if (currentStepKey === "type") {
        setInteractionModeLocked(true)
      }
      setCreateStep(createStep + 1)
    }
  }

  // Calculate progress percentage
  const progressPercent = (createStep / WIZARD_STEP_COUNT) * 100

  // Check if a specific step is completed
  const isStepCompleted = (stepId: number) => {
    const key = activeWizardSteps[stepId - 1]?.key
    switch (key) {
      case "type":
        return config.interactionMode !== null
      case "agent":
        if (config.interactionMode === "non_conversational") {
          return isValidAgentName(config.name) && (config.greetingMessage?.trim().length ?? 0) > 0
        }
        return isValidAgentName(config.name) && config.systemPrompt.length > 0
      case "llm":
        if (config.llmProvider === "kenpath") {
          return !!config.llmProvider
        }
        if (config.llmProvider === "custom_llm") {
          return !!config.llmProvider && !!config.customLlmId
        }
        return !!(config.llmProvider && config.llmModel)
      case "audio":
        if (config.interactionMode === "non_conversational") {
          return !!(
            primaryLanguage &&
            config.ttsProvider &&
            config.ttsModel &&
            config.ttsVoice &&
            config.ttsVoice.length > 0
          )
        }
        return !!(
          primaryLanguage &&
          config.sttProvider &&
          config.sttModel &&
          config.ttsProvider &&
          config.ttsModel &&
          config.ttsVoice &&
          config.ttsVoice.length > 0
        )
      case "telephony":
        return !!config.telephonyProvider
      case "call_mgmt":
        return config.callTimeoutSeconds >= 60
      case "review":
        return true
      default:
        return false
    }
  }

  // Check if can proceed to next step
  const canProceed = () => {
    return isStepCompleted(createStep)
  }

  // Check if a step is accessible
  const canAccessStep = (stepId: number) => {
    if (stepId === createStep) return true
    if (stepId < createStep) return true
    for (let i = 1; i < stepId; i++) {
      if (!isStepCompleted(i)) return false
    }
    return true
  }

  // Get next step label
  const getNextStepLabel = () => {
    if (createStep === WIZARD_STEP_COUNT) return "Create Agent"
    const nextStep = activeWizardSteps.find((s) => s.id === createStep + 1)
    return nextStep ? `Continue to ${nextStep.title}` : "Continue"
  }

  // Language combobox state
  const [languageOpen, setLanguageOpen] = useState(false)

  // Render list view
  if (view === "list") {
  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header */}
      <header className="flex h-14 items-center gap-3 border-b border-slate-200 bg-white page-shell-x sticky top-0 z-10 shrink-0">
        <SidebarTrigger className="h-9 w-9 shrink-0" />
          <nav className="flex items-center gap-1.5 text-sm min-w-0">
            <span className="text-slate-500 hidden sm:inline">Dashboard</span>
          <ChevronRight className="h-4 w-4 text-slate-400 hidden sm:inline" />
            <span className="text-slate-900 font-medium truncate">Agents</span>
        </nav>
      </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto p-4 sm:p-6 lg:p-8">
          {/* Greeting Section */}
          <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between mb-6 sm:mb-8">
            <div className="min-w-0">
              <h1 className="text-2xl sm:text-3xl font-semibold text-slate-900 mb-1 truncate">Hi {user?.name}</h1>
              <p className="text-slate-500">Let&apos;s get your agents inline.</p>
            </div>
            <div className="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center gap-3 w-full sm:w-auto">
              {dataUpdatedAt > 0 && (
                <span className="text-xs text-slate-500 whitespace-nowrap order-last sm:order-first w-full sm:w-auto text-center sm:text-left">
                  Last updated{" "}
                  {formatDistanceToNow(dataUpdatedAt, { addSuffix: true })}
                </span>
              )}
              <div className="flex items-center gap-3 w-full sm:w-auto">
              <Button
                type="button"
                variant="outline"
                size="icon"
                className="h-10 w-10 shrink-0 rounded-lg border-slate-200"
                aria-label="Refresh agents"
                disabled={isFetchingAgents || !user?.org_id}
                onClick={() => refetchAgents()}
              >
                <RefreshCw
                  className={`h-4 w-4 ${isFetchingAgents ? "animate-spin" : ""}`}
                />
              </Button>
              <Select
                value={agentSortOrder}
                onValueChange={(v) =>
                  setAgentSortOrder(
                    v as "newest" | "oldest" | "active-first" | "inactive-first"
                  )
                }
              >
                <SelectTrigger
                  aria-label="Sort agents"
                  className="h-10 w-full sm:w-[180px] rounded-lg border-slate-200 bg-white text-sm focus:ring-1 focus:ring-slate-200"
                >
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="newest">Newest first</SelectItem>
                  <SelectItem value="oldest">Oldest first</SelectItem>
                  <SelectItem value="active-first">Active first</SelectItem>
                  <SelectItem value="inactive-first">Inactive first</SelectItem>
                </SelectContent>
              </Select>
              </div>
              <div className="relative w-full sm:w-auto sm:min-w-[16rem]">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              type="text"
                  placeholder="Search Assistant"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-10 pl-9 pr-4 w-full rounded-lg border-slate-200 bg-white focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
            />
              </div>
            </div>
          </div>

          {isAgentsError && (
            <div className="mb-4 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
              Failed to load agents.{" "}
              <button
                type="button"
                className="font-medium underline"
                onClick={() => refetchAgents()}
              >
                Try again
              </button>
            </div>
          )}

          {/* Assistant Cards Grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {/* Create New Assistant Card */}
            <CreateNewAgentCard onCreateNew={handleCreateNew} />

            {/* Loading State */}
            {isLoadingAgents && (
              <div className="col-span-full text-center py-8 text-slate-500">
                Loading agents...
              </div>
            )}

            {/* Empty State */}
            {!isLoadingAgents && filteredAgents.length === 0 && (
              <div className="col-span-full text-center py-12">
                <p className="text-slate-500 mb-2">No agents found</p>
                <p className="text-sm text-slate-400">Create your first agent to get started</p>
              </div>
            )}

            {/* Existing Assistant Cards */}
            {!isLoadingAgents && sortedAgents.map((agent) => (
              <AgentCard
                key={String(agent.id || agent.org_id + agent.agent_type + agent.created_at)}
                agent={agent}
                getAgentDisplayName={getAgentDisplayName}
                getAgentDescription={getAgentDescription}
                onViewConfig={viewConfig}
                onTestCall={handleTestCall}
                onTestBrowser={handleTestBrowser}
                onViewHistory={handleViewHistory}
                onDelete={handleDelete}
                isDeleting={deletingAgentType === agent.agent_type}
              />
            ))}
          </div>
        </main>

      

        {/* Test Call Sheet */}
        <TestCallSheet
          open={isTestCallSheetOpen}
          onOpenChange={setIsTestCallSheetOpen}
          agent={selectedAgentForTest}
          getAgentDisplayName={getAgentDisplayName}
        />

        <TestBrowserDialog
          open={isTestBrowserDialogOpen}
          onOpenChange={setIsTestBrowserDialogOpen}
          agent={selectedAgentForTest}
          getAgentDisplayName={getAgentDisplayName}
        />

        {/* Delete Success Toast */}
        {showDeleteSuccessToast && (
          <div className="mobile-toast animate-in slide-in-from-top-5 fade-in-0 bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3">
            <CheckCircle2 className="h-5 w-5 text-emerald-600 shrink-0" />
            <p className="font-medium">Agent deleted successfully</p>
          </div>
        )}
      </div>
    )
  }

  // Render create wizard view
  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header with Progress */}
      <header className="flex h-auto min-h-14 flex-col gap-3 sm:flex-row sm:items-center sm:justify-between border-b border-slate-200 bg-white px-4 sm:px-6 sticky top-0 z-10 py-3 sm:py-0">
        <div className="flex items-center gap-2 sm:gap-4 min-w-0">
          <SidebarTrigger className="h-9 w-9 shrink-0" />
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBackToList}
            className="h-8 px-3 text-slate-600 hover:bg-slate-100 gap-1.5 shrink-0"
          >
            <ChevronLeft className="h-4 w-4" />
            <span className="sr-only sm:not-sr-only sm:inline">Back</span>
          </Button>
          <Separator orientation="vertical" className="h-5 hidden sm:block" />
          <h1 className="text-sm font-semibold text-slate-900 truncate">Create Telephony Agent</h1>
        </div>
        <div className="flex items-center gap-3 w-full sm:w-auto sm:shrink-0">
          <span className="text-xs text-slate-500 shrink-0">Progress</span>
          <div className="flex-1 sm:w-32 h-1 bg-slate-200 rounded-full overflow-hidden min-w-[5rem]">
            <div
              className="h-full bg-slate-900 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <span className="text-xs font-medium text-slate-700 shrink-0">{Math.round(progressPercent)}%</span>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Row - Progress Stepper */}
        <aside className="bg-white border-b border-slate-100 p-2 sm:p-3">
          <h3 className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-2 text-center">Setup Progress</h3>
          <div className="flex gap-1.5 overflow-x-auto pb-0.5 justify-center">
            {activeWizardSteps.map((step) => {
              const Icon = step.icon
              const isActive = createStep === step.id
              const isCompleted = isStepCompleted(step.id) && createStep > step.id
              const isAccessible = canAccessStep(step.id)

              return (
                <button
                  key={step.id}
                  onClick={() => isAccessible && setCreateStep(step.id)}
                  disabled={!isAccessible}
                  className={`shrink-0 min-w-[128px] sm:min-w-[148px] flex items-center gap-1.5 px-2.5 py-2 rounded-lg text-left transition-all duration-150 ${
                    isActive
                      ? "bg-slate-100"
                      : isAccessible
                      ? "hover:bg-slate-50 cursor-pointer"
                      : "opacity-50 cursor-not-allowed"
                }`}
              >
                <div
                  className={`h-7 w-7 rounded-md flex items-center justify-center transition-all duration-150 shrink-0 ${
                    isActive
                      ? "bg-slate-900 text-white"
                      : isCompleted
                      ? "bg-slate-200 text-slate-600"
                      : "bg-slate-100 text-slate-400"
                  }`}
                >
                  <Icon className="h-3.5 w-3.5" />
                </div>
                <div className="flex-1 min-w-0">
                  <p
                    className={`text-[13px] font-medium leading-tight truncate ${
                      isActive ? "text-slate-900" : isCompleted ? "text-slate-700" : "text-slate-500"
                    }`}
                  >
                    {step.title}
                  </p>
                  <p className="text-[10px] text-slate-400 truncate">{step.subtitle}</p>
                </div>
                {isCompleted && (
                  <CheckCircle2 className="h-3.5 w-3.5 text-emerald-500 flex-shrink-0" />
                )}
              </button>
              )
            })}
          </div>
        </aside>

        {/* Step Content */}
        <main className="flex-1 overflow-auto p-6 sm:p-8">
          <div className="w-full max-w-4xl mx-auto">
            {/* Step: Type selection */}
            {currentStepKey === "type" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-6">
                  <div>
                    <h2 className="text-xl font-bold text-slate-900 mb-1">Choose Agent Type</h2>
                    <p className="text-slate-500">
                      Select how this agent should interact on calls. You can change your choice
                      until you continue. After that, the type is locked for this agent.
                    </p>
                  </div>
                  {interactionModeLocked && (
                    <p className="text-sm text-amber-700 bg-amber-50 border border-amber-200 rounded-lg px-4 py-3">
                      Agent type is locked. You cannot switch between conversational and
                      non-conversational during setup.
                    </p>
                  )}
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <button
                      type="button"
                      disabled={interactionModeLocked}
                      onClick={() => updateConfig("interactionMode", "conversational")}
                      className={`text-left rounded-xl border-2 p-6 transition-all ${
                        config.interactionMode === "conversational"
                          ? "border-slate-900 bg-slate-50"
                          : "border-slate-200 hover:border-slate-300"
                      } ${interactionModeLocked ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                      <div className="flex items-center gap-3 mb-3">
                        <div className="h-10 w-10 rounded-lg bg-slate-900 text-white flex items-center justify-center">
                          <MessageSquare className="h-5 w-5" />
                        </div>
                        <h3 className="text-lg font-bold text-slate-900">Conversational</h3>
                      </div>
                      <p className="text-sm text-slate-600">
                        For an interactable voice agent that listens and responds.
                      </p>
                    </button>
                    <button
                      type="button"
                      disabled={interactionModeLocked}
                      onClick={() => updateConfig("interactionMode", "non_conversational")}
                      className={`text-left rounded-xl border-2 p-6 transition-all ${
                        config.interactionMode === "non_conversational"
                          ? "border-slate-900 bg-slate-50"
                          : "border-slate-200 hover:border-slate-300"
                      } ${interactionModeLocked ? "opacity-50 cursor-not-allowed" : ""}`}
                    >
                      <div className="flex items-center gap-3 mb-3">
                        <div className="h-10 w-10 rounded-lg bg-amber-600 text-white flex items-center justify-center">
                          <Bell className="h-5 w-5" />
                        </div>
                        <h3 className="text-lg font-bold text-slate-900">Non-conversational</h3>
                      </div>
                      <p className="text-sm text-slate-600">
                        For alerts and one-way messages. Plays your message and ends the call.
                      </p>
                    </button>
                  </div>
                </div>

                <Button
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step: Agent */}
            {currentStepKey === "agent" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-8">
                  {/* Agent Name */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">Agent Name</label>
                    <Input
                      value={config.name}
                      onChange={(e) => updateConfig("name", sanitizeAgentNameInput(e.target.value))}
                      placeholder="Enter agent name"
                      className="h-12 rounded-lg border-slate-200 bg-white text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                    />
                    <p className="text-sm text-slate-500">
                      Use letters, numbers, underscores, or hyphens only. No spaces.
                    </p>
                  </div>

                  {/* Agent Welcome / Alert Message */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">
                      {config.interactionMode === "non_conversational"
                        ? "Alert Message"
                        : "Agent Welcome Message"}
                    </label>
                    {config.interactionMode === "non_conversational" ? (
                      <SpokenMessageTextarea
                        value={config.greetingMessage ?? ""}
                        onChange={(value) => updateConfig("greetingMessage", value)}
                        placeholder="Your payment is due tomorrow Please pay to avoid late fees"
                        className="min-h-[120px] rounded-lg border-slate-200 bg-white resize-none text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                      />
                    ) : (
                      <SpokenMessageInput
                        value={config.greetingMessage ?? ""}
                        onChange={(value) => updateConfig("greetingMessage", value)}
                        placeholder="Hello"
                        className="h-12 rounded-lg border-slate-200 bg-white text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                      />
                    )}
                    <GreetingCommaBanner context="greeting" />
                    <p className="text-sm text-slate-500">
                      {config.interactionMode === "non_conversational"
                        ? "This message will be spoken on the call and the call will end when finished."
                        : "This will be the initial message from the agent."}
                    </p>
                    {config.interactionMode !== "non_conversational" && (
                    <div className="flex items-center justify-between gap-4 pt-2">
                      <div>
                        <p className="text-sm font-semibold text-slate-800">
                          Ignore user speech before welcome
                        </p>
                        <p className="text-sm text-slate-500 mt-1">
                          {config.greetingMessage?.trim()
                            ? "Block barge-in while the welcome message is playing."
                            : "Add a welcome message to enable this."}
                        </p>
                      </div>
                      <label
                        className={`relative inline-flex shrink-0 items-center ${
                          config.greetingMessage?.trim()
                            ? "cursor-pointer"
                            : "cursor-not-allowed opacity-50"
                        }`}
                      >
                        <input
                          type="checkbox"
                          className="sr-only peer"
                          checked={config.ignoreUserSpeechBeforeGreeting}
                          disabled={!config.greetingMessage?.trim()}
                          onChange={() =>
                            updateConfig(
                              "ignoreUserSpeechBeforeGreeting",
                              !config.ignoreUserSpeechBeforeGreeting
                            )
                          }
                        />
                        <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer-checked:bg-emerald-600 transition-colors" />
                        <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5" />
                      </label>
                    </div>
                    )}
                  </div>

                  {/* Agent Prompt */}
                  {config.interactionMode !== "non_conversational" && (
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <label className="text-base font-bold text-slate-900">Agent Prompt</label>
                    </div>
                    <Textarea
                      value={config.systemPrompt}
                      onChange={(e) => updateConfig("systemPrompt", e.target.value)}
                      placeholder="You are a helpful assistant that..."
                      className="min-h-[200px] rounded-lg border-slate-200 bg-white resize-none text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                    />
                  </div>
                  )}
                </div>

                <Button 
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step: LLM Settings */}
            {currentStepKey === "llm" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-6">
                  <div className="space-y-4">
                    <label className="text-base font-bold text-slate-900">Choose LLM model</label>
                    {availableLLMProviders.length === 0 ? (
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4 mt-4">
                        <p className="text-sm font-medium text-slate-900">No LLM provider available</p>
                        <p className="text-sm text-slate-600 mt-1">
                          Add an API key in{" "}
                          <button
                            type="button"
                            onClick={() => router.push("/integrations")}
                            className="text-blue-600 underline hover:text-blue-700"
                          >
                            Integrations
                          </button>
                          , or start a self-hosted model server.
                        </p>
                      </div>
                    ) : (
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <Select value={config.llmProvider} onValueChange={(v) => updateConfig("llmProvider", v)}>
                        <SelectTrigger className="h-12 rounded-lg w-full border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                          <div className="flex items-center gap-2">
                            <SelectValue placeholder="Select provider" />
                          </div>
                        </SelectTrigger>
                        <SelectContent className="rounded-lg">
                        {availableLLMProviders
                          .map(([id, provider]) => (
                            <SelectItem key={id} value={id} className="py-3 hover:bg-slate-100 transition">
                              <div className="flex items-center gap-2.5">
                                <span className="font-medium">{provider.name}</span>
                              </div>
                            </SelectItem>
                          ))}
                     
                        </SelectContent>
                      </Select>

                      {config.llmProvider === "kenpath" ? (
                        <Select
                          value={config.kenpathVariant}
                          onValueChange={(v) =>
                            updateConfig("kenpathVariant", v as KenpathVariant)
                          }
                        >
                          <SelectTrigger className="h-12 rounded-lg w-full border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                            <SelectValue placeholder="Select Kenpath environment" />
                          </SelectTrigger>
                          <SelectContent className="rounded-lg">
                            <SelectItem value="prod" className="py-2.5">
                              Production
                            </SelectItem>
                            <SelectItem value="dev" className="py-2.5">
                              Development
                            </SelectItem>
                            <SelectItem value="bharatvistaar" className="py-2.5">
                              Bharat Vistaar
                            </SelectItem>
                            <SelectItem value="bharatvistaar_dev" className="py-2.5">
                              Bharat Vistaar Dev API
                            </SelectItem>
                          </SelectContent>
                        </Select>
                      ) : config.llmProvider === "custom_llm" ? (
                        <Select
                          value={config.customLlmId}
                          onValueChange={(v) => updateConfig("customLlmId", v)}
                          disabled={customLLMIntegrations.length === 0}
                        >
                          <SelectTrigger className="h-12 rounded-lg w-full border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                            <SelectValue placeholder="Select custom LLM" />
                          </SelectTrigger>
                          <SelectContent className="rounded-lg">
                            {customLLMIntegrations.map((integration) => (
                              <SelectItem key={integration.id} value={integration.id} className="py-2.5">
                                <div className="flex flex-col items-start">
                                  <span className="font-medium">{integration.name}</span>
                                  <span className="font-mono text-xs text-slate-500">{integration.model}</span>
                                </div>
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      ) : (
                        <Select value={config.llmModel} onValueChange={(v) => updateConfig("llmModel", v)} disabled={!config.llmProvider || availableLLMModels.length === 0}>
                          <SelectTrigger className="h-12 rounded-lg w-full border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                            <SelectValue placeholder="Select model" />
                          </SelectTrigger>
                          <SelectContent className="rounded-lg max-h-[280px]">
                            {availableLLMModels.map((model) => (
                              <SelectItem key={model} value={model} className="py-2.5 font-mono text-sm">
                                {model}
                              </SelectItem>
                            ))}
                          </SelectContent>
                        </Select>
                      )}
                    </div>
                    )}
                  </div>

                  {config.llmProvider === "custom_llm" && config.customLlmId && (
                    <p className="text-sm text-slate-600">
                      Model: <span className="font-mono">{config.llmModel}</span> — configured in Integrations.
                    </p>
                  )}

                  {config.llmProvider === "kenpath" && (
                    <p className="text-sm text-blue-600">
                      {kenpathVariantHelpText(config.kenpathVariant)}
                    </p>
                  )}

                  {config.llmProvider !== "kenpath" && (
                    <div className="grid grid-cols-2 gap-6 pt-4">
                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <label className="text-base font-bold text-slate-900">Tokens generated on each LLM output</label>
                          <span className="text-base font-semibold text-slate-900">{config.maxTokens}</span>
                        </div>
                        <Slider
                          value={[config.maxTokens]}
                          onValueChange={([v]) => updateConfig("maxTokens", v)}
                          min={50}
                          max={2000}
                          step={10}
                          className="[&_[role=slider]]:bg-blue-600 [&_[role=slider]]:border-blue-600 [&_.range]:bg-blue-600"
                        />
                        <p className="text-sm text-blue-600">
                          Increasing tokens enables longer responses to be queued for speech generation but increases latency
                        </p>
                      </div>

                      <div className="space-y-4">
                        <div className="flex items-center justify-between">
                          <label className="text-base font-bold text-slate-900">Temperature</label>
                          <span className="text-base font-semibold text-slate-900">{config.temperature.toFixed(1)}</span>
                        </div>
                        <Slider
                          value={[config.temperature]}
                          onValueChange={([v]) => updateConfig("temperature", v)}
                          min={0}
                          max={2}
                          step={0.1}
                          className="[&_[role=slider]]:bg-blue-600 [&_[role=slider]]:border-blue-600 [&_.range]:bg-blue-600"
                        />
                        <p className="text-sm text-blue-600">
                          Increasing temperature enables heightened creativity, but increases chance of deviation from prompt
                        </p>
                      </div>
                    </div>
                  )}

                  {config.llmProvider === "openai" && (
                    <div className="space-y-4 pt-4 border-t border-slate-100">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-base font-bold text-slate-900">Knowledge Base</p>
                          <p className="text-sm text-slate-500">
                            Enable retrieval from selected knowledge documents.
                          </p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={config.knowledgeEnabled}
                            onChange={() =>
                              updateConfig(
                                "knowledgeEnabled",
                                !config.knowledgeEnabled
                              )
                            }
                          />
                          <div
                            className="w-11 h-6 bg-slate-200 dark:bg-slate-800 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer-checked:bg-emerald-600 transition-colors"
                          />
                          <div
                            className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"
                          />
                        </label>
                      </div>
                      {config.knowledgeEnabled && (
                        <div className="space-y-3">
                          <div className="flex items-center justify-between">
                            <label className="text-sm font-semibold text-slate-700">
                              Select knowledge documents
                            </label>
                            <span className="text-xs text-slate-500">
                              {selectedKnowledgeDocs.length} selected
                            </span>
                          </div>
                          {isKnowledgeLoading ? (
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              Loading knowledge documents...
                            </div>
                          ) : knowledgeDocs.length === 0 ? (
                            <p className="text-sm text-slate-500">
                              No ready knowledge documents found. Upload and process files in Knowledge Base.
                            </p>
                          ) : (
                            <div className="max-h-44 overflow-auto rounded-lg border border-slate-200 bg-slate-50 divide-y divide-slate-200">
                              {knowledgeDocs.map((doc) => {
                                const checked = config.knowledgeDocumentIds.includes(doc.document_id)
                                return (
                                  <button
                                    key={doc.document_id}
                                    type="button"
                                    onClick={() => toggleKnowledgeDocument(doc.document_id)}
                                    className="w-full px-3 py-2 text-left hover:bg-slate-100 transition flex items-center gap-3"
                                  >
                                    <span
                                      aria-hidden
                                      className={[
                                        "h-4 w-4 rounded border flex items-center justify-center shrink-0 transition-colors",
                                        checked
                                          ? "bg-emerald-600 border-emerald-600"
                                          : "bg-white border-slate-300",
                                      ].join(" ")}
                                    >
                                      {checked && (
                                        <Check className="h-3 w-3 text-white" />
                                      )}
                                    </span>
                                    <span className="text-sm text-slate-800 truncate">
                                      {doc.original_filename}
                                    </span>
                                  </button>
                                )
                              })}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}

                  {config.llmProvider && availableLLMModels.length === 0 && (
                    <div className="rounded-lg bg-slate-50 border border-slate-200 p-4">
                      <p className="text-slate-600 text-sm">
                        {llmProviders[config.llmProvider as keyof typeof llmProviders]?.name} uses custom model configurations.
                      </p>
                    </div>
                  )}
                </div>

                <Button
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step: Audio Settings */}
            {currentStepKey === "audio" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-8">
                  {/* Language Selection */}
                  <LanguageSelectionSection
                    selectedLanguages={config.selectedLanguages}
                    allLanguages={allLanguages}
                    llmProvider={config.llmProvider}
                    kenpathVariant={config.kenpathVariant}
                    sttProvider={config.sttProvider}
                    sttModel={config.sttModel}
                    ttsProvider={config.ttsProvider}
                    ttsModel={config.ttsModel}
                    open={languageOpen}
                    onOpenChange={setLanguageOpen}
                    onLanguagesChange={setSelectedLanguages}
                  />

                  {/* STT Settings */}
                  {activeLanguages.length > 0 && config.interactionMode !== "non_conversational" && (
                    <div className="space-y-4 pt-6 border-t border-slate-100">
                      <label className="text-base font-bold text-slate-900 italic">Select transcriber</label>
                      {availableSTTProviders.length === 0 ? (
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <p className="text-sm font-medium text-slate-900">
                          No transcriber available for {activeLanguages.join(", ")}
                        </p>
                        <p className="text-sm text-slate-600 mt-1">
                          Add an API key in{" "}
                          <button
                            type="button"
                            onClick={() => router.push("/integrations")}
                            className="text-blue-600 underline hover:text-blue-700"
                          >
                            Integrations
                          </button>
                          , or start the AI4Bharat STT server.
                        </p>
                      </div>
                      ) : (
                      <div className="grid grid-cols-2 gap-4">
                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Provider</label>
                          <Select value={config.sttProvider} onValueChange={(v) => updateConfig("sttProvider", v)}>
                            <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                              <div className="flex items-center gap-2">
                                <Mic className="h-4 w-4 text-slate-400" />
                                <SelectValue placeholder="Select provider" />
                              </div>
                            </SelectTrigger>
                            <SelectContent className="rounded-lg">
                              <div className="px-2 py-1.5 text-xs font-semibold text-slate-500">STT Providers</div>
                              {availableSTTProviders
                                .map((provider) => (
                                  <SelectItem key={provider.id} value={provider.id} className="py-2.5">
                                    <span className="font-medium">{provider.name}</span>
                                  </SelectItem>
                                ))}

                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Model</label>
                          <Select value={config.sttModel} onValueChange={(v) => updateConfig("sttModel", v)} disabled={!config.sttProvider}>
                            <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                              <SelectValue placeholder="Select model" />
                            </SelectTrigger>
                            <SelectContent className="rounded-lg">
                            {(getAllSTTModelsForProvider[config.sttProvider] || [])
                              .filter((model) => supportedSTTModels.has(model))
                              .map((model) => (
                                <SelectItem key={model} value={model} className="py-2.5 font-mono text-sm">
                                  {model}
                                </SelectItem>
                              ))}

                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      )}
                    </div>
                  )}

                  {/* TTS Settings */}
                  {activeLanguages.length > 0 && (
                    <div className="space-y-4 pt-6 border-t border-slate-100">
                      <label className="text-base font-bold text-slate-900 italic">Select synthesizer</label>
                      {availableTTSProviders.length === 0 ? (
                      <div className="rounded-lg border border-slate-200 bg-slate-50 p-4">
                        <p className="text-sm font-medium text-slate-900">
                          No synthesizer available for {activeLanguages.join(", ")}
                        </p>
                        <p className="text-sm text-slate-600 mt-1">
                          Add an API key in{" "}
                          <button
                            type="button"
                            onClick={() => router.push("/integrations")}
                            className="text-blue-600 underline hover:text-blue-700"
                          >
                            Integrations
                          </button>
                          , or start the AI4Bharat TTS server.
                        </p>
                      </div>
                      ) : (
                      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Provider</label>
                          <Select value={config.ttsProvider} onValueChange={(v) => updateConfig("ttsProvider", v)}>
                            <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                              <div className="flex items-center gap-2">
                                <Volume2 className="h-4 w-4 text-slate-400" />
                                <SelectValue placeholder="Select provider" />
                              </div>
                            </SelectTrigger>
                            <SelectContent className="rounded-lg">
                              <div className="px-2 py-1.5 text-xs font-semibold text-slate-500">TTS Providers</div>
                              {availableTTSProviders
                                .map((provider) => (
                                  <SelectItem key={provider.id} value={provider.id} className="py-2.5">
                                    <span className="font-medium">{provider.name}</span>
                                  </SelectItem>
                                ))}

                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Model</label>
                          <Select value={config.ttsModel} onValueChange={(v) => updateConfig("ttsModel", v)} disabled={!config.ttsProvider}>
                            <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                              <SelectValue placeholder="Select model" />
                            </SelectTrigger>
                            <SelectContent className="rounded-lg">
                            {(getAllTTSModelsForProvider[config.ttsProvider] || [])
                              .filter((model) => supportedTTSModels.has(model))
                              .map((model) => (
                                <SelectItem key={model} value={model} className="py-2.5 font-mono text-sm">
                                  {model}
                                </SelectItem>
                              ))}

                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Voice</label>
                          <div className="flex items-center gap-2">
                            {(config.ttsProvider === "gcp" || config.ttsProvider === "cartesia" || config.ttsProvider === "elevenlabs") ? (
                              <Input
                                value={config.ttsVoice}
                                onChange={(e) => updateConfig("ttsVoice", e.target.value)}
                                placeholder="Enter voice ID"
                                className="h-12 rounded-lg border-slate-200 bg-white font-medium focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all flex-1"
                              />
                            ) : (
                              <Select value={config.ttsVoice} onValueChange={(v) => updateConfig("ttsVoice", v)} disabled={!config.ttsProvider || availableTTSVoices.length === 0}>
                                <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all flex-1">
                                  <SelectValue placeholder="Select voice" />
                                </SelectTrigger>
                                <SelectContent className="rounded-lg max-h-[200px]">
                                  {availableTTSVoices.map((voice) => (
                                    <SelectItem key={voice} value={voice} className="py-2.5">
                                      <span className="font-medium">{voice}</span>
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            )}
                          </div>
                        </div>
                      </div>
                      )}

                      {/* TTS Description for AI4Bharat and Bhashini */}
                      {(config.ttsProvider === "ai4bharat" || config.ttsProvider === "bhashini") && (
                        <div className="space-y-2 pt-3">
                          <label className="text-sm font-semibold text-slate-700">Voice Description</label>
                          <Select value={config.ttsDescription} onValueChange={(v) => updateConfig("ttsDescription", v)}>
                            <SelectTrigger className="min-h-[48px] py-3 px-4 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all text-left [&>span]:whitespace-normal [&>span]:line-clamp-2 [&>span]:text-left">
                              <SelectValue placeholder="Select a voice description to customize voice characteristics" className="whitespace-normal" />
                            </SelectTrigger>
                            <SelectContent className="rounded-lg max-h-[300px] w-[min(600px,calc(100vw-2rem))]">
                              {descriptionsData.map((item, index) => (
                                <SelectItem key={index} value={item.description} className="py-3 px-3">
                                  <span className="text-sm leading-relaxed block whitespace-normal">{item.description}</span>
                                </SelectItem>
                              ))}
                            </SelectContent>
                          </Select>
                          <p className="text-sm text-blue-600">
                            This description defines the pitch, pace, expression, and overall voice characteristics.
                          </p>
                        </div>
                      )}
                    </div>
                  )}

                  {/* Voice Settings */}
                  {activeLanguages.length > 0 && config.ttsProvider && (
                    <div className="space-y-5 pt-6 border-t border-slate-100">
                      <div className="flex items-center gap-2">
                        <div className="h-7 w-7 rounded bg-slate-100 flex items-center justify-center">
                          <Settings className="h-4 w-4 text-slate-500" />
                        </div>
                        <h3 className="text-sm font-semibold text-slate-900">Voice Settings</h3>
                      </div>

                      {config.ttsProvider === "deepgram" && (
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-100">
                          <p className="text-sm text-slate-600">
                            <span className="font-medium">Deepgram</span> voices are pre-configured with optimal settings for natural speech.
                          </p>
                        </div>
                      )}

                      {config.ttsProvider === "gcp" && (
                        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6 bg-slate-50 rounded-lg p-4 sm:p-5 border border-slate-100">
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Speaking Rate</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.speedRate.toFixed(2)}x</span>
                            </div>
                            <Slider value={[config.speedRate]} onValueChange={([v]) => updateConfig("speedRate", v)} min={0.25} max={4.0} step={0.05} />
                            <p className="text-xs text-slate-500">Speed of speech (1.0 = normal)</p>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Pitch</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.stability > 50 ? '+' : ''}{(config.stability - 50) * 0.4} st</span>
                            </div>
                            <Slider value={[config.stability]} onValueChange={([v]) => updateConfig("stability", v)} min={0} max={100} step={1} />
                            <p className="text-xs text-slate-500">Voice pitch in semitones (-20 to +20)</p>
                          </div>
                        </div>
                      )}

                      {config.ttsProvider === "cartesia" && (
                        <div className="grid grid-cols-2 gap-6 bg-slate-50 rounded-lg p-5 border border-slate-100">
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Speed</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.speedRate.toFixed(1)}x</span>
                            </div>
                            <Slider value={[config.speedRate]} onValueChange={([v]) => updateConfig("speedRate", v)} min={0.5} max={2.0} step={0.1} />
                            <p className="text-xs text-slate-500">Playback speed multiplier</p>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Emotion Intensity</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.similarityBoost}%</span>
                            </div>
                            <Slider value={[config.similarityBoost]} onValueChange={([v]) => updateConfig("similarityBoost", v)} min={0} max={100} step={1} />
                            <p className="text-xs text-slate-500">Emotional expressiveness level</p>
                          </div>
                        </div>
                      )}

                      {config.ttsProvider === "sarvam" && (
                        <div className="grid grid-cols-2 gap-6 bg-slate-50 rounded-lg p-5 border border-slate-100">
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Pace</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.speedRate.toFixed(1)}x</span>
                            </div>
                            <Slider value={[config.speedRate]} onValueChange={([v]) => updateConfig("speedRate", v)} min={0.5} max={2.0} step={0.1} />
                            <p className="text-xs text-slate-500">Speaking pace (1.0 = normal)</p>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Pitch</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{((config.stability - 50) / 25).toFixed(1)}</span>
                            </div>
                            <Slider value={[config.stability]} onValueChange={([v]) => updateConfig("stability", v)} min={0} max={100} step={1} />
                            <p className="text-xs text-slate-500">Voice pitch adjustment (-2 to +2)</p>
                          </div>
                          <div className="space-y-3">
                            <div className="flex items-center justify-between">
                              <label className="text-sm font-medium text-slate-700">Loudness</label>
                              <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{(config.similarityBoost / 50).toFixed(1)}x</span>
                            </div>
                            <Slider value={[config.similarityBoost]} onValueChange={([v]) => updateConfig("similarityBoost", v)} min={25} max={100} step={1} />
                            <p className="text-xs text-slate-500">Audio volume level</p>
                          </div>
                        </div>
                      )}

                      {(config.ttsProvider === "ai4bharat" || config.ttsProvider === "bhashini") && (
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-100">
                          <p className="text-sm text-slate-600">
                            <span className="font-medium">{config.ttsProvider === "ai4bharat" ? "AI4Bharat" : "Bhashini"}</span> uses description-based voice control. Select a voice description above to customize pitch, pace, and expression characteristics.
                          </p>
                        </div>
                      )}

                      {/* Buffer Size */}
                      <div className="pt-5 border-t border-slate-100">
                        <div className="max-w-sm space-y-3 bg-slate-50 rounded-lg p-4 border border-slate-100">
                          <div className="flex items-center justify-between">
                            <label className="text-sm font-medium text-slate-700">Buffer Size</label>
                            <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{config.bufferSize}</span>
                          </div>
                          <Slider value={[config.bufferSize]} onValueChange={([v]) => updateConfig("bufferSize", v)} min={0} max={100} step={1} />
                          <p className="text-xs text-slate-500">Higher values reduce latency but may affect quality</p>
                        </div>
                      </div>
                    </div>
                  )}
                </div>

                <Button
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step: Delivery */}
            {currentStepKey === "telephony" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-3">
                  <label className="text-base font-bold text-slate-900">Delivery Mode</label>
                  <Select
                    value={config.telephonyProvider}
                    onValueChange={(v) => updateConfig("telephonyProvider", v)}
                  >
                    <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                      <div className="flex items-center gap-2">
                        <Phone className="h-4 w-4 text-slate-400" />
                        <SelectValue placeholder="Select delivery mode" />
                      </div>
                    </SelectTrigger>
                    <SelectContent className="rounded-lg">
                      {availableTelephonyProviders.map((provider) => (
                        <SelectItem key={provider} value={provider} className="py-3">
                          <span className="font-medium">{provider}</span>
                        </SelectItem>
                      ))}
                      <SelectItem value={WEBSOCKET_PROVIDER} className="py-3">
                        <span className="font-medium">WebSocket</span>
                      </SelectItem>
                    </SelectContent>
                  </Select>
                  <p className="text-sm text-slate-500">
                    {config.telephonyProvider === WEBSOCKET_PROVIDER
                      ? "Browser test only — no phone provider or phone number required."
                      : "Phone calls via the selected telephony provider."}
                  </p>
                  {availableTelephonyProviders.length === 0 && (
                    <p className="text-sm text-slate-600">
                      Phone delivery needs telephony credentials. Add Vobiz or Plivo in{" "}
                      <button
                        type="button"
                        onClick={() => router.push("/integrations")}
                        className="text-blue-600 underline hover:text-blue-700"
                      >
                        Integrations
                      </button>
                      .
                    </p>
                  )}
                </div>

                <Button
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step 5: Call Management */}
            {currentStepKey === "call_mgmt" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <h2 className="text-xl font-bold text-slate-900 mb-1">Call Management</h2>
                <p className="text-slate-500 mb-8">
                  Control interruptions, welcome behavior, silence handling, and call duration.
                </p>

                <div className="space-y-10">
                  {/* Interruption */}
                  <div className="space-y-4">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Interruption
                    </p>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <label className="text-base font-bold text-slate-900">
                          Words before interrupting
                        </label>
                        <p className="text-sm text-slate-500 mt-1">
                          Minimum words the caller must speak before the bot stops its audio.
                        </p>
                      </div>
                      <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                        {config.interruptionMinWords} {config.interruptionMinWords === 1 ? "word" : "words"}
                      </span>
                    </div>
                    <Slider
                      value={[config.interruptionMinWords]}
                      onValueChange={([value]) => updateConfig("interruptionMinWords", value)}
                      min={1}
                      max={10}
                      step={1}
                      className="w-full"
                    />
                  </div>

                  <div className="space-y-4 pt-4 border-t border-slate-100">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      User online detection
                    </p>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <p className="text-base font-bold text-slate-900">
                          Enable user online detection
                        </p>
                        <p className="text-sm text-slate-500 mt-1">
                          Ask whether the caller is still on the line after they stay silent.
                        </p>
                      </div>
                      <label className="relative inline-flex shrink-0 items-center cursor-pointer">
                        <input
                          type="checkbox"
                          className="sr-only peer"
                          checked={config.userOnlineDetectionEnabled}
                          onChange={() =>
                            updateConfig(
                              "userOnlineDetectionEnabled",
                              !config.userOnlineDetectionEnabled
                            )
                          }
                        />
                        <div className="w-11 h-6 bg-slate-200 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer-checked:bg-emerald-600 transition-colors" />
                        <div className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5" />
                      </label>
                    </div>

                    {config.userOnlineDetectionEnabled && (
                      <div className="space-y-4">
                        <div className="space-y-2">
                          <label className="text-base font-bold text-slate-900">
                            Detection message
                          </label>
                          <Textarea
                            value={config.userOnlineDetectionMessage}
                            onChange={(e) =>
                              updateConfig("userOnlineDetectionMessage", e.target.value)
                            }
                            placeholder="e.g. Hello, are you still on the call?"
                            rows={2}
                          />
                        </div>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between gap-4">
                            <label className="text-base font-bold text-slate-900">
                              Silence before prompt
                            </label>
                            <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                              {formatDurationSeconds(config.userOnlineDetectionSeconds)}
                            </span>
                          </div>
                          <Slider
                            value={[config.userOnlineDetectionSeconds]}
                            onValueChange={([value]) =>
                              updateConfig("userOnlineDetectionSeconds", value)
                            }
                            min={5}
                            max={60}
                            step={5}
                            className="w-full"
                          />
                          <p className="text-sm text-slate-500">
                            Seconds of user silence after the bot finishes speaking.
                          </p>
                        </div>
                        <div className="space-y-3">
                          <div className="flex items-center justify-between gap-4">
                            <label className="text-base font-bold text-slate-900">
                              Prompt repeats
                            </label>
                            <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                              {config.userOnlineDetectionRepeats}
                            </span>
                          </div>
                          <Slider
                            value={[config.userOnlineDetectionRepeats]}
                            onValueChange={([value]) =>
                              updateConfig("userOnlineDetectionRepeats", value)
                            }
                            min={1}
                            max={10}
                            step={1}
                            className="w-full"
                          />
                          <p className="text-sm text-slate-500">
                            How many times to ask before the closing message and hangup.
                          </p>
                        </div>
                        <div className="space-y-2">
                          <label className="text-base font-bold text-slate-900">
                            Closing message
                          </label>
                          <Textarea
                            value={config.userOnlineDetectionClosingMessage}
                            onChange={(e) =>
                              updateConfig(
                                "userOnlineDetectionClosingMessage",
                                e.target.value
                              )
                            }
                            placeholder="e.g. We could not hear you. Ending the call now. Goodbye."
                            rows={2}
                          />
                          <p className="text-sm text-slate-500">
                            Spoken after the last detection prompt, then the call ends. Leave
                            empty to hang up immediately after the last prompt.
                          </p>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* End of call */}
                  <div className="space-y-8 pt-4 border-t border-slate-100">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      End of call
                    </p>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <label className="text-base font-bold text-slate-900">
                            Hangup on user silence
                          </label>
                          <p className="text-sm text-slate-500 mt-1">
                            End the call if the user stays silent after the bot finishes speaking.
                          </p>
                        </div>
                        <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                          {formatDurationSeconds(config.userSilenceHangupSeconds)}
                        </span>
                      </div>
                      <Slider
                        value={[config.userSilenceHangupSeconds]}
                        onValueChange={([value]) =>
                          updateConfig("userSilenceHangupSeconds", value)
                        }
                        min={0}
                        max={120}
                        step={5}
                        className="w-full"
                      />
                      <p className="text-sm text-slate-500">
                        Set to 0 to disable. Range: 0–120 seconds.
                      </p>
                    </div>

                    <div className="space-y-4">
                      <div className="flex items-center justify-between gap-4">
                        <div>
                          <label className="text-base font-bold text-slate-900">
                            Total call timeout
                          </label>
                          <p className="text-sm text-slate-500 mt-1">
                            Maximum duration of a call before it is ended automatically.
                          </p>
                        </div>
                        <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                          {formatDurationSeconds(config.callTimeoutSeconds)}
                        </span>
                      </div>
                      <Slider
                        value={[config.callTimeoutSeconds]}
                        onValueChange={([value]) =>
                          updateConfig("callTimeoutSeconds", value)
                        }
                        min={60}
                        max={3600}
                        step={30}
                        className="w-full"
                      />
                      <p className="text-sm text-slate-500">
                        Range: 1 minute to 60 minutes.
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4 pt-4 border-t border-slate-100">
                    <p className="text-xs font-semibold uppercase tracking-wide text-slate-400">
                      Hold messages
                    </p>
                    <p className="text-sm text-slate-500">
                      Played while waiting for a Kenpath LLM response. Leave empty to disable.
                      Messages rotate on each delay.
                    </p>
                    <GreetingCommaBanner context="hold" />

                    <div className="space-y-3">
                      {config.holdMessages.map((message, index) => (
                        <div key={index} className="flex items-center gap-2">
                          <SpokenMessageInput
                            value={message}
                            onChange={(value) => {
                              const next = [...config.holdMessages]
                              next[index] = value
                              updateConfig("holdMessages", next)
                            }}
                            placeholder="e.g. Please wait I am looking up the information"
                            className="flex-1"
                          />
                          <Button
                            type="button"
                            variant="outline"
                            size="icon"
                            className="shrink-0"
                            onClick={() => {
                              updateConfig(
                                "holdMessages",
                                config.holdMessages.filter((_, i) => i !== index)
                              )
                            }}
                            aria-label="Remove hold message"
                          >
                            <Trash2 className="h-4 w-4" />
                          </Button>
                        </div>
                      ))}
                      <Button
                        type="button"
                        variant="outline"
                        className="gap-2"
                        onClick={() =>
                          updateConfig("holdMessages", [...config.holdMessages, ""])
                        }
                      >
                        <Plus className="h-4 w-4" />
                        Add message
                      </Button>
                    </div>

                    <div className="space-y-3 pt-2">
                      <div className="flex items-center justify-between gap-4">
                        <label className="text-base font-bold text-slate-900">
                          Hold message delay
                        </label>
                        <span className="text-sm font-semibold text-slate-700 whitespace-nowrap tabular-nums">
                          {config.holdMessageTimeoutSeconds.toFixed(1)}s
                        </span>
                      </div>
                      <Slider
                        value={[config.holdMessageTimeoutSeconds]}
                        onValueChange={([value]) =>
                          updateConfig("holdMessageTimeoutSeconds", value)
                        }
                        min={0.1}
                        max={3}
                        step={0.1}
                        className="w-full"
                      />
                      <p className="text-sm text-slate-500">
                        Seconds to wait for the LLM before playing the next hold message.
                      </p>
                    </div>
                  </div>
                </div>

                <Button
                  onClick={handleNextStep}
                  disabled={!canProceed()}
                  className="mt-8 h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400 transition-all"
                >
                  {getNextStepLabel()}
                  <ChevronRight className="h-4 w-4" />
                </Button>
              </div>
            )}

            {/* Step 6: Review */}
            {currentStepKey === "review" && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <h2 className="text-xl font-bold text-slate-900 mb-1">Review Configuration</h2>
                <p className="text-slate-500 mb-8">Review your agent settings before creating.</p>

                <div className="space-y-0 bg-slate-50 rounded-lg border border-slate-100 overflow-hidden">
                  {/* Agent Type */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div className="flex-1 mr-4">
                      <p className="text-sm font-bold text-slate-900 mb-2">Agent Type</p>
                      <p className="text-sm text-slate-600">
                        {config.interactionMode === "non_conversational"
                          ? "Non-conversational (Alert)"
                          : "Conversational"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("type"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* Agent Details */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div className="flex-1 mr-4">
                      <p className="text-sm font-bold text-slate-900 mb-2">Agent Details</p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Name:</span> {config.name || "—"}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">
                          {config.interactionMode === "non_conversational" ? "Alert:" : "Welcome:"}
                        </span>{" "}
                        {config.greetingMessage || "—"}
                      </p>
                      <GreetingTranslateSuggestion
                        greeting={config.greetingMessage ?? ""}
                        primaryLanguage={primaryLanguage}
                        onTranslated={(text) =>
                          updateConfig("greetingMessage", text)
                        }
                        className="mb-2"
                      />
                      {config.interactionMode !== "non_conversational" && (
                        <>
                          <p className="text-sm text-slate-600 mb-1">
                            <span className="font-semibold">Ignore speech during welcome:</span>{" "}
                            {config.greetingMessage?.trim()
                              ? config.ignoreUserSpeechBeforeGreeting
                                ? "Yes"
                                : "No"
                              : "—"}
                          </p>
                          <p className="text-sm text-slate-600 line-clamp-2">
                            <span className="font-semibold">Prompt:</span> {config.systemPrompt || "—"}
                          </p>
                        </>
                      )}
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("agent"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* LLM Settings */}
                  {config.interactionMode !== "non_conversational" && (
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">LLM Model</p>
                      <p className="text-sm font-medium text-slate-700">
                        {getProviderOfficialName(config.llmProvider) || "—"}
                        {config.llmProvider === "kenpath"
                          ? ` / ${kenpathVariantLabel(config.kenpathVariant)}`
                          : ` / ${config.llmModel || "—"}`}
                      </p>
                      {config.llmProvider !== "kenpath" && (
                        <p className="text-sm text-slate-500 mt-1">
                          Tokens: {config.maxTokens} • Temperature: {config.temperature.toFixed(1)}
                        </p>
                      )}
                      {config.llmProvider === "openai" && (
                        <p className="text-sm text-slate-500 mt-1">
                          Knowledge Base:{" "}
                          {config.knowledgeEnabled
                            ? `${selectedKnowledgeDocs.length} document(s) selected`
                            : "Disabled"}
                        </p>
                      )}
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("llm"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>
                  )}

                  {/* Audio Settings */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">Audio Configuration</p>
                      <p className="text-sm font-medium text-slate-700">
                        {activeLanguages.length > 0
                          ? activeLanguages.map((l) => displayLanguageName(l)).join(", ")
                          : "—"}
                        {hasMultipleLanguages(activeLanguages) && (
                          <span className="ml-2 inline-flex rounded-full border border-emerald-200 bg-emerald-50 px-2 py-0.5 text-[10px] font-semibold uppercase tracking-wide text-emerald-700">
                            Switching
                          </span>
                        )}
                      </p>
                      {config.interactionMode !== "non_conversational" && (
                        <p className="text-sm text-slate-500 mt-1">
                          <span className="font-medium">STT:</span> {getProviderOfficialName(config.sttProvider) || "—"} / {config.sttModel || "—"}
                        </p>
                      )}
                      <p className="text-sm text-slate-500">
                        <span className="font-medium">TTS:</span> {getProviderOfficialName(config.ttsProvider) || "—"} / {config.ttsModel || "—"} / {config.ttsVoice || "—"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("audio"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* Delivery Settings */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">Delivery Mode</p>
                      <p className="text-sm font-medium text-slate-700">
                        {config.telephonyProvider || "—"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("telephony"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* Call Management */}
                  {config.interactionMode !== "non_conversational" && (
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">Call Management</p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Words before interrupt:</span> {config.interruptionMinWords}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">User online detection:</span>{" "}
                        {config.userOnlineDetectionEnabled &&
                        config.userOnlineDetectionMessage.trim()
                          ? `Enabled (${formatDurationSeconds(config.userOnlineDetectionSeconds)}, ${config.userOnlineDetectionRepeats}×)`
                          : "Disabled"}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Silence hangup:</span>{" "}
                        {formatDurationSeconds(config.userSilenceHangupSeconds)}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Call timeout:</span>{" "}
                        {formatDurationSeconds(config.callTimeoutSeconds)}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Hold messages:</span>{" "}
                        {config.holdMessages.map((m) => m.trim()).filter(Boolean).length > 0
                          ? `${config.holdMessages.map((m) => m.trim()).filter(Boolean).length} message(s)`
                          : "Disabled"}
                      </p>
                      <p className="text-sm text-slate-600">
                        <span className="font-semibold">Hold message delay:</span>{" "}
                        {config.holdMessageTimeoutSeconds.toFixed(1)}s
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(getStepIdByKey("call_mgmt"))} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>
                  )}
                </div>



                <Button
                  onClick={handleSaveAgent}
                  disabled={isCreatingAgent}
                  className="mt-8 h-11 px-6 rounded-lg bg-blue-600 hover:bg-blue-700 text-white font-medium gap-2 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isCreatingAgent ? (
                    <>
                      <Loader2 className="h-4 w-4 animate-spin" />
                      Creating...
                    </>
                  ) : (
                    <>
                      <Plus className="h-4 w-4" />
                      Create Agent
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        </main>
      </div>

      {/* Delete Success Toast */}
      {showDeleteSuccessToast && (
        <div className="mobile-toast animate-in slide-in-from-top-5 fade-in-0 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3">
          <CheckCircle2 className="h-5 w-5 text-red-600 shrink-0" />
          <p className="font-medium">Agent deleted successfully</p>
        </div>
      )}
    </div>
  )
} 
