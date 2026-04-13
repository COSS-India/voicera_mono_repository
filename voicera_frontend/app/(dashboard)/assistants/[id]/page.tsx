"use client"

import { useState, useEffect, useMemo, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
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
import {
  ChevronLeft,
  ChevronRight,
  ChevronDown,
  Phone,
  FileText,
  Save,
  Loader2,
  Volume2,
  Mic,
  Settings,
  Languages,
  Check,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { getCurrentUser, getAgent, updateAgent, getIntegrations, getKnowledgeDocuments, type User, type Agent, type CreateAgentRequest, type Integration, type KnowledgeDocument } from "@/lib/api"

// Import JSON data
import sttData from "@/stt.json"
import { displayLanguageName } from "@/lib/languageLabels"
import ttsData from "@/tts.json"
import descriptionsData from "@/descriptions.json"
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
  }
  return nameMap[providerId] || providerId.charAt(0).toUpperCase() + providerId.slice(1)
}

// Convert official provider name back to lowercase ID for internal use
const getProviderIdFromName = (providerName: string): string => {
  const reverseMap: Record<string, string> = {
    "Assembly": "assembly",
    "Anthropic": "anthropic",
    "Azure": "azure",
    "Deepgram": "deepgram",
    "Elevenlabs": "elevenlabs",
    "Gladia": "gladia",
    "Google": "gcp", // Google maps to "gcp" internally
    "GCP": "gcp", // Handle legacy "GCP" name
    "Kenpath": "kenpath",
    "Pixa": "pixa",
    "Sarvam": "sarvam",
    "Smallest": "smallest",
    "AI4Bharat": "ai4bharat",
    "Bhashini": "bhashini",
    "Cartesia": "cartesia",
    "OpenAI": "openai",
    "Qwen": "qwen",
    "PlayHT": "playht",
    "Groq": "groq",
    "Grok": "grok",
  }
  return reverseMap[providerName] || providerName.toLowerCase()
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
}

const editWizardSteps = [
  { id: 1, title: "Agent", subtitle: "Name & Prompt", icon: FileText },
  { id: 2, title: "LLM", subtitle: "Model Config", icon: Settings },
  { id: 3, title: "Audio", subtitle: "STT & TTS", icon: Volume2 },
  { id: 4, title: "Telephony", subtitle: "Select Provider", icon: Phone },
]

export default function AgentDetailPage() {
  const router = useRouter()
  const params = useParams()
  // Decode the agentId from URL
  const agentId = params.id ? decodeURIComponent(params.id as string) : ""
  const [showSuccess, setShowSuccess] = useState(false)
  const [errorMessage, setErrorMessage] = useState("")
  const [showConfirmModal, setShowConfirmModal] = useState(false)

  const [user, setUser] = useState<User | null>(null)
  const [agent, setAgent] = useState<Agent | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [isSaving, setIsSaving] = useState(false)
  const [hasChanges, setHasChanges] = useState(false)
  const [originalConfig, setOriginalConfig] = useState<any>(null)
  const [integratedProviders, setIntegratedProviders] = useState<Set<string>>(new Set())
  const [knowledgeDocs, setKnowledgeDocs] = useState<KnowledgeDocument[]>([])
  const [isKnowledgeLoading, setIsKnowledgeLoading] = useState(false)

  // Form state
  const [systemPrompt, setSystemPrompt] = useState("")
  const [greetingMessage, setGreetingMessage] = useState("")
  const [language, setLanguage] = useState("")
  const [llmProvider, setLlmProvider] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [knowledgeEnabled, setKnowledgeEnabled] = useState(false)
  const [knowledgeDocumentIds, setKnowledgeDocumentIds] = useState<string[]>([])
  const [knowledgeTopK, setKnowledgeTopK] = useState(3)
  const [sttProvider, setSttProvider] = useState("")
  const [sttModel, setSttModel] = useState("")
  const [ttsProvider, setTtsProvider] = useState("")
  const [ttsModel, setTtsModel] = useState("")
  const [ttsVoice, setTtsVoice] = useState("")
  const [ttsDescription, setTtsDescription] = useState("")
  const [speed, setSpeed] = useState(1.0)

  // Collapsible states
  const [llmSettingsOpen, setLlmSettingsOpen] = useState(true)
  const [languageOpen, setLanguageOpen] = useState(false)
  const [editStep, setEditStep] = useState(1)

  // Track if we're in the initial data loading phase to prevent validation from clearing values
  const isInitialLoadRef = useRef(true)

  // Get all unique languages from both STT and TTS (keys are now language names)
  const allLanguages = useMemo(() => {
    const sttLangs = Object.keys(sttData.stt.languages)
    const ttsLangs = Object.keys(ttsData.tts.languages)
    const merged = new Set([...sttLangs, ...ttsLangs])
    return Array.from(merged)
      .sort()
      .map((code) => ({ code, label: displayLanguageName(code) }))
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

  // Get supported STT providers for selected language
  const supportedSTTProviders = useMemo(() => {
    if (!language) return new Set<string>()
    const langData =
      sttData.stt.languages[language as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()

    return new Set(
      Object.entries(langData.models)
        .filter(([, models]) => Array.isArray(models) && models.length > 0)
        .map(([provider]) => provider)
    )
  }, [language])

  // Get supported STT models for selected provider
  const supportedSTTModels = useMemo(() => {
    if (!language || !sttProvider) return new Set<string>()
    const langData =
      sttData.stt.languages[language as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()

    const models = langData.models[sttProvider as keyof typeof langData.models]
    return new Set(Array.isArray(models) ? models : [])
  }, [language, sttProvider])

  // Get supported TTS providers for selected language
  const supportedTTSProviders = useMemo(() => {
    if (!language) return new Set<string>()
    const langData =
      ttsData.tts.languages[language as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()

    return new Set(
      Object.entries(langData.models)
        .filter(([, data]) => {
          const modelData = data as { available?: boolean }
          return modelData.available === true
        })
        .map(([provider]) => provider)
    )
  }, [language])

  // Get supported TTS models for selected provider
  const supportedTTSModels = useMemo(() => {
    if (!language || !ttsProvider) return new Set<string>()
    const langData =
      ttsData.tts.languages[language as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()

    const providerData = langData.models[ttsProvider as keyof typeof langData.models] as {
      model?: string
      models?: string[]
      available?: boolean
    }
    if (!providerData || !providerData.available) return new Set<string>()

    const models: string[] = []
    if (providerData.models && Array.isArray(providerData.models)) {
      models.push(...providerData.models)
    }
    if (providerData.model) {
      models.push(providerData.model)
    }
    return new Set(models)
  }, [language, ttsProvider])

  // Get available TTS voices for selected provider/model (Sarvam: voices vary by model)
  const availableTTSVoices = useMemo(() => {
    if (!language || !ttsProvider) return []
    const langData =
      ttsData.tts.languages[language as keyof typeof ttsData.tts.languages]
    if (!langData) return []

    const providerData = langData.models[ttsProvider as keyof typeof langData.models] as {
      voices?: string | string[]
      voices_by_model?: Record<string, string[]>
    }
    if (!providerData) return []

    if (ttsProvider === "sarvam" && ttsModel && providerData.voices_by_model?.[ttsModel]) {
      return providerData.voices_by_model[ttsModel]
    }
    if (Array.isArray(providerData.voices)) {
      return providerData.voices
    }
    return []
  }, [language, ttsProvider, ttsModel])

  // Get available TTS descriptions for AI4Bharat and Bhashini providers
  const availableTTSDescriptions = useMemo(() => {
    if (ttsProvider !== "ai4bharat" && ttsProvider !== "bhashini") return []
    return descriptionsData.map((item) => item.description)
  }, [ttsProvider])

  // Get LLM models for selected provider
  const availableLLMModels = useMemo(() => {
    if (!llmProvider) return []
    const provider = llmProviders[llmProvider as keyof typeof llmProviders]
    return provider?.models || []
  }, [llmProvider])
  const selectedKnowledgeDocs = useMemo(
    () => knowledgeDocs.filter((d) => knowledgeDocumentIds.includes(d.document_id)),
    [knowledgeDocs, knowledgeDocumentIds]
  )

  const toggleKnowledgeDocument = (documentId: string) => {
    setKnowledgeDocumentIds((prev) =>
      prev.includes(documentId)
        ? prev.filter((id) => id !== documentId)
        : [...prev, documentId]
    )
  }

  // Load agent data
  useEffect(() => {
    // Reset all state when agentId changes
    isInitialLoadRef.current = true
    setIsLoading(true)
    setAgent(null)
    setSystemPrompt("")
    setGreetingMessage("")
    setLanguage("")
    setLlmProvider("")
    setLlmModel("")
    setSttProvider("")
    setSttModel("")
    setTtsProvider("")
    setTtsModel("")
    setTtsVoice("")
    setTtsDescription("")
    setSpeed(1.0)
    setOriginalConfig(null)
    setHasChanges(false)
    setShowSuccess(false)
    setErrorMessage("")

    if (!agentId) {
      setIsLoading(false)
      return
    }

    async function loadData() {
      try {
        const userData = await getCurrentUser()
        setUser(userData)

        // Fetch integrations to know which providers have API keys
        try {
          const integrations = await getIntegrations()
          const integrated = new Set<string>()
          integrations.forEach((integration: Integration) => {
            // Store lowercase version for matching with provider IDs
            integrated.add(integration.model.toLowerCase())
          })
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

        if (userData.org_id) {
          const agentData = await getAgent(agentId, userData.org_id)
          console.log("Full agent data received:", JSON.stringify(agentData, null, 2))
          setAgent(agentData)

          setSystemPrompt(agentData.agent_config?.system_prompt || "")
          setGreetingMessage(agentData.agent_config?.greeting_message || "")

          // Load LLM settings - convert official name to internal ID
          const llmProviderName = agentData.agent_config?.llm_model?.name || ""
          setLlmProvider(getProviderIdFromName(llmProviderName))
          setLlmModel(agentData.agent_config?.llm_model?.model || "")
          setKnowledgeEnabled(Boolean((agentData.agent_config as any)?.knowledge_base_enabled))
          setKnowledgeDocumentIds(
            Array.isArray((agentData.agent_config as any)?.knowledge_document_ids)
              ? (agentData.agent_config as any).knowledge_document_ids
              : []
          )
          setKnowledgeTopK(Number((agentData.agent_config as any)?.knowledge_top_k || 3))

          // Load language - use language name directly (no conversion needed)
          // Priority: agent_config.language > stt_model.language > tts_model.language
          const configLangName = (agentData.agent_config as any)?.language || ""
          const sttLangName = agentData.agent_config?.stt_model?.language || ""
          const ttsLangName = agentData.agent_config?.tts_model?.language || ""

          // Check if language exists in JSON (more reliable than checking allLanguages array)
          const languageExistsInJSON = (langName: string) => {
            if (!langName) return false
            return langName in sttData.stt.languages || langName in ttsData.tts.languages
          }

          // Use the first available language name, verifying it exists in JSON
          // Priority: agent_config.language > stt_model.language > tts_model.language
          let selectedLanguage = ""
          if (configLangName && languageExistsInJSON(configLangName)) {
            selectedLanguage = configLangName.trim()
          } else if (sttLangName && languageExistsInJSON(sttLangName)) {
            selectedLanguage = sttLangName.trim()
          } else if (ttsLangName && languageExistsInJSON(ttsLangName)) {
            selectedLanguage = ttsLangName.trim()
          }

          if (selectedLanguage) {
            setLanguage(selectedLanguage)
          }

          // Load STT settings - convert official name to internal ID
          const sttProviderName = agentData.agent_config?.stt_model?.name || ""
          setSttProvider(getProviderIdFromName(sttProviderName))
          setSttModel(agentData.agent_config?.stt_model?.model || "")

          // Load TTS settings - convert official name to internal ID
          const ttsProviderName = agentData.agent_config?.tts_model?.name || ""
          const ttsProviderId = getProviderIdFromName(ttsProviderName)
          setTtsProvider(ttsProviderId)
          // For Cartesia, Google, and ElevenLabs, load from args; for others, load from top level
          const ttsModelConfig = agentData.agent_config?.tts_model as any
          const ttsArgs = ttsModelConfig?.args || {}
          const usesArgsForModel = ttsProviderId === "cartesia" || ttsProviderId === "gcp" || ttsProviderId === "elevenlabs"
          const modelValue = usesArgsForModel
            ? (ttsArgs.model || ttsModelConfig?.model || "")
            : (ttsModelConfig?.model || "")
          setTtsModel(modelValue)
          // For Cartesia, Google, and ElevenLabs, load voice_id from args; for others, load from speaker
          const usesArgsForVoice = ttsProviderId === "cartesia" || ttsProviderId === "gcp" || ttsProviderId === "elevenlabs"
          const voiceValue = usesArgsForVoice
            ? (ttsArgs.voice_id || ttsModelConfig?.voice_id || ttsArgs.voice || "")
            : (ttsModelConfig?.speaker || "")
          setTtsVoice(voiceValue)
          // Load TTS description for AI4Bharat and Bhashini
          if (ttsProviderId === "ai4bharat" || ttsProviderId === "bhashini") {
            setTtsDescription(ttsModelConfig?.description || "")
          } else {
            setTtsDescription("")
          }
          setSpeed(agentData.agent_config?.tts_model?.speed || 1.0)

          if (agentData.agent_config && typeof agentData.agent_config === 'object') {
            try {
              setOriginalConfig(JSON.parse(JSON.stringify(agentData.agent_config)))
            } catch (e) {
              console.error("Error parsing Agent configuration on load:", e)
            }
          }
        }
      } catch (error) {
        console.error("Failed to load agent:", error)
        setErrorMessage("Failed to load agent details")
        setTimeout(() => router.push("/assistants"), 2000)
      } finally {
        setIsLoading(false)
        // Mark initial load as complete after a brief delay to ensure all state updates are processed
        setTimeout(() => {
          isInitialLoadRef.current = false
        }, 100)
      }
    }
    loadData()
  }, [agentId, router, allLanguages])


  // Validate and clear invalid models when language or provider changes
  // Only validate after initial load is complete (not during loading)
  useEffect(() => {
    // Don't validate during initial load
    if (isLoading || isInitialLoadRef.current || !language) return

    // Clear STT model if it's not supported for current language/provider
    if (sttProvider && sttModel) {
      if (!supportedSTTModels.has(sttModel)) {
        setSttModel("")
      }
    }

    // Clear TTS model if it's not supported for current language/provider
    if (ttsProvider && ttsModel) {
      if (!supportedTTSModels.has(ttsModel)) {
        setTtsModel("")
      }
    }

    // Clear TTS voice if it's not available for current provider; for Sarvam set to first voice of model
    if (ttsProvider && ttsVoice && availableTTSVoices.length > 0) {
      if (!availableTTSVoices.includes(ttsVoice)) {
        if (ttsProvider === "sarvam") {
          setTtsVoice(availableTTSVoices[0])
        } else {
          setTtsVoice("")
        }
      }
    }
  }, [language, sttProvider, ttsProvider, ttsModel, supportedSTTModels, supportedTTSModels, availableTTSVoices, isLoading])

  // Detect changes
  useEffect(() => {
    if (!originalConfig || !agent) {
      setHasChanges(false)
      return
    }

    // Language is already a name, use it directly
    const languageName = language || ""

    // Build current config with same structure as original
    const currentConfig: any = {
      language: languageName || "", // Include top-level language field
      system_prompt: systemPrompt || "",
      greeting_message: greetingMessage || "",
      knowledge_base_enabled: llmProvider === "openai" ? knowledgeEnabled : false,
      knowledge_document_ids:
        llmProvider === "openai" && knowledgeEnabled ? knowledgeDocumentIds : [],
      knowledge_top_k: knowledgeTopK,
      llm_model: {
        name: llmProvider || "",
        ...(llmProvider && llmProvider !== "kenpath" && llmModel && { model: llmModel }),
      },
      stt_model: {
        name: sttProvider || "",
        ...(sttModel && { model: sttModel }),
        language: languageName || "",
        ...(agent.agent_config?.stt_model?.keywords && { keywords: agent.agent_config.stt_model.keywords }),
      },
      tts_model: {
        name: ttsProvider || "",
        ...(ttsModel && { model: ttsModel }),
        language: languageName || "",
        ...((ttsProvider === "cartesia" || ttsProvider === "gcp" || ttsProvider === "elevenlabs") && ttsVoice && { voice_id: ttsVoice }),
        speaker: (ttsProvider === "cartesia" || ttsProvider === "gcp" || ttsProvider === "elevenlabs") ? "" : (ttsVoice || ""),
        speed: speed || 1.0,
        ...(agent.agent_config?.tts_model?.description && { description: agent.agent_config.tts_model.description }),
        ...(agent.agent_config?.tts_model?.pitch !== undefined && { pitch: agent.agent_config.tts_model.pitch }),
        ...(agent.agent_config?.tts_model?.emotion_intensity !== undefined && { emotion_intensity: agent.agent_config.tts_model.emotion_intensity }),
        ...(agent.agent_config?.tts_model?.loudness !== undefined && { loudness: agent.agent_config.tts_model.loudness }),
      },
    }

    // Normalize configs by removing undefined/null/empty values and sorting keys
    const normalize = (obj: any): any => {
      if (obj === null || obj === undefined) return null
      if (typeof obj !== "object") return obj
      if (Array.isArray(obj)) return obj.map(normalize)

      const normalized: any = {}
      const sortedKeys = Object.keys(obj).sort()
      for (const key of sortedKeys) {
        const value = obj[key]
        if (value !== undefined && value !== null && value !== "") {
          normalized[key] = normalize(value)
        }
      }
      return normalized
    }

    const originalNormalized = JSON.stringify(normalize(originalConfig))
    const currentNormalized = JSON.stringify(normalize(currentConfig))

    const hasChanged = originalNormalized !== currentNormalized
    setHasChanges(hasChanged)
  }, [systemPrompt, greetingMessage, language, llmProvider, llmModel, knowledgeEnabled, knowledgeDocumentIds, knowledgeTopK, sttProvider, sttModel, ttsProvider, ttsModel, ttsVoice, speed, originalConfig, agent])

  const handleSaveClick = () => {
    setShowConfirmModal(true)
  }

  const handleSave = async () => {
    if (!agent || !user) return

    setShowConfirmModal(false)
    setIsSaving(true)
    try {
      const languageName = language || ""
      // Generate agent_id from agent_type (required field)
      const agentId = agent.agent_id || agent.agent_type.replace(/\s+/g, '_').toLowerCase()

      const updatedConfig: CreateAgentRequest = {
        org_id: user.org_id,
        agent_type: agent.agent_type, // Use original agent_type - required for backend to find the agent
        agent_id: agentId,
        agent_category: (agent as any).agent_category || "voicera_telephony",
        agent_config: {
          ...agent.agent_config,
          language: languageName, // Update the top-level language field
          system_prompt: systemPrompt,
          greeting_message: greetingMessage,
          knowledge_base_enabled: llmProvider === "openai" ? knowledgeEnabled : false,
          knowledge_document_ids:
            llmProvider === "openai" && knowledgeEnabled ? knowledgeDocumentIds : [],
          knowledge_top_k: knowledgeTopK,
          llm_model: {
            name: getProviderOfficialName(llmProvider),
            ...(llmProvider !== "kenpath" && { model: llmModel }),
          },
          stt_model: {
            name: getProviderOfficialName(sttProvider),
            ...(sttModel && { model: sttModel }),
            // language: languageName,
            ...(agent.agent_config?.stt_model?.keywords && { keywords: agent.agent_config.stt_model.keywords }),
          },
          tts_model: {
            name: getProviderOfficialName(ttsProvider),
            // language: languageName,
            ...((ttsProvider === "cartesia" || ttsProvider === "gcp" || ttsProvider === "elevenlabs") && {
              args: {
                ...(ttsModel && { model: ttsModel }),
                ...(ttsVoice && { voice_id: ttsVoice }),
              },
            }),
            ...(ttsProvider !== "cartesia" && ttsProvider !== "gcp" && ttsProvider !== "elevenlabs" && ttsModel && { model: ttsModel }),
            speaker: (ttsProvider === "cartesia" || ttsProvider === "gcp" || ttsProvider === "elevenlabs") ? "" : (ttsVoice || ""),
            speed: speed,
            ...((ttsProvider === "ai4bharat" || ttsProvider === "bhashini") && ttsDescription && { description: ttsDescription }),
            ...(agent.agent_config?.tts_model?.pitch !== undefined && { pitch: agent.agent_config.tts_model.pitch }),
            ...(agent.agent_config?.tts_model?.emotion_intensity !== undefined && { emotion_intensity: agent.agent_config.tts_model.emotion_intensity }),
            ...(agent.agent_config?.tts_model?.loudness !== undefined && { loudness: agent.agent_config.tts_model.loudness }),
          },
        },
      }

      const updatedAgent = await updateAgent(agentId, updatedConfig)

      if (user?.org_id) {
        const refreshedAgent = await getAgent(agentId, user.org_id)
        setAgent(refreshedAgent)

        if (refreshedAgent?.agent_config && typeof refreshedAgent.agent_config === 'object') {
          try {
            setOriginalConfig(JSON.parse(JSON.stringify(refreshedAgent.agent_config)))
          } catch (e) {
            console.error("Error parsing Agent configuration:", e)
            // Fallback to the config we sent
            if (updatedConfig?.agent_config && typeof updatedConfig.agent_config === 'object') {
              setOriginalConfig(JSON.parse(JSON.stringify(updatedConfig.agent_config)))
            }
          }
        } else if (updatedConfig?.agent_config && typeof updatedConfig.agent_config === 'object') {
          // If refreshed agent doesn't have config, use what we sent
          setOriginalConfig(JSON.parse(JSON.stringify(updatedConfig.agent_config)))
        }
      } else if (updatedAgent?.agent_config && typeof updatedAgent.agent_config === 'object') {
        setAgent(updatedAgent)
        try {
          setOriginalConfig(JSON.parse(JSON.stringify(updatedAgent.agent_config)))
        } catch (e) {
          console.error("Error parsing Agent configuration:", e)
          if (updatedConfig?.agent_config && typeof updatedConfig.agent_config === 'object') {
            setOriginalConfig(JSON.parse(JSON.stringify(updatedConfig.agent_config)))
          }
        }
      } else if (updatedConfig?.agent_config && typeof updatedConfig.agent_config === 'object') {
        setOriginalConfig(JSON.parse(JSON.stringify(updatedConfig.agent_config)))
      }

      setHasChanges(false)
      setShowSuccess(true)
      setErrorMessage("")
      setTimeout(() => setShowSuccess(false), 3000)
    } catch (error) {
      console.error("Failed to update agent:", error)
      setErrorMessage(error instanceof Error ? error.message : "Failed to update assistant")
      setShowSuccess(false)
      setTimeout(() => setErrorMessage(""), 5000)
    } finally {
      setIsSaving(false)
    }
  }

  const handleBackToList = () => {
    router.push("/assistants")
  }

  const handleNextStep = () => {
    setEditStep((prev) => Math.min(prev + 1, editWizardSteps.length))
  }

  const handlePreviousStep = () => {
    setEditStep((prev) => Math.max(prev - 1, 1))
  }

  const progressPercent = (editStep / editWizardSteps.length) * 100


  if (isLoading) {
    return (
      <div className="flex flex-col h-screen bg-slate-50/50">
        <div className="flex-1 flex items-center justify-center">
          <Loader2 className="h-8 w-8 animate-spin text-slate-400" />
        </div>
      </div>
    )
  }

  if (!agent) {
    return null
  }

  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header with Progress */}
      <header className="flex h-14 items-center justify-between border-b border-slate-200 bg-white px-6 sticky top-0 z-10">
        <div className="flex items-center gap-4">
          <Button
            variant="ghost"
            size="sm"
            onClick={handleBackToList}
            className="h-8 px-3 text-slate-600 hover:bg-slate-100 gap-1.5"
          >
            <ChevronLeft className="h-4 w-4" />
            Back
          </Button>
          <Separator orientation="vertical" className="h-5" />
          <h1 className="text-sm font-semibold text-slate-900">Configure Telephony Agent</h1>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-xs text-slate-500">Progress</span>
          <div className="w-32 h-1.5 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-slate-900 rounded-full transition-all duration-300"
              style={{ width: `${progressPercent}%` }}
            />
          </div>
          <span className="text-xs font-medium text-slate-700">{Math.round(progressPercent)}%</span>
        </div>
      </header>

      {/* Main Content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Top Row - Progress Stepper */}
        <aside className="bg-white border-b border-slate-100 p-3 sm:p-4">
          <div className="flex gap-2 overflow-x-auto pb-1 justify-center">
            {editWizardSteps.map((step) => {
              const Icon = step.icon
              const isActive = editStep === step.id
              const isCompleted = editStep > step.id

              return (
                <button
                  key={step.id}
                  onClick={() => setEditStep(step.id)}
                  className={`shrink-0 min-w-[140px] sm:min-w-[160px] flex items-center gap-2 px-3 py-2.5 rounded-lg text-left transition-all duration-150 ${
                    isActive ? "bg-slate-100" : "hover:bg-slate-50 cursor-pointer"
                  }`}
                >
                  <div
                    className={`h-8 w-8 rounded-md flex items-center justify-center transition-all duration-150 shrink-0 ${
                      isActive
                        ? "bg-slate-900 text-white"
                        : isCompleted
                        ? "bg-slate-200 text-slate-600"
                        : "bg-slate-100 text-slate-400"
                    }`}
                  >
                    <Icon className="h-4 w-4" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <p
                      className={`text-sm font-medium leading-tight truncate ${
                        isActive ? "text-slate-900" : isCompleted ? "text-slate-700" : "text-slate-500"
                      }`}
                    >
                      {step.title}
                    </p>
                    <p className="text-[11px] text-slate-400 truncate">{step.subtitle}</p>
                  </div>
                </button>
              )
            })}
          </div>
        </aside>

        {/* Step Content */}
        <main className="flex-1 overflow-auto p-6 sm:p-8">
          <div className="w-full max-w-4xl mx-auto">
        {/* Configure Layout */}
        <div className="space-y-4">

          {/* Section Content */}
          <div className="grid grid-cols-1 gap-6">
            {/* Left Column - Settings */}
            <div className="space-y-6">
            {/* LLM Settings */}
            <div className={`bg-white rounded-xl border border-slate-200 overflow-hidden ${editStep === 2 ? "" : "hidden"}`}>
              <button
                onClick={() => setLlmSettingsOpen(!llmSettingsOpen)}
                className="w-full p-4 flex items-center justify-between hover:bg-slate-50 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <Settings className="h-5 w-5 text-slate-600" />
                  <span className="font-semibold text-slate-900">LLM Settings</span>
                </div>
                {llmSettingsOpen ? (
                  <ChevronDown className="h-4 w-4 text-slate-400" />
                ) : (
                  <ChevronRight className="h-4 w-4 text-slate-400" />
                )}
              </button>
              {llmSettingsOpen && (
                <div className="p-6 space-y-6 border-t border-slate-200 bg-slate-50 rounded-b-xl">
                  <div className="">
                    <label className="text-sm font-semibold text-slate-700 mb-2 block tracking-wide">
                      <span className="inline-flex items-center gap-2">
                        LLM Provider
                      </span>
                    </label>
                    <Select
                      value={llmProvider}
                      onValueChange={(v) => {
                        setLlmProvider(v);
                        setLlmModel("");
                        if (v !== "openai") {
                          setKnowledgeEnabled(false)
                          setKnowledgeDocumentIds([])
                        }
                      }}
                    >
                      <SelectTrigger className="border-slate-200 h-11 shadow-sm rounded-md focus:ring-slate-300 transition focus:border-slate-500 bg-white">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent className="z-[100] rounded-md shadow-lg">
                        {Object.entries(llmProviders).map(([id, provider]) => {
                          // OpenAI, Qwen, and Kenpath are always available (built-in)
                          const isBuiltIn = id === "openai" || id === "qwen" || id === "kenpath"
                          // Check if provider has integration (API key configured)
                          const isIntegrated = integratedProviders.has(id) || integratedProviders.has(provider.name.toLowerCase())
                          const isAvailable = isBuiltIn || isIntegrated
                          
                          return (
                            <SelectItem
                              key={id}
                              value={id}
                              className="font-medium hover:bg-slate-100 transition"
                              disabled={!isAvailable}
                            >
                              <div className="flex items-center gap-2">
                                <span>{provider.name}</span>
                                {!isAvailable && (
                                  <span className="text-xs text-slate-400">(not integrated)</span>
                                )}
                              </div>
                            </SelectItem>
                          )
                        })}
                      </SelectContent>
                    </Select>
                  </div>

                  {llmProvider && llmProvider !== "kenpath" && (
                    <div>
                      <label className="text-sm font-semibold text-slate-700 mb-2 block">
                        <span className="inline-flex items-center gap-2">
                          LLM Model
                        </span>
                      </label>
                      <Select
                        value={llmModel}
                        onValueChange={setLlmModel}
                        disabled={availableLLMModels.length === 0}
                      >
                        <SelectTrigger className="border-slate-200 h-11 shadow-sm rounded-md focus:ring-slate-300 transition focus:border-slate-500 bg-white">
                          <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent className="z-[100] rounded-md shadow-lg">
                          {availableLLMModels.map((model) => (
                            <SelectItem
                              key={model}
                              value={model}
                              className="font-mono text-sm hover:bg-slate-100 transition"
                            >
                              {model}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      {availableLLMModels.length === 0 && (
                        <div className="text-xs text-slate-400 mt-2 pl-1">
                          No models available for this provider.
                        </div>
                      )}
                    </div>
                  )}

                  {llmProvider === "openai" && (
                    <div className="border border-slate-200 rounded-lg p-4 space-y-3 bg-slate-50">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-sm font-semibold text-slate-800">Knowledge Base</p>
                          <p className="text-xs text-slate-500">Use selected knowledge files during responses.</p>
                        </div>
                        <label className="relative inline-flex items-center cursor-pointer">
                          <input
                            type="checkbox"
                            className="sr-only peer"
                            checked={knowledgeEnabled}
                            onChange={() => setKnowledgeEnabled((v) => !v)}
                          />
                          <div
                            className="w-11 h-6 bg-slate-200 dark:bg-slate-800 peer-focus:outline-none peer-focus:ring-2 peer-focus:ring-blue-500 rounded-full peer-checked:bg-emerald-600 transition-colors"
                          />
                          <div
                            className="absolute left-1 top-1 w-4 h-4 bg-white rounded-full transition-transform peer-checked:translate-x-5"
                          />
                        </label>
                      </div>
                      {knowledgeEnabled && (
                        <div className="space-y-2">
                          <div className="text-xs text-slate-500">
                            {selectedKnowledgeDocs.length} document(s) selected
                          </div>
                          {isKnowledgeLoading ? (
                            <div className="flex items-center gap-2 text-sm text-slate-500">
                              <Loader2 className="h-4 w-4 animate-spin" />
                              Loading knowledge documents...
                            </div>
                          ) : knowledgeDocs.length === 0 ? (
                            <p className="text-sm text-slate-500">No ready knowledge documents found.</p>
                          ) : (
                            <div className="max-h-40 overflow-auto rounded-md border border-slate-200 bg-white divide-y divide-slate-100">
                              {knowledgeDocs.map((doc) => {
                                const checked = knowledgeDocumentIds.includes(doc.document_id)
                                return (
                                  <button
                                    key={doc.document_id}
                                    type="button"
                                    onClick={() => toggleKnowledgeDocument(doc.document_id)}
                                    className="w-full px-3 py-2 text-left hover:bg-slate-50 flex items-center gap-3"
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
                                    <span className="text-sm text-slate-700 truncate">
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
                </div>
              )}
            </div>

            {/* Audio Settings */}
            <div className={`${editStep === 3 ? "space-y-4" : "hidden"}`}>
              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <h3 className="text-2xl font-semibold text-slate-900 mb-4 flex items-center gap-2">
                  <Languages className="h-5 w-5 text-slate-400" />
                  Configure Language
                </h3>
                <div className="space-y-3">
                  <label className="text-base font-bold text-slate-900">Language</label>
                  {allLanguages.length > 0 ? (
                    <Popover open={languageOpen} onOpenChange={setLanguageOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          role="combobox"
                          aria-expanded={languageOpen}
                          className="w-full min-h-[48px] py-3 px-4 justify-between rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 text-left [&>div]:whitespace-normal"
                        >
                          <div className="flex items-center gap-2 flex-1 min-w-0">
                            <Languages className="h-4 w-4 text-blue-500 flex-shrink-0" />
                            <span className="truncate">
                              {language ? displayLanguageName(language) : "Select language..."}
                            </span>
                          </div>
                        </Button>
                      </PopoverTrigger>
                      <PopoverContent className="w-[400px] p-0" align="start">
                        <Command>
                          <CommandInput placeholder="Search languages..." />
                          <CommandList>
                            <CommandEmpty>No language found.</CommandEmpty>
                            <CommandGroup heading="Languages">
                              {allLanguages.map((lang) => (
                                <CommandItem
                                  key={lang.code}
                                  value={`${lang.code} ${lang.label}`}
                                  onSelect={() => {
                                    setLanguage(lang.code)
                                    if (lang.code && lang.code !== "English (United States)" && lang.code !== "English (India)") {
                                      const sttLangData = sttData.stt.languages[lang.code as keyof typeof sttData.stt.languages]
                                      if (sttLangData?.models?.ai4bharat && Array.isArray(sttLangData.models.ai4bharat) && sttLangData.models.ai4bharat.length > 0) {
                                        setSttProvider("ai4bharat")
                                        setSttModel(sttLangData.models.ai4bharat[0])
                                      } else {
                                        setSttProvider("")
                                        setSttModel("")
                                      }

                                      const ttsLangData = ttsData.tts.languages[lang.code as keyof typeof ttsData.tts.languages]
                                      const ttsAi4bharatData = ttsLangData?.models?.ai4bharat as { available?: boolean; model?: string; voices?: string[] } | undefined
                                      if (ttsAi4bharatData?.available && ttsAi4bharatData.model) {
                                        setTtsProvider("ai4bharat")
                                        setTtsModel(ttsAi4bharatData.model)
                                        if (ttsAi4bharatData.voices && Array.isArray(ttsAi4bharatData.voices) && ttsAi4bharatData.voices.length > 0) {
                                          setTtsVoice(ttsAi4bharatData.voices[0])
                                          setTtsDescription(
                                            descriptionsData && descriptionsData.length > 0
                                              ? descriptionsData[0].description
                                              : ""
                                          )
                                        } else {
                                          setTtsVoice("")
                                          setTtsDescription("")
                                        }
                                      } else {
                                        setTtsProvider("")
                                        setTtsModel("")
                                        setTtsVoice("")
                                        setTtsDescription("")
                                      }
                                    } else {
                                      setSttProvider("")
                                      setSttModel("")
                                      setTtsProvider("")
                                      setTtsModel("")
                                      setTtsVoice("")
                                      setTtsDescription("")
                                    }
                                    setLanguageOpen(false)
                                  }}
                                  className="py-2.5"
                                >
                                  <span className="font-medium">{lang.label}</span>
                                </CommandItem>
                              ))}
                            </CommandGroup>
                          </CommandList>
                        </Command>
                      </PopoverContent>
                    </Popover>
                  ) : (
                    <div className="px-3 py-2 text-base text-slate-500 border border-slate-200 rounded-lg bg-slate-50">
                      Loading languages...
                    </div>
                  )}
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <h3 className="text-2xl font-semibold text-slate-900 mb-5 flex items-center gap-2">
                  <Mic className="h-5 w-5 text-slate-400" />
                  Speech-to-Text
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Provider</label>
                    <Select
                      value={sttProvider}
                      onValueChange={(v) => {
                        setSttProvider(v)
                        setSttModel("")
                      }}
                    >
                      <SelectTrigger className="border-slate-200 rounded-md h-11 bg-white">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {allSTTProviders
                          .filter((p) => supportedSTTProviders.has(p.id))
                          .map((provider) => {
                            const isOnPrem = provider.id === "ai4bharat"
                            const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                            return (
                              <SelectItem key={provider.id} value={provider.id} disabled={!isIntegrated}>
                                <div className="flex items-center gap-2">
                                  <span>{provider.name}</span>
                                  {!isIntegrated && <span className="text-xs text-slate-400">(not integrated)</span>}
                                </div>
                              </SelectItem>
                            )
                          })}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Model</label>
                    <Select value={sttModel} onValueChange={setSttModel} disabled={!sttProvider || supportedSTTModels.size === 0}>
                      <SelectTrigger className="border-slate-200 rounded-md h-11 bg-white">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from(supportedSTTModels).map((model) => (
                          <SelectItem key={model} value={model} className="font-mono text-sm">
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-xl border border-slate-200 p-6">
                <h3 className="text-2xl font-semibold text-slate-900 mb-5 flex items-center gap-2">
                  <Volume2 className="h-5 w-5 text-slate-400" />
                  Text-to-Speech
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  <div>
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Provider</label>
                    <Select
                      value={ttsProvider}
                      onValueChange={(v) => {
                        setTtsProvider(v)
                        setTtsModel("")
                        setTtsVoice("")
                        setTtsDescription("")
                      }}
                    >
                      <SelectTrigger className="border-slate-200 rounded-md h-11 bg-white">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent>
                        {allTTSProviders
                          .filter((p) => supportedTTSProviders.has(p.id))
                          .map((provider) => {
                            const isOnPrem = provider.id === "ai4bharat"
                            const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                            return (
                              <SelectItem key={provider.id} value={provider.id} disabled={!isIntegrated}>
                                <div className="flex items-center gap-2">
                                  <span>{provider.name}</span>
                                  {!isIntegrated && <span className="text-xs text-slate-400">(not integrated)</span>}
                                </div>
                              </SelectItem>
                            )
                          })}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Model</label>
                    <Select value={ttsModel} onValueChange={setTtsModel} disabled={!ttsProvider || supportedTTSModels.size === 0}>
                      <SelectTrigger className="border-slate-200 rounded-md h-11 bg-white">
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        {Array.from(supportedTTSModels).map((model) => (
                          <SelectItem key={model} value={model} className="font-mono text-sm">
                            {model}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div>
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Voice</label>
                    {(ttsProvider === "gcp" || ttsProvider === "cartesia" || ttsProvider === "elevenlabs") ? (
                      <Input
                        value={ttsVoice}
                        onChange={(e) => setTtsVoice(e.target.value)}
                        placeholder={ttsProvider === "elevenlabs" ? "Enter voice ID" : "Enter voice ID"}
                        className="h-11 border-slate-200 rounded-md bg-white"
                      />
                    ) : (
                      <Select value={ttsVoice} onValueChange={setTtsVoice} disabled={!ttsProvider || availableTTSVoices.length === 0}>
                        <SelectTrigger className="border-slate-200 rounded-md h-11 bg-white">
                          <SelectValue placeholder="Select voice" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableTTSVoices.map((voice) => (
                            <SelectItem key={voice} value={voice}>
                              {voice}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    )}
                  </div>
                </div>

                {(ttsProvider === "ai4bharat" || ttsProvider === "bhashini") && (
                  <div className="mt-4">
                    <label className="text-sm font-semibold text-slate-700 mb-2 block">Voice Description</label>
                    <Select value={ttsDescription} onValueChange={setTtsDescription} disabled={availableTTSDescriptions.length === 0}>
                      <SelectTrigger className="min-h-[64px] w-full py-3 px-4 rounded-lg border-slate-200 bg-white text-left">
                        <SelectValue>
                          {ttsDescription
                            ? (ttsDescription.length > 25 ? `${ttsDescription.slice(0, 25)}...` : ttsDescription)
                            : "Select a voice description to customize voice characteristics"}
                        </SelectValue>
                      </SelectTrigger>
                      <SelectContent className="rounded-lg max-h-[300px] w-[600px]">
                        {availableTTSDescriptions.map((description) => (
                          <SelectItem key={description} value={description} className="py-3 px-3">
                            <span className="text-sm leading-relaxed block whitespace-normal">{description}</span>
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                )}

                {ttsModel && ttsProvider && (
                  <div className="mt-5">
                    <div className="flex items-center justify-between mb-2">
                      <label className="text-sm font-semibold text-slate-700">Speed rate</label>
                      <span className="text-sm font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{speed.toFixed(1)}</span>
                    </div>
                    <div className="flex items-center gap-3">
                      <span className="text-xs text-slate-500 min-w-[2.5rem]">0.5</span>
                      <Slider value={[speed]} onValueChange={([value]) => setSpeed(value)} min={0.5} max={2.0} step={0.1} className="flex-1" />
                      <span className="text-xs text-slate-500 min-w-[2.5rem] text-right">2.0</span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Right Column - Agent configuration */}
          <div className="space-y-6">
            <div className={`bg-white rounded-xl border border-slate-200 p-6 sm:p-8 ${editStep === 1 ? "" : "hidden"}`}>
              <h2 className="text-lg font-semibold text-slate-900 mb-4">
                Agent configuration
              </h2>

              <div className="space-y-4">
                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">
                    Greeting Message
                  </label>
                  <Input
                    value={greetingMessage}
                    onChange={(e) => setGreetingMessage(e.target.value)}
                    className="border-slate-200 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                    placeholder="Hello from Framewise"
                  />
                  <p className="text-xs text-slate-500 mt-1">
                    This will be the initial message from the agent. You can use variables here using {"{variable_name}"}
                  </p>
                </div>

                <div>
                  <label className="text-sm font-medium text-slate-700 mb-2 block">
                    System Prompt
                  </label>
                  <Textarea
                    value={systemPrompt}
                    onChange={(e) => setSystemPrompt(e.target.value)}
                    className="min-h-[120px] border-slate-200 focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
                    placeholder="Enter the system prompt for your assistant..."
                  />
                </div>
              </div>


            </div>
            <div className={`bg-white rounded-xl border border-slate-200 p-6 sm:p-8 ${editStep === 4 ? "" : "hidden"}`}>
              <h2 className="text-lg font-semibold text-slate-900 mb-4 flex items-center gap-2">
                <Phone size={20} className="text-blue-500" />
                Telephony Info
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="p-4 rounded-lg border border-slate-100 bg-slate-50 flex items-center gap-3">
                  <span className="bg-blue-100 rounded-full p-2">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 20 20"
                      fill="none"
                      className="text-blue-500"
                    >
                      <Phone size={20} />
                    </svg>
                  </span>
                  <div>
                    <div className="text-xs text-slate-500">Provider</div>
                    <div className="text-base font-bold text-slate-900">
                      {agent.telephony_provider}
                    </div>
                  </div>
                </div>
                <div className="p-4 rounded-lg border border-slate-100 bg-slate-50 flex items-center gap-3">
                  <span className="bg-blue-100 rounded-full p-2">
                    <svg
                      width="20"
                      height="20"
                      viewBox="0 0 20 20"
                      fill="none"
                      className="text-blue-500"
                    >
                      <Phone size={20} />
                    </svg>
                  </span>
                  <div>
                    <div className="text-xs text-slate-500">Phone Number</div>
                    <div className="text-base font-bold text-slate-900">
                      {agent.phone_number ? agent.phone_number : <span className="italic text-slate-400">Not linked</span>}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <div className="flex items-center justify-between pt-2">
            <Button
              variant="outline"
              onClick={handlePreviousStep}
              disabled={editStep === 1}
              className="h-11 px-6 rounded-lg border-slate-200"
            >
              Previous
            </Button>

            {editStep < editWizardSteps.length ? (
              <Button
                onClick={handleNextStep}
                className="h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2"
              >
                Next
                <ChevronRight className="h-4 w-4" />
              </Button>
            ) : (
              <Button
                onClick={handleSaveClick}
                disabled={isSaving || !hasChanges}
                className="h-11 px-6 rounded-lg bg-slate-900 hover:bg-slate-800 text-white font-medium gap-2 disabled:bg-slate-200 disabled:text-slate-400"
              >
                {isSaving ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save Changes
                  </>
                )}
              </Button>
            )}
          </div>
          </div>
        </div>
        </div>
      </main>
      </div>

      {/* Confirmation Modal */}
      <Dialog open={showConfirmModal} onOpenChange={setShowConfirmModal}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Confirm Changes</DialogTitle>
            <DialogDescription>
              Are you sure you want to save these changes?
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="gap-2">
            <Button
              variant="outline"
              onClick={() => setShowConfirmModal(false)}
              disabled={isSaving}
              className="border-slate-200"
            >
              Cancel
            </Button>
            <Button
              onClick={handleSave}
              disabled={isSaving}
              className="bg-slate-900 hover:bg-slate-800 text-white"
            >
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                  Saving...
                </>
              ) : (
                "Yes, Save Changes"
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      {/* Success Notification */}
      {showSuccess && (
        <div className="fixed top-20 right-6 z-50 bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-lg shadow-lg">
          <p className="font-medium">Agent updated successfully</p>
        </div>
      )}

      {/* Error Notification */}
      {errorMessage && (
        <div className="fixed top-20 right-6 z-50 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg shadow-lg">
          <p className="font-medium">{errorMessage}</p>
        </div>
      )}
    </div>
  )
}
