"use client"

import { useState, useEffect, useMemo, useRef } from "react"
import { useRouter, useParams } from "next/navigation"
import Link from "next/link"
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
  ChevronRight,
  ChevronDown,
  Phone,
  Code,
  Save,
  Loader2,
  Globe,
  Volume2,
  Mic,
  Settings,
  Languages,
} from "lucide-react"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { getCurrentUser, getAgent, updateAgent, getIntegrations, type User, type Agent, type CreateAgentRequest, type Integration } from "@/lib/api"

// Import JSON data
import sttData from "@/stt.json"
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
    deepgram: "Deepgram",
    elevenlabs: "Elevenlabs",
    gladia: "Gladia",
    google: "Google",
    gcp: "Google", // GCP is officially called Google
    pixa: "Pixa",
    sarvam: "Sarvam",
    smallest: "Smallest",
    ai4bharat: "AI4Bharat",
    bhashini: "Bhashini",
    cartesia: "Cartesia",
    openai: "OpenAI",
    playht: "PlayHT",
  }
  return nameMap[providerId] || providerId.charAt(0).toUpperCase() + providerId.slice(1)
}

// Convert official provider name back to lowercase ID for internal use
const getProviderIdFromName = (providerName: string): string => {
  const reverseMap: Record<string, string> = {
    "Assembly": "assembly",
    "Azure": "azure",
    "Deepgram": "deepgram",
    "Elevenlabs": "elevenlabs",
    "Gladia": "gladia",
    "Google": "gcp", // Google maps to "gcp" internally
    "GCP": "gcp", // Handle legacy "GCP" name
    "Pixa": "pixa",
    "Sarvam": "sarvam",
    "Smallest": "smallest",
    "AI4Bharat": "ai4bharat",
    "Bhashini": "bhashini",
    "Cartesia": "cartesia",
    "OpenAI": "openai",
    "PlayHT": "playht",
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
      // GPT-5 series (latest)
      "gpt-5.2",
      "gpt-5.1",
      "gpt-5",
      "gpt-5-mini",
      "gpt-5-nano",
      // GPT-4.1 series
      "gpt-4.1",
      "gpt-4.1-mini",
      "gpt-4.1-nano",
      // GPT-4o series
      "gpt-4o",
      "gpt-4o-mini",
      // Legacy
      "gpt-4-turbo",
      "gpt-4",
      "gpt-3.5-turbo",
    ],
  },
  kenpath: {
    name: "Kenpath",
    models: [],
  },
  anthropic: {
    name: "Anthropic",
    models: [
      "claude-sonnet-4-20250514",
      "claude-3-5-sonnet-20241022",
      "claude-3-5-haiku-20241022",
      "claude-3-opus-20240229",
    ],
  },
  gemini: {
    name: "Gemini",
    models: [
      // Gemini 3 series (latest - preview)
      "gemini-3.0-pro",
      "gemini-3.0-flash",
      // Gemini 2.5 series (GA)
      "gemini-2.5-pro",
      "gemini-2.5-flash",
      "gemini-2.5-flash-lite",
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
}

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

  // Form state
  const [agentName, setAgentName] = useState("")
  const [systemPrompt, setSystemPrompt] = useState("")
  const [greetingMessage, setGreetingMessage] = useState("")
  const [language, setLanguage] = useState("")
  const [llmProvider, setLlmProvider] = useState("")
  const [llmModel, setLlmModel] = useState("")
  const [sttProvider, setSttProvider] = useState("")
  const [sttModel, setSttModel] = useState("")
  const [ttsProvider, setTtsProvider] = useState("")
  const [ttsModel, setTtsModel] = useState("")
  const [ttsVoice, setTtsVoice] = useState("")
  const [ttsDescription, setTtsDescription] = useState("")
  const [speed, setSpeed] = useState(1.0)

  // Collapsible states
  const [voiceSettingsOpen, setVoiceSettingsOpen] = useState(true)
  const [llmSettingsOpen, setLlmSettingsOpen] = useState(true)
  const [languageOpen, setLanguageOpen] = useState(false)

  // Track if we're in the initial data loading phase to prevent validation from clearing values
  const isInitialLoadRef = useRef(true)

  // Get all unique languages from both STT and TTS providers
  const allLanguages = useMemo(() => {
    const langSet = new Set<string>()
    Object.values(sttData.stt.providers).forEach((provider: any) => {
      Object.values(provider.models).forEach((model: any) => {
        if (Array.isArray(model.languages)) {
          model.languages.forEach((lang: string) => langSet.add(lang))
        }
      })
    })
    Object.values(ttsData.tts.providers).forEach((provider: any) => {
      Object.values(provider.models).forEach((model: any) => {
        if (model.languages) {
          Object.keys(model.languages).forEach((lang: string) => langSet.add(lang))
        }
      })
    })
    return Array.from(langSet).sort()
  }, [])

  // Derive all STT providers from JSON
  const allSTTProviders = useMemo(() => {
    return Object.entries(sttData.stt.providers).map(([id, provider]: [string, any]) => ({
      id,
      name: provider.display_name || getProviderDisplayName(id),
    }))
  }, [])

  // Derive all TTS providers from JSON
  const allTTSProviders = useMemo(() => {
    return Object.entries(ttsData.tts.providers).map(([id, provider]: [string, any]) => ({
      id,
      name: provider.display_name || getProviderDisplayName(id),
    }))
  }, [])

  // Get supported STT providers for selected language
  const supportedSTTProviders = useMemo(() => {
    if (!language) return new Set<string>()
    const supported = new Set<string>()
    Object.entries(sttData.stt.providers).forEach(([providerId, provider]: [string, any]) => {
      const hasLanguage = Object.values(provider.models).some((model: any) =>
        Array.isArray(model.languages) && model.languages.includes(language)
      )
      if (hasLanguage) supported.add(providerId)
    })
    return supported
  }, [language])

  // Get supported STT models for selected provider + language
  const supportedSTTModels = useMemo(() => {
    if (!language || !sttProvider) return new Set<string>()
    const provider = (sttData.stt.providers as any)[sttProvider]
    if (!provider) return new Set<string>()
    const models = new Set<string>()
    Object.entries(provider.models).forEach(([modelId, model]: [string, any]) => {
      if (Array.isArray(model.languages) && model.languages.includes(language)) {
        models.add(modelId)
      }
    })
    return models
  }, [language, sttProvider])

  // Get supported TTS providers for selected language
  const supportedTTSProviders = useMemo(() => {
    if (!language) return new Set<string>()
    const supported = new Set<string>()
    Object.entries(ttsData.tts.providers).forEach(([providerId, provider]: [string, any]) => {
      const hasLanguage = Object.values(provider.models).some((model: any) =>
        model.languages && language in model.languages
      )
      if (hasLanguage) supported.add(providerId)
    })
    return supported
  }, [language])

  // Get supported TTS models for selected provider + language
  const supportedTTSModels = useMemo(() => {
    if (!language || !ttsProvider) return new Set<string>()
    const provider = (ttsData.tts.providers as any)[ttsProvider]
    if (!provider) return new Set<string>()
    const models = new Set<string>()
    Object.entries(provider.models).forEach(([modelId, model]: [string, any]) => {
      if (model.languages && language in model.languages) {
        models.add(modelId)
      }
    })
    return models
  }, [language, ttsProvider])

  // Get available TTS voices for selected provider/model/language
  const availableTTSVoices = useMemo(() => {
    if (!language || !ttsProvider || !ttsModel) return []
    const provider = (ttsData.tts.providers as any)[ttsProvider]
    if (!provider) return []
    const model = provider.models[ttsModel]
    if (!model || !model.languages) return []
    const langData = model.languages[language]
    if (!langData || !Array.isArray(langData.voices)) return []
    return langData.voices
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

  // Load agent data
  useEffect(() => {
    // Reset all state when agentId changes
    isInitialLoadRef.current = true
    setIsLoading(true)
    setAgent(null)
    setAgentName("")
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

        if (userData.org_id) {
          const agentData = await getAgent(agentId, userData.org_id)
          console.log("Full agent data received:", JSON.stringify(agentData, null, 2))
          setAgent(agentData)

          // Extract agent name from agentId first (most reliable since it's in the URL)
          // The agentId format is: "org_id-agent_type-timestamp" where agent_type may contain spaces/hyphens
          let name = ""
          if (agentId) {
            // Decode the agentId in case it's URL encoded
            const decodedId = decodeURIComponent(agentId)
            const parts = decodedId.split('-')
            // The last part is always the timestamp (numeric), everything between first and last is agent_type
            if (parts.length >= 3) {
              // Join all parts except the first (org_id) and last (timestamp) to get the agent_type
              name = parts.slice(1, -1).join('-')
            } else if (parts.length === 2) {
              // Fallback: if only 2 parts, the second might be the agent_type
              name = parts[1]
            }
          }

          // Fallback to agent_type from backend if extraction from agentId failed
          if (!name || name === "voicera_telephony") {
            name = agentData.agent_type || ""
          }

          // Capitalize first letter for display
          const finalName = name
            ? name.charAt(0).toUpperCase() + name.slice(1)
            : "Unnamed Agent"
          setAgentName(finalName)
          setSystemPrompt(agentData.agent_config?.system_prompt || "")
          setGreetingMessage(agentData.agent_config?.greeting_message || "")

          // Load LLM settings - convert official name to internal ID
          const llmProviderName = agentData.agent_config?.llm_model?.name || ""
          setLlmProvider(getProviderIdFromName(llmProviderName))
          setLlmModel(agentData.agent_config?.llm_model?.model || "")

          // Load language - use language name directly (no conversion needed)
          // Priority: agent_config.language > stt_model.language > tts_model.language
          const configLangName = (agentData.agent_config as any)?.language || ""
          const sttLangName = agentData.agent_config?.stt_model?.language || ""
          const ttsLangName = agentData.agent_config?.tts_model?.language || ""

          // Check if language exists in JSON (more reliable than checking allLanguages array)
          const languageExistsInJSON = (langName: string) => {
            if (!langName) return false
            // Check STT providers for this language
            const inSTT = Object.values(sttData.stt.providers).some((provider: any) =>
              Object.values(provider.models).some((model: any) =>
                Array.isArray(model.languages) && model.languages.includes(langName)
              )
            )
            if (inSTT) return true
            // Check TTS providers for this language
            const inTTS = Object.values(ttsData.tts.providers).some((provider: any) =>
              Object.values(provider.models).some((model: any) =>
                model.languages && langName in model.languages
              )
            )
            return inTTS
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
          // For Cartesia and Google, load from args; for others, load from top level
          const ttsModelConfig = agentData.agent_config?.tts_model as any
          const ttsArgs = ttsModelConfig?.args || {}
          const modelValue = (ttsProviderId === "cartesia" || ttsProviderId === "gcp")
            ? (ttsArgs.model || ttsModelConfig?.model || "")
            : (ttsModelConfig?.model || "")
          setTtsModel(modelValue)
          // For Cartesia and Google, load voice_id from args; for others, load from speaker
          const voiceValue = (ttsProviderId === "cartesia" || ttsProviderId === "gcp")
            ? (ttsArgs.voice_id || ttsModelConfig?.voice_id || "")
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

    // Clear TTS voice if it's not available for current provider
    if (ttsProvider && ttsVoice && availableTTSVoices.length > 0) {
      if (!availableTTSVoices.includes(ttsVoice)) {
        setTtsVoice("")
      }
    }
  }, [language, sttProvider, ttsProvider, supportedSTTModels, supportedTTSModels, availableTTSVoices, isLoading])

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
        ...(ttsProvider === "cartesia" && ttsVoice && { voice_id: ttsVoice }),
        speaker: ttsProvider === "cartesia" ? "" : (ttsVoice || ""),
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
  }, [systemPrompt, greetingMessage, language, llmProvider, llmModel, sttProvider, sttModel, ttsProvider, ttsModel, ttsVoice, speed, originalConfig, agent])

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
            ...((ttsProvider === "cartesia" || ttsProvider === "gcp") && {
              args: {
                ...(ttsModel && { model: ttsModel }),
                ...(ttsVoice && { voice_id: ttsVoice }),
              },
            }),
            ...(ttsProvider !== "cartesia" && ttsProvider !== "gcp" && ttsModel && { model: ttsModel }),
            speaker: (ttsProvider === "cartesia" || ttsProvider === "gcp") ? "" : (ttsVoice || ""),
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

  const formatDate = (dateString: string) => {
    const utcDate = dateString.endsWith("Z")
      ? dateString
      : `${dateString}Z`

    const date = new Date(utcDate)

    return date.toLocaleString("en-IN", {
      day: "2-digit",
      month: "short",
      year: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      hour12: true,
      timeZone: "Asia/Kolkata",
    })
  }


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
      {/* Header */}
      <header className="flex h-14 items-center gap-4 border-b border-slate-200 bg-white px-5 lg:px-8 sticky top-0 z-10">
        <nav className="flex items-center gap-1.5 text-sm">
          <Link href="/assistants" className="text-slate-500 hover:text-slate-900">
            Dashboard
          </Link>
          <ChevronRight className="h-4 w-4 text-slate-400" />
          <Link href="/assistants" className="text-slate-500 hover:text-slate-900">
            Assistants
          </Link>
          <ChevronRight className="h-4 w-4 text-slate-400" />
          <span className="text-slate-900 font-medium">{agentName || agentId.slice(0, 8)}</span>
        </nav>

      </header>

      {/* Main Content */}
      <main className="flex-1 overflow-auto p-6 lg:p-8">
        {/* Agent Header */}
        <div className="mb-6">
          <div className="flex items-start justify-between mb-2">
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-semibold text-slate-900">
                {agentName ? agentName : (agentId ? agentId.slice(0, 8) : "Unnamed Agent")}
              </h1>
            </div>
            {hasChanges && (
              <Button
                onClick={handleSaveClick}
                disabled={isSaving}
                className="h-11 px-6 bg-slate-900 hover:bg-slate-800 text-white"
              >
                {isSaving ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Saving...
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4 mr-2" />
                    Save Changes
                  </>
                )}
              </Button>
            )}
          </div>
          <div className="flex justify-between mr-3">
            <p className="text-sm text-slate-500">
              Last updated at {agent?.updated_at ? formatDate(agent.updated_at) : "N/A"}
            </p>
          </div>
        </div>




        {/* Two Column Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">


          {/* Left Column - Settings */}
          <div className="space-y-4">
            {/* LLM Settings */}
            <div className="bg-white rounded-xl border border-slate-200 overflow-hidden">
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
                      }}
                    >
                      <SelectTrigger className="border-slate-200 h-11 shadow-sm rounded-md focus:ring-slate-300 transition focus:border-slate-500 bg-white">
                        <SelectValue placeholder="Select provider" />
                      </SelectTrigger>
                      <SelectContent className="z-[100] rounded-md shadow-lg">
                        {Object.entries(llmProviders).map(([id, provider]) => {
                          // OpenAI and Kenpath are always available (built-in)
                          const isBuiltIn = id === "openai" || id === "kenpath"
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
                </div>
              )}
            </div>

            {/* Voice Settings */}
            <div className="bg-white rounded-2xl border border-slate-200 overflow-hidden shadow-sm transition-all duration-200">
              <button
                onClick={() => setVoiceSettingsOpen(!voiceSettingsOpen)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-slate-50 transition-colors group outline-none focus-visible:ring focus-visible:ring-slate-300"
                aria-expanded={voiceSettingsOpen}
                aria-controls="voiceSettings-content"
              >
                <div className="flex items-center gap-4">
                  <span className="bg-slate-100 rounded-lg flex items-center justify-center h-9 w-9">
                    <Volume2 className="h-5 w-5 text-slate-700" />
                  </span>
                  <span className="font-semibold text-slate-900 text-lg tracking-tight">Voice Settings</span>
                </div>
                <span
                  className={`transition-transform duration-200 ${voiceSettingsOpen ? "rotate-180" : ""
                    }`}
                >
                  <ChevronDown className="h-5 w-5 text-slate-400 group-hover:text-slate-600" />
                </span>
              </button>
              {voiceSettingsOpen && (
                <div
                  id="voiceSettings-content"
                  className="px-8 py-6 space-y-7 border-t border-slate-100 grid gap-8"
                >
                  {/* Language */}
                  <div className="grid gap-2">
                    <label className="text-base font-semibold text-slate-800 mb-2 flex items-center gap-2">
                      <Globe className="h-4 w-4 text-slate-400 mb-0.5" />
                      Language<span className="text-red-500 text-base ml-1">*</span>
                    </label>
                    {allLanguages.length > 0 ? (
                      <Popover open={languageOpen} onOpenChange={setLanguageOpen}>
                        <PopoverTrigger asChild>
                          <Button
                            variant="outline"
                            role="combobox"
                            aria-expanded={languageOpen}
                            className="w-full max-w-md min-h-[48px] py-3 px-4 justify-between rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 text-left [&>div]:whitespace-normal"
                          >
                            <div className="flex items-center gap-2 flex-1 min-w-0">
                              <Languages className="h-4 w-4 text-blue-500 flex-shrink-0" />
                              <span className="truncate">{language || "Select language..."}</span>
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
                                    key={lang}
                                    value={lang}
                                    onSelect={() => {
                                      setLanguage(lang);

                                      // Auto-select ai4bharat for languages other than English (United States) and English (India)
                                      if (lang && lang !== "English (United States)" && lang !== "English (India)") {
                                        // Check if ai4bharat is available for STT in the new provider > model > language structure
                                        const sttAi4bharat = (sttData.stt.providers as any)["ai4bharat"]
                                        if (sttAi4bharat) {
                                          // Find first model that supports this language
                                          const sttModelEntry = Object.entries(sttAi4bharat.models).find(([, model]: [string, any]) =>
                                            Array.isArray(model.languages) && model.languages.includes(lang)
                                          )
                                          if (sttModelEntry) {
                                            setSttProvider("ai4bharat")
                                            setSttModel(sttModelEntry[0])
                                          } else {
                                            setSttProvider("")
                                            setSttModel("")
                                          }
                                        } else {
                                          setSttProvider("")
                                          setSttModel("")
                                        }

                                        // Check if ai4bharat is available for TTS in the new provider > model > language structure
                                        const ttsAi4bharat = (ttsData.tts.providers as any)["ai4bharat"]
                                        if (ttsAi4bharat) {
                                          // Find first model that supports this language
                                          const ttsModelEntry = Object.entries(ttsAi4bharat.models).find(([, model]: [string, any]) =>
                                            model.languages && lang in model.languages
                                          )
                                          if (ttsModelEntry) {
                                            const [modelId, modelData] = ttsModelEntry as [string, any]
                                            setTtsProvider("ai4bharat")
                                            setTtsModel(modelId)
                                            // Set first voice if available
                                            const langVoices = modelData.languages[lang]?.voices
                                            if (langVoices && Array.isArray(langVoices) && langVoices.length > 0) {
                                              setTtsVoice(langVoices[0])
                                              // Set first description from descriptionsData if available
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
                                          setTtsProvider("")
                                          setTtsModel("")
                                          setTtsVoice("")
                                          setTtsDescription("")
                                        }
                                      } else {
                                        // Reset providers for English languages
                                        setSttProvider("")
                                        setSttModel("")
                                        setTtsProvider("")
                                        setTtsModel("")
                                        setTtsVoice("")
                                        setTtsDescription("")
                                      }

                                      setLanguageOpen(false);
                                    }}
                                    className="py-2.5"
                                  >
                                    <span className="font-medium">{lang}</span>
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

                  {/* STT and TTS stacked (one below other) */}
                  {language && (
                    <div className="flex flex-col gap-8">
                      {/* STT Section */}
                      <section>
                        <label className="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
                          <Mic className="h-4 w-4 text-slate-400 mb-0.5" />
                          Speech-to-Text (STT)
                        </label>
                        <div className="space-y-5 mt-3">
                          {/* STT Provider */}
                          <div>
                            <label className="text-xs font-semibold text-slate-500 mb-2 block">Provider</label>
                            <Select
                              value={sttProvider}
                              onValueChange={(v) => {
                                setSttProvider(v);
                                setSttModel("");
                              }}
                            >
                              <SelectTrigger className="border-slate-200 focus:ring-2 focus:ring-slate-200 rounded-md h-11 bg-slate-50 hover:bg-slate-100 transition-colors">
                                <SelectValue placeholder="Select provider" />
                              </SelectTrigger>
                              <SelectContent>
                                {allSTTProviders
                                  .filter((p) => supportedSTTProviders.has(p.id))
                                  .map((provider) => {
                                    // AI4Bharat is on-prem, always available (no API key needed)
                                    const isOnPrem = provider.id === "ai4bharat"
                                    // Check if provider has integration (API key configured)
                                    const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                                    
                                    return (
                                      <SelectItem
                                        key={provider.id}
                                        value={provider.id}
                                        className="text-base px-3 py-2 rounded-md data-[state=checked]:bg-slate-100 data-[highlighted]:bg-slate-50"
                                        disabled={!isIntegrated}
                                      >
                                        <div className="flex items-center gap-2">
                                          <span>{provider.name}</span>
                                          {!isIntegrated && (
                                            <span className="text-xs text-slate-400">(not integrated)</span>
                                          )}
                                        </div>
                                      </SelectItem>
                                    )
                                  })}
                              </SelectContent>
                            </Select>
                          </div>
                          {/* STT Model */}
                          {sttProvider && (
                            <div>
                              <label className="text-xs font-semibold text-slate-500 mb-2 block">Model</label>
                              <Select
                                value={sttModel}
                                onValueChange={setSttModel}
                                disabled={supportedSTTModels.size === 0}
                              >
                                <SelectTrigger className="border-slate-200 rounded-md h-11 bg-slate-50 hover:bg-slate-100 transition-colors">
                                  <SelectValue placeholder="Select model" />
                                </SelectTrigger>
                                <SelectContent>
                                  {Array.from(supportedSTTModels).map((model) => (
                                    <SelectItem
                                      key={model}
                                      value={model}
                                      className="font-mono text-base px-3 py-2 rounded-md data-[state=checked]:bg-slate-100 data-[highlighted]:bg-slate-50"
                                    >
                                      {model}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                        </div>
                      </section>
                      {/* TTS Section */}
                      <section>
                        <label className="text-base font-semibold text-slate-800 mb-4 flex items-center gap-2">
                          <Volume2 className="h-4 w-4 text-slate-400 mb-0.5" />
                          Text-to-Speech (TTS)
                        </label>
                        <div className="space-y-5 mt-3">
                          {/* TTS Provider */}
                          <div>
                            <label className="text-xs font-semibold text-slate-500 mb-2 block">Provider</label>
                            <Select
                              value={ttsProvider}
                              onValueChange={(v) => {
                                setTtsProvider(v);
                                setTtsModel("");
                                setTtsVoice("");
                                setTtsDescription("");
                              }}
                            >
                              <SelectTrigger className="border-slate-200 focus:ring-2 focus:ring-slate-200 rounded-md h-11 bg-slate-50 hover:bg-slate-100 transition-colors">
                                <SelectValue placeholder="Select provider" />
                              </SelectTrigger>
                              <SelectContent>
                                {allTTSProviders
                                  .filter((p) => supportedTTSProviders.has(p.id))
                                  .map((provider) => {
                                    // AI4Bharat is on-prem, always available (no API key needed)
                                    const isOnPrem = provider.id === "ai4bharat"
                                    // Check if provider has integration (API key configured)
                                    const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                                    
                                    return (
                                      <SelectItem
                                        key={provider.id}
                                        value={provider.id}
                                        className="text-base px-3 py-2 rounded-md data-[state=checked]:bg-slate-100 data-[highlighted]:bg-slate-50"
                                        disabled={!isIntegrated}
                                      >
                                        <div className="flex items-center gap-2">
                                          <span>{provider.name}</span>
                                          {!isIntegrated && (
                                            <span className="text-xs text-slate-400">(not integrated)</span>
                                          )}
                                        </div>
                                      </SelectItem>
                                    )
                                  })}
                              </SelectContent>
                            </Select>
                          </div>
                          {/* TTS Model */}
                          {ttsProvider && (
                            <div>
                              <label className="text-xs font-semibold text-slate-500 mb-2 block">Model</label>
                              <Select
                                value={ttsModel}
                                onValueChange={(v) => {
                                  setTtsModel(v);
                                  // Reset voice when model changes (different models may have different voice lists)
                                  setTtsVoice("");
                                }}
                                disabled={supportedTTSModels.size === 0}
                              >
                                <SelectTrigger className="border-slate-200 rounded-md h-11 bg-slate-50 hover:bg-slate-100 transition-colors">
                                  <SelectValue placeholder="Select model" />
                                </SelectTrigger>
                                <SelectContent>
                                  {Array.from(supportedTTSModels).map((model) => (
                                    <SelectItem
                                      key={model}
                                      value={model}
                                      className="font-mono text-base px-3 py-2 rounded-md data-[state=checked]:bg-slate-100 data-[highlighted]:bg-slate-50"
                                    >
                                      {model}
                                    </SelectItem>
                                  ))}
                                </SelectContent>
                              </Select>
                            </div>
                          )}
                          {/* TTS Voice & Speed */}
                          {ttsModel && ttsProvider && (
                            <>
                              {/* Voice or Voice ID */}
                              {(ttsProvider === "gcp" || ttsProvider === "cartesia") ? (
                                <div>
                                  <label className="text-xs font-semibold text-slate-500 mb-2 block">
                                    Voice ID
                                    <span className="ml-2 text-slate-400 text-xs tracking-normal">
                                      (copy from provider's dashboard)
                                    </span>
                                  </label>
                                  <Input
                                    value={ttsVoice}
                                    onChange={(e) => setTtsVoice(e.target.value)}
                                    placeholder="Enter voice ID"
                                    className="h-11 border-slate-200 rounded-md bg-slate-50 focus:border-slate-400"
                                  />
                                </div>
                              ) : (
                                <div>
                                  <label className="text-xs font-semibold text-slate-500 mb-2 block">Voice</label>
                                  <Select
                                    value={ttsVoice}
                                    onValueChange={setTtsVoice}
                                    disabled={availableTTSVoices.length === 0}
                                  >
                                    <SelectTrigger className="border-slate-200 rounded-md h-11 bg-slate-50 hover:bg-slate-100 transition-colors">
                                      <SelectValue placeholder="Select voice" />
                                    </SelectTrigger>
                                    <SelectContent>
                                      {availableTTSVoices.map((voice) => (
                                        <SelectItem
                                          key={voice}
                                          value={voice}
                                          className="text-base px-3 py-2 rounded-md data-[state=checked]:bg-slate-100 data-[highlighted]:bg-slate-50"
                                        >
                                          {voice}
                                        </SelectItem>
                                      ))}
                                    </SelectContent>
                                  </Select>

                                  {/* TTS Description for AI4Bharat and Bhashini */}
                                  {(ttsProvider === "ai4bharat" || ttsProvider === "bhashini") && (
                                    <div className="mt-3">
                                      <label className="text-xs font-semibold text-slate-500 mb-2 block">
                                        Voice Description
                                      </label>

                                      <Select
                                        value={ttsDescription}
                                        onValueChange={setTtsDescription}
                                        disabled={availableTTSDescriptions.length === 0}
                                      >
                                        <SelectTrigger className="min-h-[64px] w-full py-3 px-4 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all text-left">
                                          <SelectValue>
                                            {ttsDescription
                                              ? (ttsDescription.length > 25
                                                ? `${ttsDescription.slice(0, 25)}...`
                                                : ttsDescription)
                                              : "Select a voice description to customize voice characteristics"}
                                          </SelectValue>
                                        </SelectTrigger>

                                        <SelectContent className="rounded-lg max-h-[300px] w-[600px]">
                                          {availableTTSDescriptions.map((description) => (
                                            <SelectItem
                                              key={description}
                                              value={description}
                                              className="py-3 px-3"
                                            >
                                              <span className="text-sm leading-relaxed block whitespace-normal">
                                                {description}
                                              </span>
                                            </SelectItem>
                                          ))}
                                        </SelectContent>
                                      </Select>
                                    </div>
                                  )}



                                </div>
                              )}

                              {/* Speed */}
                              <div className="mt-3">
                                <div className="flex items-center justify-between mb-2">
                                  <label className="text-xs font-semibold text-slate-500">Speed</label>
                                  <span className="text-xs font-mono text-slate-700 bg-white px-2.5 py-0.5 rounded border border-slate-200">{speed.toFixed(1)}x</span>
                                </div>
                                <div className="flex items-center gap-3">
                                  <span className="text-xs text-slate-500 min-w-[2.5rem]">0.5x</span>
                                  <Slider
                                    value={[speed]}
                                    onValueChange={([value]) => setSpeed(value)}
                                    min={0.5}
                                    max={2.0}
                                    step={0.1}
                                    className="flex-1"
                                  />
                                  <span className="text-xs text-slate-500 min-w-[2.5rem] text-right">2.0x</span>
                                </div>
                                <p className="text-xs text-slate-500 mt-1">Speaking pace (1.0 = normal)</p>
                              </div>
                            </>
                          )}
                        </div>
                      </section>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>

          {/* Right Column - Agent configuration */}
          <div className="lg:col-span-2 space-y-6">
            <div className="bg-white rounded-xl border border-slate-200 p-6">
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
            <div className="bg-white rounded-xl border border-slate-200 p-6">
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
        </div>
      </main>

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

