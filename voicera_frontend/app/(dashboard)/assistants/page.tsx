"use client"

import { useState, useMemo, useEffect } from "react"
import { useRouter } from "next/navigation"
import { getCurrentUser, getAgents, createAgent, createVobizApplication, deleteVobizApplication, deleteAgent, unlinkVobizNumber, fetchApiRoute, getIntegrations, type User, type Agent, type CreateAgentRequest, type Integration } from "@/lib/api"
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
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { TestCallSheet } from "@/components/assistants/test-call-sheet"
import { AgentCard } from "@/components/assistants/agent-card"
import { CreateNewAgentCard } from "@/components/assistants/create-new-agent-card"
import {
  ChevronRight,
  ChevronLeft,
  Plus,
  Search,
  Phone,
  BarChart3,
  FileText,
  Settings,
  Volume2,
  CheckCircle2,
  Languages,
  Mic,
  Loader2,
  X,
} from "lucide-react"

// Import JSON data
import sttData from "@/stt.json"
import ttsData from "@/tts.json"
import descriptionsData from "@/descriptions.json"

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
  const prompt = agent.agent_config?.system_prompt || ""
  if (prompt.length > 0) {
    return prompt.slice(0, 50) + (prompt.length > 50 ? "..." : "")
  }
  return "Voice Assistant"
}

// Types
interface AgentConfig {
  id: string
  name?: string
  greetingMessage?: string
  systemPrompt: string
  llmProvider: string
  llmModel: string
  temperature: number
  maxTokens: number
  language: string
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

const defaultConfig: AgentConfig = {
  id: "",
  name: "",
  greetingMessage: "",
  systemPrompt: "You are a helpful agent. You will help the customer with their queries and doubts. You will never speak more than 2 sentences. Keep your responses concise",
  llmProvider: "openai",
  llmModel: "gpt-4o",
  temperature: 0.2,
  maxTokens: 450,
  language: "Hindi",
  sttProvider: "ai4bharat",
  sttModel: "indic-conformer-stt",
  keywords: "",
  ttsProvider: "ai4bharat",
  ttsModel: "indic-parler-tts",
  ttsVoice: "Rohit",
  ttsDescription: "Speaks at a fast pace with a slightly low-pitched voice, captured clearly in a close-sounding environment with excellent recording quality.",
  bufferSize: 50,
  speedRate: 1,
  similarityBoost: 75,
  stability: 50,
  telephonyProvider: "Vobiz",
}

// Wizard steps configuration
const wizardSteps = [
  { id: 1, title: "Agent", subtitle: "Name & Prompt", icon: FileText },
  { id: 2, title: "LLM", subtitle: "Model Config", icon: Settings },
  { id: 3, title: "Audio", subtitle: "STT & TTS", icon: Volume2 },
  { id: 4, title: "Telephony", subtitle: "Select Provider", icon: Phone },
  { id: 5, title: "Review", subtitle: "Confirm", icon: CheckCircle2 },
]

export default function AssistantsPage() {
  const router = useRouter()
  const [agents, setAgents] = useState<Agent[]>([])
  const [searchQuery, setSearchQuery] = useState("")
  const [config, setConfig] = useState<AgentConfig>(defaultConfig)
  const [view, setView] = useState<"list" | "create">("list")
  const [createStep, setCreateStep] = useState(1)
  const [user, setUser] = useState<User | null>(null)
  const [isLoadingAgents, setIsLoadingAgents] = useState(true)
  const [isCreatingAgent, setIsCreatingAgent] = useState(false)
  const [isTestCallSheetOpen, setIsTestCallSheetOpen] = useState(false)
  const [selectedAgentForTest, setSelectedAgentForTest] = useState<Agent | null>(null)
  const [showDeleteSuccessToast, setShowDeleteSuccessToast] = useState(false)
  const [integratedProviders, setIntegratedProviders] = useState<Set<string>>(new Set())

  // Fetch user data, agents, and integrations on mount
  useEffect(() => {
    async function fetchData() {
      try {
        const userData = await getCurrentUser()
        setUser(userData)
        
        // Fetch agents for this org
        if (userData.org_id) {
          const agentsData = await getAgents(userData.org_id)
          setAgents(agentsData)
        }
        
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
      } catch (error) {
        console.error("Failed to fetch data:", error)
        router.push("/")
      } finally {
        setIsLoadingAgents(false)
      }
    }
    fetchData()
  }, [router])

  // Refresh agents when window regains focus (user navigates back from detail page)
  useEffect(() => {
    const handleFocus = async () => {
      if (view === "list") {
        try {
          // Re-fetch current user to ensure we have the correct org_id matching the token
          const currentUser = await getCurrentUser()
          if (currentUser?.org_id) {
            setUser(currentUser)
            const agentsData = await getAgents(currentUser.org_id)
            setAgents(agentsData)
          }
        } catch (error) {
          console.error("Failed to refresh agents:", error)
        }
      }
    }

    window.addEventListener('focus', handleFocus)
    return () => window.removeEventListener('focus', handleFocus)
  }, [view])

  // Filter agents based on search
  const filteredAgents = agents.filter(
    (agent) => {
      const name = getAgentDisplayName(agent).toLowerCase()
      const description = getAgentDescription(agent).toLowerCase()
      const query = searchQuery.toLowerCase()
      return name.includes(query) || description.includes(query)
    }
  )

  const viewConfig = (agent: Agent) => {
    // Use agent.id or agent._id (MongoDB) if available, otherwise construct a unique identifier
    let agentId = agent.id || agent._id
    if (!agentId) {
      // Construct ID only if we have the necessary fields
      if (agent.org_id && agent.agent_type) {
        const timestamp = agent.created_at || Date.now().toString()
        agentId = `${agent.org_id}-${agent.agent_type}-${timestamp}`
      }
    }
    if (!agentId || agentId === 'undefined' || agentId.includes('undefined')) {
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
    return Array.from(merged).sort().map(name => ({ code: name, name }))
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

  // Get supported STT providers for selected language
  const supportedSTTProviders = useMemo(() => {
    if (!config.language) return new Set<string>()
    const langData =
      sttData.stt.languages[config.language as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()

    return new Set(
      Object.entries(langData.models)
        .filter(([, models]) => Array.isArray(models) && models.length > 0)
        .map(([provider]) => provider)
    )
  }, [config.language])

  // Get supported STT models for selected provider
  const supportedSTTModels = useMemo(() => {
    if (!config.language || !config.sttProvider) return new Set<string>()
    const langData =
      sttData.stt.languages[config.language as keyof typeof sttData.stt.languages]
    if (!langData) return new Set<string>()

    const models = langData.models[config.sttProvider as keyof typeof langData.models]
    return new Set(Array.isArray(models) ? models : [])
  }, [config.language, config.sttProvider])

  // Get supported TTS providers for selected language
  const supportedTTSProviders = useMemo(() => {
    if (!config.language) return new Set<string>()
    const langData =
      ttsData.tts.languages[config.language as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()

    return new Set(
      Object.entries(langData.models)
        .filter(([, data]) => {
          const modelData = data as { available?: boolean }
          return modelData.available === true
        })
        .map(([provider]) => provider)
    )
  }, [config.language])

  // Get supported TTS models for selected provider
  const supportedTTSModels = useMemo(() => {
    if (!config.language || !config.ttsProvider) return new Set<string>()
    const langData =
      ttsData.tts.languages[config.language as keyof typeof ttsData.tts.languages]
    if (!langData) return new Set<string>()

    const providerData = langData.models[config.ttsProvider as keyof typeof langData.models] as {
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
  }, [config.language, config.ttsProvider])

  // Get available TTS voices for selected provider/model
  const availableTTSVoices = useMemo(() => {
    if (!config.language || !config.ttsProvider) return []
    const langData =
      ttsData.tts.languages[config.language as keyof typeof ttsData.tts.languages]
    if (!langData) return []

    const providerData = langData.models[config.ttsProvider as keyof typeof langData.models] as {
      voices?: string | string[]
    }
    if (!providerData) return []

    if (Array.isArray(providerData.voices)) {
      return providerData.voices
    }
    return []
  }, [config.language, config.ttsProvider])

  // Get LLM models for selected provider
  const availableLLMModels = useMemo(() => {
    if (!config.llmProvider) return []
    const provider = llmProviders[config.llmProvider as keyof typeof llmProviders]
    return provider?.models || []
  }, [config.llmProvider])

  // Handle create new agent
  const handleCreateNew = () => {
    setConfig({ ...defaultConfig, id: "new", telephonyProvider: "Vobiz" })
    setCreateStep(1)
    setView("create")
  }

  // Handle back to list
  const handleBackToList = () => {
    setView("list")
    setCreateStep(1)
    setConfig({ ...defaultConfig, telephonyProvider: "Vobiz" })
  }


  const handleTestCall = (agent: Agent) => {
    setSelectedAgentForTest(agent)
    setIsTestCallSheetOpen(true)
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

    try {
      // Step 1: Detach phone number if agent has one
      if (agent.phone_number) {
        try {
          // If provider is Vobiz, unlink from Vobiz application first
          if (agent.telephony_provider === "Vobiz") {
            await unlinkVobizNumber(agent.phone_number)
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
      if (agent.telephony_provider === "Vobiz" && agent.vobiz_app_id) {
        try {
          await deleteVobizApplication(agent.vobiz_app_id)
        } catch (error) {
          console.error("Failed to delete Vobiz application:", error)
          // Continue with agent deletion even if Vobiz deletion fails
        }
      }

      // Step 3: Delete the agent
      const agentId = agent.id || agent._id || agent.agent_type
      if (!agentId) {
        throw new Error("Agent ID is missing")
      }
      await deleteAgent(agentId)

      // Refresh the agents list after deletion
      const agentsData = await getAgents(user.org_id)
      setAgents(agentsData)
      
      // Show success toast
      setShowDeleteSuccessToast(true)
      setTimeout(() => {
        setShowDeleteSuccessToast(false)
      }, 3000)
    } catch (error) {
      console.error("Failed to delete agent:", error)
      alert(error instanceof Error ? error.message : "Failed to delete agent")
    }
  }

  // Update config helper
  const updateConfig = <K extends keyof AgentConfig>(
    key: K,
    value: AgentConfig[K]
  ) => {
    setConfig((prev) => {
      const updated = { ...prev, [key]: value }
      if (key === "language") {
        const newLanguage = value as string
        updated.sttProvider = ""
        updated.sttModel = ""
        updated.ttsProvider = ""
        updated.ttsModel = ""
        updated.ttsVoice = ""
        
        // Auto-select ai4bharat for languages other than English (United States) and English (India)
        if (newLanguage && newLanguage !== "English (United States)" && newLanguage !== "English (India)") {
          // Check if ai4bharat is available for STT
          const sttLangData = sttData.stt.languages[newLanguage as keyof typeof sttData.stt.languages]
          if (sttLangData?.models?.ai4bharat && Array.isArray(sttLangData.models.ai4bharat) && sttLangData.models.ai4bharat.length > 0) {
            updated.sttProvider = "ai4bharat"
            updated.sttModel = sttLangData.models.ai4bharat[0] // Use first available model
          }
          
          // Check if ai4bharat is available for TTS
          const ttsLangData = ttsData.tts.languages[newLanguage as keyof typeof ttsData.tts.languages]
          const ttsAi4bharatData = ttsLangData?.models?.ai4bharat as { available?: boolean; model?: string; voices?: string[] } | undefined
          if (ttsAi4bharatData?.available && ttsAi4bharatData.model) {
            updated.ttsProvider = "ai4bharat"
            updated.ttsModel = ttsAi4bharatData.model
            // Set first voice if available
            if (ttsAi4bharatData.voices && Array.isArray(ttsAi4bharatData.voices) && ttsAi4bharatData.voices.length > 0) {
              updated.ttsVoice = ttsAi4bharatData.voices[0]
            }
          }
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
      }
      return updated
    })
  }

  const nameSnakeCase = config.name.toLowerCase().replace(/ /g, "_")

  // Handle save agent
  const handleSaveAgent = async () => {
    if (!user?.org_id) {
      console.error("No org_id found")
      return
    }

    setIsCreatingAgent(true)

    try {
      // Generate agent_id from agent_type: replace spaces with underscores and convert to lowercase
      const agentId = config.name.replace(/\s+/g, '_').toLowerCase()

      const languageName = config.language // Already the name, no lookup needed

      // Build LLM model object with official provider name
      const llmModel: { name: string; model?: string } = {
        name: getProviderOfficialName(config.llmProvider),
      }
      if (config.llmProvider !== "kenpath") {
        llmModel.model = config.llmModel
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
      const ttsModel: any = {
        name: getProviderOfficialName(config.ttsProvider),
        ...((config.ttsProvider === "cartesia" || config.ttsProvider === "gcp") && {
          args: {
            ...(config.ttsModel && { model: config.ttsModel }),
            ...(config.ttsVoice && { voice_id: config.ttsVoice }),
          },
        }),
        ...(config.ttsProvider !== "cartesia" && config.ttsProvider !== "gcp" && config.ttsModel && { model: config.ttsModel }),
        speaker: (config.ttsProvider === "cartesia" || config.ttsProvider === "gcp") ? "" : (config.ttsVoice || ""),
      }
      if (config.ttsProvider === "ai4bharat" && config.ttsDescription) {
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
      
      if (config.telephonyProvider === "Vobiz") {
        vobizAnswerUrl = `${process.env.NEXT_PUBLIC_JOHNAIC_SERVER_URL}/answer?agent_id=${agentId}`
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

      const agentData: CreateAgentRequest = {
        org_id: user.org_id,
        agent_category: "voicera_telephony",
        agent_type: config.name,
        agent_id: agentId,
        agent_config: {
          system_prompt: config.systemPrompt,
          greeting_message: config.greetingMessage,
          session_timeout_minutes: 10,
          language: languageName,
          llm_model: llmModel,
          stt_model: sttModel,
          tts_model: ttsModel,
        },
        telephony_provider: config.telephonyProvider as any,
        ...(config.telephonyProvider === "Vobiz" && {
          vobiz_app_id: vobizAppId,
          vobiz_answer_url: vobizAnswerUrl,
        }),
      }

      // Create agent via API
      const newAgent = await createAgent(agentData)

      
      // Refresh agents list to get all agents with proper data
      if (user.org_id) {
        const agentsData = await getAgents(user.org_id)
        setAgents(agentsData)
      } else {
        // Fallback: add new agent to the list
        setAgents([...agents, newAgent])
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
    if (createStep < 5) {
      setCreateStep(createStep + 1)
    }
  }

  // Calculate progress percentage
  const progressPercent = (createStep / 5) * 100

  // Check if a specific step is completed
  const isStepCompleted = (stepId: number) => {
    switch (stepId) {
      case 1:
        return config.name.length > 0 && config.systemPrompt.length > 0 
      case 2:
        if (config.llmProvider === "kenpath") {
          return !!config.llmProvider
        }
        return config.llmProvider && config.llmModel
      case 3:
        return config.language && config.sttProvider && config.sttModel && config.ttsProvider && config.ttsModel && config.ttsVoice && config.ttsVoice.length > 0
      case 4:
        return !!config.telephonyProvider
      case 5:
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
    if (createStep === 5) return "Create Agent"
    const nextStep = wizardSteps.find(s => s.id === createStep + 1)
    return nextStep ? `Continue to ${nextStep.title}` : "Continue"
  }

  // Language combobox state
  const [languageOpen, setLanguageOpen] = useState(false)

  // Render list view
  if (view === "list") {
  return (
    <div className="flex flex-col h-screen bg-slate-50/50">
      {/* Header */}
      <header className="flex h-14 items-center gap-4 border-b border-slate-200 bg-white px-5 lg:px-8 sticky top-0 z-10">
          <nav className="flex items-center gap-1.5 text-sm">
            <span className="text-slate-500">Dashboard</span>
          <ChevronRight className="h-4 w-4 text-slate-400" />
            <span className="text-slate-900 font-medium">Agents</span>
        </nav>
      </header>

        {/* Main Content */}
        <main className="flex-1 overflow-auto p-6 lg:p-8">
          {/* Greeting Section */}
          <div className="flex items-start justify-between mb-8">
            <div>
              <h1 className="text-3xl font-semibold text-slate-900 mb-1">Hi {user?.name}</h1>
              <p className="text-slate-500">Let&apos;s get your agents inline.</p>
            </div>
            <div className="flex items-center gap-3">
              
              <div className="relative">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-400" />
            <Input
              type="text"
                  placeholder="Search Assistant"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
                  className="h-10 pl-9 pr-4 w-64 rounded-lg border-slate-200 bg-white focus:border-slate-400 focus:ring-1 focus:ring-slate-200"
            />
              </div>
            </div>
          </div>

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
            {!isLoadingAgents && filteredAgents.map((agent) => (
              <AgentCard
                key={String(agent.id || agent.org_id + agent.agent_type + agent.created_at)}
                agent={agent}
                getAgentDisplayName={getAgentDisplayName}
                getAgentDescription={getAgentDescription}
                onViewConfig={viewConfig}
                onTestCall={handleTestCall}
                onViewHistory={handleViewHistory}
                onDelete={handleDelete}
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

        {/* Delete Success Toast */}
        {showDeleteSuccessToast && (
          <div className="fixed top-20 right-6 z-50 animate-in slide-in-from-top-5 fade-in-0 bg-emerald-50 border border-emerald-200 text-emerald-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 min-w-[300px]">
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
          <h1 className="text-sm font-semibold text-slate-900">Create Telephony Agent</h1>
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

      {/* Main Content - Two Column Layout */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left Column - Progress Stepper */}
        <aside className="w-72 bg-white border-r border-slate-100 p-5">
          <h3 className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-5">Setup Progress</h3>
          <div className="space-y-1">
            {wizardSteps.map((step) => {
              const Icon = step.icon
              const isActive = createStep === step.id
              const isCompleted = isStepCompleted(step.id) && createStep > step.id
              const isAccessible = canAccessStep(step.id)

              return (
              <button
                  key={step.id}
                  onClick={() => isAccessible && setCreateStep(step.id)}
                  disabled={!isAccessible}
                  className={`w-full flex items-center gap-3 px-3 py-3 rounded-lg text-left transition-all duration-150 ${
                    isActive
                      ? "bg-slate-100"
                      : isAccessible
                      ? "hover:bg-slate-50 cursor-pointer"
                      : "opacity-50 cursor-not-allowed"
                }`}
              >
                <div
                    className={`h-9 w-9 rounded-lg flex items-center justify-center transition-all duration-150 ${
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
                      className={`text-sm font-medium truncate ${
                        isActive ? "text-slate-900" : isCompleted ? "text-slate-700" : "text-slate-500"
                      }`}
                    >
                      {step.title}
                    </p>
                    <p className="text-xs text-slate-400 truncate">{step.subtitle}</p>
                  </div>
                  {isCompleted && (
                    <CheckCircle2 className="h-4 w-4 text-emerald-500 flex-shrink-0" />
                  )}
              </button>
              )
            })}
          </div>
        </aside>

        {/* Right Column - Step Content */}
        <main className="flex-1 overflow-auto p-8">
          <div className="max-w-4xl">
            {/* Step 1: Agent Creation */}
            {createStep === 1 && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-8">
                  {/* Agent Name */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">Agent Name</label>
                    <Input
                      value={config.name}
                      onChange={(e) => updateConfig("name", e.target.value)}
                      placeholder="Enter agent name"
                      className="h-12 rounded-lg border-slate-200 bg-white text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                    />
                    <p className="text-sm text-slate-500">
                      Give your agent a unique name to identify it.
                    </p>
                  </div>

                  {/* Agent Welcome Message */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">Agent Welcome Message</label>
                    <Input
                      value={config.greetingMessage}
                      onChange={(e) => updateConfig("greetingMessage", e.target.value)}
                      placeholder="Hello from EkStep"
                      className="h-12 rounded-lg border-slate-200 bg-white text-base focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all"
                    />
                    <p className="text-sm text-slate-500">
                      This will be the initial message from the agent. You can use variables here using {"{variable_name}"}
                    </p>
                  </div>

                  {/* Agent Prompt */}
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

            {/* Step 2: LLM Settings */}
            {createStep === 2 && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-6">
                  <div className="space-y-4">
                    <label className="text-base font-bold text-slate-900">Choose LLM model</label>
                    <div className="grid grid-cols-2 gap-4 mt-4">
                      <Select value={config.llmProvider} onValueChange={(v) => updateConfig("llmProvider", v)}>
                        <SelectTrigger className="h-12 rounded-lg w-full border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                          <div className="flex items-center gap-2">
                            <SelectValue placeholder="Select provider" />
                          </div>
                        </SelectTrigger>
                        <SelectContent className="rounded-lg">
                          {Object.entries(llmProviders).map(([id, provider]) => {
                            // OpenAI and Kenpath are always available (built-in)
                            const isBuiltIn = id === "openai" || id === "kenpath"
                            // Check if provider has integration (API key configured)
                            const isIntegrated = integratedProviders.has(id) || integratedProviders.has(provider.name.toLowerCase())
                            const isAvailable = isBuiltIn || isIntegrated
                            
                            return (
                              <SelectItem key={id} value={id} className="py-3" disabled={!isAvailable}>
                                <div className="flex items-center gap-2.5">
                                  <span className="font-medium">{provider.name}</span>
                                  {!isAvailable && (
                                    <span className="ml-2 text-xs text-slate-400">(not integrated)</span>
                                  )}
                                </div>
                              </SelectItem>
                            )
                          })}
                        </SelectContent>
                      </Select>

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
                    </div>
                  </div>

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

            {/* Step 3: Audio Settings */}
            {createStep === 3 && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-8">
                  {/* Language Selection */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">Language: </label>
                    <Popover open={languageOpen} onOpenChange={setLanguageOpen}>
                      <PopoverTrigger asChild>
                        <Button
                          variant="outline"
                          role="combobox"
                          aria-expanded={languageOpen}
                          className="w-full max-w-md h-12 justify-between rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100"
                        >
                          <div className="flex items-center gap-2">
                            <Languages className="h-4 w-4 text-blue-500" />
                            {config.language || "Select language..."}
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
                                  value={lang.name}
                                  onSelect={() => {
                                    updateConfig("language", lang.name)
                                    setLanguageOpen(false)
                                  }}
                                  className="py-2.5"
                                >
                                  <span className="font-medium">{lang.name}</span>
                                </CommandItem>
                              ))}
                            </CommandGroup>
                          </CommandList>
                        </Command>
                      </PopoverContent>
                    </Popover>
                  </div>

                  {/* STT Settings */}
                  {config.language && (
                    <div className="space-y-4 pt-6 border-t border-slate-100">
                      <label className="text-base font-bold text-slate-900 italic">Select transcriber</label>
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
                              {[...allSTTProviders]
                                .sort((a, b) => {
                                  const aSupported = supportedSTTProviders.has(a.id)
                                  const bSupported = supportedSTTProviders.has(b.id)
                                  if (aSupported && !bSupported) return -1
                                  if (!aSupported && bSupported) return 1
                                  return 0
                                })
                                .map((provider) => {
                                  const isSupported = supportedSTTProviders.has(provider.id)
                                  // AI4Bharat is on-prem, always available (no API key needed)
                                  const isOnPrem = provider.id === "ai4bharat"
                                  // Check if provider has integration (API key configured)
                                  const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                                  // Determine availability status
                                  const isAvailable = isSupported && isIntegrated
                                  // Determine the reason for unavailability
                                  const unavailableReason = !isSupported ? "not supported" : !isIntegrated ? "not integrated" : ""
                                  
                                  return (
                                    <SelectItem key={provider.id} value={provider.id} disabled={!isAvailable} className="py-2.5">
                                      <span className="flex items-center gap-2">
                                        <span className={`font-medium ${!isAvailable ? "text-slate-400" : ""}`}>{provider.name}</span>
                                        {unavailableReason && (
                                          <span className="text-xs text-slate-400">({unavailableReason})</span>
                                        )}
                                      </span>
                                    </SelectItem>
                                  )
                                })}
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
                              {[...(getAllSTTModelsForProvider[config.sttProvider] || [])]
                                .sort((a, b) => {
                                  const aSupported = supportedSTTModels.has(a)
                                  const bSupported = supportedSTTModels.has(b)
                                  if (aSupported && !bSupported) return -1
                                  if (!aSupported && bSupported) return 1
                                  return 0
                                })
                                .map((model) => {
                                  const isSupported = supportedSTTModels.has(model)
                                  return (
                                    <SelectItem key={model} value={model} disabled={!isSupported} className="py-2.5 font-mono text-sm">
                                      <span className="flex items-center gap-2">
                                        <span className={!isSupported ? "text-slate-400" : ""}>{model}</span>
                                      </span>
                                    </SelectItem>
                                  )
                                })}
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                    </div>
                  )}

                  {/* TTS Settings */}
                  {config.language && (
                    <div className="space-y-4 pt-6 border-t border-slate-100">
                      <label className="text-base font-bold text-slate-900 italic">Select synthesizer</label>
                      <div className="grid grid-cols-3 gap-4">
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
                              {[...allTTSProviders]
                                .sort((a, b) => {
                                  const aSupported = supportedTTSProviders.has(a.id)
                                  const bSupported = supportedTTSProviders.has(b.id)
                                  if (aSupported && !bSupported) return -1
                                  if (!aSupported && bSupported) return 1
                                  return 0
                                })
                                .map((provider) => {
                                  const isSupported = supportedTTSProviders.has(provider.id)
                                  // AI4Bharat is on-prem, always available (no API key needed)
                                  const isOnPrem = provider.id === "ai4bharat"
                                  // Check if provider has integration (API key configured)
                                  const isIntegrated = isOnPrem || integratedProviders.has(provider.id) || integratedProviders.has(provider.name.toLowerCase())
                                  // Determine availability status
                                  const isAvailable = isSupported && isIntegrated
                                  // Determine the reason for unavailability
                                  const unavailableReason = !isSupported ? "not supported" : !isIntegrated ? "not integrated" : ""
                                  
                                  return (
                                    <SelectItem key={provider.id} value={provider.id} disabled={!isAvailable} className="py-2.5">
                                      <span className="flex items-center gap-2">
                                        <span className={`font-medium ${!isAvailable ? "text-slate-400" : ""}`}>{provider.name}</span>
                                        {unavailableReason && (
                                          <span className="text-xs text-slate-400">({unavailableReason})</span>
                                        )}
                                      </span>
                                    </SelectItem>
                                  )
                                })}
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
                              {[...(getAllTTSModelsForProvider[config.ttsProvider] || [])]
                                .sort((a, b) => {
                                  const aSupported = supportedTTSModels.has(a)
                                  const bSupported = supportedTTSModels.has(b)
                                  if (aSupported && !bSupported) return -1
                                  if (!aSupported && bSupported) return 1
                                  return 0
                                })
                                .map((model) => {
                                  const isSupported = supportedTTSModels.has(model)
                                  return (
                                    <SelectItem key={model} value={model} disabled={!isSupported} className="py-2.5 font-mono text-sm">
                                      <span className="flex items-center gap-2">
                                        <span className={!isSupported ? "text-slate-400" : ""}>{model}</span>
                                      </span>
                                    </SelectItem>
                                  )
                                })}
                            </SelectContent>
                          </Select>
                        </div>

                        <div className="space-y-2">
                          <label className="text-sm font-semibold text-slate-700">Voice</label>
                          <div className="flex items-center gap-2">
                            {(config.ttsProvider === "gcp" || config.ttsProvider === "cartesia") ? (
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

                      {/* TTS Description for AI4Bharat */}
                      {config.ttsProvider === "ai4bharat" && (
                        <div className="space-y-2 pt-3">
                          <label className="text-sm font-semibold text-slate-700">Voice Description</label>
                          <Select value={config.ttsDescription} onValueChange={(v) => updateConfig("ttsDescription", v)}>
                            <SelectTrigger className="min-h-[48px] py-3 px-4 rounded-lg border-slate-200 bg-white font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all text-left [&>span]:whitespace-normal [&>span]:line-clamp-2 [&>span]:text-left">
                              <SelectValue placeholder="Select a voice description to customize voice characteristics" className="whitespace-normal" />
                            </SelectTrigger>
                            <SelectContent className="rounded-lg max-h-[300px] w-[600px]">
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
                  {config.language && config.ttsProvider && (
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
                        <div className="grid grid-cols-3 gap-6 bg-slate-50 rounded-lg p-5 border border-slate-100">
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

                      {config.ttsProvider === "ai4bharat" && (
                        <div className="bg-slate-50 rounded-lg p-4 border border-slate-100">
                          <p className="text-sm text-slate-600">
                            <span className="font-medium">AI4Bharat</span> uses description-based voice control. Select a voice description above to customize pitch, pace, and expression characteristics.
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

            {/* Step 4: Telephony */}
            {createStep === 4 && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <div className="space-y-8">
                  {/* Telephony Provider Selection */}
                  <div className="space-y-3">
                    <label className="text-base font-bold text-slate-900">Select Telephone Provider</label>
                    <Select 
                      value={config.telephonyProvider} 
                      onValueChange={(v) => updateConfig("telephonyProvider", v)}
                    >
                      <SelectTrigger className="h-12 rounded-lg border-slate-200 bg-white text-base font-medium hover:bg-slate-50 focus:border-blue-400 focus:ring-2 focus:ring-blue-100 transition-all">
                        <div className="flex items-center gap-2">
                          <Phone className="h-4 w-4 text-slate-400" />
                          <SelectValue placeholder="Select telephone provider" />
                        </div>
                      </SelectTrigger>
                      <SelectContent className="rounded-lg">
                        <SelectItem value="Vobiz" className="py-3">
                          <span className="font-medium">Vobiz</span>
                        </SelectItem>
                        <SelectItem disabled value="Plivo" className="py-3">
                          <span className="font-medium">Plivo</span>
                        </SelectItem>
                      </SelectContent>
                    </Select>
                    <p className="text-sm text-slate-500">
                      Choose the telephone provider for your agent calls.
                    </p>
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

            {/* Step 5: Review */}
            {createStep === 5 && (
              <div className="bg-white rounded-xl border border-slate-200 p-8">
                <h2 className="text-xl font-bold text-slate-900 mb-1">Review Configuration</h2>
                <p className="text-slate-500 mb-8">Review your agent settings before creating.</p>

                <div className="space-y-0 bg-slate-50 rounded-lg border border-slate-100 overflow-hidden">
                  {/* Agent Details */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div className="flex-1 mr-4">
                      <p className="text-sm font-bold text-slate-900 mb-2">Agent Details</p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Name:</span> {config.name || "—"}
                      </p>
                      <p className="text-sm text-slate-600 mb-1">
                        <span className="font-semibold">Welcome:</span> {config.greetingMessage || "—"}
                      </p>
                      <p className="text-sm text-slate-600 line-clamp-2">
                        <span className="font-semibold">Prompt:</span> {config.systemPrompt || "—"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(1)} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* LLM Settings */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">LLM Model</p>
                      <p className="text-sm font-medium text-slate-700">
                        {getProviderOfficialName(config.llmProvider) || "—"} / {config.llmModel || "—"}
                      </p>
                      {config.llmProvider !== "kenpath" && (
                        <p className="text-sm text-slate-500 mt-1">
                          Tokens: {config.maxTokens} • Temperature: {config.temperature.toFixed(1)}
                        </p>
                      )}
                    </div>
                    <button onClick={() => setCreateStep(2)} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* Audio Settings */}
                  <div className="flex items-start justify-between p-4 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">Audio Configuration</p>
                      <p className="text-sm font-medium text-slate-700">
                        {config.language || "—"}
                      </p>
                      <p className="text-sm text-slate-500 mt-1">
                        <span className="font-medium">STT:</span> {getProviderOfficialName(config.sttProvider) || "—"} / {config.sttModel || "—"}
                      </p>
                      <p className="text-sm text-slate-500">
                        <span className="font-medium">TTS:</span> {getProviderOfficialName(config.ttsProvider) || "—"} / {config.ttsModel || "—"} / {config.ttsVoice || "—"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(3)} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>

                  {/* Telephony Settings */}
                  <div className="flex items-start justify-between p-4 border-b border-slate-200 hover:bg-slate-100/50 transition-colors">
                    <div>
                      <p className="text-sm font-bold text-slate-900 mb-2">Telephony Provider</p>
                      <p className="text-sm font-medium text-slate-700">
                        {config.telephonyProvider ? config.telephonyProvider.charAt(0).toUpperCase() + config.telephonyProvider.slice(1) : "—"}
                      </p>
                    </div>
                    <button onClick={() => setCreateStep(4)} className="text-sm font-semibold text-blue-600 hover:text-blue-700 transition-colors">
                      Edit
                    </button>
                  </div>
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
        <div className="fixed top-20 right-6 z-50 animate-in slide-in-from-top-5 fade-in-0 bg-red-50 border border-red-200 text-red-800 px-4 py-3 rounded-lg shadow-lg flex items-center gap-3 min-w-[300px]">
          <CheckCircle2 className="h-5 w-5 text-red-600 shrink-0" />
          <p className="font-medium">Agent deleted successfully</p>
        </div>
      )}
    </div>
  )
} 

