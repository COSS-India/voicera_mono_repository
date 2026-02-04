"use client"

import { useState, useMemo, useEffect } from "react"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Search,
  Eye,
  EyeOff,
  Save,
  Mic,
  Volume2,
  Brain,
  Check,
  Settings2,
  Plus,
  Trash2,
  Loader2,
} from "lucide-react"
import { getOrgId, getIntegrations, createIntegration, deleteIntegration, Integration } from "@/lib/api"

// Provider type definitions
type ProviderCapability = "stt" | "tts" | "llm"

interface Provider {
  id: string
  name: string
  capabilities: ProviderCapability[]
  description: string
}

// All providers with their capabilities
const providers: Provider[] = [
  {
    id: "deepgram",
    name: "Deepgram",
    capabilities: ["stt", "tts"],
    description: "Real-time speech recognition and text-to-speech",
  },
  {
    id: "sarvam",
    name: "Sarvam",
    capabilities: ["stt", "tts"],
    description: "Indian language speech recognition and synthesis",
  },
  {
    id: "bhashini",
    name: "Bhashini",
    capabilities: ["stt", "tts"],
    description: "Indic language AI models",
  },
  {
    id: "cartesia",
    name: "Cartesia",
    capabilities: ["tts"],
    description: "Low-latency voice synthesis with emotion control",
  },
  {
    id: "openai",
    name: "OpenAI",
    capabilities: ["llm"],
    description: "GPT-4o, GPT-4, and other advanced models",
  },
  {
    id: "azure",
    name: "Azure",
    capabilities: ["llm"],
    description: "Azure OpenAI Service models",
  },
  {
    id: "anthropic",
    name: "Anthropic",
    capabilities: ["llm"],
    description: "Claude family of AI assistants",
  },
  {
    id: "groq",
    name: "Groq",
    capabilities: ["llm"],
    description: "Ultra-fast LLM inference",
  },
  {
    id: "google",
    name: "Google",
    capabilities: ["llm"],
    description: "Gemini family of multimodal AI models",
  },
]

// Capability configuration
const capabilityConfig: Record<
  ProviderCapability,
  { label: string; fullLabel: string; icon: React.ReactNode; color: string; badgeClass: string }
> = {
  llm: {
    label: "LLM",
    fullLabel: "Language Models",
    icon: <Brain className="h-4 w-4" />,
    color: "text-purple-500",
    badgeClass: "bg-purple-500/10 text-purple-600 border-purple-500/20",
  },
  stt: {
    label: "STT",
    fullLabel: "Speech-to-Text",
    icon: <Mic className="h-4 w-4" />,
    color: "text-blue-500",
    badgeClass: "bg-blue-500/10 text-blue-600 border-blue-500/20",
  },
  tts: {
    label: "TTS",
    fullLabel: "Text-to-Speech",
    icon: <Volume2 className="h-4 w-4" />,
    color: "text-green-500",
    badgeClass: "bg-green-500/10 text-green-600 border-green-500/20",
  },
}

export default function IntegrationsPage() {
  const [searchQuery, setSearchQuery] = useState("")
  const [activeTab, setActiveTab] = useState<"all" | ProviderCapability>("all")
  
  // Connected providers state (fetched from backend)
  const [connectedProviders, setConnectedProviders] = useState<Record<string, boolean>>({})
  
  // API keys state
  const [apiKeys, setApiKeys] = useState<Record<string, string>>({})
  const [visibleKeys, setVisibleKeys] = useState<Record<string, boolean>>({})
  
  // Modal state
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState<Provider | null>(null)
  const [modalApiKey, setModalApiKey] = useState("")
  const [isModalKeyVisible, setIsModalKeyVisible] = useState(false)
  const [isSaving, setIsSaving] = useState(false)
  
  // Loading state
  const [isLoading, setIsLoading] = useState(true)

  // Fetch integrations on mount
  useEffect(() => {
    fetchIntegrations()
  }, [])

  const fetchIntegrations = async () => {
    try {
      setIsLoading(true)
      const integrations = await getIntegrations()
      
      // Convert integrations array to connected providers map and api keys map
      const connected: Record<string, boolean> = {}
      const keys: Record<string, string> = {}
      
      integrations.forEach((integration: Integration) => {
        // Find the provider by matching the model name (case-insensitive)
        const provider = providers.find(
          (p) => p.name.toLowerCase() === integration.model.toLowerCase()
        )
        if (provider) {
          connected[provider.id] = true
          keys[provider.id] = integration.api_key
        }
      })
      
      setConnectedProviders(connected)
      setApiKeys(keys)
    } catch (error) {
      console.error("Error fetching integrations:", error)
    } finally {
      setIsLoading(false)
    }
  }

  // Get connected providers list
  const connectedProvidersList = useMemo(() => {
    return providers.filter((p) => connectedProviders[p.id])
  }, [connectedProviders])

  // Get available (not connected) providers filtered by search and tab
  const availableProviders = useMemo(() => {
    return providers.filter((provider) => {
      // Exclude already connected
      if (connectedProviders[provider.id]) return false

      // Search filter
      const matchesSearch =
        searchQuery === "" ||
        provider.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        provider.description.toLowerCase().includes(searchQuery.toLowerCase())

      // Tab filter
      const matchesTab =
        activeTab === "all" || provider.capabilities.includes(activeTab)

      return matchesSearch && matchesTab
    })
  }, [searchQuery, activeTab, connectedProviders])

  // Open connect modal
  const openConnectModal = (provider: Provider) => {
    setSelectedProvider(provider)
    setModalApiKey("")
    setIsModalKeyVisible(false)
    setIsModalOpen(true)
  }

  // Open manage modal for connected provider
  const openManageModal = (provider: Provider) => {
    setSelectedProvider(provider)
    setModalApiKey(apiKeys[provider.id] || "••••••••••••••••")
    setIsModalKeyVisible(false)
    setIsModalOpen(true)
  }

  // Handle save from modal
  const handleModalSave = async () => {
    if (!selectedProvider || !modalApiKey) return

    setIsSaving(true)

    try {
      const orgId = getOrgId()
      if (!orgId) {
        throw new Error("Organization ID not found")
      }

      await createIntegration({
        org_id: orgId,
        model: selectedProvider.name,  // Use official name (e.g., "Deepgram", "OpenAI")
        api_key: modalApiKey,
      })

      // Update state
      setApiKeys((prev) => ({ ...prev, [selectedProvider.id]: modalApiKey }))
      setConnectedProviders((prev) => ({ ...prev, [selectedProvider.id]: true }))
      setIsModalOpen(false)
    } catch (error) {
      console.error("Error saving integration:", error)
      // Could add toast notification here
    } finally {
      setIsSaving(false)
    }
  }

  // Handle disconnect
  const handleDisconnect = async () => {
    if (!selectedProvider) return

    setIsSaving(true)

    try {
      await deleteIntegration(selectedProvider.name)  // Use official name

      // Update state
      setConnectedProviders((prev) => {
        const updated = { ...prev }
        delete updated[selectedProvider.id]
        return updated
      })
      setApiKeys((prev) => {
        const updated = { ...prev }
        delete updated[selectedProvider.id]
        return updated
      })
      setIsModalOpen(false)
    } catch (error) {
      console.error("Error disconnecting integration:", error)
      // Could add toast notification here
    } finally {
      setIsSaving(false)
    }
  }

  const isEditing = selectedProvider && connectedProviders[selectedProvider.id]

  return (
    <div className="flex flex-1 flex-col h-full overflow-hidden">
      {/* Fixed Header */}
      <div className="flex-shrink-0 p-6 pb-0 space-y-4">
        {/* Page Header */}
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold tracking-tight">Integrations</h1>
          <p className="text-muted-foreground">
            Connect API providers to enable speech and language capabilities
          </p>
        </div>

        {/* Connected Integrations Section */}
        {isLoading ? (
          <div className="flex items-center justify-center py-8">
            <Loader2 className="h-6 w-6 animate-spin text-muted-foreground" />
            <span className="ml-2 text-muted-foreground">Loading integrations...</span>
          </div>
        ) : connectedProvidersList.length > 0 && (
          <section className="space-y-3">
            <div className="flex items-center gap-2">
              <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
                Connected
              </h2>
              <Badge variant="secondary" className="h-5 px-1.5 text-xs">
                {connectedProvidersList.length}
              </Badge>
            </div>

            <Card>
              <CardContent className="p-0">
                <div className="divide-y">
                  {connectedProvidersList.map((provider) => (
                    <div
                      key={provider.id}
                      className="flex items-center justify-between px-4 py-3 hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center gap-3">
                        {/* Connected indicator */}
                        <div className="flex items-center justify-center h-8 w-8 rounded-full bg-green-500/10">
                          <Check className="h-4 w-4 text-green-600" />
                        </div>

                        {/* Provider info */}
                        <div className="min-w-0">
                          <div className="flex items-center gap-2">
                            <span className="font-medium">{provider.name}</span>
                            <div className="flex gap-1">
                              {provider.capabilities.map((cap) => (
                                <Tooltip key={cap}>
                                  <TooltipTrigger asChild>
                                    <Badge
                                      variant="outline"
                                      className={`${capabilityConfig[cap].badgeClass} text-[10px] px-1.5 py-0 cursor-default`}
                                    >
                                      {capabilityConfig[cap].label}
                                    </Badge>
                                  </TooltipTrigger>
                                  <TooltipContent>
                                    {capabilityConfig[cap].fullLabel}
                                  </TooltipContent>
                                </Tooltip>
                              ))}
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground truncate max-w-[300px]">
                            {provider.description}
                          </p>
                        </div>
                      </div>

                      {/* Manage button */}
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={() => openManageModal(provider)}
                        className="gap-1.5"
                      >
                        <Settings2 className="h-3.5 w-3.5" />
                        Manage
                      </Button>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </section>
        )}

        {!isLoading && <Separator />}

        {/* Available Header and Filters - Fixed */}
        {!isLoading && (
        <section className="space-y-4">
          <div className="flex items-center gap-2">
            <h2 className="text-sm font-medium text-muted-foreground uppercase tracking-wider">
              Available
            </h2>
          </div>

          {/* Search and Filter */}
          <div className="flex flex-col sm:flex-row gap-3">
            {/* Search */}
            <div className="relative flex-1 max-w-sm">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
              <Input
                placeholder="Search providers..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 h-9"
              />
            </div>

            {/* Filter Tabs */}
            <Tabs
              value={activeTab}
              onValueChange={(v) => setActiveTab(v as typeof activeTab)}
            >
              <TabsList>
                <TabsTrigger value="all">All</TabsTrigger>
                <TabsTrigger value="llm" className="gap-1.5">
                  <Brain className="h-3.5 w-3.5" />
                  LLM
                </TabsTrigger>
                <TabsTrigger value="stt" className="gap-1.5">
                  <Mic className="h-3.5 w-3.5" />
                  STT
                </TabsTrigger>
                <TabsTrigger value="tts" className="gap-1.5">
                  <Volume2 className="h-3.5 w-3.5" />
                  TTS
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>
        </section>
        )}
      </div>

      {/* Scrollable Provider Grid */}
      {!isLoading && (
      <ScrollArea className="flex-1 px-6 pb-6">
        <div className="pt-4">
          {availableProviders.length > 0 ? (
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
              {availableProviders.map((provider) => (
                <Card
                  key={provider.id}
                  className="group hover:shadow-md transition-shadow cursor-pointer"
                  onClick={() => openConnectModal(provider)}
                >
                  <CardContent className="p-4">
                    <div className="flex items-start justify-between gap-2">
                      <div className="min-w-0 flex-1 space-y-1">
                        <div className="flex items-center gap-2">
                          <span className="font-medium">{provider.name}</span>
                        </div>
                        <p className="text-xs text-muted-foreground line-clamp-1">
                          {provider.description}
                        </p>
                        <div className="flex gap-1 pt-1">
                          {provider.capabilities.map((cap) => (
                            <Badge
                              key={cap}
                              variant="outline"
                              className={`${capabilityConfig[cap].badgeClass} text-[10px] px-1.5 py-0`}
                            >
                              {capabilityConfig[cap].label}
                            </Badge>
                          ))}
                        </div>
                      </div>

                      {/* Connect button */}
                      <Button
                        variant="outline"
                        size="sm"
                        className="opacity-0 group-hover:opacity-100 transition-opacity shrink-0"
                        onClick={(e) => {
                          e.stopPropagation()
                          openConnectModal(provider)
                        }}
                      >
                        <Plus className="h-3.5 w-3.5 mr-1" />
                        Connect
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          ) : (
            <div className="flex flex-col items-center justify-center py-12 text-center border rounded-lg border-dashed">
              <div className="rounded-full bg-muted p-3 mb-3">
                <Search className="h-5 w-5 text-muted-foreground" />
              </div>
              <p className="text-sm text-muted-foreground mb-2">
                {searchQuery
                  ? `No providers match "${searchQuery}"`
                  : "All providers are connected"}
              </p>
              {searchQuery && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSearchQuery("")}
                >
                  Clear search
                </Button>
              )}
            </div>
          )}
        </div>
      </ScrollArea>
      )}

      {/* Connect/Manage Modal */}
      <Dialog open={isModalOpen} onOpenChange={setIsModalOpen}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle className="flex items-center gap-2">
              {isEditing ? "Manage" : "Connect"} {selectedProvider?.name}
            </DialogTitle>
            <DialogDescription>
              {isEditing
                ? "Update your API key or disconnect this integration."
                : `Enter your ${selectedProvider?.name} API key to enable ${selectedProvider?.capabilities
                    .map((c) => capabilityConfig[c].fullLabel)
                    .join(", ")}.`}
            </DialogDescription>
          </DialogHeader>

          <div className="space-y-4 py-2">
            {/* Capability badges */}
            <div className="flex gap-1.5">
              {selectedProvider?.capabilities.map((cap) => (
                <Badge
                  key={cap}
                  variant="outline"
                  className={`${capabilityConfig[cap].badgeClass}`}
                >
                  {capabilityConfig[cap].icon}
                  <span className="ml-1">{capabilityConfig[cap].fullLabel}</span>
                </Badge>
              ))}
            </div>

            {/* API Key Input */}
            <div className="space-y-2">
              <Label htmlFor="api-key">API Key</Label>
              <div className="relative">
                <Input
                  id="api-key"
                  type={isModalKeyVisible ? "text" : "password"}
                  placeholder="Enter your API key"
                  value={modalApiKey}
                  onChange={(e) => setModalApiKey(e.target.value)}
                  className="pr-10"
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="icon"
                  className="absolute right-0 top-0 h-full w-10 hover:bg-transparent"
                  onClick={() => setIsModalKeyVisible(!isModalKeyVisible)}
                  tabIndex={-1}
                >
                  {isModalKeyVisible ? (
                    <EyeOff className="h-4 w-4 text-muted-foreground" />
                  ) : (
                    <Eye className="h-4 w-4 text-muted-foreground" />
                  )}
                </Button>
              </div>
              <p className="text-xs text-muted-foreground">
                Your API key is encrypted and stored securely.
              </p>
            </div>
          </div>

          <DialogFooter className="flex-col sm:flex-row gap-2">
            {isEditing && (
              <Button
                variant="outline"
                onClick={handleDisconnect}
                disabled={isSaving}
                className="text-destructive hover:text-destructive hover:bg-destructive/10 sm:mr-auto"
              >
                <Trash2 className="h-4 w-4 mr-1.5" />
                Disconnect
              </Button>
            )}
            <Button
              onClick={handleModalSave}
              disabled={!modalApiKey || isSaving}
            >
              {isSaving ? (
                <>
                  <Loader2 className="h-4 w-4 mr-1.5 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="h-4 w-4 mr-1.5" />
                  {isEditing ? "Update" : "Connect"}
                </>
              )}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  )
}
