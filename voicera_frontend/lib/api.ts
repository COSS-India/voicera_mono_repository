/**
 * API utility functions for making authenticated requests
 */

/**
 * Get the stored auth token from localStorage
 */
export function getAuthToken(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem("access_token")
}

/**
 * Get the stored org_id from localStorage
 */
export function getOrgId(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem("org_id")
}

/**
 * Get the stored email from localStorage
 */
export function getStoredEmail(): string | null {
  if (typeof window === "undefined") return null
  return localStorage.getItem("email")
}

/**
 * Check if user is authenticated
 */
export function isAuthenticated(): boolean {
  return !!getAuthToken()
}

/**
 * Clear auth data (for logout)
 */
export function clearAuth(): void {
  if (typeof window === "undefined") return
  localStorage.removeItem("access_token")
  localStorage.removeItem("org_id")
  localStorage.removeItem("email")
}

/**
 * Make an authenticated fetch request to the backend API
 */
export async function fetchWithAuth(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getAuthToken()

  if (!token) {
    throw new Error("No authentication token found")
  }

  const headers = new Headers(options.headers)
  headers.set("Authorization", `Bearer ${token}`)
  headers.set("Content-Type", "application/json")
  headers.set("Accept", "application/json")

  // Use relative URL so browser hits frontend; frontend proxies to backend (Docker: backend hostname only resolves server-side)
  const response = await fetch(endpoint, {
    ...options,
    headers,
  })

  // If unauthorized, clear auth and redirect to login
  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  return response
}

/**
 * Make an authenticated fetch request to Next.js API routes
 */
export async function fetchApiRoute(
  endpoint: string,
  options: RequestInit = {}
): Promise<Response> {
  const token = getAuthToken()

  if (!token) {
    throw new Error("No authentication token found")
  }

  const headers = new Headers(options.headers)
  headers.set("Authorization", `Bearer ${token}`)
  headers.set("Content-Type", "application/json")
  headers.set("Accept", "application/json")

  const response = await fetch(endpoint, {
    ...options,
    headers,
  })

  // If unauthorized, clear auth and redirect to login
  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  return response
}

async function parseErrorMessage(response: Response, fallbackMessage: string): Promise<string> {
  const contentType = response.headers.get("content-type") || ""
  if (contentType.includes("application/json")) {
    try {
      const error = await response.json()
      return error.detail || error.error || fallbackMessage
    } catch {
      return fallbackMessage
    }
  }
  try {
    const text = await response.text()
    if (text && text.trim().length > 0) {
      return `${fallbackMessage}: ${text.slice(0, 200)}`
    }
  } catch {
    return fallbackMessage
  }
  return fallbackMessage
}

/**
 * Get current user info (works for both org owners and members)
 */
export async function getCurrentUser(): Promise<User> {
  const response = await fetchWithAuth(`/api/v1/users/me`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || "Failed to fetch user")
  }
  
  return response.json()
}

/**
 * Get all agents for an organization
 */
export async function getAgents(orgId: string): Promise<Agent[]> {
  const response = await fetchApiRoute(`/api/agents?org_id=${encodeURIComponent(orgId)}`)
  const data = await response.json()
  console.log("response", data)
  if (!response.ok) {
    throw new Error(data.detail || data.error || "Failed to fetch agents")
  }
  
  return data
}

/**
 * Create a new agent
 */
export async function createAgent(agentData: CreateAgentRequest): Promise<Agent> {
  const response = await fetchApiRoute("/api/agents", {
    method: "POST",
    body: JSON.stringify(agentData),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create agent")
  }
  
  return response.json()
}

/**
 * Get a single agent by ID
 */
export async function getAgent(agentId: string, orgId: string): Promise<Agent> {
  const response = await fetchApiRoute(`/api/agents/${encodeURIComponent(agentId)}?org_id=${encodeURIComponent(orgId)}`)
  
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Failed to fetch agent"))
  }
  
  return response.json()
}

/**
 * Update an agent
 */
export async function updateAgent(agentId: string, agentData: CreateAgentRequest): Promise<Agent> {
  const response = await fetchApiRoute(`/api/agents/${encodeURIComponent(agentId)}`, {
    method: "PUT",
    body: JSON.stringify(agentData),
  })
  
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Failed to update agent"))
  }
  
  return response.json()
}

/**
 * Delete an agent
 */
export async function deleteAgent(
  agentId: string,
  options?: { agentType?: string }
): Promise<{ status: string; message: string }> {
  const trimmedAgentType = options?.agentType?.trim() || ""
  const hasAgentType = trimmedAgentType.length > 0
  const response = hasAgentType
    ? await fetchApiRoute(`/api/agents?agent_type=${encodeURIComponent(trimmedAgentType)}`, {
        method: "DELETE",
      })
    : await fetchApiRoute(`/api/agents/${encodeURIComponent(agentId)}`, {
        method: "DELETE",
      })
  
  if (!response.ok) {
    throw new Error(await parseErrorMessage(response, "Failed to delete agent"))
  }
  
  return response.json()
}

/**
 * Get all campaigns for an organization
 */
export async function getCampaigns(orgId: string): Promise<Campaign[]> {
  const response = await fetchApiRoute(`/api/get-campaigns?org_id=${encodeURIComponent(orgId)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch campaigns")
  }
  
  return response.json()
}

/**
 * Create a new campaign
 */
export async function createCampaign(campaignData: CreateCampaignRequest): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/campaigns", {
    method: "POST",
    body: JSON.stringify(campaignData),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create campaign")
  }
  
  return response.json()
}

/**
 * Get a single campaign by name
 */
export async function getCampaign(campaignName: string): Promise<Campaign> {
  const response = await fetchApiRoute(`/api/campaigns/${encodeURIComponent(campaignName)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch campaign")
  }
  
  return response.json()
}

/**
 * Get all audiences, optionally filtered by phone number
 */
export async function getAudiences(phoneNumber?: string): Promise<Audience[]> {
  const url = phoneNumber 
    ? `/api/audiences?phone_number=${encodeURIComponent(phoneNumber)}`
    : `/api/audiences`
  const response = await fetchApiRoute(url)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch audiences")
  }
  
  return response.json()
}

/**
 * Create a new audience
 */
export async function createAudience(audienceData: CreateAudienceRequest): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/audiences", {
    method: "POST",
    body: JSON.stringify(audienceData),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create audience")
  }
  
  return response.json()
}

/**
 * Get a single audience by name
 */
export async function getAudience(audienceName: string): Promise<Audience> {
  const response = await fetchApiRoute(`/api/audiences/${encodeURIComponent(audienceName)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch audience")
  }
  
  return response.json()
}

/**
 * Type definitions for API responses
 */
export interface User {
  email: string
  name: string
  org_id: string
  company_name: string
  created_at: string
}

export interface LoginResponse {
  access_token: string
  token_type: string
  org_id: string
}

export interface ApiError {
  detail: string
}

export interface Agent {
  id?: string
  _id?: string
  org_id: string
  agent_type: string
  agent_id: string
  agent_config: AgentConfig
  created_at: string
  updated_at: string
  telephony_provider: string
  phone_number?: string
  vobiz_app_id?: string
  vobiz_answer_url?: string
  plivo_app_id?: string
  plivo_answer_url?: string
}

export interface AgentConfig {
  system_prompt: string
  greeting_message: string
  session_timeout_minutes: number
  language: string
  knowledge_base_enabled?: boolean
  knowledge_document_ids?: string[]
  knowledge_top_k?: number
  llm_model: {
    name: string
    model?: string
    vistaar_environment?: "prod" | "dev"
  }
  stt_model: {
    name: string
    model?: string
    keywords?: string
  }
  tts_model: {
    voice_id?: string
    name: string
    model?: string
    speaker: string
    description?: string
    speed?: number
    pitch?: number
    emotion_intensity?: number
    loudness?: number
    args?: {
      model?: string
      voice_id?: string
      [key: string]: any
    }
  }
}

export interface CreateAgentRequest {
  org_id: string
  agent_type: string
  agent_id: string
  agent_category: string
  agent_config: AgentConfig
  /** BFF-only: original agent_type for PUT lookup when renaming */
  original_agent_type?: string
  telephony_provider?: string
  vobiz_app_id?: string
  vobiz_answer_url?: string
  plivo_app_id?: string
  plivo_answer_url?: string
}

export interface Campaign {
  campaign_name: string
  org_id?: string
  agent_type?: string
  status?: string
  campaign_information?: Record<string, any>
}

export interface CreateCampaignRequest {
  campaign_name: string
  org_id?: string
  agent_type?: string
  status?: string
  campaign_information?: Record<string, any>
}

export interface Audience {
  audience_name: string
  phone_number: string
  parameters?: Record<string, any>
}

export interface CreateAudienceRequest {
  audience_name: string
  phone_number: string
  parameters?: Record<string, any>
}

export interface MeetingsPageParams {
  page?: number
  limit?: number
  forExport?: boolean
  agent_type?: string
  from_number?: string
  to_number?: string
  inbound?: boolean
  call_status?: string
  date_from?: string
  date_to?: string
  date_sort_order?: "latest" | "oldest"
  duration_sort_order?: "longest" | "shortest" | null
}

export interface PaginatedMeetings {
  items: Meeting[]
  total: number
  page: number
  limit: number
}

export interface MeetingFilterOptions {
  agent_types: string[]
  from_numbers: string[]
  to_numbers: string[]
}

function buildMeetingsQueryString(params: MeetingsPageParams): string {
  const q = new URLSearchParams()
  q.set("page", String(params.page ?? 1))
  q.set("limit", String(params.limit ?? 50))
  if (params.forExport) q.set("for_export", "true")
  if (params.agent_type) q.set("agent_type", params.agent_type)
  if (params.from_number) q.set("from_number", params.from_number)
  if (params.to_number) q.set("to_number", params.to_number)
  if (params.inbound !== undefined) q.set("inbound", String(params.inbound))
  if (params.call_status) q.set("call_status", params.call_status)
  if (params.date_from) q.set("date_from", params.date_from)
  if (params.date_to) q.set("date_to", params.date_to)
  if (params.date_sort_order) q.set("date_sort_order", params.date_sort_order)
  if (params.duration_sort_order) {
    q.set("duration_sort_order", params.duration_sort_order)
  }
  return q.toString()
}

/**
 * Fetch a paginated page of meetings (History tab).
 */
export async function getMeetingsPage(
  params: MeetingsPageParams = {}
): Promise<PaginatedMeetings> {
  const response = await fetchApiRoute(`/api/meetings?${buildMeetingsQueryString(params)}`)

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch meetings")
  }

  return response.json()
}

export async function getMeetingFilterOptions(): Promise<MeetingFilterOptions> {
  const response = await fetchApiRoute("/api/meetings/filter-options")

  if (!response.ok) {
    const error = await response.json()
    throw new Error(
      error.detail || error.error || "Failed to fetch meeting filter options"
    )
  }

  return response.json()
}

/**
 * Get all meetings for an organization (first page only; prefer getMeetingsPage).
 */
export async function getMeetings(agentType?: string): Promise<Meeting[]> {
  const result = await getMeetingsPage({
    page: 1,
    limit: 50,
    agent_type: agentType,
  })
  return result.items
}

/**
 * Get a single meeting by ID
 */
export async function getMeeting(meetingId: string): Promise<Meeting> {
  const response = await fetchApiRoute(`/api/meetings/${encodeURIComponent(meetingId)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch meeting")
  }
  
  return response.json()
}

/**
 * Get detailed meeting data including transcript and custom variables
 */
export async function getMeetingDetails(meetingId: string): Promise<MeetingDetails> {
  // For now, use the same endpoint as getMeeting
  // If backend has a separate details endpoint, update this
  const response = await fetchApiRoute(`/api/meetings/${encodeURIComponent(meetingId)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch meeting details")
  }
  
  const data = await response.json()
  
  // Extract custom variables from agent_config if they exist
  // Custom variables are typically stored as a nested object in agent_config
  const customVars = data.agent_config?.custom_variables || data.agent_config?.variables || data.custom_variables || {}
  
  return {
    ...data,
    custom_variables: Object.keys(customVars).length > 0 ? customVars : undefined,
  }
}

/**
 * Create a Vobiz application
 */
export async function createVobizApplication(agentType: string, answerUrl: string): Promise<{ status: string; message: string; app_id?: string }> {
  console.log("createVobizApplication", agentType, answerUrl)
  const response = await fetchApiRoute("/api/vobiz/application", {
    method: "POST",
    body: JSON.stringify({
      agent_type: agentType,
      answer_url: answerUrl,
    }),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create Vobiz application")
  }
  
  return response.json()
}

/**
 * Delete a Vobiz application
 */
export async function deleteVobizApplication(applicationId: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute(`/api/vobiz/application/${applicationId}`, {
    method: "DELETE",
  })
  
  if (!response.ok) {
    const error = await response.json()
    const errorMessage = String(error.detail || error.error || "")
    if (response.status === 404 || errorMessage.toLowerCase().includes("not found")) {
      return {
        status: "success",
        message: "Vobiz application already deleted",
      }
    }
    throw new Error(error.detail || error.error || "Failed to delete Vobiz application")
  }
  
  return response.json()
}

/**
 * Get Vobiz phone numbers
 */
export async function getVobizNumbers(): Promise<{ status: string; numbers: string[] }> {
  const response = await fetchApiRoute("/api/vobiz/numbers", {
    method: "GET",
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch Vobiz numbers")
  }
  
  return response.json()
}

/**
 * Link a phone number to a Vobiz application
 */
export async function linkVobizNumber(phoneNumber: string, applicationId: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/vobiz/numbers/link", {
    method: "POST",
    body: JSON.stringify({
      phone_number: phoneNumber,
      application_id: applicationId,
    }),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to link phone number")
  }
  
  return response.json()
}

/**
 * Unlink a phone number from a Vobiz application
 */
export async function unlinkVobizNumber(phoneNumber: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/vobiz/numbers/unlink", {
    method: "DELETE",
    body: JSON.stringify({
      phone_number: phoneNumber,
    }),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to unlink phone number")
  }
  
  return response.json()
}

/**
 * Create a Plivo application
 */
export async function createPlivoApplication(agentType: string, answerUrl: string): Promise<{ status: string; message: string; app_id?: string }> {
  const response = await fetchApiRoute("/api/plivo/application", {
    method: "POST",
    body: JSON.stringify({
      agent_type: agentType,
      answer_url: answerUrl,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create Plivo application")
  }

  return response.json()
}

/**
 * Delete a Plivo application
 */
export async function deletePlivoApplication(applicationId: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute(`/api/plivo/application/${applicationId}`, {
    method: "DELETE",
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to delete Plivo application")
  }

  return response.json()
}

/**
 * Get Plivo phone numbers
 */
export async function getPlivoNumbers(): Promise<{ status: string; numbers: string[] }> {
  const response = await fetchApiRoute("/api/plivo/numbers", {
    method: "GET",
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch Plivo numbers")
  }

  return response.json()
}

/**
 * Link a phone number to a Plivo application
 */
export async function linkPlivoNumber(phoneNumber: string, applicationId: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/plivo/numbers/link", {
    method: "POST",
    body: JSON.stringify({
      phone_number: phoneNumber,
      application_id: applicationId,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to link phone number")
  }

  return response.json()
}

/**
 * Unlink a phone number from a Plivo application
 */
export async function unlinkPlivoNumber(phoneNumber: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/plivo/numbers/unlink", {
    method: "DELETE",
    body: JSON.stringify({
      phone_number: phoneNumber,
    }),
  })

  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to unlink phone number")
  }

  return response.json()
}

/**
 * Get analytics metrics for an organization
 */
export async function getAnalytics(params?: {
  agent_type?: string
  phone_number?: string
  start_date?: string
  end_date?: string
}): Promise<Analytics> {
  const queryParams = new URLSearchParams()
  if (params?.agent_type) queryParams.append("agent_type", params.agent_type)
  if (params?.phone_number) queryParams.append("phone_number", params.phone_number)
  if (params?.start_date) queryParams.append("start_date", params.start_date)
  if (params?.end_date) queryParams.append("end_date", params.end_date)
  
  const queryString = queryParams.toString()
  const url = `/api/analytics${queryString ? `?${queryString}` : ""}`
  
  const response = await fetchApiRoute(url)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch analytics")
  }
  
  return response.json()
}

export interface CallLatencyTurn {
  turn_index: number
  user_text_preview?: string | null
  stt_ms?: number | null
  llm_ttfb_ms?: number | null
  tts_first_chunk_ms?: number | null
}

export interface CallLatencySummary {
  turn_count?: number
  avg_stt_ms?: number
  avg_llm_ttfb_ms?: number
  avg_tts_first_chunk_ms?: number
  max_stt_ms?: number
  max_llm_ttfb_ms?: number
  max_tts_first_chunk_ms?: number
}

export interface CallLatencyMetrics {
  turns: CallLatencyTurn[]
  summary?: CallLatencySummary
}

export interface Meeting {
  id?: string
  _id?: string
  meeting_id: string
  agent_type?: string
  org_id?: string
  agent_category?: string
  agent_config?: Record<string, any>
  inbound?: boolean
  from_number?: string
  to_number?: string
  created_at?: string
  start_time_utc?: string
  end_time_utc?: string
  recording_url?: string
  transcript_url?: string
  transcript_content?: string
  transcript?: TranscriptMessage[]  // Parsed transcript from backend
  duration?: number
  call_busy?: boolean
  latency_metrics?: CallLatencyMetrics
}

export interface TranscriptMessage {
  role: 'user' | 'agent' | 'assistant' | 'human'
  content: string
  timestamp?: string
}

export interface MeetingDetails extends Meeting {
  // Custom variables (these come from agent_config as a nested dict)
  // The agent_config field may contain custom_variables as a nested object
  custom_variables?: {
    mobile_number?: string
    call_status_reason?: string
    classification?: string
    key_points?: string[]
    action_items?: string[]
    summary?: string
    callback_requested_time?: string | null
    // Any other custom variables that might exist
    [key: string]: any
  }
}

export interface AgentBreakdown {
  agent_type: string
  call_count: number
}

export interface Analytics {
  org_id: string
  calls_attempted: number
  calls_connected: number
  average_call_duration: number
  total_minutes_connected: number
  most_used_agent?: string | null
  most_used_agent_count: number
  agent_breakdown: AgentBreakdown[]
  calculated_at: string
  start_date?: string | null
  end_date?: string | null
}

// Integration types
export interface Integration {
  org_id: string
  model: string
  api_key: string
  created_at?: string
  updated_at?: string
}

export interface CreateIntegrationRequest {
  org_id: string
  model: string
  api_key: string
}

/**
 * Get all integrations for the current organization
 */
export async function getIntegrations(): Promise<Integration[]> {
  const response = await fetchApiRoute("/api/integrations")
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch integrations")
  }
  
  return response.json()
}

/**
 * Get a specific integration by model
 */
export async function getIntegration(model: string): Promise<Integration> {
  const response = await fetchApiRoute(`/api/integrations/${encodeURIComponent(model)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch integration")
  }
  
  return response.json()
}

/**
 * Create or update an integration
 */
export async function createIntegration(integrationData: CreateIntegrationRequest): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute("/api/integrations", {
    method: "POST",
    body: JSON.stringify(integrationData),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to create integration")
  }
  
  return response.json()
}

/**
 * Delete an integration by model
 */
export async function deleteIntegration(model: string): Promise<{ status: string; message: string }> {
  const response = await fetchApiRoute(`/api/integrations/${encodeURIComponent(model)}`, {
    method: "DELETE",
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to delete integration")
  }
  
  return response.json()
}

// Knowledge base (org-scoped PDF ingest)
export interface KnowledgeDocument {
  document_id: string
  org_id: string
  original_filename: string
  status: "processing" | "ready" | "failed"
  chunk_count?: number | null
  embedding_model?: string | null
  error_message?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface KnowledgeUploadResponse {
  document_id: string
  org_id: string
  original_filename: string
  status: string
}

export interface KnowledgeDeleteResponse {
  deleted: boolean
}

/**
 * List knowledge PDFs for the current organization (JWT).
 */
export async function getKnowledgeDocuments(): Promise<KnowledgeDocument[]> {
  const response = await fetchApiRoute("/api/knowledge-base")

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to fetch knowledge documents"
    )
  }

  return response.json()
}

/**
 * Upload a PDF for background ingest (multipart). Do not set Content-Type manually.
 */
export async function uploadKnowledgePdf(
  file: File,
  orgId: string
): Promise<KnowledgeUploadResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const formData = new FormData()
  formData.append("file", file)
  formData.append("org_id", orgId)

  const response = await fetch("/api/knowledge-base", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to upload knowledge PDF"
    )
  }

  return response.json()
}

/**
 * Delete a knowledge document and its indexed chunks (JWT org from token).
 */
export async function deleteKnowledgeDocument(
  documentId: string
): Promise<KnowledgeDeleteResponse> {
  const encodedId = encodeURIComponent(documentId)
  const response = await fetchApiRoute(`/api/knowledge-base/${encodedId}`, {
    method: "DELETE",
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to delete knowledge document"
    )
  }

  return response.json()
}

// Batches (immutable CSV uploads + parsed contacts)
export interface Batch {
  batch_id: string
  org_id: string
  batch_name: string
  agent_type: string
  concurrency: number
  original_filename: string
  status: string
  execution_status: string
  total_contacts: number
  valid_contacts: number
  invalid_contacts: number
  attempted_calls?: number
  successful_calls?: number
  failed_calls?: number
  error_message?: string | null
  schedule_mode?: "run_now" | "scheduled"
  scheduled_at_utc?: string | null
  scheduled_timezone?: string | null
  scheduled_status?: "none" | "scheduled" | "triggered" | "canceled"
  scheduled_by?: string | null
  created_at?: string | null
  updated_at?: string | null
}

export interface BatchUploadResponse {
  batch_id: string
  org_id: string
  batch_name: string
  agent_type: string
  concurrency: number
  original_filename: string
  status: string
  total_contacts: number
  valid_contacts: number
  invalid_contacts: number
  created_at?: string | null
}

export interface BatchDeleteResponse {
  deleted: boolean
}

export interface BatchActionResponse {
  status: string
  message: string
}

export interface BatchScheduleRequest {
  scheduled_at_local: string
  timezone: string
  agent_type?: string
  concurrency?: number
}

/**
 * List batches for the current organization.
 */
export async function getBatches(agentType?: string): Promise<Batch[]> {
  const query = agentType
    ? `?agent_type=${encodeURIComponent(agentType)}`
    : ""
  const response = await fetchApiRoute(`/api/batches${query}`)

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to fetch batches"
    )
  }

  return response.json()
}

/**
 * Upload a batch CSV and parse contacts server-side.
 */
export async function uploadBatchCsv(
  file: File,
  orgId: string,
  agentType: string,
  batchName: string
): Promise<BatchUploadResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const formData = new FormData()
  formData.append("file", file)
  formData.append("org_id", orgId)
  formData.append("agent_type", agentType)
  formData.append("batch_name", batchName)

  const response = await fetch("/api/batches", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
    body: formData,
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to upload batch CSV"
    )
  }

  return response.json()
}

/**
 * Delete a batch and its parsed contacts.
 */
export async function deleteBatch(batchId: string): Promise<BatchDeleteResponse> {
  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetchApiRoute(`/api/batches/${encodedBatchId}`, {
    method: "DELETE",
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to delete batch"
    )
  }

  return response.json()
}

/**
 * Start batch execution.
 */
export async function runBatch(
  batchId: string,
  agentType?: string,
  concurrency?: number
): Promise<BatchActionResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetch(`/api/batches/${encodedBatchId}/run`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      agent_type: agentType,
      concurrency,
    }),
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to run batch"
    )
  }

  return response.json()
}

/**
 * Stop batch execution.
 */
export async function stopBatch(batchId: string): Promise<BatchActionResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetch(`/api/batches/${encodedBatchId}/stop`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to stop batch"
    )
  }

  return response.json()
}

/**
 * Schedule one-time batch execution.
 */
export async function scheduleBatch(
  batchId: string,
  payload: BatchScheduleRequest
): Promise<BatchActionResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetch(`/api/batches/${encodedBatchId}/schedule`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to schedule batch"
    )
  }

  return response.json()
}

/**
 * Cancel one-time scheduled batch execution before it triggers.
 */
export async function cancelBatchSchedule(batchId: string): Promise<BatchActionResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetch(`/api/batches/${encodedBatchId}/schedule/cancel`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
    },
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to cancel schedule"
    )
  }

  return response.json()
}

/**
 * Reschedule one-time batch execution before it triggers.
 */
export async function rescheduleBatch(
  batchId: string,
  payload: BatchScheduleRequest
): Promise<BatchActionResponse> {
  const token = getAuthToken()
  if (!token) {
    throw new Error("No authentication token found")
  }

  const encodedBatchId = encodeURIComponent(batchId)
  const response = await fetch(`/api/batches/${encodedBatchId}/schedule/reschedule`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify(payload),
  })

  if (response.status === 401) {
    clearAuth()
    if (typeof window !== "undefined") {
      window.location.href = "/"
    }
  }

  if (!response.ok) {
    const error = await response.json().catch(() => ({}))
    throw new Error(
      (error as { detail?: string }).detail ||
        (error as { error?: string }).error ||
        "Failed to reschedule batch"
    )
  }

  return response.json()
}

// Member types
export interface Member {
  email: string
  name: string
  org_id: string
  company_name: string
  created_at: string
  is_owner?: boolean  // True if this member is the org owner
}

export interface JoinOrganizationRequest {
  email: string
  password: string
  name: string
  company_name: string
  org_id: string
}

/**
 * Get all members of an organization
 */
export async function getOrgMembers(orgId: string): Promise<Member[]> {
  const response = await fetchWithAuth(`/api/v1/members/${encodeURIComponent(orgId)}`)
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to fetch members")
  }
  
  const data = await response.json()
  return data.members || []
}

/**
 * Join an existing organization (public endpoint - no auth required)
 */
export async function joinOrganization(data: JoinOrganizationRequest): Promise<{ status: string; message: string }> {
  const response = await fetch("/api/v1/members/add-member", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(data),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to join organization")
  }
  
  return response.json()
}

/**
 * Delete a member from an organization
 */
export async function deleteMember(email: string, orgId: string): Promise<{ status: string; message: string }> {
  const response = await fetchWithAuth(`/api/v1/members/delete-member`, {
    method: "POST",
    body: JSON.stringify({
      email,
      org_id: orgId,
    }),
  })
  
  if (!response.ok) {
    const error = await response.json()
    throw new Error(error.detail || error.error || "Failed to delete member")
  }
  
  return response.json()
}

/**
 * Check if a user exists by email (public endpoint - no auth required)
 * Returns the user if found, null if not found
 */
export async function checkUserExists(email: string): Promise<User | null> {
  try {
    const response = await fetch(`/api/v1/users/${encodeURIComponent(email)}`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
    })
    
    if (response.status === 404) {
      return null
    }
    
    if (!response.ok) {
      return null
    }
    
    return response.json()
  } catch {
    return null
  }
}
