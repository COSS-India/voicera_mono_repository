import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// PUT - Update an agent
export async function PUT(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    // Extract id from params (Next.js 16+ params are a Promise)
    const params = await Promise.resolve(context.params)
    const body = await request.json()
    const agentId = decodeURIComponent(params.id)

    // Validate required fields
    if (!body.org_id || !body.agent_config) {
      return NextResponse.json(
        { error: "org_id and agent_config are required" },
        { status: 400 }
      )
    }

    // Backend PUT looks up by ORIGINAL agent_type. body.agent_type may be the new name when renaming.
    let agentTypeForUrl = ""

    if (typeof body.original_agent_type === "string" && body.original_agent_type.trim()) {
      agentTypeForUrl = body.original_agent_type.trim()
    } else {
      agentTypeForUrl = agentId
      const parts = agentId.split("-")
      if (parts.length >= 3) {
        const extractedAgentType = parts.slice(1, -1).join("-")
        if (extractedAgentType) {
          agentTypeForUrl = extractedAgentType
        }
      } else if (parts.length === 2) {
        agentTypeForUrl = parts[1] || agentId
      }
    }

    if (!agentTypeForUrl) {
      return NextResponse.json(
        { error: "Could not determine agent_type from agentId" },
        { status: 400 }
      )
    }

    const { original_agent_type: _original, ...backendBody } = body

    const response = await fetch(`${API_BASE_URL}/api/v1/agents/${encodeURIComponent(agentTypeForUrl)}`, {
      method: "PUT",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": authHeader,
      },
      body: JSON.stringify(backendBody),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error updating agent:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

// GET - Get a single agent by ID
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ id: string }> | { id: string } }
) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    // Extract id from params (Next.js 16+ params are a Promise)
    const params = await Promise.resolve(context.params)
    const agentId = params.id
    const { searchParams } = new URL(request.url)
    const orgId = searchParams.get("org_id")

    if (!orgId) {
      return NextResponse.json(
        { error: "org_id parameter is required" },
        { status: 400 }
      )
    }

    // Fetch all agents and find the one with matching ID
    const response = await fetch(
      `${API_BASE_URL}/api/v1/agents/org/${orgId}`,
      {
        method: "GET",
        headers: {
          "Accept": "application/json",
          "Authorization": authHeader,
        },
      }
    )

    const agents = await response.json()

    if (!response.ok) {
      return NextResponse.json(agents, { status: response.status })
    }

    // Decode the agentId in case it's URL encoded
    const decodedAgentId = decodeURIComponent(agentId)
    const decodedAgentIdLower = decodedAgentId.toLowerCase()

    // Normalize _id to id and find the agent with matching ID
    const normalizedAgents = Array.isArray(agents)
      ? agents.map((a: any) => ({
          ...a,
          id: a.id || a._id || a.agent_type,
        }))
      : []

    // First, try to find by exact ID match (MongoDB _id or id)
    let agent = normalizedAgents.find(
      (a: any) => String(a.id) === decodedAgentId || String(a._id) === decodedAgentId
    )

    // Match by agent_type or agent_id (primary navigation keys)
    if (!agent) {
      agent = normalizedAgents.find(
        (a: any) =>
          (a.agent_type && a.agent_type.toLowerCase() === decodedAgentIdLower) ||
          (a.agent_id && a.agent_id.toLowerCase() === decodedAgentIdLower)
      )
    }

    // Legacy composite IDs: "org_id-agent_type-<timestamp>" (timestamp may be ISO with hyphens)
    if (!agent) {
      agent = normalizedAgents.find((a: any) => {
        if (!a.org_id || !a.agent_type) return false
        const timestamp = a.created_at || a.updated_at || ""
        if (!timestamp) return false
        return (
          decodedAgentId === `${a.org_id}-${a.agent_type}-${timestamp}` ||
          decodedAgentId.startsWith(`${a.org_id}-${a.agent_type}-`)
        )
      })
    }

    // Fallback: hyphen-split composite (only reliable when timestamp has no hyphens)
    if (!agent) {
      const parts = decodedAgentId.split("-")
      if (parts.length >= 3) {
        const extractedAgentType = parts.slice(1, -1).join("-")
        if (extractedAgentType) {
          agent = normalizedAgents.find(
            (a: any) =>
              a.agent_type &&
              a.agent_type.toLowerCase() === extractedAgentType.toLowerCase()
          )
        }
      } else if (parts.length === 2) {
        const possibleAgentType = parts[1]
        if (possibleAgentType) {
          agent = normalizedAgents.find(
            (a: any) =>
              a.agent_type &&
              a.agent_type.toLowerCase() === possibleAgentType.toLowerCase()
          )
        }
      }
    }

    if (!agent) {
      return NextResponse.json(
        { error: "Agent not found" },
        { status: 404 }
      )
    }

    return NextResponse.json(agent)
  } catch (error) {
    console.error("Error fetching agent:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
