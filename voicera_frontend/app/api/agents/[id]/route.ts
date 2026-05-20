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

    // Extract agent_type from agentId
    // The agentId format is typically: "org_id-agent_type-timestamp"
    // We need to use the ORIGINAL agent_type from the ID to find the agent in the backend
    let agentTypeForUrl = body.agent_type || ""
    
    // Try to extract agent_type from agentId if it's in the format "org_id-agent_type-timestamp"
    const parts = agentId.split('-')
    if (parts.length >= 3) {
      // The agent_type is everything between org_id (first part) and timestamp (last part)
      const extractedAgentType = parts.slice(1, -1).join('-')
      if (extractedAgentType) {
        agentTypeForUrl = extractedAgentType
      }
    } else if (parts.length === 2) {
      // Fallback: if only 2 parts, the second might be the agent_type
      agentTypeForUrl = parts[1] || body.agent_type || ""
    }
    
    if (!agentTypeForUrl) {
      return NextResponse.json(
        { error: "Could not determine agent_type from agentId" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    // IMPORTANT: Use the original agent_type from the agentId in the URL path
    // The backend uses this to find the agent. If we use a new agent_type, it won't find the agent.
    const response = await fetch(`${API_BASE_URL}/api/v1/agents/${encodeURIComponent(agentTypeForUrl)}`, {
      method: "PUT",
      headers: {
        "Accept": "application/json",
        "Content-Type": "application/json",
        "Authorization": authHeader,
      },
      body: JSON.stringify(body),
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
          id: a.id || a._id,
        }))
      : []

    // First, try to find by exact ID match (MongoDB _id or id)
    let agent = normalizedAgents.find(
      (a: any) => String(a.id) === decodedAgentId || String(a._id) === decodedAgentId
    )

    // If not found by ID, try matching by agent_id (case-insensitive)
    if (!agent) {
      agent = normalizedAgents.find(
        (a: any) => a.agent_id && a.agent_id.toLowerCase() === decodedAgentIdLower
      )
    }

    // If not found by ID, try to extract agent_type from composite ID and match by agent_type (case-insensitive)
    // The agentId format might be: "org_id-agent_type-timestamp"
    if (!agent) {
      const parts = decodedAgentId.split('-')
      if (parts.length >= 3) {
        const extractedAgentType = parts.slice(1, -1).join('-')
        if (extractedAgentType) {
          agent = normalizedAgents.find(
            (a: any) => a.agent_type && a.agent_type.toLowerCase() === extractedAgentType.toLowerCase()
          )
        }
      } else if (parts.length === 2) {
        const possibleAgentType = parts[1]
        if (possibleAgentType) {
          agent = normalizedAgents.find(
            (a: any) => a.agent_type && a.agent_type.toLowerCase() === possibleAgentType.toLowerCase()
          )
        }
      }
      
      // If still not found, try matching by agent_type directly (case-insensitive)
      if (!agent) {
        agent = normalizedAgents.find(
          (a: any) => a.agent_type && a.agent_type.toLowerCase() === decodedAgentIdLower
        )
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

// DELETE - Delete an agent
export async function DELETE(
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
    const agentId = decodeURIComponent(params.id)
    const { searchParams } = new URL(request.url)
    const agentTypeParam = searchParams.get("agent_type")

    // Extract agent_type from agentId
    // The agentId format is typically: "org_id-agent_type-timestamp"
    // We need to extract the agent_type to call the backend
    let agentTypeForUrl = agentTypeParam ? decodeURIComponent(agentTypeParam).trim() : ""
    const parts = agentId.split("-")

    if (!agentTypeForUrl && parts.length >= 3) {
      // The agent_type is everything between org_id (first part) and timestamp (last part)
      const extractedAgentType = parts.slice(1, -1).join("-")
      if (extractedAgentType) {
        agentTypeForUrl = extractedAgentType
      }
    } else if (!agentTypeForUrl && parts.length === 2) {
      // Fallback: if only 2 parts, the second might be the agent_type
      agentTypeForUrl = parts[1] || ""
    } else {
      // If it's a single part, it might be the agent_type directly
      if (!agentTypeForUrl) {
        agentTypeForUrl = agentId
      }
    }
    
    if (!agentTypeForUrl) {
      return NextResponse.json(
        { error: "Could not determine agent_type from agentId" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/agents/${encodeURIComponent(agentTypeForUrl)}`,
      {
        method: "DELETE",
        headers: {
          "Accept": "application/json",
          "Authorization": authHeader,
        },
      }
    )

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error deleting agent:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
