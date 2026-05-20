import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Fetch agents by org_id
export async function GET(request: NextRequest) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    // Get org_id from query params
    const { searchParams } = new URL(request.url)
    const orgId = searchParams.get("org_id")

    if (!orgId) {
      return NextResponse.json(
        { error: "org_id parameter is required" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
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

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    // Normalize _id to id if present (MongoDB returns _id)
    const normalizedData = Array.isArray(data)
      ? data.map((agent: any) => ({
          ...agent,
          id: agent.id || agent._id,
        }))
      : { ...data, id: data.id || data._id }

    return NextResponse.json(normalizedData)
  } catch (error) {
    console.error("Error fetching agents:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

// POST - Create a new agent
export async function POST(request: NextRequest) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    const body = await request.json()

    // Validate required fields
    if (!body.org_id || !body.agent_type || !body.agent_id || !body.agent_config) {
      return NextResponse.json(
        { error: "org_id, agent_type, agent_id, and agent_config are required" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    const response = await fetch(`${API_BASE_URL}/api/v1/agents`, {
      method: "POST",
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
    console.error("Error creating agent:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

// DELETE - Delete an agent by agent_type (query param)
export async function DELETE(request: NextRequest) {
  try {
    const authHeader = request.headers.get("Authorization")

    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    const { searchParams } = new URL(request.url)
    const agentType = searchParams.get("agent_type")

    if (!agentType) {
      return NextResponse.json(
        { error: "agent_type parameter is required" },
        { status: 400 }
      )
    }

    const response = await fetch(
      `${API_BASE_URL}/api/v1/agents?agent_type=${encodeURIComponent(agentType)}`,
      {
        method: "DELETE",
        headers: {
          "Accept": "application/json",
          "Authorization": authHeader,
        },
      }
    )

    const contentType = response.headers.get("content-type") || ""
    const data = contentType.includes("application/json")
      ? await response.json()
      : { error: await response.text() }

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
