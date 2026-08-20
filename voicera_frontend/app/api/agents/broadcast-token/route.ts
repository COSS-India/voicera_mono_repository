import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

// POST - Mint a short-lived host broadcast token for a translation agent.
export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("Authorization")
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 },
      )
    }

    const body = await request.json()
    const agentType = typeof body?.agent_type === "string" ? body.agent_type.trim() : ""
    if (!agentType) {
      return NextResponse.json({ error: "agent_type is required" }, { status: 400 })
    }

    const response = await fetch(
      `${API_BASE_URL}/api/v1/agents/${encodeURIComponent(agentType)}/broadcast-token`,
      {
        method: "POST",
        headers: {
          Accept: "application/json",
          "Content-Type": "application/json",
          Authorization: authHeader,
        },
      },
    )

    const data = await response.json()
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error minting broadcast token:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
