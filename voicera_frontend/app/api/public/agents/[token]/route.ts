import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

// GET - Resolve a public share_token to a secret-stripped agent projection.
// Unauthenticated by design: the backend gates this on the per-agent opt-in flag.
export async function GET(
  _request: NextRequest,
  context: { params: Promise<{ token: string }> | { token: string } },
) {
  try {
    const params = await Promise.resolve(context.params)
    const token = decodeURIComponent(params.token)

    const response = await fetch(
      `${API_BASE_URL}/api/v1/public/agents/${encodeURIComponent(token)}`,
      { method: "GET", headers: { Accept: "application/json" } },
    )

    const data = await response.json()
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error resolving public agent:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
