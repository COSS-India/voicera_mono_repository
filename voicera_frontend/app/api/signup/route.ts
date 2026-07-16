import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

// Resolve the real client IP from a header set by a trusted edge that overwrites
// it outright, never one a client can seed by simply sending it (X-Forwarded-For
// is appended-to, not overwritten, so its first entry is attacker-controlled).
// The sandbox's Cloudflare Tunnel sets CF-Connecting-IP; nginx sets X-Real-IP when
// it's the front door instead (e.g. a docker-compose deployment). Falls back to ""
// if neither is present — the backend already treats that as "no real IP known"
// rather than trusting anything client-suppliable.
function resolveClientIp(request: NextRequest): string {
  return (
    request.headers.get("cf-connecting-ip") ||
    request.headers.get("x-real-ip") ||
    ""
  )
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { name, email, password, company_name } = body

    if (!name || !email || !password || !company_name) {
      return NextResponse.json(
        { detail: "Name, email, password, and company name are required" },
        { status: 400 }
      )
    }

    const clientIp = resolveClientIp(request)
    const headers: Record<string, string> = {
      "Content-Type": "application/json",
    }
    if (clientIp) {
      headers["X-Real-IP"] = clientIp
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/users/signup`, {
      method: "POST",
      headers,
      body: JSON.stringify({ name, email, password, company_name }),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(
        { detail: data.detail || "Signup failed" },
        { status: response.status }
      )
    }

    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("Error during signup:", error)
    return NextResponse.json(
      { detail: "Internal server error" },
      { status: 500 }
    )
  }
}
