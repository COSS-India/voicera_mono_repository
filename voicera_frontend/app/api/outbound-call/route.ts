import { NextRequest, NextResponse } from "next/server"

// Voice server URL - use Docker service name in container, localhost for local dev
const VOICE_SERVER_URL = process.env.VOICE_SERVER_URL || "http://localhost:7860"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const response = await fetch(`${VOICE_SERVER_URL}/outbound/call/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })
    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    const message =
      error instanceof Error && error.message.includes("fetch")
        ? `Cannot reach voice server at ${VOICE_SERVER_URL}`
        : "Internal server error"
    return NextResponse.json({ error: message }, { status: 503 })
  }
}