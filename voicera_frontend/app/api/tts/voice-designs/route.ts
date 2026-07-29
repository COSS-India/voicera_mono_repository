import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

export async function GET(request: NextRequest) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }
  try {
    const response = await fetch(`${SERVER_API_URL}/api/v1/tts/voice-designs`, {
      headers: { Authorization: authHeader, Accept: "application/json" },
    })
    const data = await response.json().catch(() => [])
    return NextResponse.json(data, { status: response.status })
  } catch {
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
