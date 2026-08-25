import { NextResponse } from "next/server"

const VOICE_SERVER_URL = process.env.VOICE_SERVER_URL || "http://localhost:7860"

export async function GET() {
  try {
    const response = await fetch(`${VOICE_SERVER_URL}/telemetry/gpu`, {
      method: "GET",
      headers: { Accept: "application/json" },
      cache: "no-store",
    })

    const data = await response.json()
    return NextResponse.json(data, { status: response.status })
  } catch {
    return NextResponse.json(
      {
        status: "error",
        detail: "Failed to fetch telemetry from voice server",
      },
      { status: 500 },
    )
  }
}
