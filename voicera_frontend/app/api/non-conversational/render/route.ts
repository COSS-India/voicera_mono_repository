import { NextRequest, NextResponse } from "next/server"

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("Authorization")
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    const voiceServerUrl =
      process.env.NEXT_PUBLIC_JOHNAIC_SERVER_URL || "http://localhost:7860"
    const internalApiKey = process.env.INTERNAL_API_KEY || ""
    const body = await request.json()

    const response = await fetch(
      `${voiceServerUrl.replace(/\/$/, "")}/internal/render-tts-wav`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          ...(internalApiKey ? { "X-API-Key": internalApiKey } : {}),
        },
        body: JSON.stringify(body),
      }
    )

    if (!response.ok) {
      const err = await response.json().catch(() => ({}))
      return NextResponse.json(
        { error: (err as { detail?: string; error?: string }).detail || (err as { error?: string }).error || "Failed to render TTS audio" },
        { status: response.status }
      )
    }

    const audioData = await response.arrayBuffer()
    return new NextResponse(audioData, {
      status: 200,
      headers: {
        "Content-Type": "audio/wav",
        "Cache-Control": "no-store",
      },
    })
  } catch (error) {
    console.error("Error rendering non-conversational TTS:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

