import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

// POST /api/tts/synthesize
// Accepts multipart FormData, proxies to backend /api/v1/tts/synthesize
// Returns WAV audio bytes with X-Audio-Duration / X-Synth-Time / X-RTF headers
export async function POST(request: NextRequest) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  try {
    const formData = await request.formData()

    const response = await fetch(`${SERVER_API_URL}/api/v1/tts/synthesize`, {
      method: "POST",
      headers: { Authorization: authHeader },
      body: formData,
    })

    if (!response.ok) {
      const err = await response.json().catch(() => ({ detail: response.statusText }))
      return NextResponse.json(err, { status: response.status })
    }

    const audioBuffer = await response.arrayBuffer()
    return new NextResponse(audioBuffer, {
      status: 200,
      headers: {
        "Content-Type": "audio/wav",
        "X-Audio-Duration": response.headers.get("X-Audio-Duration") ?? "",
        "X-Synth-Time":     response.headers.get("X-Synth-Time") ?? "",
        "X-RTF":            response.headers.get("X-RTF") ?? "",
        "Content-Disposition": "attachment; filename=synthesis.wav",
      },
    })
  } catch (error) {
    console.error("TTS synthesize error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
