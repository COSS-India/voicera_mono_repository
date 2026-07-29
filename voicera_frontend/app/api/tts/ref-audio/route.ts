import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

// GET  /api/tts/ref-audio  — list stored reference audios
// GET  /api/tts/ref-audio?key=...  — download one file
export async function GET(request: NextRequest) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  // Optional: download a single file via ?key= (avoids .wav path issues)
  const key = request.nextUrl.searchParams.get("key")
  if (key) {
    try {
      const response = await fetch(
        `${SERVER_API_URL}/api/v1/tts/ref-audio/${encodeURIComponent(key)}`,
        { headers: { Authorization: authHeader } }
      )
      if (!response.ok) {
        return NextResponse.json({ error: "Not found" }, { status: response.status })
      }
      const buffer = await response.arrayBuffer()
      return new NextResponse(buffer, {
        status: 200,
        headers: {
          "Content-Type": "audio/wav",
          "Content-Disposition": response.headers.get("Content-Disposition") ?? "",
        },
      })
    } catch (error) {
      console.error("TTS ref-audio GET error:", error)
      return NextResponse.json({ error: "Internal server error" }, { status: 500 })
    }
  }

  try {
    const response = await fetch(`${SERVER_API_URL}/api/v1/tts/ref-audio`, {
      headers: { Authorization: authHeader, Accept: "application/json" },
    })
    const data = await response.json().catch(() => [])
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("TTS ref-audio list error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// POST /api/tts/ref-audio  — upload a new reference audio file
export async function POST(request: NextRequest) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  try {
    const formData = await request.formData()
    const response = await fetch(`${SERVER_API_URL}/api/v1/tts/ref-audio`, {
      method: "POST",
      headers: { Authorization: authHeader },
      body: formData,
    })
    const data = await response.json().catch(() => ({}))
    return NextResponse.json(data, { status: response.status })
  } catch (error) {
    console.error("TTS ref-audio upload error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

// DELETE /api/tts/ref-audio?key=...  — delete a reference audio
// Query param avoids Next.js treating ".wav" paths as static files.
export async function DELETE(request: NextRequest) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  const key = request.nextUrl.searchParams.get("key")
  if (!key) {
    return NextResponse.json({ error: "key is required" }, { status: 400 })
  }

  try {
    const response = await fetch(
      `${SERVER_API_URL}/api/v1/tts/ref-audio/${encodeURIComponent(key)}`,
      { method: "DELETE", headers: { Authorization: authHeader } }
    )
    if (!response.ok && response.status !== 204) {
      const err = await response.json().catch(() => ({}))
      return NextResponse.json(err, { status: response.status })
    }
    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error("TTS ref-audio DELETE error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
