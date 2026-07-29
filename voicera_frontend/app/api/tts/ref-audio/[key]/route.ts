import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

// GET    /api/tts/ref-audio/[key]  — download a reference audio file
// DELETE /api/tts/ref-audio/[key]  — delete a reference audio file
export async function GET(
  request: NextRequest,
  { params }: { params: { key: string } }
) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  try {
    const response = await fetch(
      `${SERVER_API_URL}/api/v1/tts/ref-audio/${encodeURIComponent(params.key)}`,
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

export async function DELETE(
  request: NextRequest,
  { params }: { params: { key: string } }
) {
  const authHeader = request.headers.get("Authorization")
  if (!authHeader) {
    return NextResponse.json({ error: "Authorization required" }, { status: 401 })
  }

  try {
    const response = await fetch(
      `${SERVER_API_URL}/api/v1/tts/ref-audio/${encodeURIComponent(params.key)}`,
      { method: "DELETE", headers: { Authorization: authHeader } }
    )
    if (!response.ok) {
      const err = await response.json().catch(() => ({}))
      return NextResponse.json(err, { status: response.status })
    }
    return new NextResponse(null, { status: 204 })
  } catch (error) {
    console.error("TTS ref-audio DELETE error:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
