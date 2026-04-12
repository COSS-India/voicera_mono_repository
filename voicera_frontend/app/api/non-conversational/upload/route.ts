import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get("Authorization")
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    const formData = await request.formData()
    const response = await fetch(`${API_BASE_URL}/api/v1/agent-assets/upload`, {
      method: "POST",
      headers: {
        Authorization: authHeader,
      },
      body: formData,
    })

    const data = await response.json().catch(() => ({}))
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error uploading non-conversational audio:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}

