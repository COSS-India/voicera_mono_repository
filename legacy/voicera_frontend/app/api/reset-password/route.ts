import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { token, new_password } = body

    if (!token || !new_password) {
      return NextResponse.json(
        { detail: "Token and new password are required" },
        { status: 400 }
      )
    }

    const response = await fetch(`${API_BASE_URL}/api/v1/users/reset-password`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ token, new_password }),
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(
        { detail: data.detail || "Failed to reset password" },
        { status: response.status }
      )
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error during reset-password:", error)
    return NextResponse.json(
      { detail: "Internal server error" },
      { status: 500 }
    )
  }
}
