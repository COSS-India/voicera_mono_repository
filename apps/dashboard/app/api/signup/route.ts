import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

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

    const response = await fetch(`${API_BASE_URL}/api/v1/users/signup`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
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
