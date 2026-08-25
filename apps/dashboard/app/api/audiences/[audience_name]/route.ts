import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Get a single audience by name
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ audience_name: string }> | { audience_name: string } }
) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    // Extract audience_name from params (Next.js 16+ params are a Promise)
    const params = await Promise.resolve(context.params)
    const audienceName = decodeURIComponent(params.audience_name)

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/audience/${encodeURIComponent(audienceName)}`,
      {
        method: "GET",
        headers: {
          "Accept": "application/json",
          "Authorization": authHeader,
        },
      }
    )

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching audience:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

