import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Fetch a single meeting by meeting_id
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ meeting_id: string }> | { meeting_id: string } }
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

    // Extract meeting_id from params
    // In Next.js 15+, params are a Promise and must be awaited
    const params = await Promise.resolve(context.params)
    const meetingId = params?.meeting_id

    if (!meetingId || meetingId === 'undefined' || meetingId.trim() === '') {
      console.error("Missing meeting_id parameter. Params received:", params)
      return NextResponse.json(
        { error: "meeting_id parameter is required", detail: `Received params: ${JSON.stringify(params)}` },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/meetings/${encodeURIComponent(meetingId)}`,
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

    // Normalize _id to id if present (MongoDB returns _id)
    const normalizedData = {
      ...data,
      id: data.id || data._id,
    }

    return NextResponse.json(normalizedData)
  } catch (error) {
    console.error("Error fetching meeting:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
