// app/api/meetings/[meeting_id]/recording/route.ts
import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Stream audio recording for a meeting
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
    const params = await Promise.resolve(context.params)
    const meetingId = params?.meeting_id

    if (!meetingId || meetingId === 'undefined' || meetingId.trim() === '') {
      return NextResponse.json(
        { error: "meeting_id parameter is required" },
        { status: 400 }
      )
    }

    // Forward the request to the backend and stream the response
    const response = await fetch(
      `${API_BASE_URL}/api/v1/meetings/${encodeURIComponent(meetingId)}/recording`,
      {
        method: "GET",
        headers: {
          "Authorization": authHeader,
        },
      }
    )

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: "Failed to fetch recording" }))
      return NextResponse.json(errorData, { status: response.status })
    }

    // Get the content type from the backend response
    const contentType = response.headers.get("content-type") || "audio/wav"
    const contentDisposition = response.headers.get("content-disposition")

    // Stream the audio data
    const audioData = await response.arrayBuffer()

    // Return the audio stream with appropriate headers
    return new NextResponse(audioData, {
      status: 200,
      headers: {
        "Content-Type": contentType,
        "Content-Disposition": contentDisposition || `inline; filename="${meetingId}.wav"`,
        "Accept-Ranges": "bytes",
        "Cache-Control": "public, max-age=3600", // Cache for 1 hour
      },
    })
  } catch (error) {
    console.error("Error fetching recording:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
