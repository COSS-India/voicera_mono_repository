import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

const PAGINATION_PARAMS = [
  "page",
  "limit",
  "for_export",
  "agent_type",
  "from_number",
  "to_number",
  "inbound",
  "call_status",
  "date_from",
  "date_to",
  "date_sort_order",
  "duration_sort_order",
  "has_latency_metrics",
  "search",
] as const

function normalizeMeeting(meeting: Record<string, unknown>) {
  return {
    ...meeting,
    id: meeting.id || meeting._id,
  }
}

// GET - Fetch paginated meetings
export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get("Authorization")

    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    const { searchParams } = new URL(request.url)
    const backendParams = new URLSearchParams()

    for (const key of PAGINATION_PARAMS) {
      const value = searchParams.get(key)
      if (value !== null && value !== "") {
        backendParams.set(key, value)
      }
    }

    if (!backendParams.has("page")) {
      backendParams.set("page", "1")
    }
    if (!backendParams.has("limit")) {
      backendParams.set("limit", "50")
    }

    const backendUrl = `${API_BASE_URL}/api/v1/meetings?${backendParams.toString()}`

    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        Accept: "application/json",
        Authorization: authHeader,
      },
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    if (data && Array.isArray(data.items)) {
      return NextResponse.json({
        ...data,
        items: data.items.map((m: Record<string, unknown>) => normalizeMeeting(m)),
      })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching meetings:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
