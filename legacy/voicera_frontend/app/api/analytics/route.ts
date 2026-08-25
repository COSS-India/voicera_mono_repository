// app/api/analytics/route.ts
import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Fetch analytics metrics
export async function GET(request: NextRequest) {
  try {
    // Get the authorization header from the request
    const authHeader = request.headers.get("Authorization")
    
    if (!authHeader) {
      return NextResponse.json(
        { error: "Authorization header is required" },
        { status: 401 }
      )
    }

    // Get query params (optional filters)
    const { searchParams } = new URL(request.url)
    const agentType = searchParams.get("agent_type")
    const phoneNumber = searchParams.get("phone_number")
    const startDate = searchParams.get("start_date")
    const endDate = searchParams.get("end_date")

    // Build the backend URL with query params
    const params = new URLSearchParams()
    if (agentType) params.append("agent_type", agentType)
    if (phoneNumber) params.append("phone_number", phoneNumber)
    if (startDate) params.append("start_date", startDate)
    if (endDate) params.append("end_date", endDate)
    
    const queryString = params.toString()
    const backendUrl = `${API_BASE_URL}/api/v1/analytics${queryString ? `?${queryString}` : ""}`

    // Forward the request to the backend
    const response = await fetch(backendUrl, {
      method: "GET",
      headers: {
        "Accept": "application/json",
        "Authorization": authHeader,
      },
    })

    const data = await response.json()

    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching analytics:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
