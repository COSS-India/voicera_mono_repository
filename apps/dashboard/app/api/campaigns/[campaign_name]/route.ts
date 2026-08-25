import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Get a single campaign by name
export async function GET(
  request: NextRequest,
  context: { params: Promise<{ campaign_name: string }> | { campaign_name: string } }
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

    // Extract campaign_name from params (Next.js 16+ params are a Promise)
    const params = await Promise.resolve(context.params)
    const campaignName = decodeURIComponent(params.campaign_name)

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/campaigns/${encodeURIComponent(campaignName)}`,
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
    console.error("Error fetching campaign:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

