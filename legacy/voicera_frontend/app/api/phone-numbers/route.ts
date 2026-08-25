import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// GET - Get all phone numbers for an organization
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

    // Get org_id from query params
    const { searchParams } = new URL(request.url)
    const orgId = searchParams.get("org_id")

    if (!orgId) {
      return NextResponse.json(
        { error: "org_id parameter is required" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/phone-numbers/org/${orgId}`,
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
    const normalizedData = Array.isArray(data)
      ? data.map((phoneNumber: any) => ({
          ...phoneNumber,
          id: phoneNumber.id || phoneNumber._id,
        }))
      : { ...data, id: data.id || data._id }

    return NextResponse.json(normalizedData)
  } catch (error) {
    console.error("Error fetching phone numbers:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}