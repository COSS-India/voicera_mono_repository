import { NextRequest, NextResponse } from "next/server"

import { SERVER_API_URL } from "@/lib/api-config"
const API_BASE_URL = SERVER_API_URL

// DELETE - Delete a Vobiz application
export async function DELETE(
  request: NextRequest,
  context: { params: Promise<{ application_id: string }> | { application_id: string } }
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

    // Extract application_id from params (Next.js 16+ params are a Promise)
    const params = await Promise.resolve(context.params)
    const application_id = decodeURIComponent(params.application_id)

    if (!application_id) {
      return NextResponse.json(
        { error: "application_id parameter is required" },
        { status: 400 }
      )
    }

    // Forward the request to the backend
    const response = await fetch(
      `${API_BASE_URL}/api/v1/vobiz/application/${encodeURIComponent(application_id)}`,
      {
        method: "DELETE",
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
    console.error("Error deleting Vobiz application:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
