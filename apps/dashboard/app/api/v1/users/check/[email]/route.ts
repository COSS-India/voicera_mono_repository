import { NextRequest, NextResponse } from "next/server"
import { SERVER_API_URL } from "@/lib/api-config"

const API_BASE_URL = SERVER_API_URL

export async function GET(
  request: NextRequest,
  { params }: { params: Promise<{ email: string }> }
) {
  try {
    const { email } = await params
    const orgId = request.nextUrl.searchParams.get("org_id")
    const query = orgId ? `?org_id=${encodeURIComponent(orgId)}` : ""

    const response = await fetch(
      `${API_BASE_URL}/api/v1/users/check/${encodeURIComponent(email)}${query}`,
      {
        method: "GET",
        headers: {
          Accept: "application/json",
        },
      }
    )

    const data = await response.json()
    if (!response.ok) {
      return NextResponse.json(data, { status: response.status })
    }
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error checking join eligibility:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
