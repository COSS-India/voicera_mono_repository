import { NextRequest, NextResponse } from "next/server"

// Direct Telephony Server URL
const VOICE_SERVER_URL = "https://bhashini-voicera-telephony.bhashini.co.in"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${VOICE_SERVER_URL}/outbound/call/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    const data = await response.json()

    return NextResponse.json(data, { status: response.status })

  } catch (error) {
    console.error("Telephony call error:", error)

    return NextResponse.json(
      { error: "Telephony server not reachable" },
      { status: 500 }
    )
  }
}