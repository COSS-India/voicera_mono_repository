/**
 * API configuration for server-side and client-side usage
 * 
 * Server-side (Next.js API routes): Use API_URL (Docker service name) or fallback to NEXT_PUBLIC_API_URL
 * Client-side: Use NEXT_PUBLIC_API_URL (public URL through proxy/router)
 */

// For server-side API routes - can use Docker service name
export const SERVER_API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"