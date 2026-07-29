/**
 * Turn technical Knowledge Base / OpenAI errors into plain language for end users.
 */
export function formatKnowledgeBaseError(
  message: string | null | undefined
): string | null {
  if (!message?.trim()) return null

  const lower = message.toLowerCase()

  if (
    lower.includes("429") ||
    lower.includes("quota") ||
    lower.includes("rate limit") ||
    lower.includes("rate_limit") ||
    lower.includes("exceeded your current quota") ||
    lower.includes("insufficient_quota")
  ) {
    return (
      "Your OpenAI account has reached its usage limit. " +
      "Check your OpenAI billing or plan, then try uploading again."
    )
  }

  if (
    lower.includes("401") ||
    lower.includes("invalid api key") ||
    lower.includes("incorrect api key") ||
    lower.includes("invalid_api_key")
  ) {
    return (
      "Your OpenAI API key is not valid. " +
      "Update it in Integrations, then try uploading again."
    )
  }

  if (lower.startsWith("error code:")) {
    return "We couldn't process this file. Please try again later."
  }

  return message
}
