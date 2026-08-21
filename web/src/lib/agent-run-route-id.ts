/**
 * Converts the single dynamic route segment supplied by Next into the raw
 * Agent Run identifier expected by the API client. The client owns URL
 * encoding, so this boundary deliberately decodes at most once.
 */
export function normalizeAgentRunRouteId(routeRunId: string | readonly string[] | undefined): string | undefined {
  const value = Array.isArray(routeRunId) ? routeRunId[0] : routeRunId

  if (typeof value !== 'string') {
    return undefined
  }

  try {
    return decodeURIComponent(value)
  } catch {
    // Preserve malformed input so the API can handle/report the supplied ID
    // without the detail page crashing during route rendering.
    return value
  }
}