function cleanDiagnosticText(value: unknown): string | null {
  if (typeof value !== 'string') {
    return null
  }
  const text = value.trim()
  return text.length > 0 ? text.slice(0, 128) : null
}

function readHttpStatus(metadata: Record<string, unknown> | undefined): number | null {
  const value = metadata?.status_code ?? metadata?.statusCode
  const parsed =
    typeof value === 'number'
      ? value
      : typeof value === 'string' && /^\d{3}$/.test(value.trim())
        ? Number(value)
        : Number.NaN
  return Number.isInteger(parsed) && parsed >= 100 && parsed <= 599 ? parsed : null
}

export function formatChatErrorDiagnostics(
  errorCode: string | undefined,
  metadata: Record<string, unknown> | undefined
): string | undefined {
  const diagnostics: string[] = []
  const cleanErrorCode = cleanDiagnosticText(errorCode)
  const statusCode = readHttpStatus(metadata)
  const backendName = cleanDiagnosticText(metadata?.backend_name ?? metadata?.backendName)

  if (cleanErrorCode) {
    diagnostics.push(cleanErrorCode)
  }
  if (statusCode !== null) {
    diagnostics.push(`HTTP ${statusCode}`)
  }
  if (backendName) {
    diagnostics.push(backendName)
  }

  return diagnostics.length > 0 ? diagnostics.join(' · ') : undefined
}
