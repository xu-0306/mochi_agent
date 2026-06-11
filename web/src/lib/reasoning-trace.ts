export function getReasoningStepSource(
  metadata: Record<string, unknown> | undefined
): string | undefined {
  return typeof metadata?.source === 'string' ? metadata.source : undefined
}
