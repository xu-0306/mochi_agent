import type { ReasoningStep, TokenStats } from '@/lib/chat'

export interface ReasoningPanelSummary {
  title: string
  detail: string
  hasError: boolean
}

interface SummaryOptions {
  steps: Array<Pick<ReasoningStep, 'type' | 'toolName' | 'content' | 'status'>>
  isStreaming: boolean
  generationTimeMs?: number
}

interface NextOpenOptions {
  previousOpen: boolean
  previousStreaming: boolean
  isStreaming: boolean
  userInteracted: boolean
}

function countUniqueTools(
  steps: Array<Pick<ReasoningStep, 'type' | 'toolName'>>
): number {
  return new Set(
    steps
      .filter((step) => (step.type === 'tool_call' || step.type === 'tool_result') && step.toolName)
      .map((step) => step.toolName)
  ).size
}

export function deriveReasoningPanelSummary({
  steps,
  isStreaming,
  generationTimeMs,
}: SummaryOptions): ReasoningPanelSummary {
  const latestTool = [...steps].reverse().find((step) => step.toolName)?.toolName
  const toolCount = countUniqueTools(steps)
  const hasError = steps.some((step) => step.type === 'error' || step.status === 'error')

  if (isStreaming) {
    return {
      title: 'Thinking…',
      detail: latestTool ? `${steps.length} steps, latest: ${latestTool}` : `${steps.length} steps`,
      hasError,
    }
  }

  const duration = typeof generationTimeMs === 'number' && Number.isFinite(generationTimeMs)
    ? `Thought for ${(generationTimeMs / 1000).toFixed(1)}s`
    : 'Reasoning trace'

  const detailParts = [`${steps.length} steps`]
  if (toolCount > 0) {
    detailParts.push(`${toolCount} tool${toolCount === 1 ? '' : 's'}`)
  }
  if (hasError) {
    detailParts.push('includes issue')
  }

  return {
    title: duration,
    detail: detailParts.join(' · '),
    hasError,
  }
}

export function getNextReasoningPanelOpen({
  previousOpen,
  previousStreaming,
  isStreaming,
  userInteracted,
}: NextOpenOptions): boolean {
  if (isStreaming) {
    return true
  }

  if (previousStreaming && !isStreaming && !userInteracted) {
    return false
  }

  return previousOpen
}

export function resolveReasoningGenerationTime(
  tokenStats?: TokenStats
): number | undefined {
  if (!tokenStats) {
    return undefined
  }
  return tokenStats.generationTimeMs
}
