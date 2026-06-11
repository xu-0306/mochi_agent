import type { ReasoningStep, TokenStats } from '@/lib/chat'

export interface ReasoningPanelSummary {
  title: string
  detail: string
  hasError: boolean
  latestIssue: string | null
}

interface SummaryOptions {
  steps: Array<Pick<ReasoningStep, 'type' | 'toolName' | 'content' | 'status' | 'errorCode'>>
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

function summarizeIssue(
  steps: Array<Pick<ReasoningStep, 'type' | 'content' | 'status' | 'errorCode'>>
): string | null {
  const issueStep = [...steps].reverse().find(
    (step) => step.type === 'error' || step.status === 'error'
  )
  if (!issueStep) {
    return null
  }

  const firstLine = issueStep.content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .find((line) => line.length > 0)

  const summary = firstLine ?? issueStep.errorCode ?? 'Unknown issue'
  if (summary.length <= 96) {
    return summary
  }
  return `${summary.slice(0, 93)}...`
}

export function deriveReasoningPanelSummary({
  steps,
  isStreaming,
  generationTimeMs,
}: SummaryOptions): ReasoningPanelSummary {
  const latestTool = [...steps].reverse().find((step) => step.toolName)?.toolName
  const toolCount = countUniqueTools(steps)
  const latestIssue = summarizeIssue(steps)
  const hasError = latestIssue !== null
  const visibleStepCount = steps.filter((step) => step.type !== 'status').length
  const progressCount = steps.length - visibleStepCount

  if (isStreaming) {
    return {
      title: 'Thinking',
      detail: latestTool
        ? `${visibleStepCount} steps, latest: ${latestTool}`
        : `${visibleStepCount} steps${progressCount > 0 ? `, ${progressCount} progress` : ''}`,
      hasError,
      latestIssue,
    }
  }

  const duration = typeof generationTimeMs === 'number' && Number.isFinite(generationTimeMs)
    ? `Thought for ${(generationTimeMs / 1000).toFixed(1)}s`
    : 'Reasoning trace'

  const detailParts = [`${visibleStepCount} steps`]
  if (toolCount > 0) {
    detailParts.push(`${toolCount} tool${toolCount === 1 ? '' : 's'}`)
  }
  if (progressCount > 0) {
    detailParts.push(`${progressCount} progress`)
  }
  if (latestIssue) {
    detailParts.push(`issue: ${latestIssue}`)
  }

  return {
    title: duration,
    detail: detailParts.join(' · '),
    hasError,
    latestIssue,
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
