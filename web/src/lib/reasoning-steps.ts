import type { ReasoningStep } from './chat'

interface BuildReasoningStepIdInput {
  phase: string
  turnKey: string | null
  timestamp?: string
  index: number
  toolCallId?: string
  content?: string
}

function normalizeIdPart(value: string): string {
  return value
    .trim()
    .toLowerCase()
    .replace(/\s+/g, ' ')
}

function hashText(value: string): string {
  let hash = 2166136261

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 16777619)
  }

  return (hash >>> 0).toString(16)
}

export function buildReasoningStepId({
  phase,
  turnKey,
  timestamp,
  index,
  toolCallId,
  content,
}: BuildReasoningStepIdInput): string {
  const normalizedTurnKey = turnKey ?? 'na'
  const normalizedPhase = normalizeIdPart(phase) || 'unknown'
  const normalizedToolCallId = toolCallId ? normalizeIdPart(toolCallId) : ''

  if (normalizedToolCallId) {
    return ['reasoning', normalizedPhase, normalizedTurnKey, normalizedToolCallId].join('-')
  }

  if (timestamp) {
    return ['reasoning', normalizedPhase, normalizedTurnKey, timestamp, String(index)].join('-')
  }

  const normalizedContent = content ? normalizeIdPart(content) : ''
  const contentFingerprint = normalizedContent ? hashText(normalizedContent) : 'now'

  return ['reasoning', normalizedPhase, normalizedTurnKey, contentFingerprint, String(index)].join('-')
}

export function mergeReasoningStep(steps: ReasoningStep[], nextStep: ReasoningStep): ReasoningStep[] {
  if (nextStep.toolCallId) {
    const existingIndex = steps.findIndex(
      (step) =>
        step.toolCallId === nextStep.toolCallId &&
        (step.type === 'tool_call' || step.type === 'tool_result')
    )

    if (existingIndex !== -1) {
      const existing = steps[existingIndex]
      const merged: ReasoningStep =
        nextStep.type === 'tool_result'
          ? {
              ...existing,
              ...nextStep,
              type: 'tool_result',
              status: nextStep.status,
            }
          : { ...existing, ...nextStep }

      return steps.map((step, index) => (index === existingIndex ? merged : step))
    }
  }

  const duplicateIndex = steps.findIndex((step) => step.id === nextStep.id)
  if (duplicateIndex !== -1) {
    return steps.map((step, index) => (
      index === duplicateIndex
        ? {
            ...step,
            ...nextStep,
        }
        : step
    ))
  }

  const previous = steps.at(-1)
  if (
    previous &&
    !previous.toolCallId &&
    !nextStep.toolCallId &&
    previous.type === nextStep.type &&
    previous.content.trim() === nextStep.content.trim() &&
    (previous.source ?? null) === (nextStep.source ?? null) &&
    (previous.errorCode ?? null) === (nextStep.errorCode ?? null) &&
    (previous.toolMeta?.raw_phase ?? null) === (nextStep.toolMeta?.raw_phase ?? null)
  ) {
    return [...steps.slice(0, -1), { ...previous, ...nextStep }]
  }

  return [...steps, nextStep]
}
