import type { ReasoningStep } from '@/lib/chat'

export function getReasoningStepBadge(step: Pick<ReasoningStep, 'source'>): string | null {
  if (step.source === 'model_summary') {
    return 'model summary'
  }
  if (step.source === 'runtime_progress') {
    return 'runtime progress'
  }
  return null
}
