import type { ReasoningStep } from './chat'

const LEGACY_ITERATION_PREFIX = 'Mochi progress: ReAct iteration'

function isRoutineIterationProgress(step: ReasoningStep): boolean {
  if (step.type !== 'status' || step.source !== 'runtime_progress') {
    return false
  }

  if (step.toolMeta?.kind === 'react_iteration_progress') {
    return true
  }

  return step.content.trimStart().startsWith(LEGACY_ITERATION_PREFIX)
}

export function compactReasoningStepsForDisplay(
  steps: ReasoningStep[],
  options: { isStreaming: boolean }
): ReasoningStep[] {
  let latestIterationIndex = -1

  if (options.isStreaming) {
    for (let index = steps.length - 1; index >= 0; index -= 1) {
      if (isRoutineIterationProgress(steps[index])) {
        latestIterationIndex = index
        break
      }
    }
  }

  return steps.filter(
    (step, index) =>
      !isRoutineIterationProgress(step) ||
      index === latestIterationIndex
  )
}
