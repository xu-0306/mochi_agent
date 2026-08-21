export interface GoalExecutionRecordAttempt {
  goalId: string
  attemptId: string
  attemptIndex: number
  agentRunId: string | null | undefined
}

export interface GoalExecutionRecordTargetInput {
  selectedGoalId: string | null | undefined
  snapshotGoalId: string | null | undefined
  currentAttemptId: string | null | undefined
  attempts: readonly GoalExecutionRecordAttempt[]
}

export type GoalExecutionRecordTarget =
  | { kind: 'unavailable'; reason: 'missing_goal' | 'goal_mismatch' | 'missing_run' }
  | { kind: 'agent_run'; runId: string; href: string }

function hasNonblankString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

export function resolveGoalExecutionRecordTarget(
  input: GoalExecutionRecordTargetInput
): GoalExecutionRecordTarget {
  if (!hasNonblankString(input.selectedGoalId)) {
    return { kind: 'unavailable', reason: 'missing_goal' }
  }

  if (input.snapshotGoalId !== input.selectedGoalId) {
    return { kind: 'unavailable', reason: 'goal_mismatch' }
  }

  const eligibleAttempts = input.attempts.filter(
    (attempt) => attempt.goalId === input.selectedGoalId && hasNonblankString(attempt.agentRunId)
  )
  const currentAttempt = eligibleAttempts.find((attempt) => attempt.attemptId === input.currentAttemptId)
  const selectedAttempt =
    currentAttempt ??
    eligibleAttempts.reduce<GoalExecutionRecordAttempt | undefined>((bestAttempt, attempt) => {
      if (!Number.isFinite(attempt.attemptIndex)) {
        return bestAttempt
      }

      if (!bestAttempt || attempt.attemptIndex >= bestAttempt.attemptIndex) {
        return attempt
      }

      return bestAttempt
    }, undefined)

  if (!selectedAttempt || !hasNonblankString(selectedAttempt.agentRunId)) {
    return { kind: 'unavailable', reason: 'missing_run' }
  }

  return {
    kind: 'agent_run',
    runId: selectedAttempt.agentRunId,
    href: `/agent-runs/${encodeURIComponent(selectedAttempt.agentRunId)}`,
  }
}
