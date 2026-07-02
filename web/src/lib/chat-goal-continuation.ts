import type { GoalHealthSummary } from '@/lib/api'

export type GoalContinuationResumeStrategy = 'continue_from_checkpoint' | 'restart_attempt'

export type GoalContinuationAction =
  | 'forward_guidance'
  | 'resume_then_forward'
  | 'refresh_then_forward'
  | 'wake_after_resolution'
  | 'manual_resolution_required'
  | 'restart_attempt_with_context'
  | 'no_live_attempt_recoverable'
  | 'blocked'

export interface GoalContinuationDecision {
  action: GoalContinuationAction
  summary: string
  blocking: boolean
  resumeAfterForwarding: boolean
  approvalIds: string[]
  runId: string | null
  blockerType: string | null
  toolNames: string[]
  recommendedAction: string | null
  resumeStrategy: GoalContinuationResumeStrategy | null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function getStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .map((item) => (typeof item === 'string' ? item.trim() : ''))
        .filter((item) => item.length > 0)
    : []
}

function normalizeStatus(value: string | null | undefined): string {
  return (value ?? '').trim().toLowerCase()
}

function isGoalTerminalStatus(status: string | null | undefined): boolean {
  const normalized = normalizeStatus(status)
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'failed' ||
    normalized === 'error'
  )
}

function isLinkedRunTerminalStatus(status: string | null | undefined): boolean {
  const normalized = normalizeStatus(status)
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'failed' ||
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'partial'
  )
}

function isApprovalWaitingStatus(status: string | null | undefined): boolean {
  const normalized = normalizeStatus(status)
  return normalized === 'waiting_approval' || normalized === 'awaiting_approval'
}

function hasLiveLinkedRun(
  linkedRunId: string | null,
  linkedRunStatus: string | null
): boolean {
  return Boolean(linkedRunId && !isLinkedRunTerminalStatus(linkedRunStatus))
}

function isAttemptRecordCompatible(
  recordAttemptId: string | null,
  currentAttemptId: string | null
): boolean {
  if (!recordAttemptId || !currentAttemptId) {
    return true
  }
  return recordAttemptId === currentAttemptId
}

function hasDurableGoalRecoveryContext(health: GoalHealthSummary): boolean {
  if (!health.persisted_checkpoint || !health.memory_snapshot) {
    return false
  }

  const currentAttemptId =
    getString(health.current_attempt?.attempt_id) ??
    getString(health.current_attempt_id)
  const checkpointAttemptId = getString(health.persisted_checkpoint.attempt_id)
  const snapshotAttemptId = getString(health.memory_snapshot.attempt_id)

  if (!isAttemptRecordCompatible(checkpointAttemptId, currentAttemptId)) {
    return false
  }
  if (!isAttemptRecordCompatible(snapshotAttemptId, currentAttemptId)) {
    return false
  }
  if (checkpointAttemptId && snapshotAttemptId && checkpointAttemptId !== snapshotAttemptId) {
    return false
  }

  return true
}

function preferredResumeStrategy(
  health: GoalHealthSummary
): GoalContinuationResumeStrategy {
  return hasDurableGoalRecoveryContext(health)
    ? 'continue_from_checkpoint'
    : 'restart_attempt'
}

function buildDefaultContinuationSummary(
  health: GoalHealthSummary,
  linkedRunId: string | null,
  linkedRunStatus: string | null,
  approvalIds: string[]
): string {
  if (
    approvalIds.length > 0 ||
    isApprovalWaitingStatus(health.status) ||
    isApprovalWaitingStatus(getString(health.approval_state?.status))
  ) {
    return 'Goal is waiting on operator approval before it can continue.'
  }

  if (!isGoalTerminalStatus(health.status) && (!linkedRunId || isLinkedRunTerminalStatus(linkedRunStatus))) {
    return 'The current goal attempt is no longer actively running, so a follow-up should reopen or advance the goal before forwarding guidance.'
  }

  const normalizedGoalStatus = normalizeStatus(health.status)
  if (normalizedGoalStatus === 'paused' || normalizedGoalStatus === 'stalled' || normalizedGoalStatus === 'awaiting_resources') {
    return 'Goal is resumable but not actively progressing.'
  }

  if (normalizedGoalStatus === 'running' || normalizedGoalStatus === 'queued') {
    return 'Goal is actively progressing and can take follow-up guidance now.'
  }

  return 'Forward the follow-up to the active goal run and keep the live timeline attached.'
}

export function resolveGoalContinuationDecision(health: GoalHealthSummary): GoalContinuationDecision {
  const recommendation = health.recommended_next_action
  const recommendedAction = getString(recommendation?.action)
  const linkedRunId =
    getString(health.linked_agent_run?.run_id) ??
    health.current_attempt?.agent_run_id ??
    null
  const linkedRunStatus = getString(health.linked_agent_run?.status)
  const liveLinkedRun = hasLiveLinkedRun(linkedRunId, linkedRunStatus)
  const resumeStrategy = preferredResumeStrategy(health)
  const approvalIds = getStringArray(recommendation?.approval_ids ?? health.approval_state?.approval_ids)
  const recommendedSummary =
    getString(recommendation?.summary) ??
    buildDefaultContinuationSummary(health, linkedRunId, linkedRunStatus, approvalIds)

  if (
    recommendedAction === 'resolve_approval' ||
    (!recommendedAction &&
      (
        approvalIds.length > 0 ||
        isApprovalWaitingStatus(getString(health.approval_state?.status)) ||
        isApprovalWaitingStatus(health.status)
      ))
  ) {
    return {
      action: liveLinkedRun ? 'wake_after_resolution' : 'manual_resolution_required',
      summary: liveLinkedRun
        ? 'Goal is waiting on operator approval, but the current attempt can queue your follow-up guidance and resume with it once approval is resolved.'
        : recommendedSummary,
      blocking: true,
      resumeAfterForwarding: false,
      approvalIds,
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType) ??
        'approval',
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames ?? health.approval_state?.tool_names),
      recommendedAction,
      resumeStrategy,
    }
  }

  if (recommendedAction === 'inspect_runtime_budget') {
    return {
      action: 'blocked',
      summary: recommendedSummary,
      blocking: true,
      resumeAfterForwarding: false,
      approvalIds: [],
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType) ??
        'runtime_budget',
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
      recommendedAction,
      resumeStrategy: null,
    }
  }

  if (recommendedAction === 'refresh_worker_generation') {
    return {
      action: 'refresh_then_forward',
      summary: recommendedSummary,
      blocking: false,
      resumeAfterForwarding: true,
      approvalIds: [],
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType),
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
      recommendedAction,
      resumeStrategy: null,
    }
  }

  if (recommendedAction === 'resume_goal') {
    if (!liveLinkedRun) {
      return {
        action:
          resumeStrategy === 'continue_from_checkpoint'
            ? 'no_live_attempt_recoverable'
            : 'restart_attempt_with_context',
        summary:
          resumeStrategy === 'continue_from_checkpoint'
            ? 'The current goal attempt is no longer actively running, but the goal still has durable recovery context and can reopen from its saved checkpoint.'
            : 'The current goal attempt is no longer actively running, so the follow-up should restart a fresh attempt with the latest available goal context.',
        blocking: false,
        resumeAfterForwarding: false,
        approvalIds: [],
        runId: linkedRunId,
        blockerType:
          getString(recommendation?.blocker_type) ??
          getString(recommendation?.blockerType),
        toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
        recommendedAction,
        resumeStrategy,
      }
    }

    return {
      action: 'resume_then_forward',
      summary: recommendedSummary,
      blocking: false,
      resumeAfterForwarding: true,
      approvalIds: [],
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType),
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
      recommendedAction,
      resumeStrategy,
    }
  }

  if (recommendedAction === 'monitor' || recommendedAction === 'capture_checkpoint' || recommendedAction === 'inspect_collector_shards') {
    return {
      action: 'forward_guidance',
      summary: recommendedSummary,
      blocking: false,
      resumeAfterForwarding: false,
      approvalIds: [],
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType),
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
      recommendedAction,
      resumeStrategy: null,
    }
  }

  if (!isGoalTerminalStatus(health.status) && (!linkedRunId || isLinkedRunTerminalStatus(linkedRunStatus))) {
    return {
      action:
        resumeStrategy === 'continue_from_checkpoint'
          ? 'no_live_attempt_recoverable'
          : 'restart_attempt_with_context',
      summary:
        resumeStrategy === 'continue_from_checkpoint'
          ? 'The current goal attempt is no longer actively running, but the goal still has durable recovery context and can reopen from its saved checkpoint.'
          : 'The current goal attempt is no longer actively running, so a follow-up should restart a fresh attempt with the latest available goal context before forwarding guidance.',
      blocking: false,
      resumeAfterForwarding: false,
      approvalIds: [],
      runId: linkedRunId,
      blockerType:
        getString(recommendation?.blocker_type) ??
        getString(recommendation?.blockerType),
      toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
      recommendedAction,
      resumeStrategy,
    }
  }

  return {
    action: 'forward_guidance',
    summary: recommendedSummary,
    blocking: false,
    resumeAfterForwarding: false,
    approvalIds: [],
    runId: linkedRunId,
    blockerType:
      getString(recommendation?.blocker_type) ??
      getString(recommendation?.blockerType),
    toolNames: getStringArray(recommendation?.tool_names ?? recommendation?.toolNames),
    recommendedAction,
    resumeStrategy: null,
  }
}
