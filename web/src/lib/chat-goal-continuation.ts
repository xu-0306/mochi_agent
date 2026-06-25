import type { GoalHealthSummary } from '@/lib/api'

export type GoalContinuationAction =
  | 'forward_guidance'
  | 'resume_then_forward'
  | 'refresh_then_forward'
  | 'manual_resolution_required'
  | 'blocked'

export interface GoalContinuationDecision {
  action: GoalContinuationAction
  summary: string
  blocking: boolean
  approvalIds: string[]
  runId: string | null
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

function isGoalTerminalStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
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
  const normalized = (status ?? '').toLowerCase()
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

export function resolveGoalContinuationDecision(health: GoalHealthSummary): GoalContinuationDecision {
  const recommendation = health.recommended_next_action
  const recommendedAction = getString(recommendation?.action)
  const recommendedSummary =
    getString(recommendation?.summary) ??
    'The active goal needs operator attention before it can continue.'
  const linkedRunId =
    getString(health.linked_agent_run?.run_id) ??
    health.current_attempt?.agent_run_id ??
    null
  const linkedRunStatus = getString(health.linked_agent_run?.status)
  const approvalIds = getStringArray(recommendation?.approval_ids ?? health.approval_state?.approval_ids)

  if (recommendedAction === 'resolve_approval') {
    return {
      action: 'manual_resolution_required',
      summary: recommendedSummary,
      blocking: true,
      approvalIds,
      runId: linkedRunId,
    }
  }

  if (recommendedAction === 'inspect_runtime_budget') {
    return {
      action: 'blocked',
      summary: recommendedSummary,
      blocking: true,
      approvalIds: [],
      runId: linkedRunId,
    }
  }

  if (recommendedAction === 'refresh_worker_generation') {
    return {
      action: 'refresh_then_forward',
      summary: recommendedSummary,
      blocking: false,
      approvalIds: [],
      runId: linkedRunId,
    }
  }

  if (recommendedAction === 'resume_goal') {
    return {
      action: 'resume_then_forward',
      summary: recommendedSummary,
      blocking: false,
      approvalIds: [],
      runId: linkedRunId,
    }
  }

  if (recommendedAction === 'monitor' || recommendedAction === 'capture_checkpoint' || recommendedAction === 'inspect_collector_shards') {
    return {
      action: 'forward_guidance',
      summary: recommendedSummary,
      blocking: false,
      approvalIds: [],
      runId: linkedRunId,
    }
  }

  if (!isGoalTerminalStatus(health.status) && (!linkedRunId || isLinkedRunTerminalStatus(linkedRunStatus))) {
    return {
      action: 'resume_then_forward',
      summary: 'The current goal attempt is no longer actively running, so a follow-up should reopen or advance the goal before forwarding guidance.',
      blocking: false,
      approvalIds: [],
      runId: linkedRunId,
    }
  }

  return {
    action: 'forward_guidance',
    summary: recommendedSummary,
    blocking: false,
    approvalIds: [],
    runId: linkedRunId,
  }
}
