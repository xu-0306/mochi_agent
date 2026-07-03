import type { ActiveGoalTurnDecision } from './api'

export interface ActiveGoalTurnDecisionMetadata {
  active_goal_turn_decision: {
    lane: ActiveGoalTurnDecision['lane']
    kind: ActiveGoalTurnDecision['kind']
    confidence: number
    selection_source: ActiveGoalTurnDecision['selection_source']
    selection_reason: string
    requires_confirmation: boolean
    goal_status: string | null
    linked_run_status: string | null
    recommended_action: string | null
  }
}

export type ActiveGoalTurnPageAction =
  | { kind: 'direct_chat' }
  | { kind: 'steer_goal' }
  | { kind: 'replan_goal' }
  | { kind: 'lifecycle_goal' }
  | { kind: 'unhandled' }

export function buildActiveGoalTurnDecisionMetadata(
  decision: ActiveGoalTurnDecision
): ActiveGoalTurnDecisionMetadata {
  return {
    active_goal_turn_decision: {
      lane: decision.lane,
      kind: decision.kind,
      confidence: decision.confidence,
      selection_source: decision.selection_source,
      selection_reason: decision.selection_reason,
      requires_confirmation: decision.requires_confirmation,
      goal_status: decision.goal_status ?? null,
      linked_run_status: decision.linked_run_status ?? null,
      recommended_action: decision.recommended_action ?? null,
    },
  }
}

export function classifyActiveGoalTurnDecision(
  decision: ActiveGoalTurnDecision
): ActiveGoalTurnPageAction {
  if (
    decision.kind === 'answer_question' ||
    decision.kind === 'explain_goal_state' ||
    decision.kind === 'exit_to_chat' ||
    (decision.kind === 'clarify' && decision.requires_confirmation)
  ) {
    return { kind: 'direct_chat' }
  }
  if (decision.kind === 'steer') {
    return { kind: 'steer_goal' }
  }
  if (decision.kind === 'replan') {
    return { kind: 'replan_goal' }
  }
  if (decision.kind === 'lifecycle') {
    return { kind: 'lifecycle_goal' }
  }
  return { kind: 'unhandled' }
}
