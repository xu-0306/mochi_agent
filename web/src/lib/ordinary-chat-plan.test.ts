import assert from 'node:assert/strict'
import {
  failureLearningDisplayItems,
  hydrateOrdinaryChatPlanProjection,
  markProvisionalOrdinaryChatFinals,
  reduceOrdinaryChatPlanEvent,
  reduceOrdinaryChatPlanEvents,
  sortOrdinaryChatTurns,
} from './ordinary-chat-plan.ts'

function event(
  eventName: string,
  sequence: number,
  turnId: string,
  payload: Record<string, unknown> = {}
) {
  return {
    event_id: `${eventName}-${turnId}-${sequence}`,
    event: eventName,
    schema_version: 1,
    sequence,
    revision: sequence,
    turn_id: turnId,
    payload,
  }
}

const plan = {
  ledger_id: 'plan-1',
  revision: 1,
  status: 'active',
  objective: 'bounded task',
  reason_codes: [],
  items: [
    {
      item_id: 'item-1',
      title: 'Inspect',
      status: 'in_progress',
      dependencies: [],
      success_criteria: ['evidence'],
      evidence_refs: [],
      blocker_reason: null,
      attempts: 0,
    },
  ],
  blockers: [],
}

const verification = {
  receipt_id: 'receipt-1',
  turn_id: 'turn-1',
  verdict: 'unverified',
  hard_failure: false,
  retry_disposition: 'blocked',
  criteria: [],
}

const ordered = reduceOrdinaryChatPlanEvents('session-1', [
  event('ordinary_chat_plan_ledger_updated', 10, 'turn-1', { plan }),
  event('ordinary_chat_verification_receipt_recorded', 20, 'turn-1', { receipt: verification }),
  event('turn_execution_checkpoint', 30, 'turn-1', {
    stage: 'blocked',
    complexity: { kind: 'plan_required', score: 8 },
    retrieval: { eligible_count: 2 },
    recovery: { remaining_attempts: 0 },
    blocker_reason: 'verification_failed',
  }),
])
assert.equal(ordered.turns['turn-1']?.status, 'blocked')
assert.equal(ordered.turns['turn-1']?.plan?.ledger_id, 'plan-1')

const cancelled = reduceOrdinaryChatPlanEvents('session-1', [
  event('session_turn_timeline', 40, 'turn-2', {
    status: 'cancelled',
    terminal_outcome: 'cancelled',
    cancellation_outcome: 'cancelled_queued',
  }),
])
assert.equal(cancelled.turns['turn-2']?.status, 'cancelled')

const partialFinalization = reduceOrdinaryChatPlanEvents('session-1', [
  event('ordinary_chat_plan_ledger_updated', 10, 'turn-partial', { plan }),
  event('turn_status_hint', 20, 'turn-partial', {
    status: 'partial',
    blocker_code: 'plan_finalization_required',
  }),
  event('session_turn_timeline', 30, 'turn-partial', {
    status: 'terminal',
    terminal_outcome: 'completed',
  }),
])
assert.equal(partialFinalization.turns['turn-partial']?.status, 'partial')
assert.deepEqual(partialFinalization.turns['turn-partial']?.blockers, [
  'plan_finalization_required',
])

const deniedActivation = reduceOrdinaryChatPlanEvents('session-1', [
  event('turn_status_hint', 10, 'turn-denied', {
    activation_outcome: 'denied',
    blocker_code: 'tool_activation_denied',
  }),
  event('session_turn_timeline', 20, 'turn-denied', {
    status: 'terminal',
    terminal_outcome: 'completed',
  }),
])
assert.equal(deniedActivation.turns['turn-denied']?.status, 'blocked')

const recoveredActivation = reduceOrdinaryChatPlanEvents('session-1', [
  event('turn_status_hint', 10, 'turn-recovered', {
    activation_outcome: 'denied',
    blocker_code: 'tool_activation_denied',
  }),
  event('turn_status_hint', 20, 'turn-recovered', {
    activation_outcome: 'activated',
  }),
  event('session_turn_timeline', 30, 'turn-recovered', {
    status: 'terminal',
    terminal_outcome: 'completed',
  }),
])
assert.equal(recoveredActivation.turns['turn-recovered']?.status, 'completed')

const normalTerminal = reduceOrdinaryChatPlanEvents('session-1', [
  event('ordinary_chat_plan_ledger_updated', 10, 'turn-completed', { plan }),
  event('session_turn_timeline', 20, 'turn-completed', {
    status: 'terminal',
    terminal_outcome: 'completed',
  }),
])
assert.equal(normalTerminal.turns['turn-completed']?.status, 'completed')

const duplicateState = reduceOrdinaryChatPlanEvent(
  reduceOrdinaryChatPlanEvent(ordered, event('message', 35, 'turn-1', { status: 'blocked' })),
  event('message', 35, 'turn-1', { status: 'blocked' })
)
assert.equal(duplicateState.ignoredEventCount, 1)

const outOfOrder = reduceOrdinaryChatPlanEvents('session-1', [
  event('turn_execution_checkpoint', 30, 'turn-3', { stage: 'completed' }),
  event('ordinary_chat_plan_ledger_updated', 10, 'turn-1', { plan }),
  event('session_turn_timeline', 20, 'turn-2', {
    status: 'cancelled',
    terminal_outcome: 'cancelled',
    cancellation_outcome: 'cancelled_queued',
  }),
])
assert.deepEqual(
  sortOrdinaryChatTurns(outOfOrder).map((turn) => turn.turn_id),
  ['turn-1', 'turn-2', 'turn-3']
)
assert.equal(outOfOrder.turns['turn-1']?.plan?.ledger_id, 'plan-1')
assert.equal(outOfOrder.turns['turn-2']?.status, 'cancelled')
assert.equal(outOfOrder.turns['turn-3']?.status, 'completed')

const reloaded = hydrateOrdinaryChatPlanProjection({
  projection_version: 'ordinary-chat-adaptive-runtime-v1',
  schema_version: 1,
  session_id: 'session-1',
  revision: 30,
  latest_sequence: 30,
  events: [event('ordinary_chat_plan_ledger_updated', 10, 'turn-1', { plan })],
  turns: [],
  metrics: {},
})
assert.equal(reloaded.turns['turn-1']?.plan?.items[0]?.title, 'Inspect')

const longReloadEvents = Array.from({ length: 128 }, (_, index) =>
  event('message', index + 1, `long-turn-${index}`)
)
const longReloadTurns = Array.from({ length: 12 }, (_, index) => {
  const sourceIndex = 116 + index
  return {
    ...ordered.turns['turn-1']!,
    turn_id: `long-turn-${sourceIndex}`,
    updated_sequence: sourceIndex + 1,
  }
})
const longReload = hydrateOrdinaryChatPlanProjection({
  projection_version: 'ordinary-chat-adaptive-runtime-v1',
  schema_version: 1,
  session_id: 'session-1',
  revision: 128,
  latest_sequence: 128,
  events: longReloadEvents,
  turns: longReloadTurns,
  metrics: {},
})
assert.equal(Object.keys(longReload.turns).length, 12)
assert.deepEqual(
  sortOrdinaryChatTurns(longReload).map((turn) => turn.turn_id),
  Array.from({ length: 12 }, (_, index) => `long-turn-${116 + index}`)
)
assert.equal(longReload.turns['long-turn-116']?.plan?.ledger_id, 'plan-1')
assert.equal(longReload.seenEventIds.size, 128)

const provisional = markProvisionalOrdinaryChatFinals(
  [
    {
      id: 'assistant-1',
      type: 'assistant',
      content: 'model answer',
      eventType: 'final_answer',
      timestamp: new Date(0),
      turnId: 'turn-1',
    },
  ],
  ordered
)
assert.match(provisional[0]?.content ?? '', /^Provisional answer/)
assert.equal(provisional[0]?.eventType, 'status')

const failureTransitions = [
  event('failure_learning_attribution_recorded', 41, 'turn-4', {
    transition: 'candidate',
    status: 'pending',
  }),
  event('failure_learning_attribution_recorded', 42, 'turn-4', {
    transition: 'processed',
    status: 'processed',
  }),
  event('failure_learning_attribution_recorded', 43, 'turn-5', {
    transition: 'rejected',
    status: 'rejected',
  }),
  event('failure_learning_attribution_recorded', 44, 'turn-5', {
    transition: 'hint_selected',
    status: 'selected',
  }),
]
const liveLearning = failureTransitions.reduce(
  reduceOrdinaryChatPlanEvent,
  reduceOrdinaryChatPlanEvents('session-learning', [])
)
assert.deepEqual(liveLearning.failureLearning, {
  coverage: 'complete',
  valid_transition_count: 4,
  ignored_transition_count: 0,
  candidates: 1,
  processed: 1,
  rejected: 1,
  hints_selected: 1,
})
assert.equal(liveLearning.turns['turn-4']?.failure_learning.candidate_count, 1)
assert.equal(liveLearning.turns['turn-5']?.failure_learning.rejected_count, 1)
const duplicateLearning = reduceOrdinaryChatPlanEvent(
  liveLearning,
  failureTransitions[0]!
)
assert.equal(duplicateLearning.failureLearning.candidates, 1)
assert.equal(duplicateLearning.ignoredEventCount, 1)

const learningReload = hydrateOrdinaryChatPlanProjection({
  projection_version: 'ordinary-chat-adaptive-runtime-v1',
  schema_version: 1,
  session_id: 'session-learning',
  revision: 44,
  latest_sequence: 44,
  events: [],
  turns: [liveLearning.turns['turn-5']!],
  metrics: {
    failure_learning: {
      coverage: 'partial',
      valid_transition_count: 3,
      ignored_transition_count: 1,
      candidates: 1,
      processed: 0,
      rejected: 1,
      hints_selected: 1,
    },
    cost: {
      coverage: 'partial',
      token_coverage: 'partial',
      wall_coverage: 'complete',
      observed_turns: 2,
      expected_turns: 2,
    },
  },
})
assert.equal(learningReload.failureLearning.coverage, 'partial')
assert.deepEqual(learningReload.costCoverage, {
  coverage: 'partial',
  token_coverage: 'partial',
  wall_coverage: 'complete',
  observed_turns: 2,
  expected_turns: 2,
})
assert.equal(learningReload.failureLearning.rejected, 1)
assert.equal(
  learningReload.turns['turn-5']?.failure_learning.hints_selected_count,
  1
)
assert.deepEqual(
  failureLearningDisplayItems(learningReload).map((item) => item.label),
  ['Candidates', 'Processed', 'Rejected', 'Hints selected']
)
assert.equal(failureLearningDisplayItems(learningReload).length, 4)

const isolatedReload = hydrateOrdinaryChatPlanProjection({
  projection_version: 'ordinary-chat-adaptive-runtime-v1',
  schema_version: 1,
  session_id: 'other-session',
  revision: 0,
  latest_sequence: 0,
  events: [],
  turns: [],
  metrics: {},
})
assert.equal(isolatedReload.failureLearning.valid_transition_count, 0)
assert.equal(Object.keys(isolatedReload.turns).length, 0)

console.log('ordinary-chat-plan tests passed')
