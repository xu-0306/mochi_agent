import assert from 'node:assert/strict'
import {
  hydrateOrdinaryChatPlanProjection,
  markProvisionalOrdinaryChatFinals,
  reduceOrdinaryChatPlanEvent,
  reduceOrdinaryChatPlanEvents,
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

const duplicateState = reduceOrdinaryChatPlanEvent(
  reduceOrdinaryChatPlanEvent(ordered, event('message', 35, 'turn-1', { status: 'blocked' })),
  event('message', 35, 'turn-1', { status: 'blocked' })
)
assert.equal(duplicateState.ignoredEventCount, 1)

const outOfOrder = reduceOrdinaryChatPlanEvents('session-1', [
  event('message', 30, 'turn-3'),
  event('message', 10, 'turn-1'),
  event('message', 20, 'turn-2'),
])
assert.deepEqual(Object.keys(outOfOrder.turns), ['turn-1', 'turn-2', 'turn-3'])

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

console.log('ordinary-chat-plan tests passed')
