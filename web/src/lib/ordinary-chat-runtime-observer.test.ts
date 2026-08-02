import assert from 'node:assert/strict'
import {
  createOrdinaryChatPlanState,
  reduceOrdinaryChatPlanEvent,
  type OrdinaryChatPlanState,
  type OrdinaryChatRuntimeEvent,
} from './ordinary-chat-plan.ts'
import { observeOrdinaryChatRuntime } from './ordinary-chat-runtime-observer.ts'

let time = 0
let snapshots = 0
let notifications = 0
let streamCalls = 0
const controller = new AbortController()
await observeOrdinaryChatRuntime<number, { event_id: string }>({
  signal: controller.signal,
  fetchSnapshot: async () => ({ state: ++snapshots, lastEventId: null }),
  stream: async function* () {
    streamCalls += 1
    if (snapshots === 1) throw { status: 409 }
    yield { event_id: 'e' }
  },
  reduce: (state) => state + 1,
  onState: () => {
    notifications += 1
  },
  isStaleCursor: (error) => (error as { status?: number }).status === 409,
  now: () => time,
  sleep: async () => {
    time += 1_000
    if (time > 2_000) controller.abort()
  },
  deadlineMs: 5_000,
})
assert.equal(snapshots, 2)
assert.ok(streamCalls >= 2)
assert.ok(notifications >= 2)

const aborted = new AbortController()
let abortedFetches = 0
aborted.abort()
await observeOrdinaryChatRuntime<number, never>({
  signal: aborted.signal,
  fetchSnapshot: async () => {
    abortedFetches += 1
    return { state: 0, lastEventId: null }
  },
  stream: async function* () {},
  reduce: (state) => state,
  onState: () => assert.fail('aborted observer must not publish state'),
  isStaleCursor: () => false,
})
assert.equal(abortedFetches, 0)

const cancelledDuringBackoff = new AbortController()
let backoffStreamCalls = 0
await observeOrdinaryChatRuntime<number, { event_id: string }>({
  signal: cancelledDuringBackoff.signal,
  fetchSnapshot: async () => ({ state: 0, lastEventId: null }),
  stream: async function* () {
    backoffStreamCalls += 1
  },
  reduce: (state) => state,
  onState: () => {},
  isStaleCursor: () => false,
  now: () => 0,
  sleep: async () => {
    cancelledDuringBackoff.abort()
  },
})
assert.equal(backoffStreamCalls, 1)

const liveFailureController = new AbortController()
let liveFailureState: OrdinaryChatPlanState | null = null
await observeOrdinaryChatRuntime<
  OrdinaryChatPlanState,
  OrdinaryChatRuntimeEvent
>({
  signal: liveFailureController.signal,
  fetchSnapshot: async () => ({
    state: createOrdinaryChatPlanState('learning-session'),
    lastEventId: null,
  }),
  stream: async function* () {
    yield {
      event_id: 'failure-live-1',
      event: 'failure_learning_attribution_recorded',
      schema_version: 1,
      sequence: 1,
      revision: 0,
      turn_id: 'turn-1',
      payload: { transition: 'candidate', status: 'pending' },
    }
  },
  reduce: reduceOrdinaryChatPlanEvent,
  onState: (state) => {
    liveFailureState = state
    if (state.failureLearning.candidates === 1) {
      liveFailureController.abort()
    }
  },
  isStaleCursor: () => false,
})
assert.equal(
  (liveFailureState as OrdinaryChatPlanState | null)?.failureLearning
    .candidates,
  1
)
assert.equal(
  (liveFailureState as OrdinaryChatPlanState | null)?.failureLearning.coverage,
  'complete'
)

console.log('ordinary chat runtime observer tests passed')
