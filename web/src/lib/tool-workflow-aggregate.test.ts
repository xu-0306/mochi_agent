import assert from 'node:assert/strict'

import {
  parseToolWorkflowAggregateV1,
  toolWorkflowAggregateStore,
} from './tool-workflow-aggregate.ts'

function aggregate(seq: number, overrides: Record<string, unknown> = {}) {
  return {
    type: 'tool_workflow_aggregate',
    schema_version: 1,
    event_id: `twa:v1:${'0'.repeat(42)}${seq}`,
    seq,
    idempotency_key: `sha256:${'b'.repeat(63)}${seq}`,
    session_id: 'session-1',
    turn_id: 'turn-1',
    occurred_at: '2026-07-25T10:00:00Z',
    source_refs: {
      timeline: { timeline_version: 'timeline:v1', turn_sequence: 1, events: [] },
      checkpoint: { checkpoint_revision: null, digest: null },
      approvals: [],
      receipts: [],
    },
    state: {
      turn_status: 'running',
      integrity: 'partial',
      policy: null,
      inventory: null,
      calls: [{
        call_id: 'call-1',
        operation_id: null,
        tool_name: 'file_write',
        arguments_digest: null,
        target: null,
        activation_status: 'activated',
        review_status: 'not_observed',
        approval_id: null,
        execution_status: 'not_started',
        verification_status: 'not_observed',
        receipt_reference: null,
        changed_paths: [],
        blocker: null,
      }],
      blocker: null,
    },
    ...overrides,
  }
}

const first = parseToolWorkflowAggregateV1(aggregate(1))
assert.equal(first.state.calls[0]?.execution_status, 'not_started')
assert.throws(
  () => parseToolWorkflowAggregateV1({ ...aggregate(1), schema_version: 2 }),
  /unsupported/
)
assert.throws(
  () => parseToolWorkflowAggregateV1({
    ...aggregate(1),
    state: { ...aggregate(1).state, calls: [{ ...aggregate(1).state.calls[0], execution_status: 'unknown' }] },
  }),
  /unknown operation/
)

const store = toolWorkflowAggregateStore
store.clearTurn('session-1', 'turn-1')
const scope = {
  storage_id: 'storage:v1:one',
  session_id: 'session-1',
  turn_id: 'turn-1',
  authoritative: true,
  publication_enabled: true,
}

assert.equal(store.applyTransport({ ...scope, aggregate: aggregate(1) }).kind, 'applied')
assert.equal(store.applyTransport({ ...scope, aggregate: aggregate(1) }).kind, 'duplicate')
const gap = store.applyTransport({ ...scope, aggregate: aggregate(3) })
assert.equal(gap.kind, 'repair_required')
assert.equal(store.getCursor('session-1', 'turn-1')?.lastSeq, 1)

const range = store.applyRange({
  ...scope,
  after_seq: 1,
  limit: 100,
  events: [aggregate(2), aggregate(3)],
  next_after_seq: 3,
  has_more: false,
  contiguous: true,
})
assert.equal(range.kind, 'applied')
assert.equal(store.getCursor('session-1', 'turn-1')?.lastSeq, 3)

const conflict = store.applyTransport({
  ...scope,
  aggregate: { ...aggregate(2), idempotency_key: `sha256:${'c'.repeat(64)}` },
})
assert.equal(conflict.kind, 'repair_required')
assert.equal(store.getCursor('session-1', 'turn-1')?.lastSeq, 3)

const unsupported = store.applyTransport({
  ...scope,
  aggregate: { ...aggregate(4), schema_version: 99 },
})
assert.equal(unsupported.kind, 'unsupported')
assert.equal(store.getEntry('session-1', 'turn-1')?.aggregate?.seq, 3)

const changedScope = store.applyTransport({
  ...scope,
  storage_id: 'storage:v1:two',
  aggregate: aggregate(1),
})
assert.equal(changedScope.kind, 'applied')
assert.equal(store.getCursor('session-1', 'turn-1')?.storageId, 'storage:v1:two')
assert.equal(store.getCursor('session-1', 'turn-1')?.lastSeq, 1)

console.log('tool workflow aggregate tests passed')
