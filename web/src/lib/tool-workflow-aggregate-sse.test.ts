import assert from 'node:assert/strict'

import { parseSseFrame } from './sse-frame.ts'
import {
  parseToolWorkflowAggregateTransport,
  toolWorkflowAggregateStore,
} from './tool-workflow-aggregate.ts'

const aggregate = {
  type: 'tool_workflow_aggregate',
  schema_version: 1,
  event_id: `twa:v1:${'0'.repeat(42)}1`,
  seq: 1,
  idempotency_key: `sha256:${'b'.repeat(64)}`,
  session_id: 'session-sse-contract',
  turn_id: 'turn-sse-contract',
  occurred_at: '2026-07-26T10:00:00Z',
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
      call_id: 'call-sse-contract',
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
} as const

const serverPayload = {
  type: 'tool_workflow_aggregate',
  schema_version: 1,
  storage_id: 'storage:v1:sse-contract',
  session_id: aggregate.session_id,
  turn_id: aggregate.turn_id,
  aggregate,
  publication_enabled: true,
  authoritative: true,
}

const frame = [
  'event: tool_workflow_aggregate',
  `id: ${aggregate.event_id}`,
  `data: ${JSON.stringify(serverPayload)}`,
  '',
  '',
].join('\n')

const parsedFrame = parseSseFrame(frame)
assert.equal(parsedFrame.eventName, 'tool_workflow_aggregate')
assert.ok(parsedFrame.data)

const transport = parseToolWorkflowAggregateTransport({
  ...JSON.parse(parsedFrame.data),
  type: parsedFrame.eventName,
})
toolWorkflowAggregateStore.clearTurn(aggregate.session_id, aggregate.turn_id)
const result = toolWorkflowAggregateStore.applyTransport(transport)

assert.equal(result.kind, 'applied')
assert.equal(toolWorkflowAggregateStore.getCursor(aggregate.session_id, aggregate.turn_id)?.lastSeq, 1)
assert.equal(
  toolWorkflowAggregateStore.getCursor(aggregate.session_id, aggregate.turn_id)?.storageId,
  serverPayload.storage_id
)
assert.equal(toolWorkflowAggregateStore.getEntry(aggregate.session_id, aggregate.turn_id)?.authoritative, true)

console.log('tool workflow aggregate SSE contract tests passed')
