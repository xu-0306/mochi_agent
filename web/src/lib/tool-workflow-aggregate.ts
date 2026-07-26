import * as React from 'react'

export const TOOL_WORKFLOW_AGGREGATE_TYPE = 'tool_workflow_aggregate'
export const TOOL_WORKFLOW_AGGREGATE_SCHEMA_VERSION = 1

const EVENT_ID_PATTERN = /^twa:v1:[A-Za-z0-9_-]{43}$/
const SHA256_PATTERN = /^sha256:[0-9a-f]{64}$/
const BARE_SHA256_PATTERN = /^[0-9a-f]{64}$/

const TURN_STATUSES = new Set([
  'queued',
  'running',
  'awaiting_approval',
  'executing',
  'verifying',
  'completed',
  'blocked',
  'cancelled',
  'unknown',
])
const ACTIVATION_STATUSES = new Set(['not_observed', 'requested', 'activated', 'rejected', 'failed', 'unknown'])
const REVIEW_STATUSES = new Set([
  'not_observed',
  'pending',
  'approved',
  'rejected',
  'expired',
  'consuming',
  'consumed',
  'unknown',
])
const EXECUTION_STATUSES = new Set([
  'not_started',
  'precommitted',
  'started',
  'succeeded',
  'failed',
  'abandoned',
  'cancelled',
  'unknown',
])
const VERIFICATION_STATUSES = new Set(['not_required', 'pending', 'verified', 'failed', 'unknown', 'not_observed'])
const INTEGRITIES = new Set(['complete', 'partial', 'unsupported'])

export interface ToolWorkflowAggregateCall {
  call_id: string
  operation_id: string | null
  tool_name: string
  arguments_digest: string | null
  target: Record<string, unknown> | null
  activation_status: string
  review_status: string
  approval_id: string | null
  execution_status: string
  verification_status: string
  receipt_reference: string | null
  changed_paths: string[]
  blocker: string | null
}

export interface ToolWorkflowAggregateState {
  turn_status: string
  integrity: string
  policy: Record<string, unknown> | null
  inventory: Record<string, unknown> | null
  calls: ToolWorkflowAggregateCall[]
  blocker: string | null
}

export interface ToolWorkflowAggregate {
  type: typeof TOOL_WORKFLOW_AGGREGATE_TYPE
  schema_version: typeof TOOL_WORKFLOW_AGGREGATE_SCHEMA_VERSION
  event_id: string
  seq: number
  idempotency_key: string
  session_id: string
  turn_id: string
  occurred_at: string
  source_refs: Record<string, unknown>
  state: ToolWorkflowAggregateState
}

export interface ToolWorkflowAggregateTransport {
  type?: string
  storage_id: string
  session_id: string
  turn_id: string
  aggregate: ToolWorkflowAggregate | null
  publication_enabled?: boolean
  authoritative?: boolean
  unsupported?: boolean
  error?: string
}

export interface ToolWorkflowAggregateRangeTransport {
  type?: string
  storage_id: string
  session_id: string
  turn_id: string
  after_seq: number
  limit: number
  events: unknown[]
  next_after_seq: number
  has_more: boolean
  contiguous: boolean
  publication_enabled?: boolean
  authoritative?: boolean
}

export type ToolWorkflowAggregateEntryStatus =
  | 'empty'
  | 'ready'
  | 'partial'
  | 'stale'
  | 'repair_required'
  | 'unsupported'
  | 'scope_changed'

export interface ToolWorkflowAggregateCursor {
  storageId: string
  sessionId: string
  turnId: string
  lastSeq: number
  lastEventId: string | null
  lastIdempotencyKey: string | null
}

export interface ToolWorkflowAggregateEntry {
  cursor: ToolWorkflowAggregateCursor
  aggregate: ToolWorkflowAggregate | null
  status: ToolWorkflowAggregateEntryStatus
  authoritative: boolean
  publicationEnabled: boolean
  diagnostic: string | null
}

export interface ToolWorkflowAggregateCallView {
  authoritative: boolean
  turnStatus: string
  integrity: string
  call: ToolWorkflowAggregateCall | null
  aggregate: ToolWorkflowAggregate | null
  entryStatus: ToolWorkflowAggregateEntryStatus
  diagnostic: string | null
}

export type ToolWorkflowAggregateApplyResult =
  | { kind: 'applied'; entry: ToolWorkflowAggregateEntry }
  | { kind: 'duplicate'; entry: ToolWorkflowAggregateEntry }
  | { kind: 'repair_required'; entry: ToolWorkflowAggregateEntry }
  | { kind: 'unsupported'; entry: ToolWorkflowAggregateEntry }

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0
}

function isPositiveInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value > 0
}

function isNonNegativeInteger(value: unknown): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= 0
}

function exactKeys(value: Record<string, unknown>, expected: string[], name: string): void {
  const actual = Object.keys(value).sort()
  const required = [...expected].sort()
  if (actual.length !== required.length || actual.some((key, index) => key !== required[index])) {
    throw new Error(`${name} fields do not match tool_workflow_aggregate v1`)
  }
}

function objectOrNull(value: unknown, name: string): Record<string, unknown> | null {
  if (value === null) return null
  if (!isRecord(value)) throw new Error(`${name} must be an object or null`)
  return { ...value }
}

function digest(value: unknown, name: string): string {
  if (typeof value !== 'string' || !SHA256_PATTERN.test(value)) {
    throw new Error(`${name} must be a sha256 digest`)
  }
  return value
}

function bareDigestOrNull(value: unknown, name: string): string | null {
  if (value === null) return null
  if (typeof value !== 'string' || !BARE_SHA256_PATTERN.test(value)) {
    throw new Error(`${name} must be a bare sha256 digest or null`)
  }
  return value
}

function parseSourceRefs(value: unknown): Record<string, unknown> {
  if (!isRecord(value)) throw new Error('source_refs must be an object')
  exactKeys(value, ['timeline', 'checkpoint', 'approvals', 'receipts'], 'source_refs')

  if (!isRecord(value.timeline)) throw new Error('source_refs.timeline must be an object')
  exactKeys(value.timeline, ['timeline_version', 'turn_sequence', 'events'], 'source_refs.timeline')
  if (value.timeline.timeline_version !== null && !isNonEmptyString(value.timeline.timeline_version)) {
    throw new Error('timeline_version must be a string or null')
  }
  if (value.timeline.turn_sequence !== null && !isPositiveInteger(value.timeline.turn_sequence)) {
    throw new Error('turn_sequence must be a positive integer or null')
  }
  if (!Array.isArray(value.timeline.events)) throw new Error('timeline events must be an array')
  const timelineEvents = value.timeline.events.map((item) => {
    if (!isRecord(item)) throw new Error('timeline source ref must be an object')
    exactKeys(item, ['source_position', 'kind', 'digest'], 'timeline source ref')
    if (!isPositiveInteger(item.source_position) || !isNonEmptyString(item.kind)) {
      throw new Error('timeline source ref has invalid identity')
    }
    return {
      source_position: item.source_position,
      kind: item.kind,
      digest: digest(item.digest, 'timeline source digest'),
    }
  })
  if (timelineEvents.some((item, index) => index > 0 && item.source_position <= timelineEvents[index - 1].source_position)) {
    throw new Error('timeline source refs must be sorted and unique')
  }

  if (!isRecord(value.checkpoint)) throw new Error('source_refs.checkpoint must be an object')
  exactKeys(value.checkpoint, ['checkpoint_revision', 'digest'], 'source_refs.checkpoint')
  if ((value.checkpoint.checkpoint_revision === null) !== (value.checkpoint.digest === null)) {
    throw new Error('checkpoint source ref must be fully present or null')
  }
  if (value.checkpoint.checkpoint_revision !== null && !isNonNegativeInteger(value.checkpoint.checkpoint_revision)) {
    throw new Error('checkpoint_revision must be a non-negative integer or null')
  }
  const checkpoint = {
    checkpoint_revision: value.checkpoint.checkpoint_revision,
    digest: value.checkpoint.digest === null ? null : digest(value.checkpoint.digest, 'checkpoint digest'),
  }

  const approvals = parseApprovalRefs(value.approvals)
  const receipts = parseReceiptRefs(value.receipts)
  return {
    timeline: {
      timeline_version: value.timeline.timeline_version,
      turn_sequence: value.timeline.turn_sequence,
      events: timelineEvents,
    },
    checkpoint,
    approvals,
    receipts,
  }
}

function parseApprovalRefs(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) throw new Error('approval source refs must be an array')
  const refs = value.map((item) => {
    if (!isRecord(item)) throw new Error('approval source ref must be an object')
    exactKeys(item, ['approval_id', 'approval_revision', 'status', 'request_digest', 'context_digest', 'legacy_digest'], 'approval source ref')
    if (!isNonEmptyString(item.approval_id) || !isNonEmptyString(item.status)) throw new Error('approval source ref has invalid identity')
    if (item.approval_revision === null && item.legacy_digest === null) throw new Error('legacy approval source ref requires a digest')
    if (item.approval_revision !== null && item.legacy_digest !== null) throw new Error('approval source ref cannot contain both revision and legacy digest')
    if (item.approval_revision !== null && !isPositiveInteger(item.approval_revision)) throw new Error('approval_revision must be positive')
    if (!['pending', 'approved_once', 'rejected', 'expired', 'superseded', 'consuming', 'consumed', 'execution_failed'].includes(item.status)) throw new Error('unsupported approval source status')
    return {
      approval_id: item.approval_id,
      approval_revision: item.approval_revision,
      status: item.status,
      request_digest: item.request_digest === '' ? '' : bareDigestOrNull(item.request_digest, 'request_digest') ?? '',
      context_digest: item.context_digest === '' ? '' : bareDigestOrNull(item.context_digest, 'context_digest') ?? '',
      legacy_digest: item.legacy_digest === null ? null : digest(item.legacy_digest, 'legacy approval digest'),
    }
  })
  if (refs.some((item, index) => {
    if (index === 0) return false
    const previous = refs[index - 1]
    return item.approval_id < previous.approval_id ||
      (item.approval_id === previous.approval_id && (item.approval_revision ?? 0) < (previous.approval_revision ?? 0))
  })) {
    throw new Error('approval source refs must be sorted')
  }
  return refs
}

function parseReceiptRefs(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) throw new Error('receipt source refs must be an array')
  const refs = value.map((item) => {
    if (!isRecord(item)) throw new Error('receipt source ref must be an object')
    exactKeys(item, ['kind', 'schema_version', 'source_position', 'operation_id', 'receipt_reference', 'digest', 'verification_status'], 'receipt source ref')
    if (item.kind !== 'artifact_receipt' || ![1, 2, 3].includes(item.schema_version as number) || !isPositiveInteger(item.source_position)) throw new Error('invalid receipt source ref')
    if (!isNonEmptyString(item.operation_id) || !isNonEmptyString(item.receipt_reference) || !isNonEmptyString(item.verification_status)) throw new Error('receipt source ref has invalid identity')
    if (!['verified', 'failed', 'partial', 'not_run'].includes(item.verification_status)) throw new Error('unsupported receipt verification status')
    return {
      kind: item.kind,
      schema_version: item.schema_version,
      source_position: item.source_position,
      operation_id: item.operation_id,
      receipt_reference: item.receipt_reference,
      digest: digest(item.digest, 'receipt digest'),
      verification_status: item.verification_status,
    }
  })
  if (refs.some((item, index) => {
    if (index === 0) return false
    const previous = refs[index - 1]
    return item.operation_id < previous.operation_id ||
      (item.operation_id === previous.operation_id && item.receipt_reference < previous.receipt_reference)
  })) {
    throw new Error('receipt source refs must be sorted')
  }
  return refs
}

function parseState(value: unknown): ToolWorkflowAggregateState {
  if (!isRecord(value)) throw new Error('state must be an object')
  exactKeys(value, ['turn_status', 'integrity', 'policy', 'inventory', 'calls', 'blocker'], 'state')
  if (!isNonEmptyString(value.turn_status) || !TURN_STATUSES.has(value.turn_status)) throw new Error('unsupported turn status')
  if (!isNonEmptyString(value.integrity) || !INTEGRITIES.has(value.integrity)) throw new Error('unsupported aggregate integrity')
  if (value.blocker !== null && !isNonEmptyString(value.blocker)) throw new Error('state.blocker must be a string or null')
  if (!Array.isArray(value.calls)) throw new Error('state.calls must be an array')
  const calls = value.calls.map(parseCall)
  if (calls.some((item, index) => index > 0 && item.call_id <= calls[index - 1].call_id)) throw new Error('state calls must be sorted and unique')
  if (calls.some((call) => call.execution_status === 'unknown') && value.turn_status !== 'unknown') throw new Error('unknown operation must make the turn unknown')
  if (value.turn_status === 'completed') {
    if (value.integrity !== 'complete' || calls.some((call) => call.execution_status !== 'succeeded')) throw new Error('completed turn lacks succeeded evidence')
    if (calls.some((call) => ['pending', 'approved', 'consuming', 'unknown'].includes(call.review_status))) throw new Error('completed turn retains unresolved approval')
    if (calls.some((call) => !['verified', 'not_required'].includes(call.verification_status))) throw new Error('completed turn lacks terminal verification')
  }
  return {
    turn_status: value.turn_status,
    integrity: value.integrity,
    policy: objectOrNull(value.policy, 'state.policy'),
    inventory: objectOrNull(value.inventory, 'state.inventory'),
    calls,
    blocker: value.blocker as string | null,
  }
}

function parseCall(value: unknown): ToolWorkflowAggregateCall {
  if (!isRecord(value)) throw new Error('aggregate call must be an object')
  exactKeys(value, ['call_id', 'operation_id', 'tool_name', 'arguments_digest', 'target', 'activation_status', 'review_status', 'approval_id', 'execution_status', 'verification_status', 'receipt_reference', 'changed_paths', 'blocker'], 'aggregate call')
  if (!isNonEmptyString(value.call_id) || !isNonEmptyString(value.tool_name)) throw new Error('aggregate call has invalid identity')
  if (!isNonEmptyString(value.activation_status) || !ACTIVATION_STATUSES.has(value.activation_status)) throw new Error('unsupported activation status')
  if (!isNonEmptyString(value.review_status) || !REVIEW_STATUSES.has(value.review_status)) throw new Error('unsupported review status')
  if (!isNonEmptyString(value.execution_status) || !EXECUTION_STATUSES.has(value.execution_status)) throw new Error('unsupported execution status')
  if (!isNonEmptyString(value.verification_status) || !VERIFICATION_STATUSES.has(value.verification_status)) throw new Error('unsupported verification status')
  if (value.operation_id !== null && !isNonEmptyString(value.operation_id)) throw new Error('operation_id must be a string or null')
  if (value.approval_id !== null && !isNonEmptyString(value.approval_id)) throw new Error('approval_id must be a string or null')
  if (value.receipt_reference !== null && !isNonEmptyString(value.receipt_reference)) throw new Error('receipt_reference must be a string or null')
  if (value.blocker !== null && !isNonEmptyString(value.blocker)) throw new Error('call.blocker must be a string or null')
  if (!Array.isArray(value.changed_paths) || value.changed_paths.some((item) => !isNonEmptyString(item))) throw new Error('changed_paths must be an array of strings')
  const changedPaths = [...(value.changed_paths as string[])]
  if (changedPaths.some((item, index) => index > 0 && item < changedPaths[index - 1])) {
    throw new Error('changed_paths must be sorted')
  }
  return {
    call_id: value.call_id,
    operation_id: value.operation_id as string | null,
    tool_name: value.tool_name,
    arguments_digest: bareDigestOrNull(value.arguments_digest, 'arguments_digest'),
    target: objectOrNull(value.target, 'call.target'),
    activation_status: value.activation_status,
    review_status: value.review_status,
    approval_id: value.approval_id as string | null,
    execution_status: value.execution_status,
    verification_status: value.verification_status,
    receipt_reference: value.receipt_reference as string | null,
    changed_paths: changedPaths,
    blocker: value.blocker as string | null,
  }
}

export function parseToolWorkflowAggregateV1(value: unknown): ToolWorkflowAggregate {
  if (!isRecord(value)) throw new Error('tool workflow aggregate must be an object')
  exactKeys(value, ['type', 'schema_version', 'event_id', 'seq', 'idempotency_key', 'session_id', 'turn_id', 'occurred_at', 'source_refs', 'state'], 'aggregate')
  if (value.type !== TOOL_WORKFLOW_AGGREGATE_TYPE || value.schema_version !== TOOL_WORKFLOW_AGGREGATE_SCHEMA_VERSION) throw new Error('unsupported tool_workflow_aggregate schema version')
  if (!isNonEmptyString(value.session_id) || !isNonEmptyString(value.turn_id) || !isNonEmptyString(value.occurred_at)) throw new Error('aggregate identity is invalid')
  if (!EVENT_ID_PATTERN.test(String(value.event_id)) || !isPositiveInteger(value.seq)) throw new Error('aggregate event identity is invalid')
  if (typeof value.idempotency_key !== 'string' || !SHA256_PATTERN.test(value.idempotency_key)) throw new Error('aggregate idempotency_key is invalid')
  const occurredAt = Date.parse(value.occurred_at)
  if (!Number.isFinite(occurredAt) || !/[zZ]|[+-]\d\d:\d\d$/.test(value.occurred_at)) throw new Error('aggregate occurred_at must include a timezone')
  return {
    type: TOOL_WORKFLOW_AGGREGATE_TYPE,
    schema_version: TOOL_WORKFLOW_AGGREGATE_SCHEMA_VERSION,
    event_id: value.event_id as string,
    seq: value.seq,
    idempotency_key: value.idempotency_key,
    session_id: value.session_id,
    turn_id: value.turn_id,
    occurred_at: value.occurred_at,
    source_refs: parseSourceRefs(value.source_refs),
    state: parseState(value.state),
  }
}

export function parseToolWorkflowAggregateTransport(value: unknown): ToolWorkflowAggregateTransport {
  if (!isRecord(value) || !isNonEmptyString(value.storage_id) || !isNonEmptyString(value.session_id) || !isNonEmptyString(value.turn_id)) {
    throw new Error('tool workflow aggregate transport scope is invalid')
  }
  const aggregate = value.aggregate === null ? null : parseToolWorkflowAggregateV1(value.aggregate)
  if (aggregate && (aggregate.session_id !== value.session_id || aggregate.turn_id !== value.turn_id)) {
    throw new Error('aggregate transport scope does not match aggregate identity')
  }
  if (value.authoritative !== undefined && typeof value.authoritative !== 'boolean') throw new Error('authoritative must be boolean')
  if (value.publication_enabled !== undefined && typeof value.publication_enabled !== 'boolean') throw new Error('publication_enabled must be boolean')
  return {
    type: isNonEmptyString(value.type) ? value.type : undefined,
    storage_id: value.storage_id,
    session_id: value.session_id,
    turn_id: value.turn_id,
    aggregate,
    publication_enabled: value.publication_enabled as boolean | undefined,
    authoritative: value.authoritative as boolean | undefined,
    unsupported: value.unsupported as boolean | undefined,
    error: isNonEmptyString(value.error) ? value.error : undefined,
  }
}

function keyFor(storageId: string, sessionId: string, turnId: string): string {
  return JSON.stringify([storageId, sessionId, turnId])
}

function turnKey(sessionId: string, turnId: string): string {
  return JSON.stringify([sessionId, turnId])
}

function stableJson(value: unknown): string {
  if (value === null || typeof value !== 'object') return JSON.stringify(value)
  if (Array.isArray(value)) return `[${value.map(stableJson).join(',')}]`
  const record = value as Record<string, unknown>
  return `{${Object.keys(record).sort().map((key) => `${JSON.stringify(key)}:${stableJson(record[key])}`).join(',')}}`
}

interface InternalEntry extends ToolWorkflowAggregateEntry {
  aggregatesBySeq: Map<number, ToolWorkflowAggregate>
}

function publicEntry(entry: InternalEntry): ToolWorkflowAggregateEntry {
  return {
    cursor: { ...entry.cursor },
    aggregate: entry.aggregate,
    status: entry.status,
    authoritative: entry.authoritative,
    publicationEnabled: entry.publicationEnabled,
    diagnostic: entry.diagnostic,
  }
}

export class ToolWorkflowAggregateStore {
  private readonly entries = new Map<string, InternalEntry>()
  private readonly activeScopeByTurn = new Map<string, string>()
  private readonly listeners = new Set<() => void>()
  private revision = 0

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener)
    return () => this.listeners.delete(listener)
  }

  private notify(): void {
    this.revision += 1
    this.listeners.forEach((listener) => listener())
  }

  getRevision(): number {
    return this.revision
  }

  private scopeEntry(transport: ToolWorkflowAggregateTransport): InternalEntry {
    const key = keyFor(transport.storage_id, transport.session_id, transport.turn_id)
    const existing = this.entries.get(key)
    if (existing) return existing
    const created: InternalEntry = {
      cursor: {
        storageId: transport.storage_id,
        sessionId: transport.session_id,
        turnId: transport.turn_id,
        lastSeq: 0,
        lastEventId: null,
        lastIdempotencyKey: null,
      },
      aggregate: null,
      status: 'empty',
      authoritative: transport.authoritative === true,
      publicationEnabled: transport.publication_enabled === true,
      diagnostic: null,
      aggregatesBySeq: new Map(),
    }
    this.entries.set(key, created)
    return created
  }

  private activateScope(transport: ToolWorkflowAggregateTransport): InternalEntry {
    const scope = turnKey(transport.session_id, transport.turn_id)
    const nextKey = keyFor(transport.storage_id, transport.session_id, transport.turn_id)
    const previousKey = this.activeScopeByTurn.get(scope)
    if (previousKey && previousKey !== nextKey) {
      this.entries.delete(previousKey)
    }
    this.activeScopeByTurn.set(scope, nextKey)
    return this.scopeEntry(transport)
  }

  getEntry(sessionId: string, turnId: string): ToolWorkflowAggregateEntry | null {
    const activeKey = this.activeScopeByTurn.get(turnKey(sessionId, turnId))
    if (!activeKey) return null
    const entry = this.entries.get(activeKey)
    return entry ? publicEntry(entry) : null
  }

  getCursor(sessionId: string, turnId: string): ToolWorkflowAggregateCursor | null {
    return this.getEntry(sessionId, turnId)?.cursor ?? null
  }

  applyTransport(value: unknown, mode: 'live' | 'snapshot' = 'live'): ToolWorkflowAggregateApplyResult {
    let transport: ToolWorkflowAggregateTransport
    try {
      transport = parseToolWorkflowAggregateTransport(value)
    } catch (error) {
      const message = error instanceof Error ? error.message : 'invalid aggregate payload'
      const fallback = isRecord(value) && isNonEmptyString(value.storage_id) && isNonEmptyString(value.session_id) && isNonEmptyString(value.turn_id)
        ? this.activateScope({ storage_id: value.storage_id, session_id: value.session_id, turn_id: value.turn_id, aggregate: null, authoritative: false })
        : null
      if (fallback) {
        fallback.status = 'unsupported'
        fallback.diagnostic = message
        this.notify()
        return { kind: 'unsupported', entry: publicEntry(fallback) }
      }
      throw error
    }

    const entry = this.activateScope(transport)
    entry.authoritative = transport.authoritative === true
    entry.publicationEnabled = transport.publication_enabled === true
    if (transport.aggregate === null) {
      if (entry.aggregate === null) entry.status = transport.unsupported ? 'unsupported' : 'empty'
      else entry.status = transport.unsupported ? 'unsupported' : 'stale'
      entry.diagnostic = transport.error ?? null
      this.notify()
      return { kind: transport.unsupported ? 'unsupported' : 'duplicate', entry: publicEntry(entry) }
    }

    const aggregate = transport.aggregate
    if (mode === 'snapshot') {
      if (aggregate.seq < entry.cursor.lastSeq) {
        entry.status = 'stale'
        entry.diagnostic = `snapshot sequence ${aggregate.seq} is older than ${entry.cursor.lastSeq}`
        this.notify()
        return { kind: 'repair_required', entry: publicEntry(entry) }
      }
      entry.aggregatesBySeq.clear()
      entry.aggregatesBySeq.set(aggregate.seq, aggregate)
      entry.aggregate = aggregate
      entry.cursor.lastSeq = aggregate.seq
      entry.cursor.lastEventId = aggregate.event_id
      entry.cursor.lastIdempotencyKey = aggregate.idempotency_key
      entry.status = 'ready'
      entry.diagnostic = null
      this.notify()
      return { kind: 'applied', entry: publicEntry(entry) }
    }

    const known = entry.aggregatesBySeq.get(aggregate.seq)
    if (known) {
      if (
        known.event_id === aggregate.event_id &&
        known.idempotency_key === aggregate.idempotency_key &&
        stableJson(known) === stableJson(aggregate)
      ) {
        entry.diagnostic = null
        this.notify()
        return { kind: 'duplicate', entry: publicEntry(entry) }
      }
      entry.status = 'repair_required'
      entry.diagnostic = `conflicting aggregate at sequence ${aggregate.seq}`
      this.notify()
      return { kind: 'repair_required', entry: publicEntry(entry) }
    }

    const eventIdCollision = [...entry.aggregatesBySeq.values()].find(
      (knownAggregate) => knownAggregate.event_id === aggregate.event_id
    )
    const idempotencyCollision = [...entry.aggregatesBySeq.values()].find(
      (knownAggregate) => knownAggregate.idempotency_key === aggregate.idempotency_key
    )
    if (eventIdCollision || idempotencyCollision) {
      entry.status = 'repair_required'
      entry.diagnostic = 'aggregate event identity or idempotency key collided across sequences'
      this.notify()
      return { kind: 'repair_required', entry: publicEntry(entry) }
    }

    if (aggregate.seq !== entry.cursor.lastSeq + 1) {
      entry.status = 'repair_required'
      entry.diagnostic = aggregate.seq > entry.cursor.lastSeq
        ? `aggregate sequence gap: expected ${entry.cursor.lastSeq + 1}, received ${aggregate.seq}`
        : `out-of-order aggregate sequence ${aggregate.seq}`
      this.notify()
      return { kind: 'repair_required', entry: publicEntry(entry) }
    }

    entry.aggregatesBySeq.set(aggregate.seq, aggregate)
    entry.aggregate = aggregate
    entry.cursor.lastSeq = aggregate.seq
    entry.cursor.lastEventId = aggregate.event_id
    entry.cursor.lastIdempotencyKey = aggregate.idempotency_key
    entry.status = entry.authoritative ? 'ready' : 'stale'
    entry.diagnostic = entry.authoritative ? null : 'aggregate publication is not authoritative'
    this.notify()
    return { kind: 'applied', entry: publicEntry(entry) }
  }

  applySnapshot(value: unknown): ToolWorkflowAggregateApplyResult {
    return this.applyTransport(value, 'snapshot')
  }

  applyRange(value: unknown): ToolWorkflowAggregateApplyResult {
    if (!isRecord(value) || !Array.isArray(value.events)) throw new Error('aggregate range payload is invalid')
    const base = parseToolWorkflowAggregateTransport({
      storage_id: value.storage_id,
      session_id: value.session_id,
      turn_id: value.turn_id,
      aggregate: null,
      authoritative: value.authoritative,
      publication_enabled: value.publication_enabled,
    })
    const entry = this.activateScope(base)
    if (value.contiguous !== true) {
      entry.status = 'repair_required'
      entry.diagnostic = 'aggregate range is not contiguous'
      this.notify()
      return { kind: 'repair_required', entry: publicEntry(entry) }
    }
    let result: ToolWorkflowAggregateApplyResult = { kind: 'duplicate', entry: publicEntry(entry) }
    for (const aggregate of value.events) {
      result = this.applyTransport({ ...base, aggregate }, 'live')
      if (result.kind === 'repair_required' || result.kind === 'unsupported') return result
    }
    return result
  }

  clearTurn(sessionId: string, turnId: string): void {
    const scope = turnKey(sessionId, turnId)
    const activeKey = this.activeScopeByTurn.get(scope)
    if (activeKey) this.entries.delete(activeKey)
    this.activeScopeByTurn.delete(scope)
    this.notify()
  }
}

export const toolWorkflowAggregateStore = new ToolWorkflowAggregateStore()

export function projectToolWorkflowAggregateCall(
  sessionId: string | null | undefined,
  turnId: string | null | undefined,
  callId: string | null | undefined
): ToolWorkflowAggregateCallView | null {
  if (!sessionId || !turnId) return null
  const entry = toolWorkflowAggregateStore.getEntry(sessionId, turnId)
  if (!entry) return null
  const call = callId ? entry.aggregate?.state.calls.find((item) => item.call_id === callId) ?? null : null
  return {
    authoritative: entry.authoritative && entry.status === 'ready',
    turnStatus: entry.aggregate?.state.turn_status ?? 'unknown',
    integrity: entry.aggregate?.state.integrity ?? (entry.status === 'unsupported' ? 'unsupported' : 'partial'),
    call,
    aggregate: entry.aggregate,
    entryStatus: entry.status,
    diagnostic: entry.diagnostic,
  }
}

export function useToolWorkflowAggregateCall(
  sessionId: string | null | undefined,
  turnId: string | null | undefined,
  callId: string | null | undefined
): ToolWorkflowAggregateCallView | null {
  const revision = React.useSyncExternalStore(
    toolWorkflowAggregateStore.subscribe,
    () => toolWorkflowAggregateStore.getRevision(),
    () => 0,
  )
  void revision
  return projectToolWorkflowAggregateCall(sessionId, turnId, callId)
}
