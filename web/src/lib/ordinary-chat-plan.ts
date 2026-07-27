/** Replay-safe ordinary Chat adaptive runtime reducer.
 *
 * This state is a display projection only.  It is intentionally separate
 * from Goal/Workflow state and is never converted into assistant messages.
 */

import type { Message } from '@/lib/chat'

export type OrdinaryChatTurnStatus =
  | 'running'
  | 'awaiting_approval'
  | 'completed'
  | 'blocked'
  | 'partial'
  | 'cancelled'

export interface OrdinaryChatPlanItem {
  item_id: string | null
  title: string | null
  status: string | null
  dependencies: string[]
  success_criteria: string[]
  evidence_refs: string[]
  blocker_reason: string | null
  attempts: number | null
}

export interface OrdinaryChatPlan {
  ledger_id: string | null
  revision: number | null
  status: string | null
  objective: string | null
  reason_codes: string[]
  items: OrdinaryChatPlanItem[]
  blockers: string[]
}

export interface OrdinaryChatReceiptCriterion {
  criterion_id: string | null
  verdict: string | null
  verifier_id: string | null
  evidence_refs: string[]
  reason_code: string | null
  retry_disposition: string | null
}

export interface OrdinaryChatReceipt {
  receipt_id: string | null
  turn_id: string | null
  verdict: string
  hard_failure: boolean
  retry_disposition: string | null
  criteria: OrdinaryChatReceiptCriterion[]
}

export interface OrdinaryChatEvidence {
  status: string
  receipts: OrdinaryChatReceipt[]
}

export interface OrdinaryChatTurn {
  turn_id: string
  status: OrdinaryChatTurnStatus
  updated_sequence: number
  complexity: Record<string, unknown>
  plan: OrdinaryChatPlan | null
  retrieval: Record<string, unknown>
  evidence: OrdinaryChatEvidence
  recovery: Record<string, unknown>
  failure_learning: {
    candidate_count: number
    processed_count: number
  }
  blockers: string[]
}

export interface OrdinaryChatRuntimeEvent {
  event_id: string
  event: string
  schema_version: number
  sequence: number
  revision: number
  turn_id: string | null
  payload: Record<string, unknown>
}

export interface OrdinaryChatRuntimeProjection {
  projection_version: string
  schema_version: number
  session_id: string
  revision: number
  latest_sequence: number
  events: OrdinaryChatRuntimeEvent[]
  turns: OrdinaryChatTurn[]
  metrics: Record<string, unknown>
}

export interface OrdinaryChatPlanState {
  sessionId: string
  latestSequence: number
  turns: Record<string, OrdinaryChatTurn>
  seenEventIds: Set<string>
  ignoredEventCount: number
}

const EMPTY_EVIDENCE = (): OrdinaryChatEvidence => ({ status: 'not_observed', receipts: [] })

export function createOrdinaryChatPlanState(sessionId: string): OrdinaryChatPlanState {
  return {
    sessionId,
    latestSequence: 0,
    turns: {},
    seenEventIds: new Set<string>(),
    ignoredEventCount: 0,
  }
}

export function reduceOrdinaryChatPlanEvent(
  state: OrdinaryChatPlanState,
  event: OrdinaryChatRuntimeEvent
): OrdinaryChatPlanState {
  if (
    !event ||
    typeof event.event_id !== 'string' ||
    typeof event.sequence !== 'number' ||
    !Number.isFinite(event.sequence) ||
    (event.turn_id !== null && typeof event.turn_id !== 'string')
  ) {
    return { ...state, ignoredEventCount: state.ignoredEventCount + 1 }
  }
  if (state.seenEventIds.has(event.event_id) || event.sequence < state.latestSequence) {
    return { ...state, ignoredEventCount: state.ignoredEventCount + 1 }
  }
  const seenEventIds = new Set(state.seenEventIds)
  seenEventIds.add(event.event_id)
  const next: OrdinaryChatPlanState = {
    ...state,
    latestSequence: Math.max(state.latestSequence, event.sequence),
    seenEventIds,
    turns: { ...state.turns },
  }
  if (!event.turn_id) {
    return next
  }

  const previous = state.turns[event.turn_id] ?? createTurn(event.turn_id)
  const turn: OrdinaryChatTurn = {
    ...previous,
    complexity: { ...previous.complexity },
    plan: previous.plan
      ? {
          ...previous.plan,
          items: previous.plan.items.map((item) => ({ ...item })),
          reason_codes: [...previous.plan.reason_codes],
          blockers: [...previous.plan.blockers],
        }
      : null,
    retrieval: { ...previous.retrieval },
    evidence: {
      ...previous.evidence,
      receipts: previous.evidence.receipts.map((receipt) => ({
        ...receipt,
        criteria: receipt.criteria.map((criterion) => ({
          ...criterion,
          evidence_refs: [...criterion.evidence_refs],
        })),
      })),
    },
    recovery: { ...previous.recovery },
    failure_learning: { ...previous.failure_learning },
    blockers: [...previous.blockers],
    updated_sequence: Math.max(previous.updated_sequence, event.sequence),
  }
  applyEvent(turn, event)
  turn.status = deriveStatus(turn)
  next.turns[event.turn_id] = turn
  return next
}

export function reduceOrdinaryChatPlanEvents(
  sessionId: string,
  events: ReadonlyArray<OrdinaryChatRuntimeEvent>
): OrdinaryChatPlanState {
  return events
    .slice()
    .sort((left, right) => left.sequence - right.sequence)
    .reduce(reduceOrdinaryChatPlanEvent, createOrdinaryChatPlanState(sessionId))
}

export function hydrateOrdinaryChatPlanProjection(
  projection: OrdinaryChatRuntimeProjection | null | undefined
): OrdinaryChatPlanState {
  if (!projection || typeof projection.session_id !== 'string') {
    return createOrdinaryChatPlanState('')
  }
  const state = reduceOrdinaryChatPlanEvents(projection.session_id, projection.events ?? [])
  if (state.latestSequence > 0 || Object.keys(state.turns).length > 0) {
    return state
  }
  for (const turn of projection.turns ?? []) {
    if (turn && typeof turn.turn_id === 'string') {
      state.turns[turn.turn_id] = normalizeTurn(turn)
    }
  }
  state.latestSequence = projection.latest_sequence ?? 0
  return state
}

export function normalizeOrdinaryChatRuntimeProjection(
  value: unknown
): OrdinaryChatRuntimeProjection | null {
  if (!isRecord(value)) return null
  const sessionId = getString(value.session_id)
  if (!sessionId) return null
  const events: OrdinaryChatRuntimeEvent[] = Array.isArray(value.events)
    ? value.events.filter(isRecord).flatMap((item) => {
        const eventId = getString(item.event_id)
        const eventName = getString(item.event)
        const sequence = getNullableNumber(item.sequence)
        const revision = getNullableNumber(item.revision)
        if (!eventId || !eventName || sequence === null || revision === null) return []
        return [{
          event_id: eventId,
          event: eventName,
          schema_version: getNullableNumber(item.schema_version) ?? 1,
          sequence,
          revision,
          turn_id: item.turn_id === null ? null : getString(item.turn_id),
          payload: isRecord(item.payload) ? { ...item.payload } : {},
        }]
      })
    : []
  const turns: OrdinaryChatTurn[] = Array.isArray(value.turns)
    ? value.turns
        .filter(isRecord)
        .flatMap((item) => {
          const turnId = getString(item.turn_id)
          return turnId ? [normalizeTurn(item as unknown as OrdinaryChatTurn)] : []
        })
    : []
  return {
    projection_version: getString(value.projection_version) ?? 'ordinary-chat-adaptive-runtime-v1',
    schema_version: getNullableNumber(value.schema_version) ?? 1,
    session_id: sessionId,
    revision: getNullableNumber(value.revision) ?? 0,
    latest_sequence: getNullableNumber(value.latest_sequence) ?? 0,
    events,
    turns,
    metrics: isRecord(value.metrics) ? { ...value.metrics } : {},
  }
}

export function normalizeOrdinaryChatRuntimeEvent(
  value: unknown
): OrdinaryChatRuntimeEvent | null {
  if (!isRecord(value)) return null
  const eventId = getString(value.event_id)
  const eventName = getString(value.event)
  const sequence = getNullableNumber(value.sequence)
  const revision = getNullableNumber(value.revision)
  if (!eventId || !eventName || sequence === null || revision === null) return null
  return {
    event_id: eventId,
    event: eventName,
    schema_version: getNullableNumber(value.schema_version) ?? 1,
    sequence,
    revision,
    turn_id: value.turn_id === null ? null : getString(value.turn_id),
    payload: isRecord(value.payload) ? { ...value.payload } : {},
  }
}

export function sortOrdinaryChatTurns(state: OrdinaryChatPlanState): OrdinaryChatTurn[] {
  return Object.values(state.turns).sort(
    (left, right) => left.updated_sequence - right.updated_sequence
  )
}

export function markProvisionalOrdinaryChatFinals(
  messages: ReadonlyArray<Message>,
  state: OrdinaryChatPlanState
): Message[] {
  return messages.map((message) => {
    if (message.type !== 'assistant' || message.eventType !== 'final_answer' || !message.turnId) {
      return message
    }
    const turn = state.turns[message.turnId]
    if (!turn || !['blocked', 'partial', 'cancelled'].includes(turn.status)) return message
    return {
      ...message,
      content: `Provisional answer — verification ${turn.status}:\n\n${message.content}`,
      eventType: 'status',
      isStreaming: false,
    }
  })
}

function applyEvent(turn: OrdinaryChatTurn, event: OrdinaryChatRuntimeEvent): void {
  const payload = isRecord(event.payload) ? event.payload : {}
  if (event.event === 'ordinary_chat_plan_ledger_updated') {
    const plan = isRecord(payload.plan) ? normalizePlan(payload.plan) : null
    if (plan && (!turn.plan || (plan.revision ?? -1) >= (turn.plan.revision ?? -1))) {
      turn.plan = plan
    }
  } else if (event.event === 'complexity_decision') {
    if (isRecord(payload.decision)) turn.complexity = { ...payload.decision }
  } else if (event.event === 'tool_retrieval_result') {
    if (isRecord(payload.retrieval)) turn.retrieval = { ...payload.retrieval }
  } else if (event.event === 'recovery_decision') {
    if (isRecord(payload.recovery)) turn.recovery = { ...payload.recovery }
  } else if (event.event === 'ordinary_chat_verification_receipt_recorded') {
    const receipt = isRecord(payload.receipt) ? normalizeReceipt(payload.receipt) : null
    if (receipt) {
      turn.evidence.receipts = [
        ...turn.evidence.receipts.filter((item) => item.receipt_id !== receipt.receipt_id),
        receipt,
      ].slice(-8)
      turn.evidence.status = receipt.verdict
      if (receipt.verdict === 'failed' || receipt.verdict === 'unverified') {
        addBlocker(turn, receipt.verdict)
      }
    }
  } else if (event.event === 'turn_execution_checkpoint') {
    if (isRecord(payload.complexity)) turn.complexity = { ...payload.complexity }
    if (isRecord(payload.retrieval)) turn.retrieval = { ...payload.retrieval }
    if (isRecord(payload.recovery)) turn.recovery = { ...payload.recovery }
    const stage = getString(payload.stage)
    if (stage) (turn as OrdinaryChatTurn & { stage?: string }).stage = stage
    const verificationStatus = getString(payload.verification_status)
    if (verificationStatus) turn.evidence.status = verificationStatus
    addBlocker(turn, getString(payload.blocker_reason))
    if (stage === 'awaiting_approval') addBlocker(turn, 'awaiting_approval')
  } else if (event.event === 'session_turn_timeline') {
    const status = getString(payload.status)
    const terminalOutcome = getString(payload.terminal_outcome)
    const cancellationOutcome = getString(payload.cancellation_outcome)
    if (status === 'cancelled' || terminalOutcome === 'cancelled' || cancellationOutcome) {
      ;(turn as OrdinaryChatTurn & { timelineCancelled?: boolean }).timelineCancelled = true
      addBlocker(turn, cancellationOutcome ?? 'cancelled')
    }
  } else if (event.event === 'message') {
    addBlocker(turn, getString(payload.error_code))
    addBlocker(turn, getString(payload.status) === 'blocked' ? 'blocked' : null)
  } else if (event.event === 'failure_learning_candidate') {
    turn.failure_learning.candidate_count += 1
    const reasons = Array.isArray(payload.reason_codes) ? payload.reason_codes : []
    for (const reason of reasons) addBlocker(turn, getString(reason))
  } else if (event.event === 'failure_learning_processed') {
    turn.failure_learning.processed_count += 1
  }
}

function createTurn(turnId: string): OrdinaryChatTurn {
  return {
    turn_id: turnId,
    status: 'running',
    updated_sequence: 0,
    complexity: {},
    plan: null,
    retrieval: {},
    evidence: EMPTY_EVIDENCE(),
    recovery: {},
    failure_learning: { candidate_count: 0, processed_count: 0 },
    blockers: [],
  }
}

function normalizeTurn(value: OrdinaryChatTurn): OrdinaryChatTurn {
  const turn = createTurn(value.turn_id)
  return {
    ...turn,
    ...value,
    complexity: isRecord(value.complexity) ? { ...value.complexity } : {},
    plan: value.plan ? normalizePlan(value.plan) : null,
    retrieval: isRecord(value.retrieval) ? { ...value.retrieval } : {},
    evidence: normalizeEvidence(value.evidence),
    recovery: isRecord(value.recovery) ? { ...value.recovery } : {},
    blockers: Array.isArray(value.blockers) ? value.blockers.filter((item) => typeof item === 'string') : [],
  }
}

function normalizePlan(value: Record<string, unknown> | OrdinaryChatPlan): OrdinaryChatPlan {
  return {
    ledger_id: getNullableString(value.ledger_id),
    revision: getNullableNumber(value.revision),
    status: getNullableString(value.status),
    objective: getNullableString(value.objective),
    reason_codes: getStringArray(value.reason_codes),
    items: Array.isArray(value.items)
      ? value.items.filter(isRecord).map((item) => ({
          item_id: getNullableString(item.item_id),
          title: getNullableString(item.title),
          status: getNullableString(item.status),
          dependencies: getStringArray(item.dependencies),
          success_criteria: getStringArray(item.success_criteria),
          evidence_refs: getStringArray(item.evidence_refs),
          blocker_reason: getNullableString(item.blocker_reason),
          attempts: getNullableNumber(item.attempts),
        }))
      : [],
    blockers: getStringArray(value.blockers),
  }
}

function normalizeEvidence(value: OrdinaryChatEvidence): OrdinaryChatEvidence {
  return {
    status: typeof value?.status === 'string' ? value.status : 'not_observed',
    receipts: Array.isArray(value?.receipts)
      ? value.receipts
          .filter(isRecord)
          .map((receipt) => normalizeReceipt(receipt as unknown as Record<string, unknown>))
      : [],
  }
}

function normalizeReceipt(value: Record<string, unknown>): OrdinaryChatReceipt {
  return {
    receipt_id: getNullableString(value.receipt_id),
    turn_id: getNullableString(value.turn_id),
    verdict: getString(value.verdict) ?? 'unverified',
    hard_failure: value.hard_failure === true,
    retry_disposition: getNullableString(value.retry_disposition),
    criteria: Array.isArray(value.criteria)
      ? value.criteria.filter(isRecord).map((criterion) => ({
          criterion_id: getNullableString(criterion.criterion_id),
          verdict: getNullableString(criterion.verdict),
          verifier_id: getNullableString(criterion.verifier_id),
          evidence_refs: getStringArray(criterion.evidence_refs),
          reason_code: getNullableString(criterion.reason_code),
          retry_disposition: getNullableString(criterion.retry_disposition),
        }))
      : [],
  }
}

function deriveStatus(turn: OrdinaryChatTurn): OrdinaryChatTurnStatus {
  const extended = turn as OrdinaryChatTurn & { stage?: string; timelineCancelled?: boolean }
  if (extended.timelineCancelled) return 'cancelled'
  if (turn.evidence.status === 'failed' || turn.plan?.status === 'blocked') return 'blocked'
  if (extended.stage === 'blocked') return 'blocked'
  if (turn.evidence.status === 'unverified') return 'partial'
  if (turn.plan?.status === 'cancelled') return 'cancelled'
  if (turn.plan?.status === 'completed' || extended.stage === 'completed') return 'completed'
  if (extended.stage === 'awaiting_approval') return 'awaiting_approval'
  return 'running'
}

function addBlocker(turn: OrdinaryChatTurn, value: string | null): void {
  if (value && !turn.blockers.includes(value)) turn.blockers.push(value)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getNullableString(value: unknown): string | null {
  return getString(value)
}

function getNullableNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string')
    : []
}

export const ordinaryChatPlanTestExports = {
  normalizePlan,
  normalizeReceipt,
}
