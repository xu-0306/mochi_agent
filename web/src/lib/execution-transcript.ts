import type { ExecutionTranscriptEvent, SubagentTranscriptSummary } from '@/lib/api'
import {
  deriveSubagentEventStatus,
  deriveSubagentMessageDeliveryStatus,
} from '@/lib/subagent-protocol-events'

declare module '@/lib/api' {
  interface ExecutionTranscriptEvent {
    eventId?: string | null
    dedupeKey?: string | null
    visibility?: string | null
    durability?: string | null
    projectionLane?: string | null
  }
}

const GOAL_SURFACE_LIFECYCLE_EVENT_TYPES = new Set([
  'runtime_blocked',
  'run_completed',
  'run_failed',
  'run_finalized_partial',
  'run_paused',
  'run_resumed',
  'run_started',
  'subagent_completed',
  'subagent_interrupted',
  'subagent_started',
])

const DURABLE_WORKFLOW_LIFECYCLE_EVENT_TYPES = new Set([
  'run_completed',
  'run_failed',
  'run_finalized_partial',
  'run_paused',
  'run_resumed',
  'run_started',
])

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function getMetadataString(metadata: Record<string, unknown>, ...keys: string[]): string | null {
  for (const key of keys) {
    const value = getString(metadata[key])
    if (value) {
      return value
    }
  }
  return null
}

function getMetadataStringArray(metadata: Record<string, unknown>, ...keys: string[]): string[] {
  for (const key of keys) {
    const arrayValue = getStringArray(metadata[key])
    if (arrayValue.length > 0) {
      return arrayValue
    }
    const singleValue = getString(metadata[key])
    if (singleValue) {
      return [singleValue]
    }
  }
  return []
}

function normalizeStatus(value: string | null | undefined): string {
  const normalized = value?.trim().toLowerCase() ?? ''
  return normalized === 'canceled' ? 'cancelled' : normalized
}

function normalizeContractToken(value: string | null | undefined): string | null {
  const normalized = value?.trim().toLowerCase() ?? ''
  return normalized.length > 0 ? normalized : null
}

function getEventToolName(event: ExecutionTranscriptEvent): string | null {
  return (
    getMetadataString(event.metadata, 'tool_name', 'toolName') ??
    getMetadataStringArray(event.metadata, 'tool_names', 'toolNames')[0] ??
    null
  )
}

function getEventApprovalIds(event: ExecutionTranscriptEvent): string[] {
  return [
    ...getMetadataStringArray(event.metadata, 'approval_ids', 'approvalIds'),
    ...(getMetadataString(event.metadata, 'approval_id', 'approvalId') ?? '').split(','),
  ]
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

function getEventRecommendedAction(event: ExecutionTranscriptEvent): string | null {
  return getMetadataString(event.metadata, 'recommended_action', 'recommendedAction')
}

function getEventContractField(
  event: ExecutionTranscriptEvent,
  field: keyof Pick<
    ExecutionTranscriptEvent,
    'eventId' | 'dedupeKey' | 'visibility' | 'durability' | 'projectionLane'
  >,
  ...metadataKeys: string[]
): string | null {
  return getString(event[field]) ?? getMetadataString(event.metadata, ...metadataKeys)
}

function hasSourceContractIdentity(event: ExecutionTranscriptEvent): boolean {
  return Boolean(
    getEventContractField(event, 'eventId', 'event_id', 'eventId') ??
      getEventContractField(event, 'dedupeKey', 'dedupe_key', 'dedupeKey') ??
      getEventContractField(event, 'projectionLane', 'projection_lane', 'projectionLane')
  )
}

function isGoalSurfaceToolResult(event: ExecutionTranscriptEvent): boolean {
  if (event.type !== 'subagent_tool_result') {
    return false
  }
  const status = normalizeStatus(deriveSubagentEventStatus(event.type, event.status))
  return (
    status === 'approval_required' ||
    status === 'awaiting_approval' ||
    status === 'blocked' ||
    status === 'cancelled' ||
    status === 'failed' ||
    status === 'error'
  )
}

function shouldKeepGoalSurfaceEvent(event: ExecutionTranscriptEvent): boolean {
  const visibility = normalizeContractToken(getEventContractField(event, 'visibility', 'visibility'))
  if (visibility === 'hidden' || visibility === 'none') {
    return false
  }

  const projectionLane = normalizeContractToken(
    getEventContractField(
      event,
      'projectionLane',
      'projection_lane',
      'projectionLane'
    )
  )
  if (projectionLane) {
    return projectionLane === 'goal_surface'
  }

  if (GOAL_SURFACE_LIFECYCLE_EVENT_TYPES.has(event.type)) {
    return true
  }
  return isGoalSurfaceToolResult(event)
}

function sameGoalSurfaceScope(
  left: ExecutionTranscriptEvent,
  right: ExecutionTranscriptEvent
): boolean {
  if (left.subagentId && right.subagentId) {
    return left.subagentId === right.subagentId
  }
  return left.parentType === right.parentType && left.parentId === right.parentId
}

function isRedundantApprovalToolResult(
  current: ExecutionTranscriptEvent,
  next: ExecutionTranscriptEvent | null
): boolean {
  if (current.type !== 'subagent_tool_result' || !next || next.type !== 'runtime_blocked') {
    return false
  }
  const status = normalizeStatus(deriveSubagentEventStatus(current.type, current.status))
  if (status !== 'approval_required' && status !== 'awaiting_approval' && status !== 'blocked') {
    return false
  }
  if (!sameGoalSurfaceScope(current, next)) {
    return false
  }

  const currentApprovals = getEventApprovalIds(current)
  const nextApprovals = getEventApprovalIds(next)
  if (
    currentApprovals.length > 0 &&
    nextApprovals.length > 0 &&
    currentApprovals.some((approvalId) => nextApprovals.includes(approvalId))
  ) {
    return true
  }

  const currentToolName = getEventToolName(current)
  const nextToolName = getEventToolName(next)
  const currentRecommendedAction = getEventRecommendedAction(current)
  const nextRecommendedAction = getEventRecommendedAction(next)

  return Boolean(
    currentToolName &&
    nextToolName &&
    currentRecommendedAction &&
    nextRecommendedAction &&
    currentToolName === nextToolName &&
    currentRecommendedAction === nextRecommendedAction
  )
}

function buildGoalSurfaceFingerprint(event: ExecutionTranscriptEvent): string {
  const dedupeKey = getEventContractField(event, 'dedupeKey', 'dedupe_key', 'dedupeKey')
  if (dedupeKey) {
    return `contract:${dedupeKey}`
  }
  const eventId = getEventContractField(event, 'eventId', 'event_id', 'eventId')
  if (eventId) {
    return `event:${eventId}`
  }

  const status = normalizeStatus(deriveSubagentEventStatus(event.type, event.status))
  return [
    event.type,
    event.parentType ?? '',
    event.parentId ?? '',
    event.subagentId ?? '',
    event.roleId ?? '',
    status,
    getEventToolName(event) ?? '',
    getEventApprovalIds(event).join(','),
    getEventRecommendedAction(event) ?? '',
    event.summary ?? '',
    event.content ?? '',
  ].join('::')
}

function mergeTranscriptMetadata(
  event: Record<string, unknown>,
  metadataCandidate: Record<string, unknown>
): Record<string, unknown> {
  const metadata = { ...metadataCandidate }

  const assignStringField = (targetKey: string, sourceKeys: string[]) => {
    if (getString(metadata[targetKey])) {
      return
    }
    for (const key of sourceKeys) {
      const value = getString(event[key]) ?? getString(metadata[key])
      if (value) {
        metadata[targetKey] = value
        return
      }
    }
  }

  const assignStringArrayField = (targetKey: string, sourceKeys: string[]) => {
    const currentValue = getMetadataStringArray(metadata, targetKey)
    if (currentValue.length > 0) {
      return
    }
    for (const key of sourceKeys) {
      const eventValue = getStringArray(event[key])
      if (eventValue.length > 0) {
        metadata[targetKey] = eventValue
        return
      }
      const metadataValue = getMetadataStringArray(metadata, key)
      if (metadataValue.length > 0) {
        metadata[targetKey] = metadataValue
        return
      }
      const singleValue = getString(event[key])
      if (singleValue) {
        metadata[targetKey] = [singleValue]
        return
      }
    }
  }

  assignStringField('blocker_type', ['blocker_type', 'blockerType'])
  assignStringField('message_id', ['message_id', 'messageId'])
  assignStringField('event_id', ['event_id', 'eventId'])
  assignStringField('dedupe_key', ['dedupe_key', 'dedupeKey'])
  assignStringField('visibility', ['visibility'])
  assignStringField('durability', ['durability'])
  assignStringField('projection_lane', ['projection_lane', 'projectionLane'])
  assignStringField('delivery_mode', ['delivery_mode', 'deliveryMode'])
  assignStringField('delivery_status', ['delivery_status', 'deliveryStatus'])
  assignStringField('delivery_reason', ['delivery_reason', 'deliveryReason'])
  assignStringField('approval_id', ['approval_id', 'approvalId'])
  assignStringField('approval_kind', ['approval_kind', 'approvalKind'])
  assignStringField('approval_scope', ['approval_scope', 'approvalScope'])
  assignStringField('approval_status', ['approval_status', 'approvalStatus'])
  assignStringArrayField('approval_ids', ['approval_ids', 'approvalIds'])
  assignStringArrayField('allowed_decisions', ['allowed_decisions', 'allowedDecisions'])
  assignStringArrayField('tool_names', ['tool_names', 'toolNames'])
  assignStringField('recommended_action', ['recommended_action', 'recommendedAction'])
  assignStringField('reason', ['reason'])
  assignStringField('security_decision', ['security_decision', 'securityDecision'])
  assignStringField('policy_source', ['policy_source', 'policySource'])
  assignStringField('approval_wait_started_at', ['approval_wait_started_at', 'approvalWaitStartedAt'])
  assignStringField('approval_wait_expires_at', ['approval_wait_expires_at', 'approvalWaitExpiresAt'])
  if (metadata.replay_safe === undefined) {
    const replaySafe = event.replay_safe ?? metadata.replay_safe
    if (typeof replaySafe === 'boolean') {
      metadata.replay_safe = replaySafe
    }
  }
  if (metadata.approval_wait_timeout_sec === undefined) {
    const timeout = event.approval_wait_timeout_sec ?? metadata.approval_wait_timeout_sec
    if (typeof timeout === 'number' && Number.isFinite(timeout)) {
      metadata.approval_wait_timeout_sec = timeout
    }
  }
  if (metadata.approval_wait_elapsed_sec === undefined) {
    const elapsed = event.approval_wait_elapsed_sec ?? metadata.approval_wait_elapsed_sec
    if (typeof elapsed === 'number' && Number.isFinite(elapsed)) {
      metadata.approval_wait_elapsed_sec = elapsed
    }
  }
  if (
    metadata.pending_approvals === undefined &&
    (Array.isArray(event.pending_approvals) || Array.isArray(event.pendingApprovals))
  ) {
    metadata.pending_approvals = Array.isArray(event.pending_approvals)
      ? event.pending_approvals
      : event.pendingApprovals
  }

  return metadata
}

function buildFallbackSubagentId(event: ExecutionTranscriptEvent): string | null {
  const parentId = event.parentId ?? getMetadataString(event.metadata, 'parent_id', 'run_id', 'goal_id')
  const roleId =
    event.roleId ??
    getMetadataString(event.metadata, 'role_id', 'roleId', 'subagent_id', 'subagentId')

  if (!parentId && !roleId) {
    return null
  }
  return [parentId ?? 'runtime', roleId ?? event.type].join('::')
}

export function mapAgentRunEventToTranscriptEvent(
  event: Record<string, unknown>
): ExecutionTranscriptEvent | null {
  const rawType =
    getString(event.type) ??
    getString(event.event_type) ??
    getString(event.kind)
  if (!rawType) {
    return null
  }
  const type = mapRuntimeTranscriptEventType(rawType)

  const metadata = mergeTranscriptMetadata(
    event,
    {
      ...(isRecord(event.details) ? event.details : {}),
      ...(isRecord(event.metadata) ? event.metadata : {}),
    }
  )

  const createdAt =
    getString(event.created_at) ??
    getString(event.createdAt) ??
    getString(event.timestamp) ??
    null
  const parentType =
    getString(event.parent_type) ??
    getString(event.parentType) ??
    getString(event.scope) ??
    (getString(event.run_id) ? 'agent_run' : null)
  const parentId =
    getString(event.parent_id) ??
    getString(event.parentId) ??
    getString(event.run_id) ??
    getString(event.goal_id) ??
    null
  const roleId =
    getString(event.role_id) ??
    getString(event.roleId) ??
    getString(metadata.role_id) ??
    null
  const explicitSubagentId =
    getString(event.subagent_id) ??
    getString(event.subagentId) ??
    getString(metadata.subagent_id) ??
    null
  const fallbackSubagentId =
    type === 'runtime_blocked'
      ? null
      : roleId ?? buildFallbackSubagentId({
        type,
        metadata,
        parentId,
        parentType,
        roleId,
      })
  const deliveryStatus =
    getString(event.delivery_status) ??
    getString(event.deliveryStatus) ??
    getString(metadata.delivery_status) ??
    deriveSubagentMessageDeliveryStatus(type)
  if (deliveryStatus && !getString(metadata.delivery_status)) {
    metadata.delivery_status = deliveryStatus
  }
  const messageId =
    getString(event.message_id) ??
    getString(event.messageId) ??
    getString(metadata.message_id) ??
    null
  const deliveryMode =
    getString(event.delivery_mode) ??
    getString(event.deliveryMode) ??
    getString(metadata.delivery_mode) ??
    null
  const deliveryReason =
    getString(event.delivery_reason) ??
    getString(event.deliveryReason) ??
    getString(metadata.delivery_reason) ??
    null
  const eventId =
    getString(event.event_id) ??
    getString(event.eventId) ??
    getString(metadata.event_id) ??
    getString(metadata.eventId) ??
    null
  const dedupeKey =
    getString(event.dedupe_key) ??
    getString(event.dedupeKey) ??
    getString(metadata.dedupe_key) ??
    getString(metadata.dedupeKey) ??
    null
  const visibility =
    getString(event.visibility) ??
    getString(metadata.visibility) ??
    null
  const durability =
    getString(event.durability) ??
    getString(metadata.durability) ??
    null
  const projectionLane =
    getString(event.projection_lane) ??
    getString(event.projectionLane) ??
    getString(metadata.projection_lane) ??
    getString(metadata.projectionLane) ??
    null
  const status = deriveSubagentEventStatus(
    type,
    getString(event.status) ??
      getString(metadata.status) ??
      deliveryStatus ??
      (rawType === 'role_error' ? 'failed' : null)
  )
  if (status && !getString(metadata.status)) {
    metadata.status = status
  }

  const normalized: ExecutionTranscriptEvent = {
    type,
    seq:
      (typeof event.seq === 'number' && Number.isFinite(event.seq) ? event.seq : null) ??
      (typeof event.sequence === 'number' && Number.isFinite(event.sequence) ? event.sequence : null) ??
      (typeof event.event_seq === 'number' && Number.isFinite(event.event_seq) ? event.event_seq : null) ??
      undefined,
    parentType,
    parentId,
    subagentId: explicitSubagentId ?? fallbackSubagentId,
    roleId,
    title:
      getString(event.title) ??
      getString(event.role_title) ??
      getString(event.name) ??
      getString(metadata.title) ??
      null,
    modelId:
      getString(event.model_id) ??
      getString(event.modelId) ??
      getString(metadata.model_id) ??
      null,
    status:
      status,
    content:
      getString(event.content) ??
      getString(event.message) ??
      getString(event.text) ??
      getString(event.current_action) ??
      getString(metadata.content) ??
      null,
    summary:
      getString(event.summary) ??
      getString(event.output_preview) ??
      getString(event.prompt_preview) ??
      getString(metadata.summary) ??
      null,
    metadata,
    createdAt,
    messageId,
    deliveryMode,
    deliveryStatus,
    deliveryReason,
    eventId,
    dedupeKey,
    visibility,
    durability,
    projectionLane,
  }

  if (!normalized.subagentId && type !== 'runtime_blocked') {
    normalized.subagentId = buildFallbackSubagentId(normalized)
  }

  return normalized
}

export function projectGoalSurfaceExecutionEvents(
  events: ExecutionTranscriptEvent[]
): ExecutionTranscriptEvent[] {
  const meaningfulEvents = events.filter((event) => shouldKeepGoalSurfaceEvent(event))
  const projected: ExecutionTranscriptEvent[] = []

  for (let index = 0; index < meaningfulEvents.length; index += 1) {
    const event = meaningfulEvents[index]
    const nextEvent = meaningfulEvents[index + 1] ?? null

    if (isRedundantApprovalToolResult(event, nextEvent)) {
      if (
        !hasSourceContractIdentity(event) &&
        !(nextEvent && hasSourceContractIdentity(nextEvent))
      ) {
        continue
      }
    }

    const fingerprint = buildGoalSurfaceFingerprint(event)
    const previous = projected.length > 0 ? projected[projected.length - 1] : null
    if (previous && buildGoalSurfaceFingerprint(previous) === fingerprint) {
      projected[projected.length - 1] = event
      continue
    }

    projected.push(event)
  }

  return projected
}

function mapRuntimeTranscriptEventType(type: string): string {
  if (type === 'role_started') {
    return 'subagent_started'
  }
  if (type === 'role_progress') {
    return 'subagent_progress'
  }
  if (type === 'role_completed' || type === 'role_error') {
    return 'subagent_completed'
  }
  return type
}

export function formatWorkflowLifecycleMessage(event: Record<string, unknown>): string {
  const type = getString(event.type) ?? 'workflow_event'
  const status = getString(event.status)

  if (type === 'run_started') {
    return 'Run started.'
  }
  if (type === 'run_resumed') {
    return 'Run resumed.'
  }
  if (type === 'run_paused') {
    return 'Run paused.'
  }
  if (type === 'run_completed') {
    return 'Run completed.'
  }
  if (type === 'run_failed') {
    return `Run failed${status ? `: ${status}.` : '.'}`
  }
  if (type === 'run_finalized_partial') {
    return 'Run finalized as partial.'
  }

  return type.replaceAll('_', ' ')
}

export function projectWorkflowRunEventToSessionEvent(
  event: Record<string, unknown>,
  options: {
    runId: string
    turnId: string
  }
): Record<string, unknown> | null {
  const type = getString(event.type)
  if (!type) {
    return null
  }

  const timestamp = getString(event.timestamp) ?? new Date().toISOString()
  const nestedPayload = isRecord(event.payload) ? event.payload : null
  const metadata = {
    ...(isRecord(event.metadata) ? event.metadata : {}),
    channel: 'workflow',
    workflow_run_id: options.runId,
  }

  if (type === 'turn_event') {
    return {
      type: 'turn_event',
      phase: getString(event.phase) ?? 'status',
      timestamp,
      turn_id: options.turnId,
      payload: nestedPayload
        ? {
            ...nestedPayload,
            metadata: {
              ...(isRecord(nestedPayload.metadata) ? nestedPayload.metadata : {}),
              channel: 'workflow',
              workflow_run_id: options.runId,
            },
          }
        : {
            content: formatWorkflowLifecycleMessage(event),
            metadata,
          },
    }
  }

  if (
    type === 'thinking' ||
    type === 'status' ||
    type === 'tool_call_request' ||
    type === 'tool_call_result' ||
    type === 'error' ||
    type === 'final_answer'
  ) {
    return {
      ...event,
      type,
      timestamp,
      turn_id: options.turnId,
      metadata,
    }
  }

  if (type === 'text_chunk' && getString(event.content)) {
    return {
      type: 'text_chunk',
      content: getString(event.content) ?? '',
      timestamp,
      turn_id: options.turnId,
    }
  }

  if (type === 'operator_message') {
    return {
      type: 'message',
      role: 'user',
      content: getString(event.content) ?? '',
      attachments: Array.isArray(event.attachments) ? event.attachments : [],
      timestamp,
      turn_id: options.turnId,
      metadata,
    }
  }

  if (type === 'assistant_message') {
    const eventMetadata = isRecord(event.metadata) ? event.metadata : {}
    if (eventMetadata.acknowledgement === true) {
      return null
    }
    return {
      type: 'message',
      role: 'assistant',
      content: getString(event.content) ?? '',
      timestamp,
      turn_id: options.turnId,
      metadata,
    }
  }

  if (!DURABLE_WORKFLOW_LIFECYCLE_EVENT_TYPES.has(type)) {
    return null
  }

  return {
    type: 'turn_event',
    phase: 'status',
    timestamp,
    turn_id: options.turnId,
    payload: {
      content: formatWorkflowLifecycleMessage(event),
      status: getString(event.status),
      event_type: type,
      metadata: {
        channel: 'workflow',
        workflow_run_id: options.runId,
        raw_phase: 'workflow_status',
      },
    },
  }
}

export function groupTranscriptEventsBySubagent(
  events: ExecutionTranscriptEvent[]
): Record<string, ExecutionTranscriptEvent[]> {
  return events.reduce<Record<string, ExecutionTranscriptEvent[]>>((groups, event) => {
    const subagentId =
      event.subagentId ??
      event.roleId ??
      buildFallbackSubagentId(event) ??
      '__runtime__'
    if (!groups[subagentId]) {
      groups[subagentId] = []
    }
    groups[subagentId].push(event)
    return groups
  }, {})
}

export function summarizeRuntimeBlocker(event: ExecutionTranscriptEvent): string {
  const metadata = event.metadata
  const blockerType =
    getString(metadata.blocker_type) ??
    getString(metadata.blockerType) ??
    null
  const approvalIds = getMetadataStringArray(metadata, 'approval_ids', 'approvalIds')
  const toolNames = getMetadataStringArray(metadata, 'tool_names', 'toolNames')
  const recommendedAction =
    getString(metadata.recommended_action) ??
    getString(metadata.recommendedAction) ??
    null
  const baseSummary =
    event.summary ??
    event.content ??
    getMetadataString(metadata, 'summary', 'message') ??
    'Runtime is blocked.'

  const details: string[] = []
  if (blockerType === 'approval' && approvalIds.length > 0) {
    details.push(`approval ${approvalIds.join(', ')}`)
  }
  if (toolNames.length > 0) {
    details.push(`tool ${toolNames.join(', ')}`)
  }
  if (recommendedAction) {
    details.push(`next ${recommendedAction.replaceAll('_', ' ')}`)
  }

  return details.length > 0 ? `${baseSummary} (${details.join(' / ')})` : baseSummary
}

export function mergeSubagentSummaries(
  current: SubagentTranscriptSummary[],
  next: SubagentTranscriptSummary[]
): SubagentTranscriptSummary[] {
  const merged = new Map<string, SubagentTranscriptSummary>()
  for (const item of current) {
    merged.set(item.subagentId, item)
  }
  for (const item of next) {
    const existing = merged.get(item.subagentId)
    if (!existing) {
      merged.set(item.subagentId, item)
      continue
    }
    merged.set(item.subagentId, {
      ...existing,
      ...item,
      promptPreview: item.promptPreview ?? existing.promptPreview ?? null,
      summary: item.summary ?? existing.summary ?? null,
      outputPreview: item.outputPreview ?? existing.outputPreview ?? null,
      eventCount: Math.max(existing.eventCount ?? 0, item.eventCount ?? 0),
      createdAt: existing.createdAt ?? item.createdAt ?? null,
      updatedAt: item.updatedAt ?? existing.updatedAt ?? item.createdAt ?? existing.createdAt ?? null,
    })
  }
  return Array.from(merged.values()).sort((left, right) => {
    const leftTime = Date.parse(left.updatedAt ?? left.createdAt ?? '') || 0
    const rightTime = Date.parse(right.updatedAt ?? right.createdAt ?? '') || 0
    return rightTime - leftTime
  })
}
