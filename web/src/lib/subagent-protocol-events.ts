import type { ExecutionTranscriptEvent } from '@/lib/api'

export type SubagentStatusVariant = 'primary' | 'success' | 'warning' | 'error' | 'outline'
export type SubagentEventKind =
  | 'approval'
  | 'blocked'
  | 'lifecycle'
  | 'message'
  | 'progress'
  | 'prompt'
  | 'protocol'
  | 'thinking'
  | 'tool'

const MESSAGE_DELIVERY_STATUS_BY_TYPE: Record<string, string> = {
  subagent_message_queued: 'queued',
  subagent_message_accepted: 'accepted',
  subagent_message_applied: 'applied',
  subagent_message_deferred: 'deferred',
  subagent_message_cancelled: 'cancelled',
  subagent_message_canceled: 'cancelled',
  subagent_message_rejected: 'rejected',
}

const PROTOCOL_STATUS_BY_TYPE: Record<string, string> = {
  subagent_interrupted: 'interrupted',
  subagent_tool_cancel_requested: 'cancel_requested',
  subagent_tool_cancelled: 'cancelled',
  subagent_tool_canceled: 'cancelled',
  subagent_tool_cancel_deferred: 'cancel_deferred',
}

function nonEmptyString(value: string | null | undefined): string | null {
  const trimmed = value?.trim()
  return trimmed && trimmed.length > 0 ? trimmed : null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter((item) => item.length > 0)
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
    const list = getStringArray(metadata[key])
    if (list.length > 0) {
      return list
    }
    const singleValue = getString(metadata[key])
    if (singleValue) {
      return [singleValue]
    }
  }
  return []
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    const normalized = getString(value)
    if (!normalized || seen.has(normalized)) {
      continue
    }
    seen.add(normalized)
    result.push(normalized)
  }
  return result
}

function capitalize(value: string): string {
  return value.length > 0 ? value.charAt(0).toUpperCase() + value.slice(1) : value
}

function prettifyLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

function truncateMiddle(value: string, maxLength: number): string {
  if (value.length <= maxLength) {
    return value
  }
  const separator = '...'
  const headLength = Math.max(12, Math.ceil((maxLength - separator.length) / 2))
  const tailLength = Math.max(8, maxLength - headLength - separator.length)
  return `${value.slice(0, headLength)}${separator}${value.slice(-tailLength)}`
}

function formatMetadataChip(label: string, value: string, prettifyValue = false): string {
  return `${label} ${prettifyValue ? prettifyLabel(value) : value}`
}

function formatApprovalCount(count: number): string {
  return `${count} approval${count === 1 ? '' : 's'}`
}

function normalizeStatus(status: string | null | undefined): string {
  const normalized = nonEmptyString(status)?.toLowerCase() ?? ''
  return normalized === 'canceled' ? 'cancelled' : normalized
}

function eventToolName(event: ExecutionTranscriptEvent): string | null {
  return (
    getMetadataString(event.metadata, 'tool_name', 'toolName') ??
    getMetadataStringArray(event.metadata, 'tool_names', 'toolNames')[0] ??
    null
  )
}

function eventCallId(event: ExecutionTranscriptEvent): string | null {
  return getMetadataString(event.metadata, 'call_id', 'callId', 'tool_call_id', 'toolCallId')
}

function eventApprovalIds(event: ExecutionTranscriptEvent): string[] {
  return uniqueStrings([
    ...getMetadataStringArray(event.metadata, 'approval_ids', 'approvalIds'),
    getMetadataString(event.metadata, 'approval_id', 'approvalId'),
  ])
}

function eventBlockerType(event: ExecutionTranscriptEvent): string | null {
  return getMetadataString(event.metadata, 'blocker_type', 'blockerType')
}

function eventDeliveryMode(event: ExecutionTranscriptEvent): string | null {
  return (
    getString(event.deliveryMode) ??
    getMetadataString(event.metadata, 'delivery_mode', 'deliveryMode')
  )
}

function eventDeliveryReason(event: ExecutionTranscriptEvent): string | null {
  return (
    getString(event.deliveryReason) ??
    getMetadataString(event.metadata, 'delivery_reason', 'deliveryReason')
  )
}

function eventMessageId(event: ExecutionTranscriptEvent): string | null {
  return (
    getString(event.messageId) ??
    getMetadataString(event.metadata, 'message_id', 'messageId')
  )
}

export function deriveSubagentMessageDeliveryStatus(type: string): string | null {
  return MESSAGE_DELIVERY_STATUS_BY_TYPE[type] ?? null
}

export function deriveSubagentEventStatus(
  type: string,
  explicitStatus?: string | null
): string | null {
  const status = nonEmptyString(explicitStatus)
  if (status) {
    return status === 'canceled' ? 'cancelled' : status
  }
  return PROTOCOL_STATUS_BY_TYPE[type] ?? deriveSubagentMessageDeliveryStatus(type)
}

export function formatSubagentStatusLabel(status: string): string {
  return prettifyLabel(status)
}

export function formatSubagentEventTypeLabel(type: string): string {
  const normalized = type.replace(/^subagent_/, '').replaceAll('_', ' ')
  return capitalize(normalized)
}

export function formatSubagentEventTitle(event: ExecutionTranscriptEvent): string {
  const status = normalizeStatus(deriveSubagentEventStatus(event.type, event.status))
  const toolName = eventToolName(event)

  if (event.type === 'runtime_blocked') {
    const blockerType = eventBlockerType(event)
    if (blockerType === 'approval') {
      return 'Approval required'
    }
    if (blockerType === 'dependency') {
      return 'Waiting on dependency'
    }
    if (blockerType === 'operator_control') {
      return 'Operator action required'
    }
    return 'Execution blocked'
  }
  if (event.type === 'run_started') {
    return 'Run started'
  }
  if (event.type === 'run_resumed') {
    return 'Run resumed'
  }
  if (event.type === 'run_paused') {
    return 'Run paused'
  }
  if (event.type === 'run_completed') {
    return 'Run completed'
  }
  if (event.type === 'run_failed') {
    return 'Run failed'
  }
  if (event.type === 'run_finalized_partial') {
    return 'Run partial'
  }
  if (event.type === 'subagent_started') {
    return 'Thread started'
  }
  if (event.type === 'subagent_prompt') {
    return 'Prompt received'
  }
  if (event.type === 'subagent_thinking') {
    return 'Thinking'
  }
  if (event.type === 'subagent_progress') {
    return 'Progress update'
  }
  if (event.type === 'subagent_tool_call') {
    return toolName ? `Running ${toolName}` : 'Tool call'
  }
  if (event.type === 'subagent_tool_result' && status === 'approval_required') {
    return 'Tool waiting for approval'
  }
  if (event.type === 'subagent_tool_result' && (status === 'failed' || status === 'error')) {
    return toolName ? `${toolName} failed` : 'Tool failed'
  }
  if (event.type === 'subagent_tool_result' && status === 'cancelled') {
    return 'Tool cancelled'
  }
  if (event.type === 'subagent_tool_result') {
    return toolName ? `${toolName} returned` : 'Tool result'
  }
  if (event.type === 'subagent_tool_cancel_requested') {
    return 'Tool cancel requested'
  }
  if (event.type === 'subagent_tool_cancelled' || event.type === 'subagent_tool_canceled') {
    return 'Tool cancelled'
  }
  if (event.type === 'subagent_tool_cancel_deferred') {
    return 'Tool cancel deferred'
  }
  if (event.type === 'subagent_interrupted') {
    return 'Thread interrupted'
  }
  if (event.type === 'subagent_completed') {
    if (status === 'failed' || status === 'error') {
      return 'Thread failed'
    }
    if (status === 'cancelled') {
      return 'Thread cancelled'
    }
    if (status === 'interrupted') {
      return 'Thread interrupted'
    }
    return 'Thread completed'
  }
  if (event.type === 'subagent_message') {
    return 'Follow-up guidance'
  }
  if (event.type === 'subagent_message_queued') {
    return 'Follow-up queued'
  }
  if (event.type === 'subagent_message_accepted') {
    return 'Follow-up accepted'
  }
  if (event.type === 'subagent_message_applied') {
    return 'Follow-up delivered'
  }
  if (event.type === 'subagent_message_deferred') {
    return 'Follow-up deferred'
  }
  if (event.type === 'subagent_message_cancelled' || event.type === 'subagent_message_canceled') {
    return 'Follow-up cancelled'
  }
  if (event.type === 'subagent_message_rejected') {
    return 'Follow-up rejected'
  }
  return formatSubagentEventTypeLabel(event.type)
}

export function deriveSubagentEventKind(event: ExecutionTranscriptEvent): SubagentEventKind {
  const status = normalizeStatus(deriveSubagentEventStatus(event.type, event.status))
  if (event.type === 'runtime_blocked') {
    return eventBlockerType(event) === 'approval' ? 'approval' : 'blocked'
  }
  if (status === 'approval_required') {
    return 'approval'
  }
  if (
    event.type === 'subagent_started' ||
    event.type === 'subagent_completed' ||
    event.type.startsWith('run_')
  ) {
    return 'lifecycle'
  }
  if (event.type === 'subagent_prompt') {
    return 'prompt'
  }
  if (event.type === 'subagent_progress') {
    return 'progress'
  }
  if (event.type === 'subagent_thinking') {
    return 'thinking'
  }
  if (event.type === 'subagent_message' || event.type.startsWith('subagent_message_')) {
    return 'message'
  }
  if (event.type.includes('tool')) {
    return 'tool'
  }
  return 'protocol'
}

export function formatSubagentEventKindLabel(event: ExecutionTranscriptEvent): string {
  return capitalize(deriveSubagentEventKind(event))
}

export function describeSubagentEventMetadata(event: ExecutionTranscriptEvent): string[] {
  const status = normalizeStatus(deriveSubagentEventStatus(event.type, event.status))
  const kind = deriveSubagentEventKind(event)
  const toolName = eventToolName(event)
  const callId = eventCallId(event)
  const command = getMetadataString(event.metadata, 'command')
  const approvalIds = eventApprovalIds(event)
  const blockerType = eventBlockerType(event)
  const approvalScope = getMetadataString(event.metadata, 'approval_scope', 'approvalScope')
  const approvalKind = getMetadataString(event.metadata, 'approval_kind', 'approvalKind')
  const policySource = getMetadataString(event.metadata, 'policy_source', 'policySource')
  const recommendedAction = getMetadataString(event.metadata, 'recommended_action', 'recommendedAction')
  const deliveryMode = eventDeliveryMode(event)
  const deliveryReason = eventDeliveryReason(event)
  const messageId = eventMessageId(event)
  const values: string[] = []

  if (kind === 'approval') {
    if (approvalIds.length > 0) {
      values.push(formatApprovalCount(approvalIds.length))
    }
    if (toolName) {
      values.push(formatMetadataChip('tool', toolName))
    }
    if (callId) {
      values.push(formatMetadataChip('call', callId))
    }
    if (approvalScope) {
      values.push(formatMetadataChip('scope', approvalScope, true))
    }
    if (approvalKind) {
      values.push(formatMetadataChip('kind', approvalKind, true))
    }
    if (policySource) {
      values.push(formatMetadataChip('policy', policySource, true))
    }
  } else if (kind === 'blocked') {
    if (blockerType) {
      values.push(formatMetadataChip('blocker', blockerType, true))
    }
    if (recommendedAction) {
      values.push(formatMetadataChip('next', recommendedAction, true))
    }
  } else if (kind === 'tool') {
    if (toolName) {
      values.push(formatMetadataChip('tool', toolName))
    }
    if (callId) {
      values.push(formatMetadataChip('call', callId))
    }
    if (command) {
      values.push(formatMetadataChip('command', truncateMiddle(command, 72)))
    }
  } else if (kind === 'message') {
    if (deliveryMode) {
      values.push(formatMetadataChip('delivery', deliveryMode, true))
    }
    if (deliveryReason) {
      values.push(formatMetadataChip('reason', deliveryReason, true))
    }
    if (messageId) {
      values.push(formatMetadataChip('message', messageId))
    }
  }

  if (status && status !== 'running' && status !== 'active' && kind !== 'approval') {
    values.push(formatMetadataChip('status', status, true))
  }

  return uniqueStrings(values)
}

export function subagentStatusVariant(status: string | null | undefined): SubagentStatusVariant {
  const normalized = normalizeStatus(status)
  if (
    normalized === 'completed' ||
    normalized === 'succeeded' ||
    normalized === 'done' ||
    normalized === 'applied' ||
    normalized === 'accepted'
  ) {
    return 'success'
  }
  if (
    normalized === 'blocked' ||
    normalized === 'awaiting_approval' ||
    normalized === 'approval_required' ||
    normalized === 'cancel_deferred' ||
    normalized === 'cancel_requested' ||
    normalized === 'deferred' ||
    normalized === 'interrupted' ||
    normalized === 'paused' ||
    normalized === 'pending' ||
    normalized === 'queued' ||
    normalized === 'stalled' ||
    normalized === 'waiting' ||
    normalized === 'waiting_approval'
  ) {
    return 'warning'
  }
  if (
    normalized === 'cancelled' ||
    normalized === 'canceled' ||
    normalized === 'error' ||
    normalized === 'failed' ||
    normalized === 'rejected'
  ) {
    return 'error'
  }
  if (
    normalized === 'active' ||
    normalized === 'created' ||
    normalized === 'in_progress' ||
    normalized === 'processing' ||
    normalized === 'resumed' ||
    normalized === 'running' ||
    normalized === 'started' ||
    normalized === 'working'
  ) {
    return 'primary'
  }
  return 'outline'
}

export function isInformativeSubagentStatus(status: string | null | undefined): boolean {
  const normalized = normalizeStatus(status)
  return normalized.length > 0 && normalized !== 'running' && normalized !== 'active'
}
