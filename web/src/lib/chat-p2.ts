import type { Message, ReasoningStep } from './chat'
import {
  extractFileChangeGroupFromToolData,
  summarizeDiffStats,
  type FileChangeGroupSummary,
  type FileChangeSummary,
} from './file-change-preview.ts'
export type ChatExportFormat = 'markdown' | 'json'
export type { FileChangeGroupSummary, FileChangeSummary } from './file-change-preview.ts'

export interface ChatExportTraceEvent {
  type: string
  seq?: number
  eventId?: string | null
  dedupeKey?: string | null
  parentType?: string | null
  parentId?: string | null
  subagentId?: string | null
  roleId?: string | null
  title?: string | null
  modelId?: string | null
  status?: string | null
  content?: string | null
  summary?: string | null
  metadata?: Record<string, unknown>
  createdAt?: string | null
  messageId?: string | null
  deliveryMode?: string | null
  deliveryStatus?: string | null
  deliveryReason?: string | null
}

export interface ChatExportOptions {
  includeReasoning?: boolean
  traceEvents?: ChatExportTraceEvent[]
}

export function isConversationEffectivelyEmpty(messages: Message[]): boolean {
  return messages.every((message) => message.type === 'system')
}

export function findRegeneratePrompt(
  messages: Message[],
  targetMessageId?: string,
): string | null {
  const assistantIndex = targetMessageId
    ? messages.findIndex((message) => (
      message.id === targetMessageId && message.type === 'assistant'
    ))
    : [...messages].reverse().findIndex((message) => message.type === 'assistant')

  const resolvedAssistantIndex =
    targetMessageId
      ? assistantIndex
      : assistantIndex === -1
        ? -1
        : messages.length - 1 - assistantIndex

  if (resolvedAssistantIndex === -1) {
    return null
  }

  for (let index = resolvedAssistantIndex - 1; index >= 0; index -= 1) {
    const message = messages[index]
    if (message.type === 'user' && message.content.trim()) {
      return message.content
    }
  }

  return null
}

export function findEditForkTurnId(
  messages: Message[],
  targetMessageId: string,
): string | null {
  const targetIndex = messages.findIndex((message) => (
    message.id === targetMessageId && message.type === 'user'
  ))

  if (targetIndex === -1) {
    return null
  }

  const targetTurnId = messages[targetIndex].turnId ?? messages[targetIndex].turnKey ?? null
  if (!targetTurnId) {
    return null
  }

  for (let index = targetIndex - 1; index >= 0; index -= 1) {
    const message = messages[index]
    const candidateTurnId = message.turnId ?? message.turnKey ?? null

    if (!candidateTurnId || candidateTurnId === targetTurnId) {
      continue
    }

    if (message.type === 'assistant') {
      return candidateTurnId
    }
  }

  return null
}

function isNonEmptyRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function toExportableReasoningStep(step: ReasoningStep) {
  return {
    type: step.type,
    content: step.content,
    timestamp: step.timestamp.toISOString(),
    source: step.source,
    status: step.status,
    tool_name: step.toolName,
    tool_call_id: step.toolCallId,
    tool_args: step.toolArgs,
    tool_result: step.toolResult,
    tool_error: step.toolError,
    error_code: step.errorCode,
    metadata: step.toolMeta,
    tool_exposure: step.toolExposure,
    transport: step.transport,
  }
}

function compactExportValue(value: unknown): unknown {
  if (value === undefined) {
    return undefined
  }
  if (Array.isArray(value)) {
    return value
      .map(compactExportValue)
      .filter((item) => item !== undefined)
  }
  if (isNonEmptyRecord(value)) {
    const entries = Object.entries(value)
      .map(([key, item]) => [key, compactExportValue(item)] as const)
      .filter(([, item]) => item !== undefined)
    return Object.fromEntries(entries)
  }
  return value
}

function stringifyExportValue(value: unknown): string {
  if (value === undefined) {
    return ''
  }
  if (typeof value === 'string') {
    return value
  }
  return JSON.stringify(value, null, 2)
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function getMetadataString(metadata: Record<string, unknown> | undefined, ...keys: string[]): string | null {
  if (!metadata) {
    return null
  }
  for (const key of keys) {
    const value = getString(metadata[key])
    if (value) {
      return value
    }
  }
  return null
}

function traceEventToolName(event: ChatExportTraceEvent): string | null {
  return getMetadataString(event.metadata, 'tool_name', 'toolName', 'last_tool_name', 'lastToolName')
}

function traceEventCallId(event: ChatExportTraceEvent): string | null {
  return getMetadataString(event.metadata, 'call_id', 'callId', 'tool_call_id', 'toolCallId')
}

function traceEventTitle(event: ChatExportTraceEvent): string {
  const status = event.status?.trim().toLowerCase() ?? ''
  const toolName = traceEventToolName(event)

  if (event.type === 'subagent_tool_call') {
    return toolName ? `Running ${toolName}` : 'Tool call'
  }
  if (event.type === 'subagent_tool_result') {
    if (status === 'failed' || status === 'error') {
      return toolName ? `${toolName} failed` : 'Tool failed'
    }
    if (status === 'cancelled' || status === 'canceled') {
      return toolName ? `${toolName} cancelled` : 'Tool cancelled'
    }
    return toolName ? `${toolName} returned` : 'Tool result'
  }
  if (event.type === 'subagent_thinking') {
    return event.title?.trim() || 'Thinking'
  }
  if (event.type === 'subagent_progress') {
    return event.title?.trim() || 'Progress update'
  }
  if (event.type === 'runtime_blocked') {
    return event.title?.trim() || 'Execution blocked'
  }
  return event.title?.trim() || event.type.replaceAll('_', ' ')
}

function toExportableTraceEvent(event: ChatExportTraceEvent) {
  return compactExportValue({
    type: event.type,
    title: traceEventTitle(event),
    status: event.status,
    timestamp: event.createdAt,
    seq: event.seq,
    event_id: event.eventId,
    parent_type: event.parentType,
    parent_id: event.parentId,
    subagent_id: event.subagentId,
    role_id: event.roleId,
    model_id: event.modelId,
    tool_name: traceEventToolName(event),
    call_id: traceEventCallId(event),
    content: event.content,
    summary: event.summary,
    metadata: event.metadata,
    message_id: event.messageId,
    delivery_mode: event.deliveryMode,
    delivery_status: event.deliveryStatus,
    delivery_reason: event.deliveryReason,
  })
}

function formatTraceEventMarkdown(event: ChatExportTraceEvent): string {
  const lines = [`#### ${traceEventTitle(event)}`]
  lines.push(`- Type: \`${event.type}\``)
  if (event.createdAt) {
    lines.push(`- Time: ${event.createdAt}`)
  }
  if (event.status) {
    lines.push(`- Status: \`${event.status}\``)
  }
  const toolName = traceEventToolName(event)
  if (toolName) {
    lines.push(`- Tool: \`${toolName}\``)
  }
  const callId = traceEventCallId(event)
  if (callId) {
    lines.push(`- Call ID: \`${callId}\``)
  }
  if (event.roleId) {
    lines.push(`- Role: \`${event.roleId}\``)
  }
  if (event.subagentId) {
    lines.push(`- Subagent: \`${event.subagentId}\``)
  }

  const blocks: Array<[string, unknown]> = [
    ['Summary', event.summary],
    ['Content', event.content],
    ['Metadata', event.metadata],
  ]
  for (const [label, rawValue] of blocks) {
    const value = stringifyExportValue(rawValue)
    if (!value.trim()) {
      continue
    }
    lines.push('', `${label}:`, '```', value, '```')
  }
  return lines.join('\n')
}

function formatTraceEventsMarkdown(events: ChatExportTraceEvent[]): string | null {
  if (events.length === 0) {
    return null
  }
  return [
    '## Execution Trace',
    '',
    ...events.map(formatTraceEventMarkdown),
  ].join('\n\n')
}

function reasoningStepTitle(step: ReasoningStep): string {
  if (step.type === 'tool_call' || step.type === 'tool_result') {
    const toolName = step.toolName ?? 'tool'
    const status = step.status === 'error' ? 'failed' : step.status
    return status ? `${toolName} ${status}` : toolName
  }
  if (step.type === 'thinking') {
    return step.source === 'model_summary' ? 'Reasoning summary' : 'Thinking'
  }
  return step.type.replaceAll('_', ' ')
}

function formatReasoningStepMarkdown(step: ReasoningStep): string {
  const lines = [`#### ${reasoningStepTitle(step)}`]
  lines.push(`- Type: \`${step.type}\``)
  lines.push(`- Time: ${step.timestamp.toISOString()}`)

  if (step.source) {
    lines.push(`- Source: \`${step.source}\``)
  }
  if (step.status) {
    lines.push(`- Status: \`${step.status}\``)
  }
  if (step.toolCallId) {
    lines.push(`- Call ID: \`${step.toolCallId}\``)
  }
  if (step.toolName) {
    lines.push(`- Tool: \`${step.toolName}\``)
  }
  if (step.errorCode) {
    lines.push(`- Error code: \`${step.errorCode}\``)
  }

  const blocks: Array<[string, unknown]> = [
    ['Content', step.content],
    ['Arguments', step.toolArgs],
    ['Result', step.toolResult],
    ['Error', step.toolError],
    ['Metadata', step.toolMeta],
    ['Tool exposure', step.toolExposure],
    ['Transport', step.transport],
  ]

  for (const [label, rawValue] of blocks) {
    const value = stringifyExportValue(rawValue)
    if (!value.trim()) {
      continue
    }
    lines.push('', `${label}:`, '```', value, '```')
  }

  return lines.join('\n')
}

function formatReasoningMarkdown(message: Message): string | null {
  const steps = message.reasoningSteps ?? []
  if (steps.length === 0) {
    return null
  }
  return [
    '### Reasoning and Tool Trace',
    '',
    ...steps.map(formatReasoningStepMarkdown),
  ].join('\n\n')
}

export function buildChatExport(
  messages: Message[],
  format: ChatExportFormat,
  options: ChatExportOptions = {},
): string {
  const includeReasoning = options.includeReasoning === true
  const traceEvents = includeReasoning ? options.traceEvents ?? [] : []
  const filtered = messages.filter((message) => (
    message.type === 'user' || message.type === 'assistant'
  ))

  if (format === 'json') {
    const exportedMessages = filtered.map((message) => ({
      role: message.type,
      content: message.content,
      timestamp: message.timestamp.toISOString(),
      ...(includeReasoning && message.reasoningSteps?.length
        ? {
            reasoning_steps: message.reasoningSteps.map((step) => (
              compactExportValue(toExportableReasoningStep(step))
            )),
          }
        : {}),
    }))
    if (traceEvents.length === 0) {
      return JSON.stringify(exportedMessages, null, 2)
    }
    return JSON.stringify(
      {
        messages: exportedMessages,
        execution_trace: traceEvents.map(toExportableTraceEvent),
      },
      null,
      2
    )
  }

  const messageExport = filtered
    .map((message) => {
      const roleBlock = message.type === 'user'
        ? `## User\n${message.content}`
        : `## Assistant\n${message.content}`
      const reasoningBlock = includeReasoning ? formatReasoningMarkdown(message) : null
      return reasoningBlock ? `${roleBlock}\n\n${reasoningBlock}` : roleBlock
    })
    .join('\n\n')
  const traceBlock = formatTraceEventsMarkdown(traceEvents)
  if (traceBlock && messageExport) {
    return `${messageExport}\n\n${traceBlock}`
  }
  return traceBlock ?? messageExport
}

export { summarizeDiffStats }

export function extractFileChangeGroupFromReasoningStep(
  step: ReasoningStep,
): FileChangeGroupSummary | null {
  if (step.type !== 'tool_result') {
    return null
  }

  return extractFileChangeGroupFromToolData({
    id: step.id,
    toolName: step.toolName,
    toolArgs: step.toolArgs,
    toolMeta: step.toolMeta,
    toolResult:
      typeof step.toolResult === 'object' && step.toolResult !== null
        ? step.toolResult
        : undefined,
  })
}

export function extractFileChangeFromReasoningStep(
  step: ReasoningStep,
): FileChangeSummary | null {
  const group = extractFileChangeGroupFromReasoningStep(step)
  if (!group || group.files.length !== 1) {
    return null
  }
  return group.files[0]
}
