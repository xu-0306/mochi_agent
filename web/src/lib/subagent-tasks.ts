import type { TaskSummary } from '@/lib/api'

export const DELEGATED_SUBAGENT_TASK_TYPE = 'delegated_multi_agent'
export const DELEGATE_SUBAGENT_TOOL_NAME = 'delegate_subagent_task'

export interface DelegatedSubagentView {
  taskId: string | null
  displayName: string | null
  role: string | null
  status: string | null
  instruction: string | null
  protocol: string | null
}

export interface DelegatedSubagentTranscriptItem {
  id: string
  role: 'user' | 'assistant' | 'system'
  label: string
  content: string
  timestamp: string | null
  meta?: string
  eventType?: string
}

export type DelegatedSubagentCardState = 'creating' | 'ready' | 'error'

export interface DelegatedSubagentCardView {
  projectionId: string
  taskId: string | null
  toolCallId: string | null
  state: DelegatedSubagentCardState
  title: string
  displayName: string | null
  role: string | null
  status: string | null
  instruction: string | null
  finalAnswer: string | null
  error: string | null
  createdAt: string | null
  updatedAt: string | null
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function firstString(...values: unknown[]): string | null {
  for (const value of values) {
    const next = getString(value)
    if (next) {
      return next
    }
  }
  return null
}

function nestedRecord(record: Record<string, unknown>, ...keys: string[]): Record<string, unknown> | null {
  for (const key of keys) {
    const value = record[key]
    if (isRecord(value)) {
      return value
    }
  }
  return null
}

function humanizeToken(value: string | null | undefined): string | null {
  const source = getString(value)
  if (!source) {
    return null
  }
  return source
    .replace(/[_-]+/g, ' ')
    .split(' ')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function eventType(event: Record<string, unknown>): string | null {
  return firstString(event.type, event.event_type, event.eventType)
}

function eventPayload(event: Record<string, unknown>): Record<string, unknown> {
  return nestedRecord(event, 'payload') ?? {}
}

function eventTimestamp(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return firstString(
    event.timestamp,
    event.created_at,
    event.occurred_at,
    payload.timestamp,
    payload.created_at,
    payload.occurred_at
  )
}

function eventContent(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return firstString(
    payload.content,
    payload.summary,
    payload.message,
    payload.final_answer,
    payload.guidance,
    event.content,
    event.summary,
    event.message,
    event.final_answer,
    event.guidance
  )
}

function eventRole(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  return firstString(
    payload.target_role_id,
    payload.role_id,
    payload.role,
    event.target_role_id,
    event.role_id,
    event.role
  )
}

function buildCreatedEventMeta(event: Record<string, unknown>): string | undefined {
  const createdContext = nestedRecord(event, 'created_context', 'createdContext')
  const source = humanizeToken(firstString(createdContext?.source, event.source))
  const delivery = humanizeToken(firstString(createdContext?.delivery, event.delivery))
  const meta = [source, delivery].filter((value): value is string => Boolean(value)).join(' / ')
  return meta || undefined
}

function buildOutputMeta(event: Record<string, unknown>): string | undefined {
  const payload = eventPayload(event)
  const stage = humanizeToken(firstString(payload.stage, payload.current_stage, event.stage))
  const role = humanizeToken(eventRole(event))
  const meta = [stage, role].filter((value): value is string => Boolean(value)).join(' / ')
  return meta || undefined
}

function buildArtifactSummary(event: Record<string, unknown>): string | null {
  const payload = eventPayload(event)
  const artifactType = firstString(payload.artifact_type, payload.artifactType, event.artifact_type, event.artifactType)
  const title = firstString(payload.title, event.title)
  const body = firstString(payload.summary, payload.message, event.summary, event.message)
  if (body) {
    return body
  }
  if (title && artifactType) {
    return `${title} (${artifactType})`
  }
  if (title) {
    return title
  }
  if (artifactType) {
    return `Produced artifact: ${artifactType}`
  }
  return null
}

function resolveDelegatedSubagentDraft(input: {
  result?: unknown
  metadata?: Record<string, unknown>
  args?: Record<string, unknown>
}): Omit<DelegatedSubagentView, 'taskId'> & { taskId: null } {
  const result = isRecord(input.result) ? input.result : {}
  const metadata = input.metadata ?? {}
  const args = input.args ?? {}
  const subagent =
    nestedRecord(result, 'subagent', 'subagent_task', 'subagentTask') ??
    nestedRecord(metadata, 'subagent', 'subagent_task', 'subagentTask') ??
    nestedRecord(args, 'subagent', 'subagent_task', 'subagentTask') ??
    {}
  const selectedModelsRoles = nestedRecord(metadata, 'selected_models_roles', 'selectedModelsRoles')
  const subagents = Array.isArray(selectedModelsRoles?.subagents) ? selectedModelsRoles.subagents : []
  const firstSuggestedRole = subagents.find(isRecord)

  return {
    taskId: null,
    displayName: firstString(
      subagent.display_name,
      subagent.displayName,
      result.display_name,
      result.displayName,
      metadata.display_name,
      metadata.displayName,
      args.display_name,
      args.displayName
    ),
    role: firstString(
      subagent.role,
      result.role,
      result.subagent_role,
      result.subagentRole,
      metadata.role,
      metadata.subagent_role,
      metadata.subagentRole,
      args.role,
      args.subagent_role,
      args.subagentRole,
      firstSuggestedRole?.role
    ),
    status: firstString(result.status, metadata.status, args.status),
    instruction: firstString(
      subagent.instruction,
      result.instruction,
      metadata.instruction,
      args.instruction,
      result.objective,
      metadata.objective,
      args.objective
    ),
    protocol: firstString(result.protocol, metadata.protocol, args.protocol),
  }
}

export function resolveDelegatedSubagentView(input: {
  result?: unknown
  metadata?: Record<string, unknown>
  args?: Record<string, unknown>
  task?: TaskSummary
}): DelegatedSubagentView | null {
  const result = isRecord(input.result) ? input.result : {}
  const metadata = input.metadata ?? {}
  const args = input.args ?? {}
  const task = input.task
  const subagent =
    (task?.delegated_subagent && isRecord(task.delegated_subagent) ? task.delegated_subagent : null) ??
    nestedRecord(result, 'subagent', 'subagent_task', 'subagentTask') ??
    nestedRecord(metadata, 'subagent', 'subagent_task', 'subagentTask') ??
    {}

  const taskType = firstString(
    task?.task_type,
    result.task_type,
    result.taskType,
    metadata.task_type,
    metadata.taskType
  )
  const taskId = firstString(task?.task_id, result.task_id, result.taskId, metadata.task_id, metadata.taskId)
  const looksDelegated =
    taskType === DELEGATED_SUBAGENT_TASK_TYPE ||
    taskType === 'controlled_subagent_execution' ||
    (!task && taskId !== null)

  if (!looksDelegated) {
    return null
  }

  const selectedModelsRoles = nestedRecord(metadata, 'selected_models_roles', 'selectedModelsRoles')
  const subagents = Array.isArray(selectedModelsRoles?.subagents) ? selectedModelsRoles.subagents : []
  const firstSuggestedRole = subagents.find(isRecord)

  return {
    taskId,
    displayName: firstString(
      task?.display_name,
      subagent.display_name,
      subagent.displayName,
      result.display_name,
      result.displayName,
      metadata.display_name,
      metadata.displayName
    ),
    role: firstString(
      task?.role,
      subagent.role,
      result.role,
      result.subagent_role,
      result.subagentRole,
      metadata.role,
      metadata.subagent_role,
      firstSuggestedRole?.role
    ),
    status: firstString(task?.status, result.status, metadata.status),
    instruction: firstString(
      task?.instruction,
      subagent.instruction,
      result.instruction,
      metadata.instruction,
      task?.input_message,
      result.objective,
      task?.objective,
      metadata.objective,
      args.objective
    ),
    protocol: firstString(result.protocol, metadata.protocol),
  }
}

export function delegatedSubagentTitle(view: DelegatedSubagentView): string {
  const label = view.displayName ?? 'delegated subagent'
  return view.role ? `Created ${label} (${view.role})` : `Created ${label}`
}

function delegatedSubagentPendingTitle(view: Omit<DelegatedSubagentView, 'taskId'> & { taskId: null }): string {
  if (view.displayName && view.role) {
    return `Creating ${view.displayName} (${view.role})`
  }
  if (view.displayName) {
    return `Creating ${view.displayName}`
  }
  if (view.role) {
    return `Creating ${view.role} subagent`
  }
  return 'Creating subagent...'
}

function delegatedSubagentFailureTitle(view: Omit<DelegatedSubagentView, 'taskId'> & { taskId: null }): string {
  if (view.displayName) {
    return `Failed to create ${view.displayName}`
  }
  if (view.role) {
    return `Failed to create ${view.role} subagent`
  }
  return 'Failed to create subagent'
}

export function buildDelegatedSubagentCardView(input: {
  result?: unknown
  metadata?: Record<string, unknown>
  args?: Record<string, unknown>
  task?: TaskSummary
  toolCallId?: string | null
  projectionId?: string | null
}): DelegatedSubagentCardView | null {
  const view = resolveDelegatedSubagentView(input)
  if (!view?.taskId) {
    return null
  }

  return {
    projectionId: input.projectionId ?? view.taskId,
    taskId: view.taskId,
    toolCallId: input.toolCallId ?? null,
    state: 'ready',
    title: delegatedSubagentTitle(view),
    displayName: view.displayName,
    role: view.role,
    status: view.status,
    instruction: view.instruction,
    finalAnswer: input.task?.final_answer ?? null,
    error: input.task?.error ?? null,
    createdAt: input.task?.created_at ?? null,
    updatedAt: input.task?.updated_at ?? null,
  }
}

export function buildDelegatedSubagentPendingCardView(input: {
  toolCallId: string
  metadata?: Record<string, unknown>
  args?: Record<string, unknown>
}): DelegatedSubagentCardView {
  const view = resolveDelegatedSubagentDraft(input)
  return {
    projectionId: `subagent-delegate-${input.toolCallId}`,
    taskId: null,
    toolCallId: input.toolCallId,
    state: 'creating',
    title: delegatedSubagentPendingTitle(view),
    displayName: view.displayName,
    role: view.role,
    status: 'creating',
    instruction: view.instruction,
    finalAnswer: null,
    error: null,
    createdAt: null,
    updatedAt: null,
  }
}

export function buildDelegatedSubagentFailureCardView(input: {
  toolCallId: string
  errorMessage: string | null | undefined
  metadata?: Record<string, unknown>
  args?: Record<string, unknown>
}): DelegatedSubagentCardView {
  const view = resolveDelegatedSubagentDraft(input)
  return {
    projectionId: `subagent-delegate-${input.toolCallId}`,
    taskId: null,
    toolCallId: input.toolCallId,
    state: 'error',
    title: delegatedSubagentFailureTitle(view),
    displayName: view.displayName,
    role: view.role,
    status: 'error',
    instruction: view.instruction,
    finalAnswer: null,
    error: input.errorMessage ?? 'Subagent creation failed.',
    createdAt: null,
    updatedAt: null,
  }
}

export function buildDelegatedSubagentTranscript(
  task: TaskSummary & { events?: Array<Record<string, unknown>> }
): DelegatedSubagentTranscriptItem[] {
  const view = resolveDelegatedSubagentView({ task })
  if (!view) {
    return []
  }

  const label = view.displayName ?? 'Subagent'
  const transcript: DelegatedSubagentTranscriptItem[] = [
    {
      id: `${task.task_id}:intro`,
      role: 'user',
      label: `Main Agent -> ${label}`,
      content: view.instruction ?? 'No instruction recorded.',
      timestamp: task.created_at,
      meta: 'Initial delegated instruction.',
      eventType: 'delegated_subagent_instruction',
    },
  ]

  let hasFinalAnswerEvent = false

  ;(task.events ?? []).forEach((event, index) => {
    const type = eventType(event) ?? 'event'
    const timestamp = eventTimestamp(event)
    const content = eventContent(event)
    const roleLabel = humanizeToken(eventRole(event))

    if (type === 'delegated_subagent_created') {
      const createdLabel = firstString(event.display_name, eventPayload(event).display_name, view.displayName) ?? label
      const createdRole = firstString(event.role, eventPayload(event).role, view.role)
      transcript.push({
        id: `${task.task_id}:event:${index}`,
        role: 'system',
        label: 'Delegation created',
        content: createdRole ? `Created ${createdLabel} (${createdRole}).` : `Created ${createdLabel}.`,
        timestamp,
        meta: buildCreatedEventMeta(event),
        eventType: type,
      })
      return
    }

    if (
      type === 'delegated_subagent_message' ||
      type === 'delegated_subagent_guidance' ||
      type === 'subagent_message'
    ) {
      transcript.push({
        id: `${task.task_id}:event:${index}`,
        role: 'user',
        label: `You -> ${label}`,
        content: content ?? 'Guidance queued for the next applicable turn.',
        timestamp,
        meta: 'Guidance queued for the next applicable turn.',
        eventType: type,
      })
      return
    }

    if (type === 'role_output') {
      transcript.push({
        id: `${task.task_id}:event:${index}`,
        role: 'assistant',
        label,
        content: content ?? `${label} produced an output.`,
        timestamp,
        meta: buildOutputMeta(event),
        eventType: type,
      })
      return
    }

    if (type === 'artifact') {
      transcript.push({
        id: `${task.task_id}:event:${index}`,
        role: 'assistant',
        label: `${label} artifact`,
        content: buildArtifactSummary(event) ?? `${label} produced an artifact.`,
        timestamp,
        meta: humanizeToken(firstString(eventPayload(event).artifact_type, event.artifact_type)) ?? roleLabel ?? undefined,
        eventType: type,
      })
      return
    }

    if (type === 'final_answer') {
      hasFinalAnswerEvent = true
      transcript.push({
        id: `${task.task_id}:event:${index}`,
        role: 'assistant',
        label,
        content: content ?? task.final_answer ?? `${label} returned a final answer.`,
        timestamp,
        meta: 'Final answer',
        eventType: type,
      })
    }
  })

  if (!hasFinalAnswerEvent && task.final_answer) {
    transcript.push({
      id: `${task.task_id}:final-answer`,
      role: 'assistant',
      label,
      content: task.final_answer,
      timestamp: task.finished_at ?? task.updated_at,
      meta: 'Final answer',
      eventType: 'final_answer',
    })
  }

  return transcript
}
