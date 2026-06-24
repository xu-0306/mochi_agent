import type { WorkflowProgressCardView } from '../components/workflow/types'
import type { AgentRunDetail, TaskSummary } from './api'
import type { Message } from './chat'
import {
  buildDelegatedSubagentCardView,
  buildDelegatedSubagentFailureCardView,
  buildDelegatedSubagentPendingCardView,
  DELEGATE_SUBAGENT_TOOL_NAME,
  type DelegatedSubagentCardView,
} from './subagent-tasks.ts'

function isWorkflowCompletionReportStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'succeeded' ||
    normalized === 'failed' ||
    normalized === 'cancelled' ||
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'error' ||
    normalized === 'partial'
  )
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

export function buildWorkflowCompletionContent(run: AgentRunDetail | null): string | null {
  if (!run || !isWorkflowCompletionReportStatus(run.status)) {
    return null
  }

  const status = run.status.toLowerCase()
  const finalAnswer = getString(run.summary?.final_answer)?.trim() ?? ''
  const latestError = run.latest_error?.trim() ?? ''
  const workflowLink = `/agent-runs/${encodeURIComponent(run.run_id)}`

  if (status === 'succeeded' || status === 'completed' || status === 'done') {
    return [
      '### Workflow completed',
      '',
      finalAnswer || 'The workflow completed, but no final answer was recorded.',
      '',
      `[Open workflow details](${workflowLink})`,
    ].join('\n')
  }

  if (status === 'partial') {
    return [
      '### Workflow completed with partial results',
      '',
      finalAnswer || latestError || 'The workflow stopped after producing partial results.',
      '',
      `[Open workflow details](${workflowLink})`,
    ].join('\n')
  }

  return [
    '### Workflow stopped',
    '',
    latestError || finalAnswer || `The workflow ended with status: ${run.status}.`,
    '',
    `[Open workflow details](${workflowLink})`,
  ].join('\n')
}

function dateFromIso(value: string | null | undefined, fallback: Date): Date {
  if (!value) {
    return fallback
  }
  const parsed = Date.parse(value)
  return Number.isNaN(parsed) ? fallback : new Date(parsed)
}

function messageTimestampValue(message: Message): number {
  const timestamp = message.timestamp.getTime()
  return Number.isFinite(timestamp) ? timestamp : Date.now()
}

function insertDisplayMessageByTimestamp(messages: Message[], message: Message): Message[] {
  const nextMessages = messages.filter((candidate) => candidate.id !== message.id)
  const targetTimestamp = messageTimestampValue(message)
  const insertIndex = nextMessages.findIndex(
    (candidate) => messageTimestampValue(candidate) > targetTimestamp
  )
  if (insertIndex === -1) {
    return [...nextMessages, message]
  }
  return [
    ...nextMessages.slice(0, insertIndex),
    message,
    ...nextMessages.slice(insertIndex),
  ]
}

export function buildProjectedDisplayMessages(input: {
  messages: Message[]
  runtimeTasks: TaskSummary[]
  workflowProgressCard?: WorkflowProgressCardView | null
  workflowRun?: AgentRunDetail | null
}): Message[] {
  const { messages, runtimeTasks, workflowProgressCard = null, workflowRun = null } = input
  const workflowCompletionContent = buildWorkflowCompletionContent(workflowRun)
  const subagentTaskCards = new Map<string, { card: DelegatedSubagentCardView; timestamp: Date }>()
  const subagentCardKeyByTaskId = new Map<string, string>()
  const subagentCardKeyByToolCallId = new Map<string, string>()

  const upsertSubagentTaskCard = (card: DelegatedSubagentCardView, timestamp: Date) => {
    const existingKey =
      (card.taskId ? subagentCardKeyByTaskId.get(card.taskId) : null) ??
      (card.toolCallId ? subagentCardKeyByToolCallId.get(card.toolCallId) : null) ??
      card.projectionId
    const existing = subagentTaskCards.get(existingKey)
    const nextCard = existing
      ? {
          ...existing.card,
          ...card,
          projectionId: existing.card.projectionId,
        }
      : card

    subagentTaskCards.set(existingKey, {
      card: nextCard,
      timestamp: existing?.timestamp ?? timestamp,
    })
    if (nextCard.taskId) {
      subagentCardKeyByTaskId.set(nextCard.taskId, existingKey)
    }
    if (nextCard.toolCallId) {
      subagentCardKeyByToolCallId.set(nextCard.toolCallId, existingKey)
    }
  }

  for (const message of messages) {
    for (const step of message.reasoningSteps ?? []) {
      if (step.toolName !== DELEGATE_SUBAGENT_TOOL_NAME) {
        continue
      }
      const toolCallId = step.toolCallId ?? step.id

      if (step.type === 'tool_call') {
        upsertSubagentTaskCard(buildDelegatedSubagentPendingCardView({
          toolCallId,
          metadata: step.toolMeta,
          args: step.toolArgs,
        }), step.timestamp)
        continue
      }

      if (step.type !== 'tool_result') {
        continue
      }

      if (step.toolError) {
        upsertSubagentTaskCard(buildDelegatedSubagentFailureCardView({
          toolCallId,
          errorMessage: step.toolError,
          metadata: step.toolMeta,
          args: step.toolArgs,
        }), step.timestamp)
        continue
      }

      const card = buildDelegatedSubagentCardView({
        result: step.toolResult,
        metadata: step.toolMeta,
        args: step.toolArgs,
        toolCallId,
        projectionId: `subagent-delegate-${toolCallId}`,
      })
      if (card) {
        upsertSubagentTaskCard(card, step.timestamp)
      }
    }
  }

  for (const task of runtimeTasks) {
    const card = buildDelegatedSubagentCardView({ task })
    if (!card) {
      continue
    }
    upsertSubagentTaskCard(card, dateFromIso(task.created_at, new Date()))
  }

  if (!workflowProgressCard && !workflowCompletionContent && subagentTaskCards.size === 0) {
    return messages
  }

  let nextMessages = messages

  for (const { card, timestamp } of [...subagentTaskCards.values()].sort(
    (left, right) => left.timestamp.getTime() - right.timestamp.getTime()
  )) {
    nextMessages = insertDisplayMessageByTimestamp(nextMessages, {
      id: `subagent-task-card-${card.projectionId}`,
      type: 'assistant',
      content: '',
      timestamp,
      subagentTaskCard: card,
    })
  }

  if (workflowProgressCard) {
    nextMessages = insertDisplayMessageByTimestamp(nextMessages, {
      id: `workflow-card-${workflowProgressCard.runId}`,
      type: 'assistant',
      content: '',
      timestamp: workflowProgressCard.updatedAt
        ? new Date(workflowProgressCard.updatedAt)
        : new Date(),
      workflowCard: workflowProgressCard,
    })
  }

  if (workflowCompletionContent && workflowRun) {
    nextMessages = insertDisplayMessageByTimestamp(nextMessages, {
      id: `workflow-completion-${workflowRun.run_id}`,
      type: 'assistant',
      content: workflowCompletionContent,
      timestamp: workflowRun.finished_at
        ? new Date(workflowRun.finished_at)
        : workflowRun.updated_at
          ? new Date(workflowRun.updated_at)
          : new Date(),
      eventType: 'final_answer',
      workflowCompletion: true,
    })
  }

  return nextMessages
}
