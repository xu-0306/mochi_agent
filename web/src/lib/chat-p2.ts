import type { Message, ReasoningStep } from '@/lib/chat'
import {
  extractFileChangeGroupFromToolData,
  summarizeDiffStats,
  type FileChangeGroupSummary,
  type FileChangeSummary,
} from '@/lib/file-change-preview'
export type ChatExportFormat = 'markdown' | 'json'
export type { FileChangeGroupSummary, FileChangeSummary } from '@/lib/file-change-preview'

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

export function buildChatExport(messages: Message[], format: ChatExportFormat): string {
  const filtered = messages.filter((message) => (
    message.type === 'user' || message.type === 'assistant'
  ))

  if (format === 'json') {
    return JSON.stringify(
      filtered.map((message) => ({
        role: message.type,
        content: message.content,
        timestamp: message.timestamp.toISOString(),
      })),
      null,
      2
    )
  }

  return filtered
    .map((message) => (
      message.type === 'user'
        ? `## User\n${message.content}`
        : `## Assistant\n${message.content}`
    ))
    .join('\n\n')
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
