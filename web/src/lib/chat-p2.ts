import type { Message, ReasoningStep } from '@/lib/chat'

export type ChatExportFormat = 'markdown' | 'json'

export interface FileChangeSummary {
  filePath: string
  originalContent: string | null
  newContent: string | null
  diff: string | null
  undoAvailable: boolean
  undoAction: 'restore' | 'delete' | null
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
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

export function summarizeDiffStats(diff: string | null | undefined): {
  additions: number
  deletions: number
} {
  if (!diff) {
    return { additions: 0, deletions: 0 }
  }

  let additions = 0
  let deletions = 0
  for (const line of diff.split(/\r?\n/)) {
    if (line.startsWith('+++') || line.startsWith('---')) {
      continue
    }
    if (line.startsWith('+')) {
      additions += 1
      continue
    }
    if (line.startsWith('-')) {
      deletions += 1
    }
  }

  return { additions, deletions }
}

export function extractFileChangeFromReasoningStep(
  step: ReasoningStep,
): FileChangeSummary | null {
  if (step.type !== 'tool_result' || step.toolName !== 'file_write') {
    return null
  }

  const meta = step.toolMeta
  if (!isRecord(meta)) {
    return null
  }

  const filePath = getString(meta.file_path)
  if (!filePath) {
    return null
  }

  const undoAction = getString(meta.undo_action)

  return {
    filePath,
    originalContent: getString(meta.original_content),
    newContent: getString(meta.new_content),
    diff: getString(meta.diff),
    undoAvailable: getBoolean(meta.undo_available) ?? false,
    undoAction: undoAction === 'restore' || undoAction === 'delete' ? undoAction : null,
  }
}
