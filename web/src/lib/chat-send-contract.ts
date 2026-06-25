import type { ChatAttachment, Message } from '@/lib/chat'

function buildMessageTimestamp(timestamp?: Date): Date {
  return timestamp instanceof Date ? timestamp : new Date()
}

function buildAssistantTurnPlaceholder(
  turnKey: string,
  options?: {
    content?: string
    timestamp?: Date
  }
): Message {
  return {
    id: `assistant-turn-${turnKey}`,
    type: 'assistant',
    content: options?.content ?? '',
    timestamp: buildMessageTimestamp(options?.timestamp),
    turnKey,
    reasoningSteps: [],
    isStreaming: true,
  }
}

export function buildOptimisticConversationTurnMessages(input: {
  existingMessages: Message[]
  userContent: string
  attachments: ChatAttachment[]
  turnKey: string
  assistantPlaceholderContent?: string
  timestamp?: Date
}): {
  messages: Message[]
  lastMessageSummary: string
} {
  const timestamp = buildMessageTimestamp(input.timestamp)
  const userMessage: Message = {
    id: `user-${timestamp.getTime()}`,
    type: 'user',
    content: input.userContent,
    attachments: input.attachments,
    timestamp,
  }
  const lastMessageSummary =
    input.userContent.trim() ||
    (input.attachments.length > 0
      ? input.attachments.slice(0, 2).map((attachment) => attachment.name).join(', ')
      : '')

  return {
    messages: [
      ...input.existingMessages,
      userMessage,
      buildAssistantTurnPlaceholder(input.turnKey, {
        content: input.assistantPlaceholderContent,
        timestamp,
      }),
    ],
    lastMessageSummary,
  }
}
