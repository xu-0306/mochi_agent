import type { Message } from './chat'

type LegacyGoalCardMessage = Message & { goalCard?: unknown }

export function stripLegacyGoalCardMessage(message: Message): Message | null {
  const legacyMessage = message as LegacyGoalCardMessage
  if (!legacyMessage.goalCard) {
    return message
  }

  const { goalCard: _legacyGoalCard, ...messageWithoutGoalCard } = legacyMessage
  if (
    messageWithoutGoalCard.content.trim().length > 0 ||
    (messageWithoutGoalCard.attachments?.length ?? 0) > 0 ||
    (messageWithoutGoalCard.reasoningSteps?.length ?? 0) > 0 ||
    messageWithoutGoalCard.workflowCard ||
    messageWithoutGoalCard.workflowCompletion ||
    messageWithoutGoalCard.subagentTaskCard
  ) {
    return messageWithoutGoalCard
  }
  return null
}
