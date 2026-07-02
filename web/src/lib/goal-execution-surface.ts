export type GoalConversationBodyLayout =
  | 'empty_state'
  | 'empty_state_with_execution'
  | 'conversation'
  | 'conversation_with_execution'

export function resolveGoalConversationBodyLayout(options: {
  showEmptyState: boolean
  showExecutionTranscript: boolean
}): GoalConversationBodyLayout {
  if (options.showEmptyState) {
    return options.showExecutionTranscript ? 'empty_state_with_execution' : 'empty_state'
  }
  return options.showExecutionTranscript ? 'conversation_with_execution' : 'conversation'
}

export function layoutShowsExecutionHighlights(
  layout: GoalConversationBodyLayout
): boolean {
  return layout === 'empty_state_with_execution' || layout === 'conversation_with_execution'
}

export function layoutShowsEmptyState(
  layout: GoalConversationBodyLayout
): boolean {
  return layout === 'empty_state' || layout === 'empty_state_with_execution'
}
