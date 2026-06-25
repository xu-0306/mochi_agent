export type ChatModeCommand = {
  mode: 'workflow' | 'chat'
  content: string
}

export type GoalCommandAction = 'help' | 'proposal' | 'status' | 'pause' | 'resume' | 'stop'

export interface GoalCommand {
  action: GoalCommandAction
  content: string
  raw: string
}

export type ChatGoalWorkflowRoute =
  | { kind: 'direct_chat' }
  | { kind: 'goal_help'; raw: string }
  | { kind: 'goal_proposal'; content: string; raw: string }
  | { kind: 'workflow_proposal'; requestText: string }
  | { kind: 'natural_language_goal_proposal'; requestText: string }
  | { kind: 'goal_confirmation'; requestText: string; raw: string }
  | { kind: 'goal_revision'; requestText: string }
  | { kind: 'goal_follow_up'; requestText: string }
  | { kind: 'goal_lifecycle'; action: 'status' | 'pause' | 'resume' | 'stop'; raw: string }

export interface ChatGoalWorkflowRoutingDecision {
  modeCommand: ChatModeCommand | null
  requestText: string
  route: ChatGoalWorkflowRoute
  workflowModeRequested: boolean
  requiresSessionMaterialization: boolean
  shouldHandleGoalWorkflowRouting: boolean
}

export interface ResolveChatGoalWorkflowRoutingInput {
  text: string
  attachmentCount: number
  hasPendingProposal: boolean
  hasActiveGoal: boolean
}

export function parseChatModeCommand(value: string): ChatModeCommand | null {
  const match = value.match(/^\/(workflow|chat)(?:\s+([\s\S]*))?$/i)
  if (!match) {
    return null
  }
  return {
    mode: match[1].toLowerCase() as ChatModeCommand['mode'],
    content: (match[2] ?? '').trim(),
  }
}

export function parseGoalCommand(value: string): GoalCommand | null {
  const match = value.match(/^\/goal(?:\s+([\s\S]*))?$/i)
  if (!match) {
    return null
  }

  const content = (match[1] ?? '').trim()
  const normalized = content.toLowerCase()
  if (!normalized) {
    return {
      action: 'help',
      content: '',
      raw: value.trim(),
    }
  }

  if (normalized === 'status' || normalized === 'pause' || normalized === 'resume' || normalized === 'stop') {
    return {
      action: normalized,
      content: '',
      raw: value.trim(),
    }
  }

  return {
    action: 'proposal',
    content,
    raw: value.trim(),
  }
}

export function isGoalConfirmationText(value: string): boolean {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, ' ')
  return (
    normalized === 'start' ||
    normalized === 'go ahead' ||
    normalized === 'proceed' ||
    normalized === 'yes' ||
    normalized === 'run it'
  )
}

export function isNaturalLanguageGoalRequest(value: string): boolean {
  const normalized = value.trim().toLowerCase().replace(/\s+/g, ' ')
  if (!normalized) {
    return false
  }

  return (
    /\b(?:in the background|background task|background run)\b/.test(normalized) ||
    /\b(?:keep working on this|continue working on this|keep going on this)\b/.test(normalized) ||
    /\b(?:make progress while i(?:'m| am) away|work on this while i(?:'m| am) away)\b/.test(normalized) ||
    /\b(?:retry until|checkpointed|with checkpoints|save checkpoints)\b/.test(normalized) ||
    /\b(?:spend|work for|run for|continue for|for the next)\s+\d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b/.test(normalized) ||
    /\b\d+\s*(?:min(?:ute)?s?|hours?|hrs?)\b.*\b(?:background|keep working|continue working|come back)\b/.test(normalized) ||
    /\b(?:keep at it|stay on this|come back with progress)\b/.test(normalized)
  )
}

export function resolveChatGoalWorkflowRouting(
  input: ResolveChatGoalWorkflowRoutingInput
): ChatGoalWorkflowRoutingDecision {
  const modeCommand = parseChatModeCommand(input.text)
  const goalCommand = parseGoalCommand(input.text)
  const requestText =
    modeCommand
      ? modeCommand.content
      : goalCommand?.action === 'proposal'
        ? goalCommand.content
        : input.text
  const workflowModeRequested = modeCommand?.mode === 'workflow'
  const workflowProposalRequested = workflowModeRequested && requestText.length > 0
  const naturalLanguageGoalRequested =
    !goalCommand &&
    !modeCommand &&
    !input.hasPendingProposal &&
    input.attachmentCount === 0 &&
    isNaturalLanguageGoalRequest(requestText)
  const activeGoalFollowUpRequested =
    !goalCommand &&
    !modeCommand &&
    !input.hasPendingProposal &&
    input.hasActiveGoal &&
    (requestText.trim().length > 0 || input.attachmentCount > 0) &&
    !naturalLanguageGoalRequested
  const confirmationRequested =
    !goalCommand &&
    !modeCommand &&
    input.hasPendingProposal &&
    input.attachmentCount === 0 &&
    isGoalConfirmationText(input.text)
  const proposalRevisionRequested =
    !goalCommand &&
    !modeCommand &&
    input.hasPendingProposal &&
    !confirmationRequested &&
    (requestText.trim().length > 0 || input.attachmentCount > 0)

  let route: ChatGoalWorkflowRoute = { kind: 'direct_chat' }
  if (goalCommand?.action === 'help') {
    route = {
      kind: 'goal_help',
      raw: goalCommand.raw,
    }
  } else if (
    goalCommand?.action === 'status' ||
    goalCommand?.action === 'pause' ||
    goalCommand?.action === 'resume' ||
    goalCommand?.action === 'stop'
  ) {
    route = {
      kind: 'goal_lifecycle',
      action: goalCommand.action,
      raw: goalCommand.raw,
    }
  } else if (workflowProposalRequested) {
    route = {
      kind: 'workflow_proposal',
      requestText,
    }
  } else if (goalCommand?.action === 'proposal') {
    route = {
      kind: 'goal_proposal',
      content: goalCommand.content,
      raw: goalCommand.raw,
    }
  } else if (naturalLanguageGoalRequested) {
    route = {
      kind: 'natural_language_goal_proposal',
      requestText,
    }
  } else if (confirmationRequested) {
    route = {
      kind: 'goal_confirmation',
      requestText,
      raw: input.text.trim(),
    }
  } else if (proposalRevisionRequested) {
    route = {
      kind: 'goal_revision',
      requestText,
    }
  } else if (activeGoalFollowUpRequested) {
    route = {
      kind: 'goal_follow_up',
      requestText,
    }
  }

  return {
    modeCommand,
    requestText,
    route,
    workflowModeRequested,
    requiresSessionMaterialization: modeCommand !== null || route.kind !== 'direct_chat',
    shouldHandleGoalWorkflowRouting: route.kind !== 'direct_chat',
  }
}
