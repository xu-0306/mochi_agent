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
  | { kind: 'workflow_pending_follow_up'; requestText: string; raw: string }
  | { kind: 'goal_confirmation'; requestText: string; raw: string }
  | { kind: 'goal_revision'; requestText: string }
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

export function parseNaturalLanguageGoalActivation(
  value: string
): Pick<GoalCommand, 'content' | 'raw'> | null {
  const raw = value.trim()
  if (!raw || raw.startsWith('/')) {
    return null
  }

  const englishMatch = raw.match(
    /^(?:please\s+)?(?:create|start|set|activate)\s+(?:a\s+)?goal(?:\s*(?::|-|for)\s*|\s+to\s+)(.+)$/i
  )
  if (englishMatch?.[1]?.trim()) {
    return { content: englishMatch[1].trim(), raw }
  }

  const traditionalChineseMatch = raw.match(
    /^(?:\u8acb\s*)?(?:\u5efa\u7acb|\u555f\u52d5|\u8a2d\u5b9a|\u958b\u59cb)\s*(?:\u4e00\u500b\s*)?(?:goal|\u76ee\u6a19)\s*(?:\u70ba|\u662f|\uff1a|:)?\s*(.+)$/i
  )
  if (traditionalChineseMatch?.[1]?.trim()) {
    return { content: traditionalChineseMatch[1].trim(), raw }
  }

  // Existing routing tests already recognize an English, explicitly time-bounded
  // research/background-work request as a durable task intent.
  if (
    /^(?:research|investigate)\b[\s\S]*\bfor\s+(?:the\s+next\s+)?\d+\s+(?:minutes?|hours?)\b/i.test(raw) ||
    /^keep\s+working\b[\s\S]*\bin\s+the\s+background\b[\s\S]*\bfor\s+(?:the\s+next\s+)?\d+\s+(?:minutes?|hours?)\b/i.test(raw)
  ) {
    return { content: raw, raw }
  }

  return null
}
export function isPlainGreeting(value: string): boolean {
  const normalized = value.trim().toLowerCase().replace(/[\s!\uFF01.\u3002?\uFF1F,\uFF0C~\uFF5E]+/g, '')
  return ['hi', 'hello', 'hey', '\u4f60\u597d', '\u60a8\u597d', '\u55e8'].includes(normalized)
}
export function resolveChatGoalWorkflowRouting(
  input: ResolveChatGoalWorkflowRoutingInput
): ChatGoalWorkflowRoutingDecision {
  const modeCommand = parseChatModeCommand(input.text)
  const goalCommand = parseGoalCommand(input.text)
  const naturalGoalActivation = modeCommand ? null : parseNaturalLanguageGoalActivation(input.text)
  const requestText =
    modeCommand
      ? modeCommand.content
      : goalCommand?.action === 'proposal'
        ? goalCommand.content
        : naturalGoalActivation?.content ?? input.text
  const workflowModeRequested = modeCommand?.mode === 'workflow'
  const workflowProposalRequested = workflowModeRequested && requestText.length > 0
  const pendingProposalFollowUpRequested =
    !goalCommand &&
    !modeCommand &&
    input.hasPendingProposal &&
    input.attachmentCount === 0 &&
    !isPlainGreeting(requestText) &&
    requestText.trim().length > 0
  const proposalRevisionRequested =
    !goalCommand &&
    !modeCommand &&
    input.hasPendingProposal &&
    input.attachmentCount > 0

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
  } else if (naturalGoalActivation) {
    route = {
      kind: 'goal_proposal',
      content: naturalGoalActivation.content,
      raw: naturalGoalActivation.raw,
    }
  } else if (pendingProposalFollowUpRequested) {
    route = {
      kind: 'workflow_pending_follow_up',
      requestText,
      raw: input.text.trim(),
    }
  } else if (proposalRevisionRequested) {
    route = {
      kind: 'goal_revision',
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
