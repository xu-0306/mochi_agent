import {
  buildGoalApprovalCountLabel,
  buildGoalBlockerSummary,
  buildGoalChromeCopy,
  buildGoalPendingApprovalNotice,
  buildGoalRecommendedActionLabel,
  buildGoalReviewApprovalsLabel,
  buildGoalUiErrorMessage,
} from '@/lib/goal-proposal-copy'

export type GoalFocusCalloutAction = 'review_approvals' | 'open_console' | 'open_details'

export interface GoalFocusBlockerSnapshot {
  summary: string | null
  recommendedAction: string | null
  latestError: string | null
  approvalCount?: number
  approvalIds: string[]
  approvalToolNames: string[]
  blockedTools: string[]
  blockedDomains: string[]
  blockNetworkUsage: boolean
}

export interface GoalFocusCalloutView {
  tone: 'warning' | 'error' | 'info'
  title: string
  message: string
  detail: string | null
  meta: string[]
  action: GoalFocusCalloutAction | null
  actionLabel: string | null
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.map((value) => value?.trim() ?? '').filter((value) => value.length > 0))]
}

export function summarizeGoalRestrictionMeta(
  userMessage: string,
  blocker: GoalFocusBlockerSnapshot | null | undefined
): string[] {
  if (!blocker) {
    return []
  }

  const copy = buildGoalChromeCopy(userMessage)
  const values: Array<string | null> = []

  if (blocker.blockNetworkUsage) {
    values.push(`${copy.networkLabel}: ${copy.blockedValueLabel}`)
  }
  if (blocker.blockedTools.length > 0) {
    values.push(`${copy.toolsSectionLabel}: ${blocker.blockedTools.join(', ')}`)
  }
  if (blocker.blockedDomains.length > 0) {
    values.push(`${copy.domainsSectionLabel}: ${blocker.blockedDomains.join(', ')}`)
  }
  if (blocker.approvalToolNames.length > 0) {
    values.push(`${copy.approvalWaitLabel}: ${blocker.approvalToolNames.join(', ')}`)
  }

  return uniqueStrings(values)
}

export function buildGoalFocusCallout(options: {
  userMessage: string
  pendingApprovalCount: number
  blocker?: GoalFocusBlockerSnapshot | null
  errorMessage?: string | null
  goalDisplayState?: 'active' | 'blocked' | 'completed' | 'failed'
}): GoalFocusCalloutView | null {
  const {
    userMessage,
    pendingApprovalCount,
    blocker = null,
    errorMessage = null,
    goalDisplayState = 'active',
  } = options
  const copy = buildGoalChromeCopy(userMessage)
  const approvalCount = Math.max(
    pendingApprovalCount,
    blocker?.approvalCount ?? blocker?.approvalIds.length ?? 0
  )
  const recommendedAction =
    buildGoalRecommendedActionLabel(userMessage, blocker?.recommendedAction) ??
    blocker?.recommendedAction ??
    null
  const detail = recommendedAction
    ? `${copy.recommendedActionLabel}: ${recommendedAction}`
    : null

  if (goalDisplayState === 'completed') {
    return null
  }

  if (approvalCount > 0) {
    return {
      tone: 'warning',
      title: copy.approvalWaitLabel,
      message: blocker
        ? buildGoalBlockerSummary(userMessage, blocker.summary, blocker.latestError, {
            approvalCount,
            recommendedAction: blocker.recommendedAction,
          })
        : buildGoalPendingApprovalNotice(userMessage, approvalCount),
      detail,
      meta: uniqueStrings([
        buildGoalApprovalCountLabel(userMessage, approvalCount),
        ...summarizeGoalRestrictionMeta(userMessage, blocker),
      ]),
      action: 'review_approvals',
      actionLabel: buildGoalReviewApprovalsLabel(userMessage),
    }
  }

  if (blocker) {
    return {
      tone: 'warning',
      title: copy.blockedStatusLabel,
      message: buildGoalBlockerSummary(userMessage, blocker.summary, blocker.latestError, {
        approvalCount,
        recommendedAction: blocker.recommendedAction,
      }),
      detail,
      meta: summarizeGoalRestrictionMeta(userMessage, blocker),
      action: 'open_details',
      actionLabel: copy.openDetailsLabel,
    }
  }

  if (errorMessage) {
    return {
      tone: 'error',
      title: copy.goalStatusLabel,
      message: buildGoalUiErrorMessage(
        userMessage,
        errorMessage,
        copy.goalStatusRefreshFailedLabel
      ),
      detail: null,
      meta: [],
      action: 'open_console',
      actionLabel: copy.openConsoleLabel,
    }
  }

  return null
}
