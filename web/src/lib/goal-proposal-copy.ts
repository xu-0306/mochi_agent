export interface GoalProposalAssistantFallbackInput {
  objective: string
  execution_mode: 'single_agent' | 'workflow'
  protocol_selection?: string | null
  revision_index: number
}

export interface GoalProposalSystemCtaCopy {
  title: string
  launchLabel: string
  launchBody: string
  reviseLabel: string
  reviseBody: string
  chatLabel: string
  chatBody: string
}

export type GoalCardKind = 'proposal' | 'revised_proposal' | 'started'

export interface GoalCardChromeCopy {
  proposalLabel: string
  revisedProposalLabel: string
  startedLabel: string
  completedGoalLabel: string
  goalNeedsAttentionLabel: string
  singleAgentLabel: string
  workflowLabel: string
  executionLabel: string
  protocolLabel: string
  runtimeLabel: string
  goalIdLabel: string
  objectiveLabel: string
  modelsLabel: string
  roleSummaryLabel: string
  riskNoteLabel: string
  supersededLabel: string
  activeGoalLabel: string
  mostRecentGoalLabel: string
  goalSummaryLabel: string
  goalBlockedLabel: string
  goalStatusLabel: string
  goalUpdatedLabel: string
  goalPausedLabel: string
  goalResumedLabel: string
  goalStoppedLabel: string
  pendingSummaryIntro: string
  activeSummaryIntro: string
  recentSummaryIntro: string
  closeGoalDrawerLabel: string
  blockedStatusLabel: string
  recommendedActionLabel: string
  operatorControlsLabel: string
  approvalWaitLabel: string
  pendingApprovalsDescription: string
  pendingApprovalsCountLabel: string
  loadingPendingApprovalsLabel: string
  pendingApprovalsLoadFailedLabel: string
  pendingFileReviewLabel: string
  approvalMetadataUnavailableLabel: string
  goalStatusLoadFailedLabel: string
  goalStatusRefreshFailedLabel: string
  approvalResolveFailedLabel: string
  approveOnceLabel: string
  rejectLabel: string
  refreshLabel: string
  pauseLabel: string
  resumeLabel: string
  stopLabel: string
  openDetailsLabel: string
  openConsoleLabel: string
  networkLabel: string
  toolsSectionLabel: string
  domainsSectionLabel: string
  blockedValueLabel: string
  allowedValueLabel: string
  blockedToolsEmptyLabel: string
  blockedDomainsEmptyLabel: string
  notReportedLabel: string
  noShellLabel: string
  patchValidationLabel: string
  replaySafeLabel: string
  notStartedLabel: string
  goalNeedsAttentionBody: string
}

export type GoalLifecycleMessageKind =
  | 'goal_started'
  | 'goal_manage_hint'
  | 'pending_cleared'
  | 'no_active_goal'
  | 'status_fetched'
  | 'goal_paused'
  | 'goal_resumed'
  | 'goal_stopped'

type GoalProposalLanguageHint =
  | 'traditional_chinese'
  | 'simplified_chinese'
  | 'chinese'
  | 'japanese'
  | 'korean'
  | 'devanagari'
  | 'bengali'
  | 'gurmukhi'
  | 'gujarati'
  | 'tamil'
  | 'telugu'
  | 'kannada'
  | 'malayalam'
  | 'latin_script'
  | 'other'

const TRADITIONAL_CHINESE_HINTS =
  '\u9019\u500b\u5e6b\u8acb\u8207\u70ba\u8aaa\u660e\u555f\u52d5\u7e7c\u7e8c\u7bc4\u570d\u8f03\u9069\u5408\u8abf\u8ad6\u6587'
const SIMPLIFIED_CHINESE_HINTS =
  '\u8fd9\u4e2a\u5e2e\u8bf7\u4e0e\u4e3a\u8bf4\u660e\u542f\u52a8\u7ee7\u7eed\u8303\u56f4\u6bd4\u8f83\u9002\u5408\u8c03\u8bba\u6587'

function containsJapaneseKana(text: string): boolean {
  return /[\u3040-\u309f\u30a0-\u30ff\u31f0-\u31ff\uff66-\uff9f]/.test(text)
}

function containsHangul(text: string): boolean {
  return /[\u1100-\u11ff\u3130-\u318f\uac00-\ud7af]/.test(text)
}

function containsAsciiLetters(text: string): boolean {
  return /[A-Za-z]/.test(text)
}

function containsBlock(text: string, start: number, end: number): boolean {
  return [...text].some((char) => {
    const code = char.codePointAt(0) ?? 0
    return code >= start && code <= end
  })
}

function detectGoalProposalLanguageHint(value: string): GoalProposalLanguageHint {
  const text = value.trim()
  if (!text) {
    return 'latin_script'
  }
  if (containsJapaneseKana(text)) {
    return 'japanese'
  }
  if (containsHangul(text)) {
    return 'korean'
  }
  if (/[\u4e00-\u9fff]/.test(text)) {
    const traditionalHits = [...text].filter((char) => TRADITIONAL_CHINESE_HINTS.includes(char)).length
    const simplifiedHits = [...text].filter((char) => SIMPLIFIED_CHINESE_HINTS.includes(char)).length
    if (traditionalHits > simplifiedHits) {
      return 'traditional_chinese'
    }
    if (simplifiedHits > traditionalHits) {
      return 'simplified_chinese'
    }
    return 'chinese'
  }
  if (containsBlock(text, 0x0900, 0x097f)) {
    return 'devanagari'
  }
  if (containsBlock(text, 0x0980, 0x09ff)) {
    return 'bengali'
  }
  if (containsBlock(text, 0x0a00, 0x0a7f)) {
    return 'gurmukhi'
  }
  if (containsBlock(text, 0x0a80, 0x0aff)) {
    return 'gujarati'
  }
  if (containsBlock(text, 0x0b80, 0x0bff)) {
    return 'tamil'
  }
  if (containsBlock(text, 0x0c00, 0x0c7f)) {
    return 'telugu'
  }
  if (containsBlock(text, 0x0c80, 0x0cff)) {
    return 'kannada'
  }
  if (containsBlock(text, 0x0d00, 0x0d7f)) {
    return 'malayalam'
  }
  if (containsAsciiLetters(text)) {
    return 'latin_script'
  }
  return 'other'
}

function isGoalCopyLanguageAligned(userMessage: string, candidate: string): boolean {
  const userHint = detectGoalProposalLanguageHint(userMessage)
  const candidateHint = detectGoalProposalLanguageHint(candidate)

  if (
    userHint === 'traditional_chinese' ||
    userHint === 'simplified_chinese' ||
    userHint === 'chinese'
  ) {
    return (
      candidateHint === 'traditional_chinese' ||
      candidateHint === 'simplified_chinese' ||
      candidateHint === 'chinese'
    )
  }

  if (
    userHint === 'japanese' ||
    userHint === 'korean' ||
    userHint === 'devanagari' ||
    userHint === 'bengali' ||
    userHint === 'gurmukhi' ||
    userHint === 'gujarati' ||
    userHint === 'tamil' ||
    userHint === 'telugu' ||
    userHint === 'kannada' ||
    userHint === 'malayalam'
  ) {
    return candidateHint === userHint
  }

  return true
}

function isChineseLanguageHint(hint: GoalProposalLanguageHint): boolean {
  return (
    hint === 'traditional_chinese' ||
    hint === 'simplified_chinese' ||
    hint === 'chinese'
  )
}

function buildGoalApprovalWaitSummary(userMessage: string, approvalCount: number): string {
  const safeCount = Math.max(0, approvalCount)
  const languageHint = detectGoalProposalLanguageHint(userMessage)

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    if (safeCount > 0) {
      return `這個 goal 還在等待 ${safeCount} 個待核准項目處理完成。請到 goal drawer 或 Goal Console 處理後再繼續。`
    }
    return '這個 goal 還在等待操作者核准。請到 goal drawer 或 Goal Console 檢查後再繼續。'
  }

  if (languageHint === 'simplified_chinese') {
    if (safeCount > 0) {
      return `这个 goal 还在等待 ${safeCount} 个待批准项目处理完成。请到 goal drawer 或 Goal Console 处理后再继续。`
    }
    return '这个 goal 还在等待操作员批准。请到 goal drawer 或 Goal Console 检查后再继续。'
  }

  if (safeCount > 0) {
    return `This goal is waiting for ${safeCount} pending approval${safeCount === 1 ? '' : 's'} before it can continue. Review them from the goal drawer or Goal Console.`
  }
  return 'This goal is waiting on operator approval before it can continue. Review the pending approval from the goal drawer or Goal Console.'
}

function looksLikeApprovalBlocker(value: string): boolean {
  const normalized = value.trim().toLowerCase()
  if (!normalized) {
    return false
  }
  return (
    /\bapproval\b/.test(normalized) ||
    /\bapprove\b/.test(normalized) ||
    /\boperator approval\b/.test(normalized) ||
    /\bawaiting approval\b/.test(normalized) ||
    /\bwaiting on operator\b/.test(normalized)
  )
}

export function buildLocalGoalProposalAssistantExplanation(
  userMessage: string,
  proposal: GoalProposalAssistantFallbackInput
): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage || proposal.objective)
  const protocol = proposal.protocol_selection?.trim()
  const updated = proposal.revision_index > 0

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    const prefix = updated
      ? '\u6211\u5df2\u4f9d\u7167\u4f60\u6700\u65b0\u7684\u65b9\u5411\u66f4\u65b0\u9019\u4efd goal \u63d0\u6848\u3002'
      : '\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd goal \u8349\u7a3f\uff0c\u4f5c\u70ba\u9019\u500b\u4efb\u52d9\u7684\u57f7\u884c\u5951\u7d04\u3002'
    const detail = protocol
      ? `\u76ee\u524d\u9078\u5b9a\u7684\u57f7\u884c\u7b56\u7565\u662f ${protocol}\uff0c\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u958b\u59cb\u57f7\u884c\u3002`
      : proposal.execution_mode === 'workflow'
        ? '\u9019\u500b\u7bc4\u570d\u8f03\u9069\u5408\u9078\u7528 workflow \u4f5c\u70ba\u9019\u500b goal \u7684\u660e\u78ba\u7b56\u7565\uff0c\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u57f7\u884c\u3002'
        : '\u9019\u500b\u7bc4\u570d\u8f03\u9069\u5408\u7528 single-agent \u7b56\u7565\uff0c\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u57f7\u884c\u3002'
    return `${prefix} ${detail}`
  }

  if (languageHint === 'simplified_chinese') {
    const prefix = updated
      ? '\u6211\u5df2\u6839\u636e\u4f60\u6700\u65b0\u7684\u65b9\u5411\u66f4\u65b0\u8fd9\u4efd goal \u63d0\u6848\u3002'
      : '\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd goal \u8349\u7a3f\uff0c\u4f5c\u4e3a\u8fd9\u4e2a\u4efb\u52a1\u7684\u6267\u884c\u5951\u7ea6\u3002'
    const detail = protocol
      ? `\u76ee\u524d\u9009\u5b9a\u7684\u6267\u884c\u7b56\u7565\u662f ${protocol}\uff0c\u786e\u8ba4\u542f\u52a8\u540e\u624d\u4f1a\u5f00\u59cb\u6267\u884c\u3002`
      : proposal.execution_mode === 'workflow'
        ? '\u8fd9\u4e2a\u8303\u56f4\u66f4\u9002\u5408\u9009\u7528 workflow \u4f5c\u4e3a\u8fd9\u4e2a goal \u7684\u660e\u786e\u7b56\u7565\uff0c\u786e\u8ba4\u542f\u52a8\u540e\u624d\u4f1a\u6267\u884c\u3002'
        : '\u8fd9\u4e2a\u8303\u56f4\u66f4\u9002\u5408\u7528 single-agent \u7b56\u7565\uff0c\u786e\u8ba4\u542f\u52a8\u540e\u624d\u4f1a\u6267\u884c\u3002'
    return `${prefix} ${detail}`
  }

  const prefix = updated
    ? 'I updated this goal proposal to match your latest direction.'
    : 'I framed your request as a goal draft that acts as the contract for this task.'
  const detail = protocol
    ? `The selected execution strategy is ${protocol}, and execution begins only after you confirm the start.`
    : proposal.execution_mode === 'workflow'
      ? 'This scope fits an explicit workflow strategy under the goal, and execution begins only after you confirm the start.'
      : 'This scope fits a single-agent strategy, and execution begins only after you confirm the start.'
  return `${prefix} ${detail}`
}

export function buildGoalProposalSystemCtaCopy(userMessage: string): GoalProposalSystemCtaCopy {
  const languageHint = detectGoalProposalLanguageHint(userMessage)

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return {
      title: '\u4e0b\u4e00\u6b65',
      launchLabel: '\u555f\u52d5',
      launchBody: '\u8981\u958b\u59cb\u57f7\u884c\u6642\uff0c\u8acb\u9001\u51fa\u4e00\u5247\u7c21\u77ed\u78ba\u8a8d\u8a0a\u606f\u3002',
      reviseLabel: '\u4fee\u6539\u63d0\u6848',
      reviseBody: '\u5982\u679c\u8981\u7e2e\u5c0f\u3001\u8abf\u6574\u6216\u64f4\u5927\u7bc4\u570d\uff0c\u518d\u9001\u4e00\u5247\u8a0a\u606f\u5373\u53ef\u3002',
      chatLabel: '\u4e00\u822c\u804a\u5929',
      chatBody: '\u5982\u679c\u4f60\u60f3\u5148\u8a0e\u8ad6\uff0c\u8acb\u7528 `/chat <request>` \u66ab\u6642\u96e2\u958b goal \u8a2d\u5b9a\u3002',
    }
  }

  if (languageHint === 'simplified_chinese') {
    return {
      title: '\u4e0b\u4e00\u6b65',
      launchLabel: '\u542f\u52a8',
      launchBody: '\u8981\u5f00\u59cb\u6267\u884c\u65f6\uff0c\u8bf7\u53d1\u9001\u4e00\u6761\u7b80\u77ed\u786e\u8ba4\u6d88\u606f\u3002',
      reviseLabel: '\u4fee\u6539\u63d0\u6848',
      reviseBody: '\u5982\u679c\u8981\u7f29\u5c0f\u3001\u8c03\u6574\u6216\u6269\u5927\u8303\u56f4\uff0c\u518d\u53d1\u9001\u4e00\u6761\u6d88\u606f\u5373\u53ef\u3002',
      chatLabel: '\u666e\u901a\u804a\u5929',
      chatBody: '\u5982\u679c\u4f60\u60f3\u5148\u8ba8\u8bba\uff0c\u8bf7\u7528 `/chat <request>` \u6682\u65f6\u79bb\u5f00 goal \u8bbe\u7f6e\u3002',
    }
  }

  return {
    title: 'Next step',
    launchLabel: 'Launch',
    launchBody: 'Send a short confirmation when you want execution to begin.',
    reviseLabel: 'Revise',
    reviseBody: 'Send another message to narrow, change, or expand the draft.',
    chatLabel: 'Chat',
    chatBody: 'Use `/chat <request>` to step outside goal setup.',
  }
}

export function buildGoalLifecycleMessage(
  userMessage: string,
  kind: GoalLifecycleMessageKind
): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    const mapping: Record<GoalLifecycleMessageKind, string> = {
      goal_started:
        'Goal \u5df2\u555f\u52d5\u3002\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u5b83\u3002',
      goal_manage_hint:
        '\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u76ee\u524d\u7684 goal\u3002',
      pending_cleared:
        '\u5df2\u6e05\u9664\u9019\u4efd\u5f85\u78ba\u8a8d\u7684 goal \u63d0\u6848\u3002\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u4efb\u52d9\uff0c\u6216\u7528 `/goal <request>` \u91cd\u65b0\u6e96\u5099\u4e00\u4efd\u8349\u7a3f\u3002',
      no_active_goal:
        '\u9019\u500b\u5c0d\u8a71\u76ee\u524d\u6c92\u6709\u7d81\u5b9a\u4efb\u4f55\u9032\u884c\u4e2d\u7684 goal\u3002\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u4efb\u52d9\uff0c\u6216\u7528 `/goal <request>` \u660e\u78ba\u6e96\u5099\u65b0\u7684 goal\u3002',
      status_fetched: '\u6211\u5df2\u53d6\u56de\u6700\u65b0\u7684 goal \u72c0\u614b\u3002',
      goal_paused: '\u6211\u5df2\u66ab\u505c\u9019\u500b\u9032\u884c\u4e2d\u7684 goal\u3002',
      goal_resumed: '\u6211\u5df2\u6062\u5fa9\u9019\u500b goal \u7684\u57f7\u884c\u3002',
      goal_stopped: '\u6211\u5df2\u505c\u6b62\u9019\u500b goal\u3002',
    }
    return mapping[kind]
  }

  if (languageHint === 'simplified_chinese') {
    const mapping: Record<GoalLifecycleMessageKind, string> = {
      goal_started:
        'Goal \u5df2\u542f\u52a8\u3002\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5b83\u3002',
      goal_manage_hint:
        '\u4f60\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5f53\u524d\u7684 goal\u3002',
      pending_cleared:
        '\u5df2\u6e05\u9664\u8fd9\u4efd\u5f85\u786e\u8ba4\u7684 goal \u63d0\u6848\u3002\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u4efb\u52a1\uff0c\u6216\u7528 `/goal <request>` \u91cd\u65b0\u51c6\u5907\u4e00\u4efd\u8349\u7a3f\u3002',
      no_active_goal:
        '\u8fd9\u4e2a\u5bf9\u8bdd\u76ee\u524d\u6ca1\u6709\u7ed1\u5b9a\u4efb\u4f55\u8fdb\u884c\u4e2d\u7684 goal\u3002\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u4efb\u52a1\uff0c\u6216\u7528 `/goal <request>` \u660e\u786e\u51c6\u5907\u65b0\u7684 goal\u3002',
      status_fetched: '\u6211\u5df2\u53d6\u56de\u6700\u65b0\u7684 goal \u72b6\u6001\u3002',
      goal_paused: '\u6211\u5df2\u6682\u505c\u8fd9\u4e2a\u8fdb\u884c\u4e2d\u7684 goal\u3002',
      goal_resumed: '\u6211\u5df2\u6062\u590d\u8fd9\u4e2a goal \u7684\u6267\u884c\u3002',
      goal_stopped: '\u6211\u5df2\u505c\u6b62\u8fd9\u4e2a goal\u3002',
    }
    return mapping[kind]
  }

  const mapping: Record<GoalLifecycleMessageKind, string> = {
    goal_started:
      'Goal started. Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` to manage it.',
    goal_manage_hint:
      'Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` to manage the active goal.',
    pending_cleared:
      'Cleared the pending goal proposal. Describe the task normally, or use `/goal <request>` to prepare a new draft.',
    no_active_goal:
      'No active goal is bound to this chat. Describe the task normally, or use `/goal <request>` to prepare a new goal.',
    status_fetched: 'Fetched the latest goal status.',
    goal_paused: 'Paused the active goal.',
    goal_resumed: 'Resumed the active goal.',
    goal_stopped: 'Stopped the active goal.',
  }
  return mapping[kind]
}

export type GoalFollowUpMessageKind =
  | 'active_goal_exists'
  | 'goal_help'
  | 'queued_after_resolution'
  | 'manual_resolution_required'
  | 'blocked'
  | 'no_live_attempt'
  | 'restarted_forwarded'
  | 'refreshed_forwarded'
  | 'resumed_forwarded'
  | 'forwarded'

export function buildGoalCardChromeCopy(userMessage: string): GoalCardChromeCopy {
  const languageHint = detectGoalProposalLanguageHint(userMessage)

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return {
      proposalLabel: 'Goal \u63d0\u6848',
      revisedProposalLabel: '\u5df2\u66f4\u65b0\u7684 goal \u63d0\u6848',
      startedLabel: 'Goal \u5df2\u555f\u52d5',
      completedGoalLabel: '\u5df2\u5b8c\u6210\u7684 goal',
      goalNeedsAttentionLabel: 'Goal \u9700\u8981\u8655\u7406',
      singleAgentLabel: '\u55ae\u4ee3\u7406',
      workflowLabel: '\u5de5\u4f5c\u6d41',
      executionLabel: '\u57f7\u884c\u65b9\u5f0f',
      protocolLabel: '\u5354\u5b9a',
      runtimeLabel: '\u57f7\u884c\u6a21\u5f0f',
      goalIdLabel: 'Goal ID',
      objectiveLabel: '\u76ee\u6a19',
      modelsLabel: '\u6a21\u578b',
      roleSummaryLabel: '\u89d2\u8272\u6458\u8981',
      riskNoteLabel: '\u98a8\u96aa\u63d0\u793a',
      supersededLabel: '\u5df2\u53d6\u4ee3',
      activeGoalLabel: '\u9032\u884c\u4e2d\u7684 goal',
      mostRecentGoalLabel: '\u6700\u8fd1\u7684 goal',
      goalSummaryLabel: 'Goal \u6458\u8981',
      goalBlockedLabel: 'Goal \u53d7\u963b',
      goalStatusLabel: 'Goal \u72c0\u614b',
      goalUpdatedLabel: 'Goal \u5df2\u66f4\u65b0',
      goalPausedLabel: 'Goal \u5df2\u66ab\u505c',
      goalResumedLabel: 'Goal \u5df2\u6062\u5fa9',
      goalStoppedLabel: 'Goal \u5df2\u505c\u6b62',
      pendingSummaryIntro: '\u6b64\u5c0d\u8a71\u76ee\u524d\u6709\u4e00\u4efd\u5f85\u78ba\u8a8d\u7684 goal \u63d0\u6848\u3002',
      activeSummaryIntro: '\u4ee5\u4e0b\u662f\u9019\u500b\u5c0d\u8a71\u76ee\u524d\u9032\u884c\u4e2d\u7684 goal \u6458\u8981\u3002',
      recentSummaryIntro: '\u4ee5\u4e0b\u662f\u9019\u500b\u5c0d\u8a71\u6700\u8fd1\u4e00\u6b21\u7684 goal \u6458\u8981\u3002',
      closeGoalDrawerLabel: '\u95dc\u9589 goal drawer',
      blockedStatusLabel: '\u53d7\u963b\u72c0\u614b',
      recommendedActionLabel: '\u5efa\u8b70\u52d5\u4f5c',
      operatorControlsLabel: '\u64cd\u4f5c\u8005\u63a7\u5236',
      approvalWaitLabel: '\u6838\u51c6\u7b49\u5f85',
      pendingApprovalsDescription:
        '\u53ef\u4ee5\u76f4\u63a5\u5728\u9019\u88e1\u8655\u7406\u5f85\u6838\u51c6\u9805\u76ee\uff0c\u6216\u6253\u958b\u5b8c\u6574\u7684 Goal Console \u505a\u66f4\u6df1\u5165\u7684\u6aa2\u8996\u3002',
      pendingApprovalsCountLabel: '\u5f85\u6838\u51c6\u9805\u76ee',
      loadingPendingApprovalsLabel: '\u8f09\u5165\u5f85\u6838\u51c6\u9805\u76ee\u4e2d...',
      pendingApprovalsLoadFailedLabel: '\u8f09\u5165\u5f85\u6838\u51c6\u9805\u76ee\u5931\u6557\u3002',
      pendingFileReviewLabel: '\u5f85\u6aa2\u8996\u7684\u6a94\u6848',
      approvalMetadataUnavailableLabel:
        '\u76ee\u524d\u9019\u500b\u4ecb\u9762\u9084\u53d6\u4e0d\u5230\u5b8c\u6574\u7684\u6838\u51c6\u8a73\u7d30\u8cc7\u6599\uff0c\u4f46\u5df2\u7d93\u5075\u6e2c\u5230\u5f85\u6838\u51c6 metadata\u3002',
      goalStatusLoadFailedLabel: '\u8f09\u5165 goal \u72c0\u614b\u5931\u6557\u3002',
      goalStatusRefreshFailedLabel: '\u91cd\u65b0\u6574\u7406 goal \u72c0\u614b\u5931\u6557\u3002',
      approvalResolveFailedLabel: '\u8655\u7406\u6838\u51c6\u5931\u6557\u3002',
      approveOnceLabel: '\u55ae\u6b21\u6838\u51c6',
      rejectLabel: '\u62d2\u7d55',
      refreshLabel: '\u91cd\u65b0\u6574\u7406',
      pauseLabel: '\u66ab\u505c',
      resumeLabel: '\u6062\u5fa9',
      stopLabel: '\u505c\u6b62',
      openDetailsLabel: '\u67e5\u770b\u8a73\u7d30',
      openConsoleLabel: '\u958b\u555f console',
      networkLabel: '\u7db2\u8def',
      toolsSectionLabel: '\u5de5\u5177',
      domainsSectionLabel: '\u7db2\u57df',
      blockedValueLabel: '\u5c01\u9396',
      allowedValueLabel: '\u5141\u8a31',
      blockedToolsEmptyLabel: '\u6c92\u6709\u5de5\u5177\u5c01\u9396',
      blockedDomainsEmptyLabel: '\u6c92\u6709\u7db2\u57df\u5c01\u9396',
      notReportedLabel: '\u672a\u56de\u5831',
      noShellLabel: '\u672a\u63d0\u4f9b shell',
      patchValidationLabel: 'patch validation',
      replaySafeLabel: 'replay safe',
      notStartedLabel: '\u5c1a\u672a\u555f\u52d5',
      goalNeedsAttentionBody: '\u9019\u500b goal \u5728\u7e7c\u7e8c\u4e4b\u524d\u9700\u8981\u64cd\u4f5c\u8005\u5148\u8655\u7406\u3002',
    }
  }

  if (languageHint === 'simplified_chinese') {
    return {
      proposalLabel: 'Goal \u63d0\u6848',
      revisedProposalLabel: '\u5df2\u66f4\u65b0\u7684 goal \u63d0\u6848',
      startedLabel: 'Goal \u5df2\u542f\u52a8',
      completedGoalLabel: '\u5df2\u5b8c\u6210\u7684 goal',
      goalNeedsAttentionLabel: 'Goal \u9700\u8981\u5904\u7406',
      singleAgentLabel: '\u5355\u4ee3\u7406',
      workflowLabel: '\u5de5\u4f5c\u6d41',
      executionLabel: '\u6267\u884c\u65b9\u5f0f',
      protocolLabel: '\u534f\u8bae',
      runtimeLabel: '\u6267\u884c\u6a21\u5f0f',
      goalIdLabel: 'Goal ID',
      objectiveLabel: '\u76ee\u6807',
      modelsLabel: '\u6a21\u578b',
      roleSummaryLabel: '\u89d2\u8272\u6458\u8981',
      riskNoteLabel: '\u98ce\u9669\u63d0\u793a',
      supersededLabel: '\u5df2\u66ff\u4ee3',
      activeGoalLabel: '\u8fdb\u884c\u4e2d\u7684 goal',
      mostRecentGoalLabel: '\u6700\u8fd1\u7684 goal',
      goalSummaryLabel: 'Goal \u6458\u8981',
      goalBlockedLabel: 'Goal \u53d7\u963b',
      goalStatusLabel: 'Goal \u72b6\u6001',
      goalUpdatedLabel: 'Goal \u5df2\u66f4\u65b0',
      goalPausedLabel: 'Goal \u5df2\u6682\u505c',
      goalResumedLabel: 'Goal \u5df2\u6062\u590d',
      goalStoppedLabel: 'Goal \u5df2\u505c\u6b62',
      pendingSummaryIntro: '\u6b64\u5bf9\u8bdd\u76ee\u524d\u6709\u4e00\u4efd\u5f85\u786e\u8ba4\u7684 goal \u63d0\u6848\u3002',
      activeSummaryIntro: '\u4ee5\u4e0b\u662f\u8fd9\u4e2a\u5bf9\u8bdd\u76ee\u524d\u8fdb\u884c\u4e2d\u7684 goal \u6458\u8981\u3002',
      recentSummaryIntro: '\u4ee5\u4e0b\u662f\u8fd9\u4e2a\u5bf9\u8bdd\u6700\u8fd1\u4e00\u6b21\u7684 goal \u6458\u8981\u3002',
      closeGoalDrawerLabel: '\u5173\u95ed goal drawer',
      blockedStatusLabel: '\u53d7\u963b\u72b6\u6001',
      recommendedActionLabel: '\u5efa\u8bae\u52a8\u4f5c',
      operatorControlsLabel: '\u64cd\u4f5c\u8005\u63a7\u5236',
      approvalWaitLabel: '\u6279\u51c6\u7b49\u5f85',
      pendingApprovalsDescription:
        '\u53ef\u4ee5\u76f4\u63a5\u5728\u8fd9\u91cc\u5904\u7406\u5f85\u6279\u51c6\u9879\u76ee\uff0c\u6216\u6253\u5f00\u5b8c\u6574\u7684 Goal Console \u505a\u66f4\u6df1\u5165\u7684\u68c0\u67e5\u3002',
      pendingApprovalsCountLabel: '\u5f85\u6279\u51c6\u9879\u76ee',
      loadingPendingApprovalsLabel: '\u6b63\u5728\u52a0\u8f7d\u5f85\u6279\u51c6\u9879\u76ee...',
      pendingApprovalsLoadFailedLabel: '\u52a0\u8f7d\u5f85\u6279\u51c6\u9879\u76ee\u5931\u8d25\u3002',
      pendingFileReviewLabel: '\u5f85\u68c0\u67e5\u7684\u6587\u4ef6',
      approvalMetadataUnavailableLabel:
        '\u76ee\u524d\u8fd9\u4e2a\u754c\u9762\u8fd8\u62ff\u4e0d\u5230\u5b8c\u6574\u7684\u6279\u51c6\u8be6\u7ec6\u8d44\u6599\uff0c\u4f46\u5df2\u7ecf\u68c0\u6d4b\u5230\u5f85\u6279\u51c6 metadata\u3002',
      goalStatusLoadFailedLabel: '\u52a0\u8f7d goal \u72b6\u6001\u5931\u8d25\u3002',
      goalStatusRefreshFailedLabel: '\u5237\u65b0 goal \u72b6\u6001\u5931\u8d25\u3002',
      approvalResolveFailedLabel: '\u5904\u7406\u6279\u51c6\u5931\u8d25\u3002',
      approveOnceLabel: '\u5355\u6b21\u6279\u51c6',
      rejectLabel: '\u62d2\u7edd',
      refreshLabel: '\u5237\u65b0',
      pauseLabel: '\u6682\u505c',
      resumeLabel: '\u6062\u590d',
      stopLabel: '\u505c\u6b62',
      openDetailsLabel: '\u67e5\u770b\u8be6\u60c5',
      openConsoleLabel: '\u6253\u5f00 console',
      networkLabel: '\u7f51\u7edc',
      toolsSectionLabel: '\u5de5\u5177',
      domainsSectionLabel: '\u57df\u540d',
      blockedValueLabel: '\u5c01\u9501',
      allowedValueLabel: '\u5141\u8bb8',
      blockedToolsEmptyLabel: '\u6ca1\u6709\u5de5\u5177\u5c01\u9501',
      blockedDomainsEmptyLabel: '\u6ca1\u6709\u57df\u540d\u5c01\u9501',
      notReportedLabel: '\u672a\u62a5\u544a',
      noShellLabel: '\u672a\u63d0\u4f9b shell',
      patchValidationLabel: 'patch validation',
      replaySafeLabel: 'replay safe',
      notStartedLabel: '\u5c1a\u672a\u542f\u52a8',
      goalNeedsAttentionBody: '\u8fd9\u4e2a goal \u5728\u7ee7\u7eed\u4e4b\u524d\u9700\u8981\u64cd\u4f5c\u8005\u5148\u5904\u7406\u3002',
    }
  }

  return {
    proposalLabel: 'Goal proposal',
    revisedProposalLabel: 'Revised goal proposal',
    startedLabel: 'Goal started',
    completedGoalLabel: 'Completed goal',
    goalNeedsAttentionLabel: 'Goal needs attention',
    singleAgentLabel: 'Single agent',
    workflowLabel: 'Workflow',
    executionLabel: 'Execution',
    protocolLabel: 'Protocol',
    runtimeLabel: 'Runtime',
    goalIdLabel: 'Goal ID',
    objectiveLabel: 'Objective',
    modelsLabel: 'Models',
    roleSummaryLabel: 'Role summary',
    riskNoteLabel: 'Risk note',
    supersededLabel: 'Superseded',
    activeGoalLabel: 'Active goal',
    mostRecentGoalLabel: 'Most recent goal',
    goalSummaryLabel: 'Goal summary',
    goalBlockedLabel: 'Goal blocked',
    goalStatusLabel: 'Goal status',
    goalUpdatedLabel: 'Goal updated',
    goalPausedLabel: 'Goal paused',
    goalResumedLabel: 'Goal resumed',
    goalStoppedLabel: 'Goal stopped',
    pendingSummaryIntro: 'Pending goal proposal in this session.',
    activeSummaryIntro: 'Active goal summary for this session.',
    recentSummaryIntro: 'Most recent goal summary for this session.',
    closeGoalDrawerLabel: 'Close goal drawer',
    blockedStatusLabel: 'Blocked status',
    recommendedActionLabel: 'Recommended action',
    operatorControlsLabel: 'Operator controls',
    approvalWaitLabel: 'Approval wait',
    pendingApprovalsDescription:
      'Resolve the pending approval here, or open the full Goal Console for deeper review.',
    pendingApprovalsCountLabel: 'Pending approvals',
    loadingPendingApprovalsLabel: 'Loading pending approvals...',
    pendingApprovalsLoadFailedLabel: 'Failed to load pending approvals.',
    pendingFileReviewLabel: 'Pending file review',
    approvalMetadataUnavailableLabel:
      'Approval metadata is present, but the detailed approval payload is not available yet on this surface.',
    goalStatusLoadFailedLabel: 'Failed to load goal status.',
    goalStatusRefreshFailedLabel: 'Failed to refresh goal status.',
    approvalResolveFailedLabel: 'Failed to resolve approval.',
    approveOnceLabel: 'Approve once',
    rejectLabel: 'Reject',
    refreshLabel: 'Refresh',
    pauseLabel: 'Pause',
    resumeLabel: 'Resume',
    stopLabel: 'Stop',
    openDetailsLabel: 'Open details',
    openConsoleLabel: 'Open console',
    networkLabel: 'Network',
    toolsSectionLabel: 'Tools',
    domainsSectionLabel: 'Domains',
    blockedValueLabel: 'Blocked',
    allowedValueLabel: 'Allowed',
    blockedToolsEmptyLabel: 'No tool blocks',
    blockedDomainsEmptyLabel: 'No domain blocks',
    notReportedLabel: 'Not reported',
    noShellLabel: 'No shell',
    patchValidationLabel: 'patch validation',
    replaySafeLabel: 'replay safe',
    notStartedLabel: 'Not started',
    goalNeedsAttentionBody: 'This goal needs operator attention before it can continue.',
  }
}

export function buildGoalCardKindLabel(userMessage: string, kind: GoalCardKind): string {
  const copy = buildGoalCardChromeCopy(userMessage)
  if (kind === 'revised_proposal') {
    return copy.revisedProposalLabel
  }
  if (kind === 'started') {
    return copy.startedLabel
  }
  return copy.proposalLabel
}

export function buildGoalCardExecutionModeLabel(
  userMessage: string,
  executionMode: 'single_agent' | 'workflow'
): string {
  const copy = buildGoalCardChromeCopy(userMessage)
  return executionMode === 'single_agent' ? copy.singleAgentLabel : copy.workflowLabel
}

export function buildGoalCardStatusLabel(
  userMessage: string,
  status: string | null | undefined
): string | null {
  const normalized = (status ?? '').trim().toLowerCase()
  if (!normalized) {
    return null
  }

  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    const mapping: Record<string, string> = {
      completed: '\u5df2\u5b8c\u6210',
      succeeded: '\u5df2\u5b8c\u6210',
      done: '\u5df2\u5b8c\u6210',
      running: '\u57f7\u884c\u4e2d',
      active: '\u57f7\u884c\u4e2d',
      started: '\u5df2\u555f\u52d5',
      in_progress: '\u9032\u884c\u4e2d',
      waiting_approval: '\u7b49\u5f85\u6838\u51c6',
      awaiting_approval: '\u7b49\u5f85\u6838\u51c6',
      blocked: '\u5df2\u53d7\u963b',
      paused: '\u5df2\u66ab\u505c',
      awaiting_resources: '\u7b49\u5f85\u8cc7\u6e90',
      stalled: '\u5df2\u505c\u6eef',
      partial: '\u90e8\u5206\u5b8c\u6210',
      pending: '\u5f85\u8655\u7406',
      approved: '\u5df2\u6838\u51c6',
      approved_once: '\u5df2\u55ae\u6b21\u6838\u51c6',
      approved_and_saved_rule: '\u5df2\u6838\u51c6\u4e26\u5132\u5b58\u898f\u5247',
      rejected: '\u5df2\u62d2\u7d55',
      failed: '\u5931\u6557',
      error: '\u932f\u8aa4',
      cancelled: '\u5df2\u53d6\u6d88',
      canceled: '\u5df2\u53d6\u6d88',
      superseded: '\u5df2\u53d6\u4ee3',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }

  if (languageHint === 'simplified_chinese') {
    const mapping: Record<string, string> = {
      completed: '\u5df2\u5b8c\u6210',
      succeeded: '\u5df2\u5b8c\u6210',
      done: '\u5df2\u5b8c\u6210',
      running: '\u6267\u884c\u4e2d',
      active: '\u6267\u884c\u4e2d',
      started: '\u5df2\u542f\u52a8',
      in_progress: '\u8fdb\u884c\u4e2d',
      waiting_approval: '\u7b49\u5f85\u6279\u51c6',
      awaiting_approval: '\u7b49\u5f85\u6279\u51c6',
      blocked: '\u5df2\u53d7\u963b',
      paused: '\u5df2\u6682\u505c',
      awaiting_resources: '\u7b49\u5f85\u8d44\u6e90',
      stalled: '\u5df2\u505c\u6ede',
      partial: '\u90e8\u5206\u5b8c\u6210',
      pending: '\u5f85\u5904\u7406',
      approved: '\u5df2\u6279\u51c6',
      approved_once: '\u5df2\u5355\u6b21\u6279\u51c6',
      approved_and_saved_rule: '\u5df2\u6279\u51c6\u5e76\u4fdd\u5b58\u89c4\u5219',
      rejected: '\u5df2\u62d2\u7edd',
      failed: '\u5931\u8d25',
      error: '\u9519\u8bef',
      cancelled: '\u5df2\u53d6\u6d88',
      canceled: '\u5df2\u53d6\u6d88',
      superseded: '\u5df2\u66ff\u4ee3',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }

  return normalized.replaceAll('_', ' ')
}

export function buildGoalHiddenModelsLabel(userMessage: string, hiddenCount: number): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (
    languageHint === 'traditional_chinese' ||
    languageHint === 'chinese' ||
    languageHint === 'simplified_chinese'
  ) {
    return `+${Math.max(0, hiddenCount)} \u66f4\u591a`
  }
  return `+${Math.max(0, hiddenCount)} more`
}

export function buildGoalDisplayStateLabel(
  userMessage: string,
  state: 'active' | 'blocked' | 'completed' | 'failed'
): string {
  const copy = buildGoalCardChromeCopy(userMessage)
  if (state === 'completed') {
    return copy.completedGoalLabel
  }
  if (state === 'blocked' || state === 'failed') {
    return copy.goalNeedsAttentionLabel
  }
  return copy.activeGoalLabel
}

export function buildGoalModelCountLabel(userMessage: string, count: number): string {
  const safeCount = Math.max(0, count)
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (
    languageHint === 'traditional_chinese' ||
    languageHint === 'chinese' ||
    languageHint === 'simplified_chinese'
  ) {
    return `${safeCount} \u500b\u6a21\u578b`
  }
  return `${safeCount} model${safeCount === 1 ? '' : 's'}`
}

export function buildGoalApprovalCountLabel(userMessage: string, count: number): string {
  const safeCount = Math.max(0, count)
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return `${safeCount} \u500b\u5f85\u6838\u51c6\u9805\u76ee`
  }
  if (languageHint === 'simplified_chinese') {
    return `${safeCount} \u4e2a\u5f85\u6279\u51c6\u9879\u76ee`
  }
  return `${safeCount} approval${safeCount === 1 ? '' : 's'}`
}

export function buildGoalPendingApprovalNotice(userMessage: string, count: number): string {
  const safeCount = Math.max(0, count)
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return `\u6709 ${safeCount} \u500b\u5f85\u6838\u51c6\u9805\u76ee\u5c1a\u672a\u8655\u7406\uff0c\u80cc\u666f\u5de5\u4f5c\u5728\u6838\u51c6\u5b8c\u6210\u524d\u4e0d\u6703\u7e7c\u7e8c\u3002`
  }
  if (languageHint === 'simplified_chinese') {
    return `\u6709 ${safeCount} \u4e2a\u5f85\u6279\u51c6\u9879\u76ee\u5c1a\u672a\u5904\u7406\uff0c\u540e\u53f0\u5de5\u4f5c\u5728\u6279\u51c6\u5b8c\u6210\u524d\u4e0d\u4f1a\u7ee7\u7eed\u3002`
  }
  return `${safeCount} approval${safeCount === 1 ? ' is' : 's are'} waiting before background work can continue.`
}

export function buildGoalReviewApprovalsLabel(userMessage: string): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return '\u67e5\u770b\u6838\u51c6\u9805\u76ee'
  }
  if (languageHint === 'simplified_chinese') {
    return '\u67e5\u770b\u6279\u51c6\u9879'
  }
  return 'Review approvals'
}

export function buildGoalOpenWorkflowLabel(userMessage: string): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return '\u958b\u555f workflow'
  }
  if (languageHint === 'simplified_chinese') {
    return '\u6253\u5f00 workflow'
  }
  return 'Open workflow'
}

export function buildGoalFileCountLabel(userMessage: string, count: number): string {
  const safeCount = Math.max(0, count)
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (
    languageHint === 'traditional_chinese' ||
    languageHint === 'chinese' ||
    languageHint === 'simplified_chinese'
  ) {
    return `${safeCount} \u500b\u6a94\u6848`
  }
  return `${safeCount} file${safeCount === 1 ? '' : 's'}`
}

export function buildGoalMoreFilesLabel(userMessage: string, count: number): string {
  const safeCount = Math.max(0, count)
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (
    languageHint === 'traditional_chinese' ||
    languageHint === 'chinese' ||
    languageHint === 'simplified_chinese'
  ) {
    return `+${safeCount} \u66f4\u591a\u6a94\u6848`
  }
  return `+${safeCount} more file${safeCount === 1 ? '' : 's'}`
}

export function buildGoalApprovalScopeLabel(userMessage: string, scope: string | null | undefined): string {
  const normalized = (scope ?? '').trim().toLowerCase()
  if (!normalized) {
    const languageHint = detectGoalProposalLanguageHint(userMessage)
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u64cd\u4f5c\u8005'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u64cd\u4f5c\u8005'
    }
    return 'operator'
  }

  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    const mapping: Record<string, string> = {
      operator: '\u64cd\u4f5c\u8005',
      workspace: '\u5de5\u4f5c\u5340',
      session: '\u5c0d\u8a71',
      project: '\u5c08\u6848',
      repository: '\u7a0b\u5f0f\u78bc\u5eab',
      repo: '\u7a0b\u5f0f\u78bc\u5eab',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }
  if (languageHint === 'simplified_chinese') {
    const mapping: Record<string, string> = {
      operator: '\u64cd\u4f5c\u8005',
      workspace: '\u5de5\u4f5c\u533a',
      session: '\u5bf9\u8bdd',
      project: '\u9879\u76ee',
      repository: '\u4ee3\u7801\u5e93',
      repo: '\u4ee3\u7801\u5e93',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }
  return normalized.replaceAll('_', ' ')
}

export function buildGoalRecommendedActionLabel(
  userMessage: string,
  action: string | null | undefined
): string | null {
  const normalized = (action ?? '').trim().toLowerCase()
  if (!normalized) {
    return null
  }

  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    const mapping: Record<string, string> = {
      resolve_approval: '\u8655\u7406\u6838\u51c6',
      inspect_runtime_budget: '\u6aa2\u67e5 runtime \u984d\u5ea6',
      refresh_worker_generation: '\u91cd\u65b0\u6574\u7406 worker generation',
      resume_goal: '\u6062\u5fa9 goal',
      monitor: '\u7e7c\u7e8c\u89c0\u5bdf',
      capture_checkpoint: '\u8a18\u9304 checkpoint',
      inspect_collector_shards: '\u6aa2\u67e5 collector shards',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }
  if (languageHint === 'simplified_chinese') {
    const mapping: Record<string, string> = {
      resolve_approval: '\u5904\u7406\u6279\u51c6',
      inspect_runtime_budget: '\u68c0\u67e5 runtime \u914d\u989d',
      refresh_worker_generation: '\u5237\u65b0 worker generation',
      resume_goal: '\u6062\u590d goal',
      monitor: '\u7ee7\u7eed\u89c2\u5bdf',
      capture_checkpoint: '\u8bb0\u5f55 checkpoint',
      inspect_collector_shards: '\u68c0\u67e5 collector shards',
    }
    return mapping[normalized] ?? normalized.replaceAll('_', ' ')
  }
  return normalized.replaceAll('_', ' ')
}

export function buildGoalBlockerSummary(
  userMessage: string,
  summary: string | null | undefined,
  latestError?: string | null,
  options?: {
    approvalCount?: number
    recommendedAction?: string | null
  }
): string {
  const summaryText = (summary ?? '').trim()
  if (summaryText && isGoalCopyLanguageAligned(userMessage, summaryText)) {
    return summaryText
  }
  const latestErrorText = (latestError ?? '').trim()
  if (latestErrorText && isGoalCopyLanguageAligned(userMessage, latestErrorText)) {
    return latestErrorText
  }
  const approvalCount = Math.max(0, options?.approvalCount ?? 0)
  const recommendedAction = (options?.recommendedAction ?? '').trim().toLowerCase()
  const userHint = detectGoalProposalLanguageHint(userMessage)
  if (
    approvalCount > 0 ||
    recommendedAction === 'resolve_approval' ||
    looksLikeApprovalBlocker(summaryText) ||
    looksLikeApprovalBlocker(latestErrorText)
  ) {
    return buildGoalApprovalWaitSummary(userMessage, approvalCount)
  }
  if (isChineseLanguageHint(userHint) && recommendedAction === 'resolve_approval') {
    return buildGoalApprovalWaitSummary(userMessage, approvalCount)
  }
  return buildGoalCardChromeCopy(userMessage).goalNeedsAttentionBody
}

export function buildGoalUiErrorMessage(
  userMessage: string,
  rawMessage: string | null | undefined,
  fallbackMessage: string
): string {
  const trimmed = (rawMessage ?? '').trim()
  if (trimmed && isGoalCopyLanguageAligned(userMessage, trimmed)) {
    return trimmed
  }
  return fallbackMessage
}

function goalFollowUpBaseSummary(
  userMessage: string,
  summary: string | null | undefined,
  traditionalDefault: string,
  simplifiedDefault: string,
  englishDefault: string
): string {
  const trimmed = (summary ?? '').trim()
  if (trimmed && isGoalCopyLanguageAligned(userMessage, trimmed)) {
    return trimmed
  }

  const languageHint = detectGoalProposalLanguageHint(userMessage)
  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return traditionalDefault
  }
  if (languageHint === 'simplified_chinese') {
    return simplifiedDefault
  }
  return englishDefault
}

export function buildGoalCommandHelpMessage(userMessage: string): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)

  if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
    return (
      '\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u60f3\u5b8c\u6210\u7684\u4efb\u52d9\uff0c\u6216\u7528 `/goal <request>` \u660e\u78ba\u6e96\u5099\u4e00\u500b goal \u8349\u7a3f\u3002\n' +
      '\u82e5\u4f60\u8981\u660e\u78ba\u6307\u5b9a workflow \u7b56\u7565\uff0c\u53ef\u4ee5\u518d\u7528 `/workflow <request>` \u4f5c\u70ba\u9032\u968e\u8986\u5beb\u3002\n' +
      'Goal \u555f\u52d5\u5f8c\uff0c\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u3002'
    )
  }

  if (languageHint === 'simplified_chinese') {
    return (
      '\u4f60\u53ef\u4ee5\u76f4\u63a5\u63cf\u8ff0\u60f3\u5b8c\u6210\u7684\u4efb\u52a1\uff0c\u6216\u7528 `/goal <request>` \u660e\u786e\u51c6\u5907\u4e00\u4e2a goal \u8349\u7a3f\u3002\n' +
      '\u82e5\u4f60\u8981\u660e\u786e\u6307\u5b9a workflow \u7b56\u7565\uff0c\u53ef\u4ee5\u518d\u7528 `/workflow <request>` \u4f5c\u4e3a\u8fdb\u9636\u8986\u5199\u3002\n' +
      'Goal \u542f\u52a8\u540e\uff0c\u53ef\u4ee5\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u3002'
    )
  }

  return [
    'Describe the task normally, or use `/goal <request>` to prepare a goal draft.',
    'Use `/workflow <request>` only when you want to explicitly override the goal with a workflow strategy.',
    'Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` after a goal starts.',
  ].join('\n')
}

export function buildGoalFollowUpMessage(
  userMessage: string,
  kind: GoalFollowUpMessageKind,
  options?: {
    summary?: string | null
    approvalCount?: number
    toolNames?: string[]
    operatorControlHint?: string | null
  }
): string {
  const languageHint = detectGoalProposalLanguageHint(userMessage)
  const approvalCount = Math.max(0, options?.approvalCount ?? 0)
  const toolNames = (options?.toolNames ?? []).map((item) => item.trim()).filter(Boolean)
  const operatorControlHint =
    options?.operatorControlHint?.trim() &&
    isGoalCopyLanguageAligned(userMessage, options.operatorControlHint)
      ? options.operatorControlHint.trim()
      : ''

  if (kind === 'goal_help') {
    return buildGoalCommandHelpMessage(userMessage)
  }

  if (kind === 'active_goal_exists') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return (
        '\u9019\u500b\u5c0d\u8a71\u5df2\u7d93\u6709\u4e00\u500b\u9032\u884c\u4e2d\u7684 goal\u3002' +
        ' \u8acb\u5148\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u4f86\u7ba1\u7406\u5b83\u3002'
      )
    }
    if (languageHint === 'simplified_chinese') {
      return (
        '\u8fd9\u4e2a\u5bf9\u8bdd\u5df2\u7ecf\u6709\u4e00\u4e2a\u8fdb\u884c\u4e2d\u7684 goal\u3002' +
        ' \u8bf7\u5148\u7528 `/goal status`\u3001`/goal pause`\u3001`/goal resume` \u6216 `/goal stop` \u6765\u7ba1\u7406\u5b83\u3002'
      )
    }
    return 'This chat already has an active goal. Use `/goal status`, `/goal pause`, `/goal resume`, or `/goal stop` before starting a new one.'
  }

  if (kind === 'manual_resolution_required') {
    const base = goalFollowUpBaseSummary(
      userMessage,
      options?.summary,
      '\u9019\u500b goal \u5728\u7e7c\u7e8c\u524d\u9700\u8981\u4f60\u5148\u8655\u7406\u5f85\u6838\u51c6\u9805\u76ee\u3002',
      '\u8fd9\u4e2a goal \u5728\u7ee7\u7eed\u524d\u9700\u8981\u4f60\u5148\u5904\u7406\u5f85\u6279\u51c6\u9879\u76ee\u3002',
      'The active goal needs approval handling before it can continue.'
    )
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      const toolHint = toolNames.length > 0 ? ` \u5f85\u6838\u51c6\u5de5\u5177\uff1a${toolNames.join(', ')}\u3002` : ''
      const action =
        approvalCount > 0
          ? ` \u8acb\u5148\u5f9e goal drawer \u6216 Goal Console \u8655\u7406 ${approvalCount} \u500b\u5f85\u6838\u51c6\u9805\u76ee\u3002`
          : ' \u8acb\u5148\u6253\u958b Goal Console \u6aa2\u67e5\u76ee\u524d\u7684\u963b\u585e\u6838\u51c6\u72c0\u614b\u3002'
      return `${base}${toolHint}${action}`.trim()
    }
    if (languageHint === 'simplified_chinese') {
      const toolHint = toolNames.length > 0 ? ` \u5f85\u6279\u51c6\u5de5\u5177\uff1a${toolNames.join(', ')}\u3002` : ''
      const action =
        approvalCount > 0
          ? ` \u8bf7\u5148\u4ece goal drawer \u6216 Goal Console \u5904\u7406 ${approvalCount} \u4e2a\u5f85\u6279\u51c6\u9879\u76ee\u3002`
          : ' \u8bf7\u5148\u6253\u5f00 Goal Console \u68c0\u67e5\u5f53\u524d\u7684\u963b\u585e\u6279\u51c6\u72b6\u6001\u3002'
      return `${base}${toolHint}${action}`.trim()
    }
    const toolHint = toolNames.length > 0 ? ` Pending approval for ${toolNames.join(', ')}.` : ''
    const action =
      approvalCount > 0
        ? ` Review the pending approval${approvalCount > 1 ? 's' : ''} from the goal drawer or Goal Console before continuing.`
        : ' Open the Goal Console to inspect the blocking approval state before continuing.'
    return `${base}${toolHint}${action}`.trim()
  }

  if (kind === 'queued_after_resolution') {
    const base = goalFollowUpBaseSummary(
      userMessage,
      options?.summary,
      '\u9019\u500b goal \u76ee\u524d\u9084\u5728\u7b49\u5f85\u6838\u51c6\uff0c\u4f46\u6211\u5df2\u5148\u628a\u4f60\u7684\u65b0\u65b9\u5411\u8a18\u5230\u76ee\u524d\u7684 attempt \u88e1\u3002',
      '\u8fd9\u4e2a goal \u76ee\u524d\u8fd8\u5728\u7b49\u5f85\u6279\u51c6\uff0c\u4f46\u6211\u5df2\u7ecf\u5148\u628a\u4f60\u7684\u65b0\u65b9\u5411\u8bb0\u5230\u5f53\u524d\u7684 attempt \u91cc\u3002',
      'The active goal is still waiting on approval, but I queued your new direction on the current attempt.'
    )
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      const action =
        approvalCount > 0
          ? ` \u4f60\u5148\u8655\u7406\u5b8c ${approvalCount} \u500b\u5f85\u6838\u51c6\u9805\u76ee\u5f8c\uff0c\u5b83\u5c31\u80fd\u6cbf\u8457\u9019\u500b\u65b9\u5411\u7e7c\u7e8c\uff0c\u4e0d\u9700\u8981\u4f60\u518d\u91cd\u8aaa\u4e00\u6b21\u3002`
          : ' \u7b49\u76ee\u524d\u7684\u6838\u51c6\u72c0\u614b\u89e3\u9664\u5f8c\uff0c\u5b83\u5c31\u80fd\u6cbf\u8457\u9019\u500b\u65b9\u5411\u7e7c\u7e8c\uff0c\u4e0d\u9700\u8981\u4f60\u518d\u91cd\u8aaa\u4e00\u6b21\u3002'
      return `${base}${action}`.trim()
    }
    if (languageHint === 'simplified_chinese') {
      const action =
        approvalCount > 0
          ? ` \u4f60\u5148\u5904\u7406\u5b8c ${approvalCount} \u4e2a\u5f85\u6279\u51c6\u9879\u76ee\u540e\uff0c\u5b83\u5c31\u80fd\u6cbf\u7740\u8fd9\u4e2a\u65b9\u5411\u7ee7\u7eed\uff0c\u4e0d\u9700\u8981\u4f60\u518d\u91cd\u8bf4\u4e00\u6b21\u3002`
          : ' \u7b49\u5f53\u524d\u7684\u6279\u51c6\u72b6\u6001\u89e3\u9664\u540e\uff0c\u5b83\u5c31\u80fd\u6cbf\u7740\u8fd9\u4e2a\u65b9\u5411\u7ee7\u7eed\uff0c\u4e0d\u9700\u8981\u4f60\u518d\u91cd\u8bf4\u4e00\u6b21\u3002'
      return `${base}${action}`.trim()
    }
    const action =
      approvalCount > 0
        ? ` Resolve the pending approval${approvalCount > 1 ? 's' : ''} from the goal drawer or Goal Console, and it can continue without you restating it.`
        : ' Once the approval state clears, it can continue without you restating it.'
    return `${base}${action}`.trim()
  }


  if (kind === 'blocked') {
    const base = goalFollowUpBaseSummary(
      userMessage,
      options?.summary,
      '\u9019\u500b goal \u76ee\u524d\u8655\u65bc\u53d7\u963b\u72c0\u614b\u3002',
      '\u8fd9\u4e2a goal \u76ee\u524d\u5904\u4e8e\u53d7\u963b\u72b6\u6001\u3002',
      'The active goal is currently blocked.'
    )
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      const extra = operatorControlHint ? ` ${operatorControlHint}` : ''
      return `${base}${extra} \u8acb\u5148\u5230 Goal Console \u8abf\u6574 goal\uff0c\u518d\u7e7c\u7e8c\u9001\u51fa\u5f8c\u7e8c\u57f7\u884c\u6307\u793a\u3002`.trim()
    }
    if (languageHint === 'simplified_chinese') {
      const extra = operatorControlHint ? ` ${operatorControlHint}` : ''
      return `${base}${extra} \u8bf7\u5148\u5230 Goal Console \u8c03\u6574 goal\uff0c\u518d\u7ee7\u7eed\u53d1\u9001\u540e\u7eed\u6267\u884c\u6307\u4ee4\u3002`.trim()
    }
    const extra = operatorControlHint ? ` ${operatorControlHint}` : ''
    return `${base}${extra} Adjust the goal from the Goal Console before sending more execution guidance.`.trim()
  }

  if (kind === 'no_live_attempt') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u76ee\u524d\u9019\u500b active goal \u9084\u6c92\u6709\u53ef\u4ee5\u63a5\u6536\u5f8c\u7e8c\u6307\u793a\u7684\u6d3b\u52d5 attempt\u3002 \u8acb\u5148\u6253\u958b Goal Console \u6aa2\u67e5\u73fe\u5728\u7684\u6062\u5fa9\u72c0\u614b\u3002'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u76ee\u524d\u8fd9\u4e2a active goal \u8fd8\u6ca1\u6709\u53ef\u4ee5\u63a5\u6536\u540e\u7eed\u6307\u4ee4\u7684\u6d3b\u52a8 attempt\u3002 \u8bf7\u5148\u6253\u5f00 Goal Console \u68c0\u67e5\u73b0\u5728\u7684\u6062\u590d\u72b6\u6001\u3002'
    }
    return 'The active goal still does not have a live attempt ready to receive follow-up guidance. Use the Goal Console to inspect the current recovery state.'
  }

  if (kind === 'restarted_forwarded') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u6211\u5df2\u7528\u76ee\u524d\u53ef\u6062\u5fa9\u7684\u4e0a\u4e0b\u6587\u91cd\u65b0\u958b\u4e00\u500b goal attempt\uff0c\u4e26\u628a\u4f60\u7684\u6307\u793a\u4e00\u8d77\u5e36\u9032\u53bb\u7e7c\u7e8c\u57f7\u884c\u3002'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u6211\u5df2\u7528\u76ee\u524d\u53ef\u6062\u590d\u7684\u4e0a\u4e0b\u6587\u91cd\u65b0\u5f00\u4e00\u4e2a goal attempt\uff0c\u5e76\u628a\u4f60\u7684\u6307\u4ee4\u4e00\u8d77\u5e26\u8fdb\u53bb\u7ee7\u7eed\u6267\u884c\u3002'
    }
    return 'Restarted the goal from the latest available recovery context and forwarded your guidance into the new attempt.'
  }


  if (kind === 'refreshed_forwarded') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u6211\u5df2\u91cd\u65b0\u6574\u7406 active worker generation\uff0c\u4e26\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u5230\u66f4\u65b0\u5f8c\u7684 goal attempt\u3002'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u6211\u5df2\u91cd\u65b0\u6574\u7406 active worker generation\uff0c\u5e76\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u5230\u66f4\u65b0\u540e\u7684 goal attempt\u3002'
    }
    return 'Refreshed the active worker generation and forwarded your guidance to the updated goal attempt.'
  }

  if (kind === 'resumed_forwarded') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u6211\u5df2\u6062\u5fa9\u9019\u500b active goal\uff0c\u4e26\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u5230\u76ee\u524d\u7684 attempt\u3002'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u6211\u5df2\u6062\u590d\u8fd9\u4e2a active goal\uff0c\u5e76\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u5230\u5f53\u524d\u7684 attempt\u3002'
    }
    return 'Resumed the active goal and forwarded your guidance to the current attempt.'
  }

  if (kind === 'forwarded') {
    if (languageHint === 'traditional_chinese' || languageHint === 'chinese') {
      return '\u6211\u5df2\u628a\u4f60\u7684\u6307\u793a\u8f49\u9001\u7d66\u76ee\u524d\u7684 active goal\u3002\u5b83\u6703\u4f9d\u7167\u9019\u500b\u66f4\u65b0\u7684\u65b9\u5411\u7e7c\u7e8c\u57f7\u884c\u3002'
    }
    if (languageHint === 'simplified_chinese') {
      return '\u6211\u5df2\u628a\u4f60\u7684\u6307\u4ee4\u8f6c\u53d1\u7ed9\u5f53\u524d\u7684 active goal\u3002\u5b83\u4f1a\u6309\u7167\u8fd9\u4e2a\u66f4\u65b0\u540e\u7684\u65b9\u5411\u7ee7\u7eed\u6267\u884c\u3002'
    }
    return 'Forwarded your guidance to the active goal. It will continue working with this updated direction.'
  }

  throw new Error(`Unsupported goal follow-up message kind: ${kind}`)
}
