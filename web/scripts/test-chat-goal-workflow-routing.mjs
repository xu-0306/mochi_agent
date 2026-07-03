import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/chat-goal-routing.ts')
).href

const {
  parseGoalCommand,
  resolveChatGoalWorkflowRouting,
} = await import(moduleUrl)

const goalProposal = parseGoalCommand('/goal investigate failing routing tests')
assert.deepEqual(goalProposal, {
  action: 'proposal',
  content: 'investigate failing routing tests',
  raw: '/goal investigate failing routing tests',
})

const goalHelp = parseGoalCommand('/goal')
assert.deepEqual(goalHelp, {
  action: 'help',
  content: '',
  raw: '/goal',
})

for (const action of ['status', 'pause', 'resume', 'stop']) {
  assert.deepEqual(parseGoalCommand(`/goal ${action}`), {
    action,
    content: '',
    raw: `/goal ${action}`,
  })
}

const workflowProposal = resolveChatGoalWorkflowRouting({
  text: '/workflow draft a multi-agent plan',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: false,
})
assert.equal(workflowProposal.modeCommand?.mode, 'workflow')
assert.equal(workflowProposal.requestText, 'draft a multi-agent plan')
assert.equal(workflowProposal.workflowModeRequested, true)
assert.equal(workflowProposal.route.kind, 'workflow_proposal')
assert.equal(workflowProposal.shouldHandleGoalWorkflowRouting, true)

for (const { label, text, hasActiveGoal } of [
  {
    label: 'chinese timed research request',
    text: '\u8acb\u7814\u7a76\u9019\u500b\u4e3b\u984c 20\u5206\u9418\uff0c\u6574\u7406\u91cd\u9ede\u7d66\u6211\u3002',
    hasActiveGoal: false,
  },
  {
    label: 'english timed research request',
    text: 'Research this for 20 minutes',
    hasActiveGoal: false,
  },
  {
    label: 'english background-work request',
    text: 'Keep working on this in the background for 30 minutes and come back with progress',
    hasActiveGoal: false,
  },
  {
    label: 'spanish timed research request',
    text: 'Investiga este tema durante 20 minutos y resume los hallazgos.',
    hasActiveGoal: false,
  },
  {
    label: 'hindi timed research request',
    text: 'Is vishay par 20 minute research karke summary do.',
    hasActiveGoal: false,
  },
  {
    label: 'active goal progress question',
    text: 'What is the goal doing right now?',
    hasActiveGoal: true,
  },
  {
    label: 'active goal blocked-state explanation',
    text: 'What does this blocked state mean?',
    hasActiveGoal: true,
  },
  {
    label: 'active goal steering instruction',
    text: 'Prioritize the failing login test first and keep the patch minimal',
    hasActiveGoal: true,
  },
  {
    label: 'active goal ambiguous follow-up',
    text: 'Can you share progress?',
    hasActiveGoal: true,
  },
]) {
  const decision = resolveChatGoalWorkflowRouting({
    text,
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal,
  })
  assert.equal(decision.route.kind, 'direct_chat', label)
  assert.equal(decision.shouldHandleGoalWorkflowRouting, false, label)

  const alternateStateDecision = resolveChatGoalWorkflowRouting({
    text,
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: !hasActiveGoal,
  })
  assert.equal(alternateStateDecision.route.kind, 'direct_chat', `${label} alternate state`)
  assert.equal(
    alternateStateDecision.shouldHandleGoalWorkflowRouting,
    false,
    `${label} alternate state`
  )
}

const explicitGoalProposal = resolveChatGoalWorkflowRouting({
  text: '/goal research this for 20 minutes',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: false,
})
assert.equal(explicitGoalProposal.route.kind, 'goal_proposal')
assert.equal(explicitGoalProposal.route.content, 'research this for 20 minutes')
assert.equal(explicitGoalProposal.shouldHandleGoalWorkflowRouting, true)

const pendingConfirmation = resolveChatGoalWorkflowRouting({
  text: 'go ahead',
  attachmentCount: 0,
  hasPendingProposal: true,
  hasActiveGoal: false,
})
assert.equal(pendingConfirmation.route.kind, 'goal_pending_follow_up')
assert.equal(pendingConfirmation.shouldHandleGoalWorkflowRouting, true)

const pendingRevision = resolveChatGoalWorkflowRouting({
  text: 'tighten the scope to the chat page only',
  attachmentCount: 0,
  hasPendingProposal: true,
  hasActiveGoal: false,
})
assert.equal(pendingRevision.route.kind, 'goal_pending_follow_up')
assert.equal(pendingRevision.shouldHandleGoalWorkflowRouting, true)

const attachmentBackedRevision = resolveChatGoalWorkflowRouting({
  text: 'yes',
  attachmentCount: 1,
  hasPendingProposal: true,
  hasActiveGoal: false,
})
assert.equal(attachmentBackedRevision.route.kind, 'goal_revision')

const commandWinsOverPendingProposal = resolveChatGoalWorkflowRouting({
  text: '/goal status',
  attachmentCount: 0,
  hasPendingProposal: true,
  hasActiveGoal: true,
})
assert.equal(commandWinsOverPendingProposal.route.kind, 'goal_lifecycle')
assert.equal(commandWinsOverPendingProposal.route.action, 'status')

const chatEscapeBypassesActiveGoal = resolveChatGoalWorkflowRouting({
  text: '/chat what models are available?',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: true,
})
assert.equal(chatEscapeBypassesActiveGoal.route.kind, 'direct_chat')
assert.equal(chatEscapeBypassesActiveGoal.shouldHandleGoalWorkflowRouting, false)

console.log('ok')
