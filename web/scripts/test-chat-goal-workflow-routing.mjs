import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/chat-goal-routing.ts')
).href

const {
  isNaturalLanguageGoalRequest,
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

for (const request of [
  'Keep working on this in the background for 30 minutes and come back with progress',
  'Please continue working on this while I am away',
  'Run for 2 hours with checkpoints and retry until it passes',
]) {
  assert.equal(
    isNaturalLanguageGoalRequest(request),
    true,
    `expected ${request} to be treated as natural-language goal intent`
  )
}

assert.equal(
  isNaturalLanguageGoalRequest('What models are available right now?'),
  false
)

const naturalLanguageProposal = resolveChatGoalWorkflowRouting({
  text: 'Keep working on this in the background for 30 minutes and come back with progress',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: false,
})
assert.equal(naturalLanguageProposal.route.kind, 'natural_language_goal_proposal')
assert.equal(naturalLanguageProposal.shouldHandleGoalWorkflowRouting, true)
assert.equal(naturalLanguageProposal.requestText, 'Keep working on this in the background for 30 minutes and come back with progress')

const activeGoalFollowUp = resolveChatGoalWorkflowRouting({
  text: 'Prioritize the failing login test first and keep the patch minimal',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: true,
})
assert.equal(activeGoalFollowUp.route.kind, 'goal_follow_up')
assert.equal(activeGoalFollowUp.shouldHandleGoalWorkflowRouting, true)

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
