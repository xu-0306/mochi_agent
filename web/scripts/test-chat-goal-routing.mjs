import assert from 'node:assert/strict'
import path from 'node:path'
import { pathToFileURL } from 'node:url'

const moduleUrl = pathToFileURL(
  path.join(process.cwd(), 'src/lib/chat-goal-routing.ts')
).href

const {
  parseChatModeCommand,
  parseGoalCommand,
  resolveChatGoalWorkflowRouting,
} = await import(moduleUrl)

assert.deepEqual(parseChatModeCommand('/chat summarize this'), {
  mode: 'chat',
  content: 'summarize this',
})
assert.deepEqual(parseChatModeCommand('/workflow '), {
  mode: 'workflow',
  content: '',
})
assert.equal(parseChatModeCommand('plain text'), null)

assert.deepEqual(parseGoalCommand('/goal'), {
  action: 'help',
  content: '',
  raw: '/goal',
})
assert.deepEqual(parseGoalCommand('/goal status'), {
  action: 'status',
  content: '',
  raw: '/goal status',
})
assert.deepEqual(parseGoalCommand('/goal write a migration plan'), {
  action: 'proposal',
  content: 'write a migration plan',
  raw: '/goal write a migration plan',
})

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'Summarize the latest changes.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }),
  {
    modeCommand: null,
    requestText: 'Summarize the latest changes.',
    route: { kind: 'direct_chat' },
    workflowModeRequested: false,
    requiresSessionMaterialization: false,
    shouldHandleGoalWorkflowRouting: false,
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '/chat Summarize the latest changes.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: true,
  }),
  {
    modeCommand: { mode: 'chat', content: 'Summarize the latest changes.' },
    requestText: 'Summarize the latest changes.',
    route: { kind: 'direct_chat' },
    workflowModeRequested: false,
    requiresSessionMaterialization: true,
    shouldHandleGoalWorkflowRouting: false,
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '/goal',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }).route,
  { kind: 'goal_help', raw: '/goal' }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '/goal status',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: true,
  }).route,
  { kind: 'goal_lifecycle', action: 'status', raw: '/goal status' }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '/goal Prepare a long-running cleanup plan.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_proposal',
    content: 'Prepare a long-running cleanup plan.',
    raw: '/goal Prepare a long-running cleanup plan.',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '/workflow Research the issue and come back with sources.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'workflow_proposal',
    requestText: 'Research the issue and come back with sources.',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'Keep working on this in the background for the next 30 minutes.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'natural_language_goal_proposal',
    requestText: 'Keep working on this in the background for the next 30 minutes.',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'start',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_pending_follow_up',
    requestText: 'start',
    raw: 'start',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '開始',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_pending_follow_up',
    requestText: '開始',
    raw: '開始',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '시작해줘',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_pending_follow_up',
    requestText: '시작해줘',
    raw: '시작해줘',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'Revise the proposal to include rollback steps.',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_pending_follow_up',
    requestText: 'Revise the proposal to include rollback steps.',
    raw: 'Revise the proposal to include rollback steps.',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'Please also capture a checkpoint after the first pass.',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: true,
  }).route,
  {
    kind: 'goal_follow_up',
    requestText: 'Please also capture a checkpoint after the first pass.',
  }
)

console.log('ok')
