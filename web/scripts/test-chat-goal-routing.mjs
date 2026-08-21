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

for (const { label, text, hasActiveGoal } of [
  {
    label: 'chinese timed research request',
    text: '\u8acb\u7814\u7a76\u9019\u500b\u4e3b\u984c 20\u5206\u9418\uff0c\u6574\u7406\u91cd\u9ede\u7d66\u6211\u3002',
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
  assert.deepEqual(
    resolveChatGoalWorkflowRouting({
      text,
      attachmentCount: 0,
      hasPendingProposal: false,
      hasActiveGoal,
    }),
    {
      modeCommand: null,
      requestText: text,
      route: { kind: 'direct_chat' },
      workflowModeRequested: false,
      requiresSessionMaterialization: false,
      shouldHandleGoalWorkflowRouting: false,
    },
    `expected ${label} to stay direct_chat`
  )

  assert.deepEqual(
    resolveChatGoalWorkflowRouting({
      text,
      attachmentCount: 0,
      hasPendingProposal: false,
      hasActiveGoal: !hasActiveGoal,
    }).route,
    { kind: 'direct_chat' },
    `expected ${label} to stay direct_chat regardless of active goal state`
  )
}

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'start',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'workflow_pending_follow_up',
    requestText: 'start',
    raw: 'start',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '\u540c\u610f',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'workflow_pending_follow_up',
    requestText: '\u540c\u610f',
    raw: '\u540c\u610f',
  }
)

for (const text of ['hi', 'hello', '\u4f60\u597d', '\u55e8']) {
  assert.deepEqual(
    resolveChatGoalWorkflowRouting({
      text,
      attachmentCount: 0,
      hasPendingProposal: true,
      hasActiveGoal: false,
    }).route,
    { kind: 'direct_chat' },
    `expected greeting ${text} to stay in direct_chat even with a workflow pending proposal`
  )
}

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: '\u8abf\u6574\u7bc4\u570d',
    attachmentCount: 0,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'workflow_pending_follow_up',
    requestText: '\u8abf\u6574\u7bc4\u570d',
    raw: '\u8abf\u6574\u7bc4\u570d',
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
    kind: 'workflow_pending_follow_up',
    requestText: 'Revise the proposal to include rollback steps.',
    raw: 'Revise the proposal to include rollback steps.',
  }
)

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'yes',
    attachmentCount: 1,
    hasPendingProposal: true,
    hasActiveGoal: false,
  }).route,
  {
    kind: 'goal_revision',
    requestText: 'yes',
  }
)

const slashNaturalGoal = resolveChatGoalWorkflowRouting({
  text: '/goal Research this for 20 minutes',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: false,
})
const naturalTimedGoal = resolveChatGoalWorkflowRouting({
  text: 'Research this for 20 minutes',
  attachmentCount: 0,
  hasPendingProposal: false,
  hasActiveGoal: false,
})
assert.deepEqual(slashNaturalGoal.route, {
  kind: 'goal_proposal',
  content: 'Research this for 20 minutes',
  raw: '/goal Research this for 20 minutes',
})
assert.deepEqual(naturalTimedGoal.route, {
  kind: 'goal_proposal',
  content: 'Research this for 20 minutes',
  raw: 'Research this for 20 minutes',
})
assert.equal(
  naturalTimedGoal.route.kind === 'goal_proposal' && naturalTimedGoal.route.content,
  slashNaturalGoal.route.kind === 'goal_proposal' && slashNaturalGoal.route.content,
  'slash and natural timed activation must normalize to the same objective'
)

for (const [text, content] of [
  ['Start a goal: Review the release checklist', 'Review the release checklist'],
  ['\u8acb\u5efa\u7acb\u4e00\u500b\u76ee\u6a19\uff1a\u6574\u7406\u767c\u5e03\u6aa2\u67e5\u6e05\u55ae', '\u6574\u7406\u767c\u5e03\u6aa2\u67e5\u6e05\u55ae'],
]) {
  assert.deepEqual(
    resolveChatGoalWorkflowRouting({
      text,
      attachmentCount: 0,
      hasPendingProposal: false,
      hasActiveGoal: false,
    }).route,
    { kind: 'goal_proposal', content, raw: text },
    `${text} must use the same session-bound goal proposal route`
  )
}

assert.deepEqual(
  resolveChatGoalWorkflowRouting({
    text: 'Research this today',
    attachmentCount: 0,
    hasPendingProposal: false,
    hasActiveGoal: false,
  }).route,
  { kind: 'direct_chat' },
  'ordinary non-time-bounded research remains chat'
)

console.log('ok')
