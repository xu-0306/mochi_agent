import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath, pathToFileURL } from 'node:url'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const WEB_DIR = path.resolve(SCRIPT_DIR, '..')
const moduleUrl = pathToFileURL(path.join(WEB_DIR, 'src/lib/goal-turn-controller.ts')).href

const {
  buildActiveGoalTurnDecisionMetadata,
  classifyActiveGoalTurnDecision,
} = await import(moduleUrl)

const baseDecision = {
  lane: 'active_goal_turn',
  kind: 'answer_question',
  confidence: 0.72,
  selection_source: 'bounded_fallback',
  selection_reason: 'Operator asked a question about the active goal.',
  requires_confirmation: false,
  goal_status: 'running',
  linked_run_status: 'running',
  recommended_action: 'answer_in_chat',
}

assert.deepEqual(buildActiveGoalTurnDecisionMetadata(baseDecision), {
  active_goal_turn_decision: {
    lane: 'active_goal_turn',
    kind: 'answer_question',
    confidence: 0.72,
    selection_source: 'bounded_fallback',
    selection_reason: 'Operator asked a question about the active goal.',
    requires_confirmation: false,
    goal_status: 'running',
    linked_run_status: 'running',
    recommended_action: 'answer_in_chat',
  },
})

for (const kind of ['answer_question', 'explain_goal_state', 'exit_to_chat']) {
  assert.deepEqual(
    classifyActiveGoalTurnDecision({
      ...baseDecision,
      kind,
      requires_confirmation: false,
    }),
    { kind: 'direct_chat' },
    kind
  )
}

assert.deepEqual(
  classifyActiveGoalTurnDecision({
    ...baseDecision,
    kind: 'clarify',
    requires_confirmation: true,
  }),
  { kind: 'direct_chat' }
)

assert.deepEqual(
  classifyActiveGoalTurnDecision({
    ...baseDecision,
    kind: 'steer',
  }),
  { kind: 'steer_goal' }
)

assert.deepEqual(
  classifyActiveGoalTurnDecision({
    ...baseDecision,
    kind: 'replan',
  }),
  { kind: 'replan_goal' }
)

assert.deepEqual(
  classifyActiveGoalTurnDecision({
    ...baseDecision,
    kind: 'lifecycle',
  }),
  { kind: 'lifecycle_goal' }
)

assert.deepEqual(
  classifyActiveGoalTurnDecision({
    ...baseDecision,
    kind: 'clarify',
    requires_confirmation: false,
  }),
  { kind: 'unhandled' }
)

const pageSource = fs.readFileSync(path.join(WEB_DIR, 'src/app/page.tsx'), 'utf8')
const directChatBranchMatch = pageSource.match(
  /if\s*\(\s*decisionAction\.kind === 'direct_chat'\s*\)\s*\{([\s\S]*?)\n\s*\}/
)
assert.ok(
  directChatBranchMatch,
  'expected to find the non-mutating direct_chat decision branch'
)
const directChatBranch = directChatBranchMatch[1]

assert.match(
  pageSource,
  /if\s*\(\s*route\.kind === 'direct_chat'[\s\S]*?hasActiveNonTerminalGoal[\s\S]*?pending_proposal === null[\s\S]*?await activeGoalDirectTurnDecisionRef\.current\(/,
  'page.tsx should classify eligible active-goal natural-language direct chat turns before submitting ordinary chat'
)

assert.match(
  pageSource,
  /let decision: api\.ActiveGoalTurnDecision[\s\S]*?decision = await api\.fetchGoalTurnDecision\(activeGoalId,\s*\{\s*message:\s*requestText\s*\}\)/,
  'active-goal direct turns should call fetchGoalTurnDecision with the user message'
)

assert.match(
  pageSource,
  /buildActiveGoalTurnDecisionMetadata\(decision\)/,
  'page.tsx should build persisted active-goal turn metadata via the extracted helper'
)

assert.match(
  pageSource,
  /classifyActiveGoalTurnDecision\(decision\)/,
  'page.tsx should classify backend active-goal decisions via the extracted helper'
)

assert.match(
  pageSource,
  /let decision: api\.ActiveGoalTurnDecision[\s\S]*?catch \{\s*await submitDirectChatTurn\(/,
  'decision fetch failures should fall back to ordinary chat instead of aborting send'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'direct_chat'[\s\S]*?submitDirectChatTurn\(/,
  'question-only and explanatory active-goal turns should remain conversational'
)

assert.ok(
  !/api\.(?:appendAgentRunGuidance|resumeGoal|refreshGoal|startGoal)\(/.test(directChatBranch),
  'direct_chat active-goal turns must not mutate goal runtime state'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'steer_goal'[\s\S]*?api\.(?:appendAgentRunGuidance|resumeGoal|refreshGoal)\(/,
  'steering turns should use an existing goal continuation mutation path only after typed decision classification'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'steer_goal'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?await api\.refreshGoal\(activeGoalId,\s*\{\s*strategy:\s*continuationResumeStrategy\s*\}\)[\s\S]*?await api\.appendAgentRunGuidance\(refreshedRunId,\s*\{[\s\S]*?guidance:\s*requestText,/,
  'steering refresh_then_forward path must refresh first and then forward requestText to the refreshed run'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'steer_goal'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?if \(refreshedRunId\)[\s\S]*?appendAgentRunGuidance[\s\S]*?else \{[\s\S]*?await api\.resumeGoal\(activeGoalId,\s*\{[\s\S]*?guidanceMessage:\s*requestText,/,
  'steering refresh_then_forward path must fall back to guidance-preserving resume when no refreshed run id is available'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'replan_goal'[\s\S]*?api\.(?:resumeGoal|refreshGoal)\(/,
  'replanning turns should use an existing goal continuation mutation path only after typed decision classification'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'replan_goal'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?await api\.refreshGoal\(activeGoalId,\s*\{\s*strategy:\s*continuationResumeStrategy\s*\}\)[\s\S]*?await api\.appendAgentRunGuidance\(refreshedRunId,\s*\{[\s\S]*?guidance:\s*requestText,/,
  'replanning refresh_then_forward path must refresh first and then forward requestText to the refreshed run'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'replan_goal'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?if \(refreshedRunId\)[\s\S]*?appendAgentRunGuidance[\s\S]*?else \{[\s\S]*?await api\.resumeGoal\(activeGoalId,\s*\{[\s\S]*?guidanceMessage:\s*requestText,/,
  'replanning refresh_then_forward path must fall back to guidance-preserving resume when no refreshed run id is available'
)

assert.ok(
  !/linkedRunStatus:\s*decision\.linked_run_status\s*\?\?\s*continuation\.recommendedAction/.test(pageSource),
  'linkedRunStatus must not fall back to continuation.recommendedAction'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'steer_goal'[\s\S]*?catch \(error\)[\s\S]*?content: t\('chat\.requestFailed'\)/,
  'steer continuation failures should surface a visible chat.requestFailed error'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'steer_goal'\) \{\s*try \{[\s\S]*?const goalHealth = await api\.fetchGoalHealth\(activeGoalId\)/,
  'steer path should fetch goal health inside guarded error handling'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'replan_goal'[\s\S]*?catch \(error\)[\s\S]*?content: t\('chat\.requestFailed'\)/,
  'replan continuation failures should surface a visible chat.requestFailed error'
)

assert.match(
  pageSource,
  /decisionAction\.kind === 'replan_goal'\) \{\s*try \{[\s\S]*?const goalHealth = await api\.fetchGoalHealth\(activeGoalId\)/,
  'replan path should fetch goal health inside guarded error handling'
)

assert.match(
  pageSource,
  /if\s*\(\s*shouldHandleGoalWorkflowRouting && route\.kind !== 'direct_chat'\s*\)[\s\S]*?return[\s\S]*?const activeGoalId =/,
  'slash lifecycle and other explicit routed commands should be handled before active-goal turn classification'
)

console.log('ok')
