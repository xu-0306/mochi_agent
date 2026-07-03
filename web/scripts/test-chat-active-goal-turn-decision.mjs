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

assert.match(pageSource, /decisionAction\.kind === 'direct_chat'/)
assert.match(pageSource, /decisionAction\.kind === 'steer_goal'/)
assert.match(pageSource, /decisionAction\.kind === 'replan_goal'/)

console.log('ok')
