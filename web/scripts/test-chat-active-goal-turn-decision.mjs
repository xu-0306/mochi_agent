import assert from 'node:assert/strict'
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'

const SCRIPT_DIR = path.dirname(fileURLToPath(import.meta.url))
const WEB_DIR = path.resolve(SCRIPT_DIR, '..')
const pageSource = fs.readFileSync(path.join(WEB_DIR, 'src/app/page.tsx'), 'utf8')
const explanatoryBranchMatch = pageSource.match(
  /if\s*\(\s*decision\.kind === 'answer_question'[\s\S]*?\(decision\.kind === 'clarify' && decision\.requires_confirmation\)\s*\)\s*\{([\s\S]*?)\n\s*\}/
)
assert.ok(explanatoryBranchMatch, 'expected to find the non-mutating explanatory decision branch')
const explanatoryBranch = explanatoryBranchMatch[1]

assert.match(
  pageSource,
  /if\s*\(\s*route\.kind === 'direct_chat'[\s\S]*?hasActiveNonTerminalGoal[\s\S]*?pending_proposal === null[\s\S]*?await handleActiveGoalDirectTurnDecision\(/,
  'page.tsx should classify eligible active-goal natural-language direct chat turns before submitting ordinary chat'
)

assert.match(
  pageSource,
  /let decision: api\.ActiveGoalTurnDecision[\s\S]*?decision = await api\.fetchGoalTurnDecision\(activeGoalId,\s*\{\s*message:\s*requestText\s*\}\)/,
  'active-goal direct turns should call fetchGoalTurnDecision with the user message'
)

assert.match(
  pageSource,
  /decision\.kind === 'answer_question'[\s\S]*?submitDirectChatTurn\(/,
  'question-only active-goal turns should remain conversational'
)

assert.match(
  pageSource,
  /decision\.kind === 'explain_goal_state'[\s\S]*?submitDirectChatTurn\(/,
  'goal-state explanation turns should remain conversational'
)

assert.ok(
  !/api\.(?:appendAgentRunGuidance|resumeGoal|refreshGoal|startGoal)\(/.test(explanatoryBranch),
  'question/explanation active-goal turns must not mutate goal runtime state'
)

assert.match(
  pageSource,
  /decision\.kind === 'steer'[\s\S]*?api\.(?:appendAgentRunGuidance|resumeGoal|refreshGoal)\(/,
  'steering turns should use an existing goal continuation mutation path'
)

assert.match(
  pageSource,
  /decision\.kind === 'steer'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?await api\.refreshGoal\(activeGoalId,\s*\{\s*strategy:\s*continuationResumeStrategy\s*\}\)[\s\S]*?await api\.appendAgentRunGuidance\(refreshedRunId,\s*\{[\s\S]*?guidance:\s*requestText,/,
  'steering refresh_then_forward path must refresh first and then forward requestText to the refreshed run'
)

assert.match(
  pageSource,
  /decision\.kind === 'steer'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?if \(refreshedRunId\)[\s\S]*?appendAgentRunGuidance[\s\S]*?else \{[\s\S]*?await api\.resumeGoal\(activeGoalId,\s*\{[\s\S]*?guidanceMessage:\s*requestText,/,
  'steering refresh_then_forward path must fall back to guidance-preserving resume when no refreshed run id is available'
)

assert.match(
  pageSource,
  /decision\.kind === 'replan'[\s\S]*?api\.(?:resumeGoal|refreshGoal)\(/,
  'replanning turns should use an existing goal continuation mutation path'
)

assert.match(
  pageSource,
  /decision\.kind === 'replan'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?await api\.refreshGoal\(activeGoalId,\s*\{\s*strategy:\s*continuationResumeStrategy\s*\}\)[\s\S]*?await api\.appendAgentRunGuidance\(refreshedRunId,\s*\{[\s\S]*?guidance:\s*requestText,/,
  'replanning refresh_then_forward path must refresh first and then forward requestText to the refreshed run'
)

assert.match(
  pageSource,
  /decision\.kind === 'replan'[\s\S]*?continuation\.action === 'refresh_then_forward'[\s\S]*?if \(refreshedRunId\)[\s\S]*?appendAgentRunGuidance[\s\S]*?else \{[\s\S]*?await api\.resumeGoal\(activeGoalId,\s*\{[\s\S]*?guidanceMessage:\s*requestText,/,
  'replanning refresh_then_forward path must fall back to guidance-preserving resume when no refreshed run id is available'
)

assert.ok(
  !/linkedRunStatus:\s*decision\.linked_run_status\s*\?\?\s*continuation\.recommendedAction/.test(pageSource),
  'linkedRunStatus must not fall back to continuation.recommendedAction'
)

assert.match(
  pageSource,
  /let decision: api\.ActiveGoalTurnDecision[\s\S]*?catch \{\s*await submitDirectChatTurn\(/,
  'decision fetch failures should fall back to ordinary chat instead of aborting send'
)

assert.match(
  pageSource,
  /decision\.kind === 'steer'[\s\S]*?catch \(error\)[\s\S]*?content: t\('chat\.requestFailed'\)/,
  'steer continuation failures should surface a visible chat.requestFailed error'
)

assert.match(
  pageSource,
  /decision\.kind === 'steer'\) \{\s*try \{[\s\S]*?const goalHealth = await api\.fetchGoalHealth\(activeGoalId\)/,
  'steer path should fetch goal health inside guarded error handling'
)

assert.match(
  pageSource,
  /decision\.kind === 'replan'[\s\S]*?catch \(error\)[\s\S]*?content: t\('chat\.requestFailed'\)/,
  'replan continuation failures should surface a visible chat.requestFailed error'
)

assert.match(
  pageSource,
  /decision\.kind === 'replan'\) \{\s*try \{[\s\S]*?const goalHealth = await api\.fetchGoalHealth\(activeGoalId\)/,
  'replan path should fetch goal health inside guarded error handling'
)

assert.match(
  pageSource,
  /if\s*\(\s*shouldHandleGoalWorkflowRouting && route\.kind !== 'direct_chat'\s*\)[\s\S]*?return[\s\S]*?const activeGoalId =/,
  'slash lifecycle and other explicit routed commands should be handled before active-goal turn classification'
)

console.log('ok')
