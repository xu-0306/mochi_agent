import assert from 'node:assert/strict'

const moduleUrl = new URL('../../web/src/lib/goal-execution-record-target.ts', import.meta.url)
let targetModule = null

try {
  targetModule = await import(moduleUrl.href)
} catch {
  targetModule = null
}

assert.equal(
  typeof targetModule?.resolveGoalExecutionRecordTarget,
  'function',
  'resolveGoalExecutionRecordTarget must be exported'
)

const { resolveGoalExecutionRecordTarget } = targetModule

const attempt = (overrides = {}) => ({
  goalId: 'goal-1',
  attemptId: 'attempt-1',
  attemptIndex: 1,
  agentRunId: 'run-1',
  ...overrides,
})

const cases = [
  {
    name: 'missing goal',
    input: { selectedGoalId: '  ', snapshotGoalId: 'goal-1', currentAttemptId: null, attempts: [] },
    expected: { kind: 'unavailable', reason: 'missing_goal' },
  },
  {
    name: 'goal mismatch',
    input: { selectedGoalId: 'goal-1', snapshotGoalId: 'goal-2', currentAttemptId: null, attempts: [] },
    expected: { kind: 'unavailable', reason: 'goal_mismatch' },
  },
  {
    name: 'current attempt takes priority over a later attempt',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: 'attempt-current',
      attempts: [attempt({ attemptId: 'attempt-current', attemptIndex: 1, agentRunId: 'run-current' }), attempt({ attemptId: 'attempt-later', attemptIndex: 2, agentRunId: 'run-later' })],
    },
    expected: { kind: 'agent_run', runId: 'run-current', href: '/agent-runs/run-current' },
  },
  {
    name: 'missing current run falls back to the highest finite attempt index',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: 'attempt-current',
      attempts: [attempt({ attemptId: 'attempt-current', attemptIndex: 9, agentRunId: null }), attempt({ attemptId: 'attempt-two', attemptIndex: 2, agentRunId: 'run-two' }), attempt({ attemptId: 'attempt-five', attemptIndex: 5, agentRunId: 'run-five' }), attempt({ attemptId: 'attempt-infinite', attemptIndex: Infinity, agentRunId: 'run-infinite' })],
    },
    expected: { kind: 'agent_run', runId: 'run-five', href: '/agent-runs/run-five' },
  },
  {
    name: 'foreign attempt is excluded',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: 'foreign-current',
      attempts: [attempt({ goalId: 'goal-2', attemptId: 'foreign-current', attemptIndex: 99, agentRunId: 'foreign-run' }), attempt({ attemptId: 'local', agentRunId: 'local-run' })],
    },
    expected: { kind: 'agent_run', runId: 'local-run', href: '/agent-runs/local-run' },
  },
  {
    name: 'blank run is excluded',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: 'blank',
      attempts: [attempt({ attemptId: 'blank', attemptIndex: 2, agentRunId: '   ' }), attempt({ attemptId: 'valid', attemptIndex: 1, agentRunId: 'valid-run' })],
    },
    expected: { kind: 'agent_run', runId: 'valid-run', href: '/agent-runs/valid-run' },
  },
  {
    name: 'no eligible run',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: null,
      attempts: [attempt({ agentRunId: null }), attempt({ goalId: 'goal-2', agentRunId: 'foreign-run' })],
    },
    expected: { kind: 'unavailable', reason: 'missing_run' },
  },
  {
    name: 'run identifiers are URL encoded',
    input: { selectedGoalId: 'goal-1', snapshotGoalId: 'goal-1', currentAttemptId: 'encoded', attempts: [attempt({ attemptId: 'encoded', agentRunId: 'run/a b?' })] },
    expected: { kind: 'agent_run', runId: 'run/a b?', href: '/agent-runs/run%2Fa%20b%3F' },
  },
  {
    name: 'same attempt index selects the last eligible attempt',
    input: {
      selectedGoalId: 'goal-1',
      snapshotGoalId: 'goal-1',
      currentAttemptId: null,
      attempts: [attempt({ attemptId: 'first', attemptIndex: 3, agentRunId: 'first-run' }), attempt({ attemptId: 'last', attemptIndex: 3, agentRunId: 'last-run' })],
    },
    expected: { kind: 'agent_run', runId: 'last-run', href: '/agent-runs/last-run' },
  },
]

for (const { name, input, expected } of cases) {
  assert.deepStrictEqual(resolveGoalExecutionRecordTarget(input), expected, name)
}

const immutableInput = {
  selectedGoalId: 'goal-1',
  snapshotGoalId: 'goal-1',
  currentAttemptId: null,
  attempts: [attempt({ attemptId: 'immutable', agentRunId: 'immutable-run' })],
}
const beforeResolution = structuredClone(immutableInput)
Object.freeze(immutableInput.attempts)
Object.freeze(immutableInput)

assert.deepStrictEqual(
  resolveGoalExecutionRecordTarget(immutableInput),
  { kind: 'agent_run', runId: 'immutable-run', href: '/agent-runs/immutable-run' },
  'frozen input is supported'
)
assert.deepStrictEqual(immutableInput, beforeResolution, 'the input must not be modified')

const firstResolution = resolveGoalExecutionRecordTarget(immutableInput)
const repeatedResolution = resolveGoalExecutionRecordTarget(immutableInput)
assert.deepStrictEqual(firstResolution, repeatedResolution, 'repeated calls must be consistent')

console.log(`goal execution record target: ${cases.length + 3} assertions passed`)
