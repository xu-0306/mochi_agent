import assert from 'node:assert/strict'

const moduleUrl = new URL('../../web/src/lib/goal-drawer-model.ts', import.meta.url)
let goalDrawerModule = null

try {
  goalDrawerModule = await import(moduleUrl.href)
} catch {
  goalDrawerModule = null
}

assert.equal(
  typeof goalDrawerModule?.projectGoalDrawerState,
  'function',
  'projectGoalDrawerState must be exported'
)

const { projectGoalDrawerState } = goalDrawerModule

const actionFlagsByStatus = {
  proposed: {
    canPause: false,
    canResume: false,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
  running: {
    canPause: true,
    canResume: false,
    canStop: true,
    canEdit: true,
    canClear: true,
  },
  paused: {
    canPause: false,
    canResume: true,
    canStop: true,
    canEdit: true,
    canClear: true,
  },
  stopped: {
    canPause: false,
    canResume: true,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
  completed: {
    canPause: false,
    canResume: false,
    canStop: false,
    canEdit: false,
    canClear: true,
  },
  failed: {
    canPause: false,
    canResume: true,
    canStop: false,
    canEdit: true,
    canClear: true,
  },
}

const validCases = Object.entries(actionFlagsByStatus).flatMap(([status, actionFlags]) =>
  [false, true].map((hasLinkedAgentRun) => {
    const goalId = `goal-${status}-${hasLinkedAgentRun ? 'linked' : 'unlinked'}`
    const title = `${status} goal`

    return {
      name: `${status} goal ${hasLinkedAgentRun ? 'with' : 'without'} a linked AgentRun`,
      snapshot: { goalId, title, status, hasLinkedAgentRun },
      expected: {
        goalId,
        title,
        status,
        ...actionFlags,
        canOpenExecutionRecord: hasLinkedAgentRun,
      },
    }
  })
)

for (const { name, snapshot, expected } of validCases) {
  assert.deepStrictEqual(projectGoalDrawerState(snapshot), expected, name)
}

const invalidCases = [
  { name: 'null snapshot', snapshot: null },
  { name: 'undefined snapshot', snapshot: undefined },
  {
    name: 'missing goalId',
    snapshot: { title: 'missing identity', status: 'running', hasLinkedAgentRun: false },
  },
  {
    name: 'empty goalId',
    snapshot: { goalId: '', title: 'empty identity', status: 'running', hasLinkedAgentRun: false },
  },
  {
    name: 'whitespace goalId',
    snapshot: { goalId: '   ', title: 'blank identity', status: 'running', hasLinkedAgentRun: false },
  },
  {
    name: 'unknown status',
    snapshot: { goalId: 'goal-unknown', title: 'unknown status', status: 'unknown', hasLinkedAgentRun: false },
  },
  {
    name: 'inherited property is an unknown status',
    snapshot: { goalId: 'goal-inherited', title: 'inherited status', status: 'toString', hasLinkedAgentRun: false },
  },
]

for (const { name, snapshot } of invalidCases) {
  assert.equal(projectGoalDrawerState(snapshot), null, name)
}

const immutableSnapshot = Object.freeze({
  goalId: 'goal-immutable',
  title: 'Immutable goal',
  status: 'paused',
  hasLinkedAgentRun: true,
})
const beforeProjection = { ...immutableSnapshot }

projectGoalDrawerState(immutableSnapshot)
assert.deepStrictEqual(immutableSnapshot, beforeProjection, 'the input snapshot must not be modified')

const firstProjection = projectGoalDrawerState(immutableSnapshot)
const repeatedProjection = projectGoalDrawerState(immutableSnapshot)
assert.deepStrictEqual(firstProjection, repeatedProjection, 'repeated calls must return the same structure')

console.log(`goal drawer model: ${validCases.length + invalidCases.length + 2} assertions passed`)
