import assert from 'node:assert/strict'
const moduleUrl = new URL('../src/lib/goal-proposal-models.ts', import.meta.url).href

const {
  applyGoalProposalModelReadiness,
  buildGoalProposalProbeCandidates,
  selectGoalProposalModels,
  summarizeGoalProposalModelReadinessRisk,
} = await import(moduleUrl)

const baseOptions = [
  { id: 'gpt-5', label: 'GPT-5', status: 'configured' },
  { id: 'claude-sonnet', label: 'Claude Sonnet', status: 'configured' },
  { id: 'gemini-flash', label: 'Gemini Flash', status: 'configured' },
]

const readyOptions = applyGoalProposalModelReadiness(baseOptions, 'gpt-5', {
  'claude-sonnet': 'ready',
  'gemini-flash': 'failed',
})

assert.deepEqual(
  readyOptions.map((option) => ({ id: option.id, status: option.status })),
  [
    { id: 'gpt-5', status: 'connected' },
    { id: 'claude-sonnet', status: 'connected' },
    { id: 'gemini-flash', status: 'disconnected' },
  ],
  'Readiness application should promote ready models to connected and failed probes to disconnected'
)

assert.deepEqual(
  buildGoalProposalProbeCandidates(
    readyOptions,
    'gpt-5',
    'workflow',
    ['claude-sonnet', 'gemini-flash']
  ),
  [],
  'Already connected or failed models should not be re-probed immediately'
)

assert.deepEqual(
  buildGoalProposalProbeCandidates(
    baseOptions,
    null,
    'workflow',
    ['claude-sonnet']
  ),
  ['claude-sonnet', 'gpt-5', 'gemini-flash'],
  'Workflow probe selection should stay targeted while keeping explicit hints first'
)

assert.deepEqual(
  selectGoalProposalModels(
    readyOptions,
    'gpt-5',
    'workflow',
    []
  ),
  ['gpt-5', 'claude-sonnet', 'gemini-flash'],
  'Model proposal selection should prefer connected models before falling back to the remaining configured catalog'
)

assert.deepEqual(
  selectGoalProposalModels(
    readyOptions,
    'gpt-5',
    'single_agent',
    ['claude-sonnet']
  ),
  ['claude-sonnet'],
  'Explicit model hints should win for single-agent proposals'
)

assert.equal(
  summarizeGoalProposalModelReadinessRisk(
    readyOptions,
    ['gpt-5']
  ),
  null,
  'Ready selected models should not surface a readiness warning'
)

assert.match(
  summarizeGoalProposalModelReadinessRisk(
    baseOptions,
    ['claude-sonnet']
  ) ?? '',
  /No verified model connection is ready yet/,
  'Configured-only selections should surface a readiness warning'
)

console.log('ok')
