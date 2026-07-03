import assert from 'node:assert/strict'

const moduleUrl = new URL('../src/lib/goal-strategy-selection.ts', import.meta.url).href

const {
  buildGoalProposalState,
  resolveGoalStrategyDisplaySelection,
} = await import(moduleUrl)

const modelCandidates = [
  { id: 'gpt-5', label: 'GPT-5', status: 'connected' },
  { id: 'claude-sonnet', label: 'Claude Sonnet', status: 'configured' },
]

const keywordHeavyWorkflowDraft = buildGoalProposalState({
  objective: 'Compare the options, run a debate, and distill the answer.',
  executionMode: 'workflow',
  modelCandidates,
  currentModelId: 'gpt-5',
  workflowSelectedModelRoles: {},
  workflowTemplate: 'standard',
  workflowProtocolId: 'teacher_student_distill',
  workflowScheduleEnabled: false,
  workflowScheduleType: 'interval',
  effectiveAutonomyMode: 'trusted_workspace',
  now: new Date('2026-07-03T00:00:00.000Z'),
})

assert.equal(
  keywordHeavyWorkflowDraft.protocol_id,
  'teacher_student_distill',
  'ordinary proposal text must not switch workflow protocol away from the explicit workflow config'
)
assert.equal(
  keywordHeavyWorkflowDraft.protocol_selection,
  'teacher_student_distill'
)

const keywordHeavySingleAgentDraft = buildGoalProposalState({
  objective: 'Debate the tradeoffs and compare the outcomes.',
  executionMode: 'single_agent',
  modelCandidates,
  currentModelId: 'gpt-5',
  workflowSelectedModelRoles: {},
  workflowTemplate: 'standard',
  workflowProtocolId: 'teacher_student_distill',
  workflowScheduleEnabled: false,
  workflowScheduleType: 'interval',
  effectiveAutonomyMode: 'trusted_workspace',
  now: new Date('2026-07-03T00:00:00.000Z'),
})

assert.equal(
  keywordHeavySingleAgentDraft.protocol_id,
  null,
  'single-agent proposals must stay protocol-free regardless of objective wording'
)
assert.equal(keywordHeavySingleAgentDraft.execution_topology, 'single_agent')

const registrySelection = resolveGoalStrategyDisplaySelection({
  registry: {
    type: 'goal_strategy_registry',
    default_strategy_id: 'debate-default',
    entries: [
      {
        id: 'debate-default',
        name: 'debate-default',
        display_name: 'Debate Default',
        description: 'Registry-backed workflow choice',
        when_to_use: null,
        when_not_to_use: null,
        execution_topology: 'multi_agent',
        kind: 'workflow',
        protocol_id: 'multi_agent_debate',
        required_capabilities: [],
        approval_profile: null,
        control_scope: null,
        interrupt_policy: null,
        resume_policy: null,
        event_contract: null,
        success_signals: [],
        failure_modes: [],
        fallback_strategy_ids: [],
        requires_confirmation: true,
        is_default: true,
        available: true,
        availability_reason: null,
        deprecated: false,
        override_label: 'Debate Default',
        selection_guidance: 'Use the registry-selected debate flow.',
        raw: {},
      },
    ],
  },
  protocolId: 'multi_agent_debate',
  interactionMode: 'workflow',
})

assert.equal(registrySelection.strategyId, 'debate-default')
assert.equal(registrySelection.protocolId, 'multi_agent_debate')
assert.equal(
  registrySelection.selectionRationale,
  'Use the registry-selected debate flow.'
)

console.log('ok')
