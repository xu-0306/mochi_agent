import assert from 'node:assert/strict'

import {
  normalizeToolExposureDiagnostics,
  normalizeToolWorkflowProjection,
  projectConcreteToolWorkflow,
} from './tool-workflow-observability.ts'

const metadata = {
  tool_exposure: {
    exposed_tools: ['tool_search', 'tool_activate'],
    diagnostics: {
      capability_exposure_adapter: {
        plan_version: 'capability-plan-v1',
        activation_allowed_tool_names: ['file_write'],
        activation_broker: { deferred_tool_names: ['file_write'] },
      },
      stages: [
        {
          stage: 'turn_contract_rollout',
          capability_plan: {
            plan_version: 'capability-plan-v1',
            eligible_tools: ['tool_search', 'file_write'],
            tool_diagnostics: [
              { tool_name: 'tool_search', status: 'exposed' },
              { tool_name: 'file_write', status: 'eligible' },
            ],
          },
        },
      ],
    },
  },
}

const exposure = normalizeToolExposureDiagnostics(metadata)
assert.deepEqual(exposure?.policyCatalog, ['tool_search', 'file_write'])
assert.deepEqual(exposure?.eligibleTools, ['tool_search', 'file_write'])
assert.deepEqual(exposure?.exposedTools, ['tool_search', 'tool_activate'])
assert.deepEqual(exposure?.activationAllowedTools, ['file_write'])

const successfulButUnverified = projectConcreteToolWorkflow({
  toolName: 'file_write',
  result: { changed_paths: ['README.md'] },
  metadata: {},
  isResult: true,
})
assert.equal(successfulButUnverified.executionStatus, 'completed')
assert.equal(successfulButUnverified.autoReviewDecision, 'not_observed')
assert.equal(successfulButUnverified.verificationStatus, 'not_observed')

const activation = projectConcreteToolWorkflow({
  toolName: 'tool_activate',
  result: {
    status: 'tool_activated',
    activation_authorizes_tool_call: false,
  },
  isResult: true,
})
assert.equal(activation.activationStatus, 'tool_activated')
assert.equal(activation.activationAuthorizesCall, false)
assert.equal(activation.reviewStatus, 'not_observed')

const projection = normalizeToolWorkflowProjection({
  schema_version: 1,
  effective_policy: {
    policy_snapshot_id: 'policy-1',
    policy_version: 'effective-policy:1',
    source_chain: ['security_config'],
    autonomy_mode: 'auto_review',
    expectation_status: 'matches',
  },
  tool_inventory: {
    catalog_scope: 'policy_eligible',
    policy_catalog: ['file_write'],
    eligible_tools: ['file_write'],
    exposed_tools: ['file_write'],
  },
  activation: { status: 'not_observed', calls: [] },
  call_review: { status: 'not_observed', calls: [] },
  execution: { status: 'not_observed', calls: [] },
})
assert.equal(projection?.effectivePolicy.autonomyMode, 'auto_review')
assert.equal(projection?.effectivePolicy.expectationStatus, 'matches')
assert.equal(projection?.toolInventory.catalogScope, 'policy_eligible')

console.log('tool workflow observability tests passed')
