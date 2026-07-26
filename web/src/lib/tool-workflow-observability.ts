import type { ToolExposureDiagnostics } from './chat.ts'

export interface EffectivePolicyProjection {
  policySnapshotId: string | null
  policyVersion: string | null
  sourceChain: string[]
  autonomyMode: string
  requireApprovalForFileWrite: boolean | null
  requireApprovalForExec: boolean | null
  fileReadScope: string | null
  fileWriteScope: string | null
  hardDenies: string[]
  expectationStatus: 'matches' | 'stale' | 'not_provided'
  expectedPolicyVersion: string | null
  reviewSemantics: 'concrete_call_only'
}

export interface ToolWorkflowCallProjection {
  callId: string
  toolName: string
  status: string
  arguments?: Record<string, unknown>
  target?: string | null
  approvalId?: string | null
  approvalStatus?: string
  autoReviewDecision?: string
  autoReviewSource?: string | null
  operationId?: string | null
  changedPaths?: string[]
  verificationStatus?: string
  retryStatus?: string
  blocker?: string | null
}

export interface ToolWorkflowProjection {
  schemaVersion: number
  effectivePolicy: EffectivePolicyProjection
  toolInventory: {
    catalogScope: 'policy_eligible'
    policyCatalog: string[]
    eligibleTools: string[]
    exposedTools: string[]
  }
  activation: { status: string; calls: ToolWorkflowCallProjection[] }
  callReview: { status: string; calls: ToolWorkflowCallProjection[] }
  execution: { status: string; calls: ToolWorkflowCallProjection[] }
}

export interface ConcreteToolWorkflowView {
  activationStatus: string | null
  activationAuthorizesCall: boolean | null
  reviewStatus: string
  approvalStatus: string
  autoReviewDecision: string
  autoReviewSource: string | null
  executionStatus: string
  operationId: string | null
  changedPaths: string[]
  verificationStatus: string
  retryStatus: string
  blocker: string | null
}

function record(value: unknown): Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown>
    : {}
}

function records(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.map(record).filter((item) => Object.keys(item).length > 0) : []
}

function text(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function strings(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === 'string' && item.length > 0)
    : []
}

function optionalBoolean(...values: unknown[]): boolean | null {
  for (const value of values) {
    if (typeof value === 'boolean') return value
  }
  return null
}

function capabilityPlan(exposure: Record<string, unknown>): Record<string, unknown> {
  const diagnostics = record(exposure.diagnostics)
  const stages = records(diagnostics.stages)
  for (let index = stages.length - 1; index >= 0; index -= 1) {
    const stage = stages[index]
    if (stage.stage === 'turn_contract_rollout') {
      return record(stage.capability_plan)
    }
  }
  return {}
}

export function normalizeToolExposureDiagnostics(
  metadata?: Record<string, unknown>
): ToolExposureDiagnostics | undefined {
  const exposure = record(metadata?.tool_exposure ?? metadata?.toolExposure)
  if (Object.keys(exposure).length === 0) return undefined

  const plan = capabilityPlan(exposure)
  const rawDiagnostics = records(plan.tool_diagnostics)
  const adapter = record(record(exposure.diagnostics).capability_exposure_adapter)
  const broker = record(adapter.activation_broker)
  const exposedTools = strings(exposure.exposed_tools ?? exposure.exposedTools)
  const eligibleTools = strings(plan.eligible_tools)
  const policyCatalog = rawDiagnostics
    .map((item) => text(item.tool_name))
    .filter((item): item is string => item !== null)
  const workspaceBound = optionalBoolean(exposure.workspace_bound, exposure.workspaceBound)
  const attachmentCount = [exposure.attachment_count, exposure.attachmentCount]
    .find((value): value is number => typeof value === 'number' && Number.isFinite(value))

  return {
    catalogScope: 'policy_eligible',
    policyCatalog,
    eligibleTools,
    exposedTools,
    activationAllowedTools: strings(adapter.activation_allowed_tool_names),
    deferredTools: strings(broker.deferred_tool_names),
    planVersion: text(plan.plan_version) ?? text(adapter.plan_version) ?? undefined,
    toolDiagnostics: rawDiagnostics.map((item) => ({
      toolName: text(item.tool_name) ?? 'unknown',
      status: text(item.status) ?? 'unknown',
      includeReasons: strings(item.include_reasons),
      excludeReasons: strings(item.exclude_reasons),
    })),
    workspaceBound: workspaceBound ?? undefined,
    attachmentCount,
  }
}

function normalizeCall(value: unknown): ToolWorkflowCallProjection | null {
  const item = record(value)
  const callId = text(item.call_id)
  if (!callId) return null
  return {
    callId,
    toolName: text(item.tool_name) ?? text(item.requested_tool) ?? 'tool_activate',
    status: text(item.status) ?? 'not_observed',
    arguments: Object.keys(record(item.arguments)).length > 0 ? record(item.arguments) : undefined,
    target: text(item.target),
    approvalId: text(item.approval_id),
    approvalStatus: text(item.approval_status) ?? undefined,
    autoReviewDecision: text(item.auto_review_decision) ?? undefined,
    autoReviewSource: text(item.auto_review_source),
    operationId: text(item.operation_id),
    changedPaths: strings(item.changed_paths),
    verificationStatus: text(item.verification_status) ?? undefined,
    retryStatus: text(item.retry_status) ?? undefined,
    blocker: text(item.blocker),
  }
}

function normalizeCallGroup(value: unknown): { status: string; calls: ToolWorkflowCallProjection[] } {
  const group = record(value)
  return {
    status: text(group.status) ?? 'not_observed',
    calls: (Array.isArray(group.calls) ? group.calls : [])
      .map(normalizeCall)
      .filter((item): item is ToolWorkflowCallProjection => item !== null),
  }
}

export function normalizeToolWorkflowProjection(value: unknown): ToolWorkflowProjection | null {
  const root = record(value)
  if (Object.keys(root).length === 0) return null
  const policy = record(root.effective_policy)
  const inventory = record(root.tool_inventory)
  const expectation = text(policy.expectation_status)
  return {
    schemaVersion: typeof root.schema_version === 'number' ? root.schema_version : 1,
    effectivePolicy: {
      policySnapshotId: text(policy.policy_snapshot_id),
      policyVersion: text(policy.policy_version),
      sourceChain: strings(policy.source_chain),
      autonomyMode: text(policy.autonomy_mode) ?? 'unknown',
      requireApprovalForFileWrite: optionalBoolean(policy.require_approval_for_file_write),
      requireApprovalForExec: optionalBoolean(policy.require_approval_for_exec),
      fileReadScope: text(policy.file_read_scope),
      fileWriteScope: text(policy.file_write_scope),
      hardDenies: strings(policy.hard_denies),
      expectationStatus: expectation === 'matches' || expectation === 'stale' ? expectation : 'not_provided',
      expectedPolicyVersion: text(policy.expected_policy_version),
      reviewSemantics: 'concrete_call_only',
    },
    toolInventory: {
      catalogScope: 'policy_eligible',
      policyCatalog: strings(inventory.policy_catalog),
      eligibleTools: strings(inventory.eligible_tools),
      exposedTools: strings(inventory.exposed_tools),
    },
    activation: normalizeCallGroup(root.activation),
    callReview: normalizeCallGroup(root.call_review),
    execution: normalizeCallGroup(root.execution),
  }
}

export function projectConcreteToolWorkflow(input: {
  toolName: string
  result?: unknown
  metadata?: Record<string, unknown>
  isResult: boolean
  errorMessage?: string
}): ConcreteToolWorkflowView {
  const metadata = record(input.metadata)
  const result = record(input.result)
  const isActivation = input.toolName === 'tool_activate'
  const activationStatus = isActivation
    ? text(metadata.status) ?? text(result.status) ?? (input.errorMessage ? 'failed' : 'not_observed')
    : null
  const approvalStatus = text(metadata.approval_status) ?? text(metadata.approval_state) ?? 'not_observed'
  const autoReviewDecision = text(metadata.auto_review_decision) ?? 'not_observed'
  return {
    activationStatus,
    activationAuthorizesCall: isActivation
      ? optionalBoolean(
          metadata.activation_authorizes_tool_call,
          result.activation_authorizes_tool_call
        )
      : null,
    reviewStatus: autoReviewDecision !== 'not_observed' ? autoReviewDecision : approvalStatus,
    approvalStatus,
    autoReviewDecision,
    autoReviewSource: text(metadata.auto_review_source),
    executionStatus: input.isResult ? (input.errorMessage ? 'failed' : 'completed') : 'not_observed',
    operationId: text(metadata.operation_id) ?? text(result.operation_id),
    changedPaths: strings(metadata.changed_paths ?? result.changed_paths),
    verificationStatus: text(metadata.verification_status) ?? text(result.verification_status) ?? 'not_observed',
    retryStatus: text(metadata.retry_status) ?? 'not_observed',
    blocker: text(metadata.blocker) ?? text(metadata.reason),
  }
}
