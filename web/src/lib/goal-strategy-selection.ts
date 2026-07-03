import type {
  AgentRunProtocolId,
  GoalSummary,
  GoalExecutionMode,
  GoalExecutionTopology,
  GoalInteractionMode,
  GoalStrategyRegistryEntry,
  GoalStrategyRegistryResponse,
} from './api'
import {
  selectGoalProposalModels,
  summarizeGoalProposalModelReadinessRisk,
  type GoalProposalModelOption,
} from './goal-proposal-models.ts'

export type WorkflowTemplate = 'standard' | 'research_debate'
export type WorkflowScheduleType = 'interval' | 'once' | 'cron'

export interface GoalSessionSummaryLike {
  goal_id: string | null
  objective: string
  execution_mode: GoalExecutionMode
  interaction_mode: GoalInteractionMode
  execution_topology: GoalExecutionTopology
  strategy_id?: string | null
  selection_source?: string | null
  selection_reason?: string | null
  protocol_id: string | null
  bound_run_id: string | null
  protocol_selection: string | null
  selection_rationale: string | null
  models: string[]
  role_summary: string | null
  runtime_mode: string | null
  risk_note: string | null
  status: string | null
}

export interface GoalSessionProposalLike extends GoalSessionSummaryLike {
  proposal_id: string
  revision_index: number
  updated_at: string
  assistant_explanation: string | null
  assistant_explanation_source: string | null
}

export interface GoalStrategyDisplaySelection {
  strategyId: string | null
  protocolId: AgentRunProtocolId | null
  label: string | null
  description: string | null
  selectionRationale: string | null
  executionTopology: GoalExecutionTopology
}

export interface BuildGoalProposalStateOptions {
  objective: string
  executionMode: GoalExecutionMode
  previous?: GoalSessionProposalLike | null
  revisionText?: string | null
  modelCandidates: GoalProposalModelOption[]
  currentModelId?: string | null
  workflowSelectedModelRoles?: Record<string, string>
  workflowTemplate: WorkflowTemplate
  workflowProtocolId: AgentRunProtocolId
  workflowScheduleEnabled: boolean
  workflowScheduleType: WorkflowScheduleType
  effectiveAutonomyMode: 'trusted_workspace' | 'strict' | 'high_autonomy' | 'auto_review'
  goalStrategyRegistry?: GoalStrategyRegistryResponse | null
  defaultWorkflowProtocol?: AgentRunProtocolId
  now?: Date
}

export function normalizeGoalExecutionMode(
  value: unknown
): GoalExecutionMode | null {
  return value === 'single_agent' || value === 'workflow' ? value : null
}

export function normalizeGoalInteractionMode(
  value: unknown
): GoalInteractionMode | null {
  return value === 'goal' || value === 'workflow' ? value : null
}

export function normalizeGoalExecutionTopology(
  value: unknown
): GoalExecutionTopology | null {
  return value === 'single_agent' || value === 'multi_agent' ? value : null
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  return [...new Set(values.map((value) => (typeof value === 'string' ? value.trim() : '')).filter((value) => value.length > 0))]
}

export function normalizeSelectedModelRoles(
  value: Record<string, string> | undefined
): Record<string, string> {
  const next: Record<string, string> = {}
  for (const [role, modelId] of Object.entries(value ?? {})) {
    const normalizedRole = role.trim()
    const normalizedModelId = modelId.trim()
    if (normalizedRole && normalizedModelId) {
      next[normalizedRole] = normalizedModelId
    }
  }
  return next
}

export function mergeSelectedModelRoles(
  ...sources: Array<Record<string, string> | undefined>
): Record<string, string> {
  const next: Record<string, string> = {}
  for (const source of sources) {
    Object.assign(next, normalizeSelectedModelRoles(source))
  }
  return next
}

export function buildSelectedModelsRolesPayload(
  byRole: Record<string, string>
): Record<string, unknown> {
  const normalized = normalizeSelectedModelRoles(byRole)
  const entries = Object.entries(normalized).map(([role, model_id]) => ({ role, model_id }))
  return {
    by_role: normalized,
    entries,
    subagents: entries,
  }
}

function formatGoalRoleSummary(roles: string[]): string | null {
  if (roles.length === 0) {
    return null
  }
  return roles.map((role) => role.replaceAll('_', ' ')).join(', ')
}

function detectGoalRuntimeHint(value: string): string | null {
  const normalized = value.trim()
  const durationMatch = normalized.match(/\b\d+\s*(?:min(?:ute)?s?|hour(?:s)?|hr|hrs)\b/i)
  if (durationMatch) {
    return `Requested duration: ${durationMatch[0]}`
  }
  if (/schedule|cron|interval/i.test(normalized)) {
    return 'Scheduled goal execution requested'
  }
  return null
}

export function goalProtocolExecutionTopology(
  protocolId: AgentRunProtocolId | null
): GoalExecutionTopology {
  return protocolId === 'autonomous_single_agent' ? 'single_agent' : 'multi_agent'
}

export function defaultRolesForGoalProtocol(
  protocolId: AgentRunProtocolId | null,
  workflowTemplate: WorkflowTemplate
): string[] {
  if (workflowTemplate === 'research_debate') {
    return ['planner', 'debater_a', 'debater_b', 'judge']
  }
  if (protocolId === 'controlled_subagent_execution') {
    return ['planner', 'executor', 'controller', 'evaluator']
  }
  if (protocolId === 'multi_agent_debate') {
    return ['debater_a', 'debater_b', 'judge']
  }
  if (protocolId === 'dr_zero_self_evolve') {
    return ['proposer', 'solver', 'verifier']
  }
  return ['agent']
}

export function describeGoalProtocolSelection(
  protocolId: AgentRunProtocolId,
  interactionMode: GoalInteractionMode
): string {
  if (protocolId === 'autonomous_single_agent') {
    return 'Routed to autonomous_single_agent for direct long-running execution.'
  }
  if (protocolId === 'teacher_student_distill') {
    return 'Routed to teacher_student_distill for summarization and distillation work.'
  }
  if (protocolId === 'multi_agent_debate') {
    return 'Routed to multi_agent_debate for comparison or evaluation work.'
  }
  if (protocolId === 'controlled_subagent_execution') {
    return 'Routed to controlled_subagent_execution for operator-gated execution steps.'
  }
  if (protocolId === 'dr_zero_self_evolve') {
    return 'Routed to dr_zero_self_evolve for iterative self-evolving problem solving.'
  }
  return interactionMode === 'goal'
    ? 'Routed into the goal lane with workflow-capable execution.'
    : 'Routed into the workflow lane for orchestrated execution.'
}

export function detectGoalModelHints(
  value: string,
  candidates: GoalProposalModelOption[],
  currentModel: string | null
): string[] {
  const normalized = value.toLowerCase()
  const matches: string[] = []

  const currentModelTrimmed = currentModel?.trim() ?? ''
  if (currentModelTrimmed && normalized.includes(currentModelTrimmed.toLowerCase())) {
    matches.push(currentModelTrimmed)
  }

  for (const candidate of candidates) {
    const id = candidate.id.trim()
    const label = candidate.label.trim().toLowerCase()
    if (!id) {
      continue
    }
    if (normalized.includes(id.toLowerCase()) || (label && normalized.includes(label))) {
      matches.push(id)
    }
  }

  return uniqueStrings(matches)
}

function findGoalStrategyRegistryEntry(
  registry: GoalStrategyRegistryResponse | null | undefined,
  options: {
    strategyId?: string | null
    protocolId?: AgentRunProtocolId | null
  }
): GoalStrategyRegistryEntry | null {
  if (!registry) {
    return null
  }

  const strategyId = options.strategyId?.trim() ?? ''
  if (strategyId) {
    const entry = registry.entries.find((item) => item.id === strategyId)
    if (entry) {
      return entry
    }
  }

  const protocolId = options.protocolId?.trim() ?? ''
  if (protocolId) {
    const entry = registry.entries.find((item) => item.protocol_id === protocolId || item.id === protocolId)
    if (entry) {
      return entry
    }
  }

  const defaultStrategyId = registry.default_strategy_id?.trim() ?? ''
  if (defaultStrategyId) {
    const entry = registry.entries.find((item) => item.id === defaultStrategyId)
    if (entry) {
      return entry
    }
  }

  return registry.entries.find((item) => item.is_default) ?? null
}

export function resolveGoalStrategyDisplaySelection(options: {
  registry?: GoalStrategyRegistryResponse | null
  strategyId?: string | null
  protocolId?: AgentRunProtocolId | null
  interactionMode: GoalInteractionMode
}): GoalStrategyDisplaySelection {
  const entry = findGoalStrategyRegistryEntry(options.registry, options)
  const resolvedProtocolId = (entry?.protocol_id ?? options.protocolId ?? null) as AgentRunProtocolId | null
  const executionTopology = entry?.execution_topology ?? goalProtocolExecutionTopology(resolvedProtocolId)
  const selectionRationale =
    entry?.selection_guidance?.trim() ||
    entry?.description?.trim() ||
    (resolvedProtocolId
      ? describeGoalProtocolSelection(resolvedProtocolId, options.interactionMode)
      : null)

  return {
    strategyId: entry?.id ?? options.strategyId ?? null,
    protocolId: resolvedProtocolId,
    label: entry?.override_label ?? entry?.display_name ?? null,
    description: entry?.description ?? entry?.selection_guidance ?? null,
    selectionRationale,
    executionTopology,
  }
}

export function buildGoalSummaryFromGoal<TGoalSummary extends GoalSessionSummaryLike>(
  goal: GoalSummary,
  fallback?: TGoalSummary | null
): TGoalSummary {
  return {
    goal_id: goal.goal_id ? goal.goal_id : (fallback?.goal_id ?? null),
    objective: goal.objective ? goal.objective : (fallback?.objective ?? ''),
    execution_mode: goal.execution_mode,
    interaction_mode: goal.interaction_mode,
    execution_topology: goal.execution_topology,
    strategy_id: goal.strategy_id ?? fallback?.strategy_id ?? null,
    selection_source: goal.selection_source ?? fallback?.selection_source ?? null,
    selection_reason: goal.selection_reason ?? fallback?.selection_reason ?? null,
    protocol_id: goal.protocol_id ?? fallback?.protocol_id ?? null,
    bound_run_id: goal.bound_run_id ?? fallback?.bound_run_id ?? null,
    protocol_selection:
      goal.protocol_selection ??
      fallback?.protocol_selection ??
      goal.protocol_id ??
      null,
    selection_rationale:
      goal.selection_rationale ??
      goal.selection_reason ??
      fallback?.selection_rationale ??
      fallback?.selection_reason ??
      null,
    models: fallback?.models ?? [],
    role_summary: fallback?.role_summary ?? null,
    runtime_mode: fallback?.runtime_mode ?? null,
    risk_note: fallback?.risk_note ?? null,
    status: goal.status,
  } as TGoalSummary
}

export function buildGoalProposalState(
  options: BuildGoalProposalStateOptions
): GoalSessionProposalLike {
  const revisionText = options.revisionText?.trim() ?? ''
  const heuristicSource = revisionText || options.objective
  const interactionMode: GoalInteractionMode =
    options.executionMode === 'workflow' ? 'workflow' : 'goal'
  const defaultWorkflowProtocol =
    options.defaultWorkflowProtocol ?? 'teacher_student_distill'
  const explicitWorkflowProtocolId: AgentRunProtocolId | null =
    options.executionMode !== 'workflow'
      ? null
      : options.workflowTemplate === 'research_debate'
        ? 'multi_agent_debate'
        : options.workflowProtocolId ?? defaultWorkflowProtocol
  const selectedRoles = normalizeSelectedModelRoles(options.workflowSelectedModelRoles)
  const selectedRoleNames = Object.keys(selectedRoles)
  const fallbackWorkflowRoles = defaultRolesForGoalProtocol(
    explicitWorkflowProtocolId,
    options.workflowTemplate
  )
  const workflowModels = uniqueStrings([
    ...Object.values(selectedRoles),
    options.currentModelId,
    options.modelCandidates[0]?.id ?? null,
  ])
  const explicitModelHints = detectGoalModelHints(
    heuristicSource,
    options.modelCandidates,
    options.currentModelId ?? null
  )
  const strategySelection = resolveGoalStrategyDisplaySelection({
    registry: options.goalStrategyRegistry,
    protocolId: explicitWorkflowProtocolId,
    interactionMode,
  })
  const protocolId = strategySelection.protocolId ?? explicitWorkflowProtocolId
  const executionTopology = protocolId
    ? strategySelection.executionTopology
    : 'single_agent'
  const primaryModels =
    executionTopology === 'multi_agent'
      ? uniqueStrings([
          ...selectGoalProposalModels(
            options.modelCandidates,
            options.currentModelId ?? null,
            options.executionMode,
            explicitModelHints
          ),
          ...workflowModels,
        ]).slice(0, 3)
      : selectGoalProposalModels(
          options.modelCandidates,
          options.currentModelId ?? null,
          options.executionMode,
          explicitModelHints
        )
  const runtimeHint = detectGoalRuntimeHint(heuristicSource)
  const roleSummary =
    executionTopology === 'multi_agent'
      ? formatGoalRoleSummary(selectedRoleNames.length > 0 ? selectedRoleNames : fallbackWorkflowRoles)
      : 'Primary agent continues the task directly with the current chat tools.'
  const runtimeMode =
    runtimeHint ??
    (executionTopology === 'single_agent'
      ? 'Single-agent long-running execution'
      : interactionMode === 'workflow'
        ? options.workflowScheduleEnabled
          ? `Scheduled ${options.workflowScheduleType} workflow`
          : 'Workflow run starts immediately'
        : 'Goal-backed orchestrated execution')
  const modelReadinessRisk = summarizeGoalProposalModelReadinessRisk(
    options.modelCandidates,
    primaryModels
  )
  const autonomyRisk =
    options.effectiveAutonomyMode === 'strict'
      ? 'Runtime actions may pause for approval before execution.'
      : options.effectiveAutonomyMode === 'trusted_workspace'
        ? 'Riskier runtime actions may still require approval.'
        : null
  const riskNote = [modelReadinessRisk, autonomyRisk].filter(Boolean).join(' ') || null

  return {
    goal_id: null,
    proposal_id: options.previous?.proposal_id ?? `goal-proposal-${(options.now ?? new Date()).getTime()}`,
    objective: options.objective.trim(),
    execution_mode: options.executionMode,
    interaction_mode: interactionMode,
    execution_topology: executionTopology,
    strategy_id: strategySelection.strategyId,
    selection_source: options.previous?.selection_source ?? null,
    selection_reason: strategySelection.selectionRationale,
    protocol_id: protocolId,
    bound_run_id: null,
    protocol_selection: protocolId,
    selection_rationale: strategySelection.selectionRationale,
    models: primaryModels,
    role_summary: roleSummary,
    runtime_mode: runtimeMode,
    risk_note: riskNote,
    status: null,
    revision_index: (options.previous?.revision_index ?? -1) + 1,
    updated_at: (options.now ?? new Date()).toISOString(),
    assistant_explanation: options.previous?.assistant_explanation ?? null,
    assistant_explanation_source: options.previous?.assistant_explanation_source ?? null,
  }
}
