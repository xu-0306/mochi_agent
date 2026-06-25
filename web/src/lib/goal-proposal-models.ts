export type GoalProposalExecutionMode = 'single_agent' | 'workflow'

export interface GoalProposalModelOption {
  id: string
  label: string
  detail?: string | null
  status?: 'connected' | 'configured' | 'disconnected'
}

export type GoalProposalModelReadinessState = 'ready' | 'failed'
export type GoalProposalModelReadinessById = Record<string, GoalProposalModelReadinessState>

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    if (!value) {
      continue
    }
    const trimmed = value.trim()
    if (!trimmed || seen.has(trimmed)) {
      continue
    }
    seen.add(trimmed)
    result.push(trimmed)
  }
  return result
}

export function applyGoalProposalModelReadiness(
  modelOptions: GoalProposalModelOption[],
  currentModel: string | null,
  readinessById: GoalProposalModelReadinessById
): GoalProposalModelOption[] {
  return modelOptions.map((option) => {
    const readiness = readinessById[option.id]
    if (currentModel && option.id === currentModel) {
      return {
        ...option,
        status: 'connected',
      }
    }
    if (readiness === 'ready') {
      return {
        ...option,
        status: 'connected',
      }
    }
    if (readiness === 'failed') {
      return {
        ...option,
        status: 'disconnected',
      }
    }
    return {
      ...option,
      status: option.status ?? 'configured',
    }
  })
}

export function buildGoalProposalProbeCandidates(
  modelOptions: GoalProposalModelOption[],
  currentModel: string | null,
  executionMode: GoalProposalExecutionMode,
  explicitModelHints: string[]
): string[] {
  const probeLimit = executionMode === 'workflow' ? 3 : 2
  const configuredCandidates = modelOptions
    .filter((option) => option.status !== 'connected' && option.status !== 'disconnected')
    .map((option) => option.id)

  return uniqueStrings([
    ...explicitModelHints,
    currentModel,
    ...configuredCandidates,
  ]).filter((modelId) => {
    const candidate = modelOptions.find((option) => option.id === modelId)
    return candidate ? candidate.status !== 'connected' && candidate.status !== 'disconnected' : true
  }).slice(0, probeLimit)
}

export function selectGoalProposalModels(
  modelOptions: GoalProposalModelOption[],
  currentModel: string | null,
  executionMode: GoalProposalExecutionMode,
  explicitModelHints: string[]
): string[] {
  const selectionLimit = executionMode === 'workflow' ? 3 : 1
  const ready = modelOptions
    .filter((option) => option.status === 'connected')
    .map((option) => option.id)
  const configured = modelOptions
    .filter((option) => option.status !== 'disconnected')
    .map((option) => option.id)
  const remaining = modelOptions.map((option) => option.id)

  if (explicitModelHints.length > 0) {
    return uniqueStrings([
      ...explicitModelHints,
      ...ready,
      ...configured,
      ...remaining,
    ]).slice(0, selectionLimit)
  }

  return uniqueStrings([
    currentModel,
    ...ready,
    ...configured,
    ...remaining,
  ]).slice(0, selectionLimit)
}

export function summarizeGoalProposalModelReadinessRisk(
  modelOptions: GoalProposalModelOption[],
  selectedModels: string[]
): string | null {
  if (selectedModels.length === 0) {
    return 'No suggested model is configured yet. Review Settings before starting this goal.'
  }

  const statuses = new Map(modelOptions.map((option) => [option.id, option.status ?? 'configured']))
  const readySelected = selectedModels.filter((modelId) => statuses.get(modelId) === 'connected')

  if (readySelected.length > 0) {
    return null
  }

  return 'No verified model connection is ready yet. Review Settings or refresh the saved model connections before starting.'
}
