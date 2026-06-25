'use client'

import * as React from 'react'
import { CheckCircle2, ExternalLink, Loader2, PanelRightClose, Plus, RotateCcw, Trash2, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import {
  FloatingPanelShell,
} from '@/components/chat/FloatingPanelShell'
import { PanelSectionCard } from '@/components/chat/PanelSectionCard'
import { ThinkingLevelPanelControl } from '@/components/chat/ThinkingLevelControls'
import { Switch } from '@/components/ui/switch'
import type {
  AgentRunProtocolId,
  AgentRunRunPolicy,
  ProjectSummary,
  ReasoningEffort,
  SessionWorkflowConfig,
  SessionWorkflowState,
} from '@/lib/api'
import type { ChatInputModelOption } from '@/components/chat/ChatInput'
import { useSessionStore } from '@/lib/stores/session-store'
type WorkflowScheduleType = 'interval' | 'once' | 'cron'
type WorkflowSaveState = 'idle' | 'saving' | 'saved' | 'error'
type WorkflowTemplate = 'standard' | 'research_debate'
type WorkflowRunPolicyPreset = 'short' | 'balanced' | 'long' | 'custom'

interface WorkflowProtocolOption {
  value: AgentRunProtocolId
  label: string
  description: string
}

interface WorkflowPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  sessionId: string | null
  workflowEnabled: boolean
  workflowBusy: boolean
  workflowError: string | null
  workflowBoundRunId: string | null
  workflowState: SessionWorkflowState
  workflowConfig: SessionWorkflowConfig
  workflowTemplate: WorkflowTemplate
  workflowProtocolId: AgentRunProtocolId
  workflowReasoningEffort: ReasoningEffort | null
  workflowRunPolicyPreset: WorkflowRunPolicyPreset
  workflowRunPolicy: AgentRunRunPolicy
  workflowExecutionPolicy: Record<string, unknown>
  workflowEvidenceConfig: Record<string, unknown>
  workflowResearchConfig: Record<string, unknown>
  workflowScheduleConfig: Record<string, unknown>
  workflowScheduleType: WorkflowScheduleType
  workflowScheduleEnabled: boolean
  workflowProtocolOptions: WorkflowProtocolOption[]
  modelOptions: ChatInputModelOption[]
  supportedReasoningEfforts: ReasoningEffort[]
  effectiveProjectId: string | null
  projects: ProjectSummary[]
  workflowProjectWorkspace: string | null
  uploadTargetDir: string | null
  effectiveWorkflowWorkspace: string | null
  workflowHasUnsavedChanges: boolean
  workflowSaveState: WorkflowSaveState
  workflowLastSavedLabel: string | null
  workflowLastSaveScope: 'persisted' | 'draft' | null
  onWorkflowToggle: (enabled: boolean) => void
  onWorkflowNewRun: () => void
  onOpenRunDetail: (runId: string) => void
  onWorkflowProjectChange: (projectId: string | null) => void
  onWorkflowFieldChange: (patch: Partial<SessionWorkflowState>) => void
  onWorkflowTemplateChange: (template: WorkflowTemplate) => void
  onWorkflowRunPolicyPresetChange: (preset: WorkflowRunPolicyPreset) => void
  onWorkflowConfigPatch: (patch: Partial<SessionWorkflowConfig>) => void
  onWorkflowSave: () => void
  buildWorkflowScheduleConfig: (
    schedule: Record<string, unknown>,
    type: WorkflowScheduleType,
    enabled: boolean
  ) => Record<string, unknown>
  formatWorkflowScheduleRunAt: (value: unknown) => string
  defaultScheduleTimezone: () => string
}

interface WorkflowRoleDraft {
  id: string
  role: string
  modelId: string
}

type WorkflowRoleDefaultModel = 'lead' | 'worker' | 'neutral'

interface WorkflowRoleOption {
  value: string
  label: string
  description: string
  defaultModel: WorkflowRoleDefaultModel
}

const EXECUTION_LANE_ROLES = new Set(['executor', 'controller', 'evaluator'])

const KNOWN_WORKFLOW_ROLE_OPTIONS: Record<string, WorkflowRoleOption> = {
  teacher: {
    value: 'teacher',
    label: 'Teacher',
    description: 'Creates the strongest reference answer for the rest of the workflow.',
    defaultModel: 'neutral',
  },
  student: {
    value: 'student',
    label: 'Student',
    description: 'Distills the reference answer into a shorter, easier-to-use final response.',
    defaultModel: 'neutral',
  },
  proposer: {
    value: 'proposer',
    label: 'Proposer',
    description: 'Generates candidate tasks or directions for the run to explore.',
    defaultModel: 'neutral',
  },
  solver: {
    value: 'solver',
    label: 'Solver',
    description: 'Works through the task and produces the main solution.',
    defaultModel: 'neutral',
  },
  verifier: {
    value: 'verifier',
    label: 'Verifier',
    description: 'Checks whether the main answer is correct, complete, and well-supported.',
    defaultModel: 'lead',
  },
  planner: {
    value: 'planner',
    label: 'Planner',
    description: 'Breaks the task into steps and decides how the team should approach it.',
    defaultModel: 'lead',
  },
  executor: {
    value: 'executor',
    label: 'Executor',
    description: 'Prepares code or execution requests when the workflow needs runtime actions.',
    defaultModel: 'worker',
  },
  controller: {
    value: 'controller',
    label: 'Controller',
    description: 'Reviews and approves execution requests before they reach the shared runtime.',
    defaultModel: 'lead',
  },
  evaluator: {
    value: 'evaluator',
    label: 'Evaluator',
    description: 'Summarizes execution results, artifacts, and the next recommended step.',
    defaultModel: 'lead',
  },
  debater_a: {
    value: 'debater_a',
    label: 'Debater A',
    description: 'Argues for the strongest first candidate answer.',
    defaultModel: 'worker',
  },
  debater_b: {
    value: 'debater_b',
    label: 'Debater B',
    description: 'Challenges assumptions and proposes a competing alternative.',
    defaultModel: 'worker',
  },
  judge: {
    value: 'judge',
    label: 'Judge',
    description: 'Chooses the most reliable answer after comparing the evidence and arguments.',
    defaultModel: 'lead',
  },
  synthesizer: {
    value: 'synthesizer',
    label: 'Synthesizer',
    description: 'Combines the best findings into the final research output.',
    defaultModel: 'lead',
  },
  local_worker: {
    value: 'local_worker',
    label: 'Research worker',
    description: 'Handles evidence gathering and local research passes in parallel.',
    defaultModel: 'worker',
  },
  skeptic: {
    value: 'skeptic',
    label: 'Skeptic',
    description: 'Looks for weak evidence, missing citations, and overconfident claims.',
    defaultModel: 'worker',
  },
}

const CUSTOM_ROLE_VALUE = '__custom__'

function createWorkflowRoleDraft(role = '', modelId = ''): WorkflowRoleDraft {
  return {
    id:
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`,
    role,
    modelId,
  }
}

function getWorkflowRoleOption(role: string): WorkflowRoleOption | null {
  return KNOWN_WORKFLOW_ROLE_OPTIONS[role] ?? null
}

function defaultRolesForProtocol(protocolId: AgentRunProtocolId): string[] {
  if (protocolId === 'dr_zero_self_evolve') {
    return ['proposer', 'solver', 'verifier']
  }
  if (protocolId === 'multi_agent_debate') {
    return ['debater_a', 'debater_b', 'judge']
  }
  if (protocolId === 'controlled_subagent_execution') {
    return ['planner', 'executor', 'controller', 'evaluator']
  }
  return ['teacher', 'student']
}

function buildWorkflowRoleOptions(
  template: WorkflowTemplate,
  protocolId: AgentRunProtocolId,
  controlledExecutionEnabled: boolean
): WorkflowRoleOption[] {
  if (template === 'research_debate') {
    const researchRoles = [
      'planner',
      'judge',
      'verifier',
      'synthesizer',
      'debater_a',
      'debater_b',
      'local_worker',
      'skeptic',
      ...(controlledExecutionEnabled ? ['executor', 'controller', 'evaluator'] : []),
    ]
    return researchRoles.map((role) => KNOWN_WORKFLOW_ROLE_OPTIONS[role]).filter(Boolean)
  }

  const seen = new Set<string>()
  const options: WorkflowRoleOption[] = []
  const pushRole = (role: string) => {
    if (seen.has(role)) {
      return
    }
    const option = KNOWN_WORKFLOW_ROLE_OPTIONS[role]
    if (!option) {
      return
    }
    seen.add(role)
    options.push(option)
  }

  defaultRolesForProtocol(protocolId).forEach(pushRole)
  if (controlledExecutionEnabled) {
    ;['planner', 'executor', 'controller', 'evaluator'].forEach(pushRole)
  }
  ;['teacher', 'student', 'proposer', 'solver', 'verifier', 'judge'].forEach(pushRole)
  return options
}

function getRoleSelectValue(role: string, options: WorkflowRoleOption[]): string {
  return options.some((option) => option.value === role) ? role : CUSTOM_ROLE_VALUE
}

function getRoleDefaultModelLabel(defaultModel: WorkflowRoleDefaultModel): string {
  if (defaultModel === 'lead') {
    return 'Smart model by default'
  }
  if (defaultModel === 'worker') {
    return 'Research worker model by default'
  }
  return 'No recommended default'
}

function isExecutionLaneRole(role: string): boolean {
  return EXECUTION_LANE_ROLES.has(role)
}

function getRoleCapabilityLabel(role: string): string {
  return isExecutionLaneRole(role) ? 'Execution lane' : 'Research / verify only'
}

function getRoleCapabilityDescription(role: string): string {
  if (role === 'executor') {
    return 'Can prepare file, code, or command requests when controlled execution is enabled.'
  }
  if (role === 'controller') {
    return 'Can approve or reject execution requests before anything writes files or runs commands.'
  }
  if (role === 'evaluator') {
    return 'Can review execution output and decide what should happen next.'
  }
  return 'Stays in read, research, debate, planning, or verification work. No write/run access.'
}

function getRoleCapabilityClassName(role: string): string {
  return isExecutionLaneRole(role)
    ? 'border-amber-400/20 bg-amber-500/10 text-amber-200'
    : 'border-emerald-400/20 bg-emerald-500/10 text-emerald-200'
}

function normalizeSelectedModelRoles(
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

function serializeSelectedModelRoles(value: Record<string, string> | undefined): string {
  return JSON.stringify(
    Object.entries(normalizeSelectedModelRoles(value)).sort(([left], [right]) => left.localeCompare(right))
  )
}

function buildRoleDraftsFromSelection(value: Record<string, string> | undefined): WorkflowRoleDraft[] {
  return Object.entries(normalizeSelectedModelRoles(value))
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([role, modelId]) => createWorkflowRoleDraft(role, modelId))
}

function buildSelectedModelRolesFromDrafts(drafts: WorkflowRoleDraft[]): Record<string, string> {
  const next: Record<string, string> = {}
  for (const draft of drafts) {
    const role = draft.role.trim()
    const modelId = draft.modelId.trim()
    if (role && modelId) {
      next[role] = modelId
    }
  }
  return next
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function normalizeEvidenceCollectionMode(value: unknown): string {
  if (value === 'local_only') {
    return 'rag'
  }
  if (value === 'web_only') {
    return 'web'
  }
  return typeof value === 'string' && value.trim().length > 0 ? value : 'hybrid'
}

function getGoalExecutionMode(goal: Record<string, unknown> | null | undefined): 'single_agent' | 'workflow' | null {
  const executionMode = goal?.execution_mode
  return executionMode === 'single_agent' || executionMode === 'workflow' ? executionMode : null
}

function WorkflowPanelBody({
  sessionId,
  workflowEnabled,
  workflowBusy,
  workflowError,
  workflowBoundRunId,
  workflowState,
  workflowConfig,
  workflowTemplate,
  workflowProtocolId,
  workflowReasoningEffort,
  workflowRunPolicyPreset,
  workflowRunPolicy,
  workflowExecutionPolicy,
  workflowEvidenceConfig,
  workflowResearchConfig,
  workflowScheduleConfig,
  workflowScheduleType,
  workflowScheduleEnabled,
  workflowProtocolOptions,
  modelOptions,
  supportedReasoningEfforts,
  effectiveProjectId,
  projects,
  workflowProjectWorkspace,
  uploadTargetDir,
  effectiveWorkflowWorkspace,
  workflowHasUnsavedChanges,
  workflowSaveState,
  workflowLastSavedLabel,
  workflowLastSaveScope,
  onWorkflowToggle,
  onWorkflowNewRun,
  onOpenRunDetail,
  onWorkflowProjectChange,
  onWorkflowFieldChange,
  onWorkflowTemplateChange,
  onWorkflowRunPolicyPresetChange,
  onWorkflowConfigPatch,
  onWorkflowSave,
  buildWorkflowScheduleConfig,
  formatWorkflowScheduleRunAt,
  defaultScheduleTimezone,
  onClose,
}: Omit<WorkflowPanelProps, 'open' | 'onOpenChange'> & {
  onClose?: () => void
}) {
  const currentSessionGoal = useSessionStore((state) => {
    if (sessionId && state.currentSessionDetail?.id === sessionId) {
      return state.currentSessionDetail.goal ?? null
    }
    if (sessionId) {
      return state.sessions.find((session) => session.id === sessionId)?.goal ?? null
    }
    return state.currentSessionDetail?.goal ?? null
  })
  const sessionGoalExecutionMode = getGoalExecutionMode(currentSessionGoal)
  const workflowUiSuppressed = sessionGoalExecutionMode === 'single_agent'
  const selectedRolesKey = serializeSelectedModelRoles(workflowConfig.selected_models_roles)
  const roleDraftScopeKey = `${sessionId ?? '__no_session__'}:${selectedRolesKey}`
  const selectedRolesSyncRef = React.useRef(roleDraftScopeKey)
  const [roleDrafts, setRoleDrafts] = React.useState<WorkflowRoleDraft[]>(() =>
    buildRoleDraftsFromSelection(workflowConfig.selected_models_roles)
  )
  const evidenceQueriesText = React.useMemo(
    () => getStringArray(workflowEvidenceConfig.queries).join('\n'),
    [workflowEvidenceConfig]
  )
  const ragMcpServersText = React.useMemo(
    () => getStringArray(workflowEvidenceConfig.rag_mcp_servers).join('\n'),
    [workflowEvidenceConfig]
  )
  const researchOutputTargets = React.useMemo(
    () => getStringArray(workflowResearchConfig.output_targets),
    [workflowResearchConfig]
  )
  const researchSourceMode = typeof workflowResearchConfig.source_mode === 'string'
    ? workflowResearchConfig.source_mode
    : 'hybrid'
  const citationPolicy = typeof workflowResearchConfig.citation_policy === 'string'
    ? workflowResearchConfig.citation_policy
    : 'claim_level_required'
  const controlledExecutionEnabled = workflowExecutionPolicy.mode === 'controlled'
  const smartModelId = typeof workflowResearchConfig.smart_model_id === 'string'
    ? workflowResearchConfig.smart_model_id
    : ''
  const localWorkerModelId = typeof workflowResearchConfig.local_worker_model_id === 'string'
    ? workflowResearchConfig.local_worker_model_id
    : ''
  const localWorkerCount = String(workflowResearchConfig.local_worker_count ?? 3)
  const debateRounds = String(workflowResearchConfig.debate_rounds ?? 2)
  const roleOptions = React.useMemo(
    () => buildWorkflowRoleOptions(workflowTemplate, workflowProtocolId, controlledExecutionEnabled),
    [controlledExecutionEnabled, workflowProtocolId, workflowTemplate]
  )
  const researchLeadRoles = React.useMemo(
    () =>
      roleOptions.filter((option) =>
        ['planner', 'judge', 'verifier', 'synthesizer', 'controller', 'evaluator'].includes(option.value)
      ),
    [roleOptions]
  )
  const researchWorkerRoles = React.useMemo(
    () =>
      roleOptions.filter((option) =>
        ['debater_a', 'debater_b', 'local_worker', 'skeptic', 'executor'].includes(option.value)
      ),
    [roleOptions]
  )
  const researchOnlyRoles = React.useMemo(
    () => roleOptions.filter((option) => !isExecutionLaneRole(option.value)),
    [roleOptions]
  )
  const executionLaneRoles = React.useMemo(
    () => roleOptions.filter((option) => isExecutionLaneRole(option.value)),
    [roleOptions]
  )
  React.useEffect(() => {
    if (selectedRolesSyncRef.current === roleDraftScopeKey) {
      return
    }
    selectedRolesSyncRef.current = roleDraftScopeKey
    setRoleDrafts(buildRoleDraftsFromSelection(workflowConfig.selected_models_roles))
  }, [roleDraftScopeKey, workflowConfig.selected_models_roles])

  const updateRoleDrafts = React.useCallback((updater: (current: WorkflowRoleDraft[]) => WorkflowRoleDraft[]) => {
    const nextDrafts = updater(roleDrafts)
    const selectedModelsRoles = buildSelectedModelRolesFromDrafts(nextDrafts)
    selectedRolesSyncRef.current = `${sessionId ?? '__no_session__'}:${serializeSelectedModelRoles(selectedModelsRoles)}`
    setRoleDrafts(nextDrafts)
    onWorkflowConfigPatch({
      selected_models_roles: selectedModelsRoles,
    })
  }, [onWorkflowConfigPatch, roleDrafts, sessionId])

  const handleSeedProtocolRoles = React.useCallback(() => {
    updateRoleDrafts(() => {
      if (workflowTemplate === 'research_debate') {
        return roleOptions.map((option) =>
          createWorkflowRoleDraft(
            option.value,
            option.defaultModel === 'lead'
              ? smartModelId
              : option.defaultModel === 'worker'
                ? localWorkerModelId
                : ''
          )
        )
      }
      return defaultRolesForProtocol(workflowProtocolId).map((role) => createWorkflowRoleDraft(role, ''))
    })
  }, [localWorkerModelId, roleOptions, smartModelId, updateRoleDrafts, workflowProtocolId, workflowTemplate])

  const handleAddRole = React.useCallback(() => {
    updateRoleDrafts((current) => {
      const nextSuggestedRole =
        roleOptions.find((option) => current.every((draft) => draft.role !== option.value))?.value ?? ''
      return [...current, createWorkflowRoleDraft(nextSuggestedRole, '')]
    })
  }, [roleOptions, updateRoleDrafts])

  const handleRoleDraftChange = React.useCallback((
    id: string,
    key: 'role' | 'modelId',
    value: string
  ) => {
    updateRoleDrafts((current) =>
      current.map((draft) => (draft.id === id ? { ...draft, [key]: value } : draft))
    )
  }, [updateRoleDrafts])

  const handleRemoveRole = React.useCallback((id: string) => {
    updateRoleDrafts((current) => current.filter((draft) => draft.id !== id))
  }, [updateRoleDrafts])

  const patchRunPolicy = React.useCallback((patch: Partial<AgentRunRunPolicy>) => {
    onWorkflowConfigPatch({
      run_policy_preset: 'custom',
      run_policy: {
        ...workflowRunPolicy,
        ...patch,
      },
    })
  }, [onWorkflowConfigPatch, workflowRunPolicy])

  const toggleResearchOutputTarget = React.useCallback(
    (target: 'research_brief' | 'dataset_package') => {
      const nextTargets = researchOutputTargets.includes(target)
        ? researchOutputTargets.filter((item) => item !== target)
        : [...researchOutputTargets, target]
      onWorkflowConfigPatch({
        research: {
          ...workflowResearchConfig,
          output_targets: nextTargets.length > 0 ? nextTargets : [target],
        },
      })
    },
    [onWorkflowConfigPatch, researchOutputTargets, workflowResearchConfig]
  )

  const saveStatusMessage = (() => {
    if (workflowSaveState === 'saving') {
      return 'Saving workflow settings...'
    }
    if (workflowSaveState === 'error') {
      return workflowError ?? 'Unable to save workflow settings.'
    }
    if (workflowHasUnsavedChanges) {
      return 'Unsaved changes'
    }
    if (workflowSaveState === 'saved') {
      if (workflowLastSaveScope === 'draft') {
        return 'Saved in this draft only. Send a chat message to persist it.'
      }
      return workflowLastSavedLabel ? `Saved at ${workflowLastSavedLabel}` : 'Settings saved'
    }
    return 'Changes are stored per chat session.'
  })()

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.18),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
              <Workflow className="h-3.5 w-3.5" />
              Workflow Desk
            </div>
            <h2 className="text-base font-semibold text-foreground">Workflow controls</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Keep this conversation bound to one orchestrated runtime without leaving the main chat.
            </p>
          </div>
          {onClose ? (
            <Button
              type="button"
              size="icon-sm"
              variant="ghost"
              onClick={onClose}
              title="Hide workflow controls"
              aria-label="Hide workflow controls"
              className="mt-0.5 rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        {workflowUiSuppressed ? (
          <div className="space-y-4">
            <PanelSectionCard
              title="Workflow override"
              description="This chat is currently bound to a single-agent goal, so workflow-native controls stay out of the main path."
            >
              <div className="space-y-3 rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-3 text-sm text-muted-foreground">
                <p>
                  Single-agent goals should stay chat-first. Workflow routing, bound-run controls, and role configuration are hidden until you explicitly prepare a workflow goal.
                </p>
                <p>
                  Use <code>/workflow &lt;request&gt;</code> when you want to promote the next long-running task into a workflow proposal.
                </p>
                <p>
                  The existing workflow settings remain stored for this chat, but they are not active while the current goal stays in single-agent mode.
                </p>
              </div>
            </PanelSectionCard>
            {onClose ? (
              <div className="flex justify-end">
                <Button type="button" variant="outline" onClick={onClose}>
                  Close
                </Button>
              </div>
            ) : null}
          </div>
        ) : (
        <div className="space-y-4">
          <PanelSectionCard
            title="Workflow mode"
            description="Route new chat turns into the workflow runtime for this session."
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-foreground">Enable workflow</p>
                  <p className="text-xs text-muted-foreground">
                    Switch between direct chat replies and the bound workflow lane.
                  </p>
                </div>
                <Switch checked={workflowEnabled} onCheckedChange={onWorkflowToggle} />
              </div>
              <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-2 text-xs text-muted-foreground">
                {workflowEnabled
                  ? 'Workflow mode is active for this chat session.'
                  : 'Workflow mode is off. Use /workflow or this switch to enable it.'}
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Bound run"
            description="This chat keeps appending to the same workflow run unless you intentionally rotate it."
          >
            <div className="space-y-3">
              <div className="flex items-center justify-between gap-2">
                <div className="min-w-0">
                  <p className="truncate text-sm font-medium text-foreground">
                    {workflowBoundRunId ?? 'No run bound yet'}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {workflowBoundRunId
                      ? `Synced events: ${workflowState.synced_run_event_count ?? 0}`
                      : 'The first workflow message will create and bind a run.'}
                  </p>
                </div>
                <Button type="button" variant="outline" size="sm" onClick={onWorkflowNewRun}>
                  <RotateCcw className="h-3.5 w-3.5" />
                  New run
                </Button>
              </div>
              {workflowBoundRunId ? (
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="rounded-full px-0"
                  onClick={() => onOpenRunDetail(workflowBoundRunId)}
                >
                  <ExternalLink className="h-3.5 w-3.5" />
                  Open run detail
                </Button>
              ) : null}
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Project / workspace"
            description="Files and execution resolve from the selected project unless you override the path."
          >
            <div className="space-y-3">
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Project</span>
                <Select
                  value={effectiveProjectId ?? '__none__'}
                  onValueChange={(value) => onWorkflowProjectChange(value === '__none__' ? null : value)}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="No project" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="__none__">No project</SelectItem>
                    {projects.map((project) => (
                      <SelectItem key={project.id} value={project.id}>
                        {project.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Workspace override</span>
                <Input
                  value={workflowState.workspace_dir_override ?? ''}
                  placeholder={workflowProjectWorkspace ?? uploadTargetDir ?? 'Use project workspace'}
                  onChange={(event) =>
                    onWorkflowFieldChange({
                      workspace_dir_override: event.target.value || null,
                      config: {
                        ...(workflowState.config ?? {}),
                        workspace_dir_override: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-2 text-xs text-muted-foreground">
                Effective workspace:{' '}
                <span className="break-all text-foreground">{effectiveWorkflowWorkspace || 'Not set'}</span>
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Protocol / reasoning"
            description="Session-scoped defaults applied whenever a new bound run is created."
          >
            <div className="space-y-3">
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Template</span>
                <Select
                  value={workflowTemplate}
                  onValueChange={(value) => onWorkflowTemplateChange(value as WorkflowTemplate)}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">Standard workflow</SelectItem>
                    <SelectItem value="research_debate">Research debate</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {workflowTemplate === 'research_debate'
                    ? 'Uses the multi-agent debate flow with Smart Judge research defaults.'
                    : 'General-purpose workflow configuration for chat-bound runs.'}
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Title</span>
                <Input
                  value={workflowConfig.title ?? ''}
                  placeholder="Optional workflow title"
                  onChange={(event) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        title: event.target.value || null,
                      },
                    })
                  }
                />
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Protocol</span>
                <Select
                  value={workflowProtocolId}
                  disabled={workflowTemplate === 'research_debate'}
                  onValueChange={(value) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        protocol_id: value as AgentRunProtocolId,
                      },
                    })
                  }
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {workflowProtocolOptions.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {workflowProtocolOptions.find((option) => option.value === workflowProtocolId)?.description}
                </p>
              </div>
              <div className="space-y-2">
                <span className="text-xs font-medium text-muted-foreground">Thinking Level</span>
                <ThinkingLevelPanelControl
                  supportedEfforts={supportedReasoningEfforts}
                  value={workflowReasoningEffort}
                  allowInherit
                  onInherit={() =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        reasoning_effort: null,
                      },
                    })
                  }
                  onChange={(next) =>
                    onWorkflowFieldChange({
                      config: {
                        ...(workflowState.config ?? {}),
                        reasoning_effort: next,
                      },
                    })
                  }
                />
              </div>
            </div>
          </PanelSectionCard>

          {workflowTemplate === 'research_debate' ? (
            <PanelSectionCard
              title="Research team"
              description="Set the default models for your research workflow, then override specific roles only when needed."
            >
              <div className="space-y-4">
                <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-3">
                  <p className="text-sm font-medium text-foreground">Default team assignment</p>
                  <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
                    Most users only need these two defaults. The Smart model is the shared default for planner,
                    judge, verifier, and synthesizer. The research worker model is the shared default for debate,
                    evidence gathering, and skeptical review.
                  </p>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Smart model</span>
                    <Select
                      value={smartModelId || '__unassigned__'}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            smart_model_id: value === '__unassigned__' ? '' : value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__unassigned__">Unassigned</SelectItem>
                        {modelOptions.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Shared default for planner, judge, verifier, and synthesizer.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Research worker model</span>
                    <Select
                      value={localWorkerModelId || '__unassigned__'}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            local_worker_model_id: value === '__unassigned__' ? '' : value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue placeholder="Select model" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="__unassigned__">Unassigned</SelectItem>
                        {modelOptions.map((model) => (
                          <SelectItem key={model.id} value={model.id}>
                            {model.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                    <p className="text-xs text-muted-foreground">
                      Used by the debate agents, research worker, and skeptic by default.
                    </p>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-white/8 bg-surface-layer/50 px-3 py-3">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-primary-300">
                      Lead responsibilities
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {researchLeadRoles.map((role) => (
                        <span
                          key={role.value}
                          className="inline-flex rounded-full border border-white/10 bg-canvas/50 px-2.5 py-1 text-[11px] text-muted-foreground"
                        >
                          {role.label}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-xl border border-white/8 bg-surface-layer/50 px-3 py-3">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-emerald-300">
                      Worker responsibilities
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {researchWorkerRoles.map((role) => (
                        <span
                          key={role.value}
                          className="inline-flex rounded-full border border-white/10 bg-canvas/50 px-2.5 py-1 text-[11px] text-muted-foreground"
                        >
                          {role.label}
                        </span>
                      ))}
                    </div>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="rounded-xl border border-emerald-400/15 bg-emerald-500/5 px-3 py-3">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-emerald-300">
                      Research-only roles
                    </p>
                    <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                      These roles can read, compare, debate, and verify evidence, but they do not write files or run commands.
                    </p>
                    <div className="mt-2 flex flex-wrap gap-2">
                      {researchOnlyRoles.map((role) => (
                        <span
                          key={role.value}
                          className="inline-flex rounded-full border border-white/10 bg-canvas/50 px-2.5 py-1 text-[11px] text-muted-foreground"
                        >
                          {role.label}
                        </span>
                      ))}
                    </div>
                  </div>
                  <div className="rounded-xl border border-amber-400/15 bg-amber-500/5 px-3 py-3">
                    <p className="text-xs font-medium uppercase tracking-[0.14em] text-amber-200">
                      Execution lane
                    </p>
                    {executionLaneRoles.length > 0 ? (
                      <>
                        <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                          Only this lane can prepare, approve, and assess write/run actions when controlled execution is enabled.
                        </p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          {executionLaneRoles.map((role) => (
                            <span
                              key={role.value}
                              className="inline-flex rounded-full border border-white/10 bg-canvas/50 px-2.5 py-1 text-[11px] text-muted-foreground"
                            >
                              {role.label}
                            </span>
                          ))}
                        </div>
                      </>
                    ) : (
                      <p className="mt-2 text-xs leading-relaxed text-muted-foreground">
                        Controlled execution is off, so this workflow stays research-only.
                      </p>
                    )}
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Parallel research workers</span>
                    <Input
                      value={localWorkerCount}
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            local_worker_count: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 3,
                          },
                        })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Higher values let the worker model explore more sources in parallel.
                    </p>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Debate rounds</span>
                    <Input
                      value={debateRounds}
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            debate_rounds: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 2,
                          },
                        })
                      }
                    />
                    <p className="text-xs text-muted-foreground">
                      Two rounds works well for most topics. Increase only when a topic needs more challenge and rebuttal.
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">Output targets</span>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      type="button"
                      variant={researchOutputTargets.includes('research_brief') ? 'secondary' : 'ghost'}
                      size="sm"
                      onClick={() => toggleResearchOutputTarget('research_brief')}
                    >
                      Research brief
                    </Button>
                    <Button
                      type="button"
                      variant={researchOutputTargets.includes('dataset_package') ? 'secondary' : 'ghost'}
                      size="sm"
                      onClick={() => toggleResearchOutputTarget('dataset_package')}
                    >
                      Dataset package
                    </Button>
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Source mode</span>
                    <Select
                      value={researchSourceMode}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            source_mode: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                        <SelectItem value="local_only">Local only</SelectItem>
                        <SelectItem value="web_first">Web first</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Citation policy</span>
                    <Select
                      value={citationPolicy}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          research: {
                            ...workflowResearchConfig,
                            citation_policy: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="claim_level_required">Claim level required</SelectItem>
                        <SelectItem value="best_effort">Best effort</SelectItem>
                        <SelectItem value="strict_fail">Strict fail</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>
            </PanelSectionCard>
          ) : null}

          <PanelSectionCard
            title={workflowTemplate === 'research_debate' ? 'Role overrides' : 'Agent roles'}
            description={
              workflowTemplate === 'research_debate'
                ? 'Pick a role preset, then override it only when that role should use a different model than the Smart or worker default.'
                : 'Pick role presets and models for each workflow role before a new bound run starts.'
            }
          >
            <div className="space-y-3">
              <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
                <p className="min-w-0 text-xs leading-relaxed text-muted-foreground sm:max-w-[14rem]">
                  {workflowTemplate === 'research_debate'
                    ? 'These preset overrides still feed `selected_models_roles`, but the shared Smart and worker defaults above cover most research roles automatically.'
                    : 'These preset mappings feed `selected_models_roles` for newly created workflow runs.'}
                </p>
                <div className="flex flex-wrap items-center gap-2 sm:justify-end">
                  <Button type="button" variant="ghost" size="sm" onClick={handleSeedProtocolRoles}>
                    {workflowTemplate === 'research_debate' ? 'Seed common roles' : 'Seed defaults'}
                  </Button>
                  <Button type="button" variant="outline" size="sm" onClick={handleAddRole}>
                    <Plus className="h-3.5 w-3.5" />
                    {workflowTemplate === 'research_debate' ? 'Add override' : 'Add role'}
                  </Button>
                </div>
              </div>
              {roleDrafts.length === 0 ? (
                <div className="rounded-xl border border-dashed border-white/12 bg-surface-layer/40 px-3 py-3 text-xs text-muted-foreground">
                  {workflowTemplate === 'research_debate'
                    ? 'No overrides yet. Your lead and worker defaults will be used automatically.'
                    : 'No roles assigned yet. Seed protocol defaults or add a custom role.'}
                </div>
              ) : (
                <div className="space-y-3">
                  {roleDrafts.map((draft, index) => {
                    const selectedRoleOption = getWorkflowRoleOption(draft.role)
                    const roleSelectValue = getRoleSelectValue(draft.role, roleOptions)
                    return (
                      <div key={draft.id} className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                        <div className="mb-3 flex items-start justify-between gap-3">
                          <div className="min-w-0">
                            <p className="text-xs font-medium text-muted-foreground">
                              {workflowTemplate === 'research_debate' ? `Override ${index + 1}` : `Role ${index + 1}`}
                            </p>
                            {selectedRoleOption ? (
                              <div className="mt-1 space-y-1">
                                <span
                                  className={`inline-flex rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.14em] ${getRoleCapabilityClassName(selectedRoleOption.value)}`}
                                >
                                  {getRoleCapabilityLabel(selectedRoleOption.value)}
                                </span>
                                <p className="text-xs leading-relaxed text-muted-foreground">
                                  {selectedRoleOption.description}
                                </p>
                                <p className="text-xs leading-relaxed text-muted-foreground">
                                  {getRoleCapabilityDescription(selectedRoleOption.value)}
                                </p>
                              </div>
                            ) : null}
                          </div>
                          <Button
                            type="button"
                            variant="ghost"
                            size="icon-sm"
                            onClick={() => handleRemoveRole(draft.id)}
                            title="Remove role"
                            aria-label="Remove role"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </Button>
                        </div>
                        <div className="grid gap-3 sm:grid-cols-2">
                          <div className="space-y-2">
                            <span className="text-xs font-medium text-muted-foreground">Role preset</span>
                            <Select
                              value={roleSelectValue}
                              onValueChange={(value) => {
                                if (value === CUSTOM_ROLE_VALUE) {
                                  handleRoleDraftChange(draft.id, 'role', '')
                                  return
                                }
                                handleRoleDraftChange(draft.id, 'role', value)
                              }}
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Choose role preset">
                                  {roleSelectValue === CUSTOM_ROLE_VALUE
                                    ? 'Custom role'
                                    : selectedRoleOption?.label}
                                </SelectValue>
                              </SelectTrigger>
                              <SelectContent>
                                {roleOptions.map((option) => (
                                  <SelectItem key={option.value} value={option.value} textValue={option.label}>
                                    <div className="flex flex-col py-0.5">
                                      <span>{option.label}</span>
                                      <span className="text-xs text-muted-foreground">{option.description}</span>
                                      <span className="text-[10px] uppercase tracking-[0.14em] text-muted-foreground/80">
                                        {getRoleCapabilityLabel(option.value)}
                                      </span>
                                    </div>
                                  </SelectItem>
                                ))}
                                <SelectItem value={CUSTOM_ROLE_VALUE} textValue="Custom role">
                                  <div className="flex flex-col py-0.5">
                                    <span>Custom role</span>
                                    <span className="text-xs text-muted-foreground">
                                      Use only when the runtime expects a role ID outside the built-in presets.
                                    </span>
                                  </div>
                                </SelectItem>
                              </SelectContent>
                            </Select>
                            {roleSelectValue === CUSTOM_ROLE_VALUE ? (
                              <div className="space-y-2">
                                <Input
                                  value={draft.role}
                                  placeholder="custom_role_id"
                                  onChange={(event) => handleRoleDraftChange(draft.id, 'role', event.target.value)}
                                />
                                <p className="text-xs text-muted-foreground">
                                  Custom roles are not labeled by the UI, so use them sparingly and name them exactly as the workflow expects.
                                </p>
                              </div>
                            ) : selectedRoleOption ? (
                              <div className="rounded-md border border-white/8 bg-canvas/35 px-3 py-2 text-xs text-muted-foreground">
                                <p>{getRoleDefaultModelLabel(selectedRoleOption.defaultModel)}</p>
                                <p className="mt-1">{getRoleCapabilityDescription(selectedRoleOption.value)}</p>
                              </div>
                            ) : null}
                          </div>
                          <div className="space-y-2">
                            <span className="text-xs font-medium text-muted-foreground">Model</span>
                            <Select
                              value={draft.modelId || '__unassigned__'}
                              onValueChange={(value) =>
                                handleRoleDraftChange(
                                  draft.id,
                                  'modelId',
                                  value === '__unassigned__' ? '' : value
                                )
                              }
                            >
                              <SelectTrigger>
                                <SelectValue placeholder="Select model" />
                              </SelectTrigger>
                              <SelectContent>
                                <SelectItem value="__unassigned__">Unassigned</SelectItem>
                                {modelOptions.map((model) => (
                                  <SelectItem key={model.id} value={model.id}>
                                    {model.label}
                                  </SelectItem>
                                ))}
                              </SelectContent>
                            </Select>
                          </div>
                        </div>
                      </div>
                    )
                  })}
                </div>
              )}
              {modelOptions.length === 0 ? (
                <p className="text-xs text-amber-300">
                  No configured models are available yet. Add or refresh models in Settings before assigning roles.
                </p>
              ) : null}
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Execution / schedule"
            description="These are defaults only. Controlled execution boundaries remain enforced by runtime policy."
          >
            <div className="space-y-3">
              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="mb-3 space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">Run policy preset</span>
                  <Select
                    value={workflowRunPolicyPreset}
                    onValueChange={(value) => onWorkflowRunPolicyPresetChange(value as WorkflowRunPolicyPreset)}
                  >
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="short">Short</SelectItem>
                      <SelectItem value="balanced">Balanced</SelectItem>
                      <SelectItem value="long">Long</SelectItem>
                      <SelectItem value="custom">Custom</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max wall clock (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.max_wall_clock_sec ?? '')}
                      placeholder="1800"
                      onChange={(event) =>
                        patchRunPolicy({
                          max_wall_clock_sec: event.target.value
                            ? Number.parseInt(event.target.value, 10)
                            : null,
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Heartbeat timeout (sec)</span>
                    <Input
                      value={String(workflowRunPolicy.heartbeat_timeout_sec ?? '')}
                      placeholder="90"
                      onChange={(event) =>
                        patchRunPolicy({
                          heartbeat_timeout_sec: event.target.value
                            ? Number.parseInt(event.target.value, 10)
                            : null,
                        })
                      }
                    />
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Checkpoint steps</span>
                    <Input
                      value={String(workflowRunPolicy.checkpoint_interval_steps ?? '')}
                      placeholder="1"
                      onChange={(event) =>
                        patchRunPolicy({
                          checkpoint_interval_steps: event.target.value
                            ? Number.parseInt(event.target.value, 10)
                            : undefined,
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max subagent failures</span>
                    <Input
                      value={String(workflowRunPolicy.max_subagent_failures_per_role ?? '')}
                      placeholder="2"
                      onChange={(event) =>
                        patchRunPolicy({
                          max_subagent_failures_per_role: event.target.value
                            ? Number.parseInt(event.target.value, 10)
                            : undefined,
                        })
                      }
                    />
                  </div>
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On budget exhausted</span>
                    <Select
                      value={workflowRunPolicy.on_budget_exhausted ?? 'finalize_partial'}
                      onValueChange={(value) =>
                        patchRunPolicy({
                          on_budget_exhausted: value as 'pause' | 'finalize_partial',
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="finalize_partial">Finalize partial</SelectItem>
                        <SelectItem value="pause">Pause</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">On subagent disconnect</span>
                    <Select
                      value={workflowRunPolicy.on_subagent_disconnect ?? 'pause'}
                      onValueChange={(value) =>
                        patchRunPolicy({
                          on_subagent_disconnect: value as 'retry_then_degrade' | 'pause' | 'fail',
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pause">Pause</SelectItem>
                        <SelectItem value="retry_then_degrade">Retry then degrade</SelectItem>
                        <SelectItem value="fail">Fail</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Controlled execution</p>
                    <p className="text-xs text-muted-foreground">
                      Keep subagents proposal-only while the controller owns runtime execution.
                    </p>
                  </div>
                  <Switch
                    checked={workflowExecutionPolicy.mode === 'controlled'}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        execution_policy: checked
                          ? {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests:
                                Number(workflowExecutionPolicy.max_execution_requests) || 1,
                              max_commands_per_request:
                                Number(workflowExecutionPolicy.max_commands_per_request) || 1,
                              default_timeout_sec:
                                Number(workflowExecutionPolicy.default_timeout_sec) || 300,
                              background_allowed:
                                workflowExecutionPolicy.background_allowed !== false,
                            }
                          : {},
                      })
                    }
                  />
                </div>
                {workflowExecutionPolicy.mode === 'controlled' ? (
                  <div className="mt-3 grid grid-cols-2 gap-3">
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max exec requests</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_execution_requests ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_execution_requests: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Max commands / request</span>
                      <Input
                        value={String(workflowExecutionPolicy.max_commands_per_request ?? '')}
                        placeholder="1"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              max_commands_per_request: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 1,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Default exec timeout</span>
                      <Input
                        value={String(workflowExecutionPolicy.default_timeout_sec ?? '')}
                        placeholder="300"
                        onChange={(event) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              default_timeout_sec: event.target.value
                                ? Number.parseInt(event.target.value, 10)
                                : 300,
                            },
                          })
                        }
                      />
                    </div>
                    <div className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Allow background execution</span>
                      <Switch
                        checked={workflowExecutionPolicy.background_allowed !== false}
                        onCheckedChange={(checked) =>
                          onWorkflowConfigPatch({
                            execution_policy: {
                              ...workflowExecutionPolicy,
                              mode: 'controlled',
                              background_allowed: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </div>
                ) : null}
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Evidence collection</p>
                    <p className="text-xs text-muted-foreground">
                      Configure retrieval defaults for new workflow runs.
                    </p>
                  </div>
                  <Switch
                    checked={workflowEvidenceConfig.enabled !== false}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        evidence: {
                          ...workflowEvidenceConfig,
                          enabled: checked,
                          mode: normalizeEvidenceCollectionMode(workflowEvidenceConfig.mode),
                        },
                      })
                    }
                  />
                </div>
                <div className="mt-3 space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">Evidence queries</span>
                  <Textarea
                    value={evidenceQueriesText}
                    rows={4}
                    placeholder={'approved deployment note\nsecurity review checklist'}
                    onChange={(event) =>
                      onWorkflowConfigPatch({
                        evidence: {
                          ...workflowEvidenceConfig,
                          enabled: workflowEvidenceConfig.enabled !== false,
                          queries: event.target.value
                            .split('\n')
                            .map((item) => item.trim())
                            .filter((item) => item.length > 0),
                        },
                      })
                    }
                  />
                </div>
                <div className="mt-3 grid grid-cols-2 gap-3">
                  {workflowTemplate !== 'research_debate' ? (
                    <div className="space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Mode</span>
                      <Select
                        value={normalizeEvidenceCollectionMode(workflowEvidenceConfig.mode)}
                        onValueChange={(value) =>
                          onWorkflowConfigPatch({
                            evidence: {
                              ...workflowEvidenceConfig,
                              enabled: workflowEvidenceConfig.enabled !== false,
                              mode: value,
                            },
                          })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="hybrid">Hybrid</SelectItem>
                          <SelectItem value="web">Web</SelectItem>
                          <SelectItem value="rag">RAG</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  ) : (
                    <div className="rounded-md border border-white/8 bg-surface-layer/50 px-3 py-2 text-xs text-muted-foreground sm:col-span-2">
                      Research debate uses the source mode above to choose between hybrid, web-first, or local-first evidence gathering.
                    </div>
                  )}
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">RAG provider</span>
                    <Select
                      value={String(workflowEvidenceConfig.rag_provider ?? 'memory')}
                      onValueChange={(value) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            rag_provider: value,
                          },
                        })
                      }
                    >
                      <SelectTrigger>
                        <SelectValue />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="memory">Memory</SelectItem>
                        <SelectItem value="mcp_resource">MCP resource</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max results / query</span>
                    <Input
                      value={String(workflowEvidenceConfig.max_results_per_query ?? '')}
                      placeholder={workflowTemplate === 'research_debate' ? '4' : '3'}
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            max_results_per_query: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : workflowTemplate === 'research_debate'
                                ? 4
                                : 3,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max fetch / query</span>
                    <Input
                      value={String(workflowEvidenceConfig.max_fetch_per_query ?? '')}
                      placeholder="2"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            max_fetch_per_query: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 2,
                          },
                        })
                      }
                    />
                  </div>
                  <div className="space-y-2">
                    <span className="text-xs font-medium text-muted-foreground">Max content chars</span>
                    <Input
                      value={String(workflowEvidenceConfig.max_content_chars ?? '')}
                      placeholder="2000"
                      onChange={(event) =>
                        onWorkflowConfigPatch({
                          evidence: {
                            ...workflowEvidenceConfig,
                            enabled: workflowEvidenceConfig.enabled !== false,
                            max_content_chars: event.target.value
                              ? Number.parseInt(event.target.value, 10)
                              : 2000,
                          },
                        })
                      }
                    />
                  </div>
                </div>
                <div className="mt-3 space-y-2">
                  <span className="text-xs font-medium text-muted-foreground">MCP servers</span>
                  <Textarea
                    value={ragMcpServersText}
                    rows={2}
                    placeholder={'docs\nknowledge-base'}
                    onChange={(event) =>
                      onWorkflowConfigPatch({
                        evidence: {
                          ...workflowEvidenceConfig,
                          enabled: workflowEvidenceConfig.enabled !== false,
                          rag_mcp_servers: event.target.value
                            .split('\n')
                            .map((item) => item.trim())
                            .filter((item) => item.length > 0),
                        },
                      })
                    }
                  />
                  <p className="text-xs text-muted-foreground">
                    Used when the RAG provider is set to <code>mcp_resource</code> or <code>auto</code>.
                  </p>
                </div>
              </div>

              <div className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <p className="text-sm font-medium text-foreground">Schedule</p>
                    <p className="text-xs text-muted-foreground">
                      Let this workflow execute via backend scheduling instead of immediate start.
                    </p>
                  </div>
                  <Switch
                    checked={workflowScheduleEnabled}
                    onCheckedChange={(checked) =>
                      onWorkflowConfigPatch({
                        schedule: buildWorkflowScheduleConfig(
                          workflowScheduleConfig,
                          workflowScheduleType,
                          checked
                        ),
                      })
                    }
                  />
                </div>
                {workflowScheduleEnabled ? (
                  <>
                    <div className="mt-3 space-y-2">
                      <span className="text-xs font-medium text-muted-foreground">Schedule type</span>
                      <Select
                        value={workflowScheduleType}
                        onValueChange={(value) =>
                          onWorkflowConfigPatch({
                            schedule: buildWorkflowScheduleConfig(
                              workflowScheduleConfig,
                              value as WorkflowScheduleType,
                              true
                            ),
                          })
                        }
                      >
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="interval">Interval</SelectItem>
                          <SelectItem value="once">One-shot</SelectItem>
                          <SelectItem value="cron">Cron</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    {workflowScheduleType === 'interval' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Interval seconds</span>
                        <Input
                          value={String(workflowScheduleConfig.interval_seconds ?? '')}
                          placeholder="3600"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                interval_seconds: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : 3600,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'once' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Run at</span>
                        <Input
                          type="datetime-local"
                          value={formatWorkflowScheduleRunAt(workflowScheduleConfig.run_at)}
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                run_at: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    {workflowScheduleType === 'cron' ? (
                      <div className="mt-3 space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Cron</span>
                        <Input
                          value={String(workflowScheduleConfig.cron ?? '')}
                          placeholder="0 9 * * 1"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                cron: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="mt-3 grid grid-cols-2 gap-3">
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Timezone</span>
                        <Input
                          value={String(workflowScheduleConfig.timezone ?? '')}
                          placeholder={defaultScheduleTimezone()}
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                timezone: event.target.value,
                              },
                            })
                          }
                        />
                      </div>
                      <div className="space-y-2">
                        <span className="text-xs font-medium text-muted-foreground">Max runs</span>
                        <Input
                          value={
                            workflowScheduleConfig.max_runs === null || workflowScheduleConfig.max_runs === undefined
                              ? ''
                              : String(workflowScheduleConfig.max_runs)
                          }
                          placeholder="Optional"
                          onChange={(event) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                max_runs: event.target.value
                                  ? Number.parseInt(event.target.value, 10)
                                  : null,
                              },
                            })
                          }
                        />
                      </div>
                    </div>
                    {workflowScheduleType !== 'once' ? (
                      <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                        <span className="text-xs text-muted-foreground">Start immediately</span>
                        <Switch
                          checked={workflowScheduleConfig.start_immediately !== false}
                          onCheckedChange={(checked) =>
                            onWorkflowConfigPatch({
                              schedule: {
                                ...workflowScheduleConfig,
                                enabled: true,
                                start_immediately: checked,
                              },
                            })
                          }
                        />
                      </div>
                    ) : null}
                    <div className="mt-3 flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2">
                      <span className="text-xs text-muted-foreground">Auto-pause on failure</span>
                      <Switch
                        checked={workflowScheduleConfig.auto_pause_on_failure !== false}
                        onCheckedChange={(checked) =>
                          onWorkflowConfigPatch({
                            schedule: {
                              ...workflowScheduleConfig,
                              enabled: true,
                              auto_pause_on_failure: checked,
                            },
                          })
                        }
                      />
                    </div>
                  </>
                ) : null}
              </div>
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Status / recovery"
            description="Full artifacts, logs, and recovery actions stay available from the run detail page."
          >
            <div className="rounded-xl border border-white/8 bg-surface-layer/60 px-3 py-3 text-xs text-muted-foreground">
              <p>Workflow busy: {workflowBusy ? 'Yes' : 'No'}</p>
              <p>Last error: {workflowError ?? 'None'}</p>
            </div>
          </PanelSectionCard>

          <div className="flex items-center justify-end gap-2 pt-1">
            <div
              aria-live="polite"
              className="mr-auto inline-flex min-h-9 items-center gap-2 rounded-full border border-white/8 bg-surface-layer/60 px-3 text-xs text-muted-foreground"
            >
              {workflowSaveState === 'saving' ? (
                <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-300" />
              ) : workflowSaveState === 'saved' && !workflowHasUnsavedChanges ? (
                <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
              ) : null}
              <span>{saveStatusMessage}</span>
            </div>
            {onClose ? (
              <Button type="button" variant="outline" onClick={onClose}>
                Close
              </Button>
            ) : null}
            <Button type="button" onClick={onWorkflowSave} disabled={workflowSaveState === 'saving'}>
              {workflowSaveState === 'saving' ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Saving...
                </>
              ) : (
                'Save settings'
              )}
            </Button>
          </div>
        </div>
        )}
      </div>
    </div>
  )
}

export function WorkflowPanel({
  open,
  onOpenChange,
  ...bodyProps
}: WorkflowPanelProps) {
  return (
    <FloatingPanelShell
      open={open}
      onOpenChange={onOpenChange}
      desktopSide="right"
      desktopWidthClass="w-[26rem]"
      desktopBreakpoint="lg"
    >
      <WorkflowPanelBody {...bodyProps} onClose={() => onOpenChange(false)} />
    </FloatingPanelShell>
  )
}
