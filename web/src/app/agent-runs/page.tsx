'use client'

import * as React from 'react'
import Link from 'next/link'
import {
  Clock3,
  FolderKanban,
  MessageSquare,
  PanelRight,
  Plus,
  RefreshCw,
  Rocket,
  Settings2,
  Trash2,
} from 'lucide-react'
import * as api from '@/lib/api'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/lib/i18n'
import { useProjectStore } from '@/lib/stores/project-store'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Switch } from '@/components/ui/switch'
import { Textarea } from '@/components/ui/textarea'
import { cn } from '@/lib/utils'

interface SubagentDraft {
  id: string
  role: string
  modelId: string
}

interface WorkflowTimelineItem {
  id: string
  role: 'system' | 'assistant' | 'operator'
  title: string
  body: string
  timestamp: string
  status?: 'ready' | 'pending' | 'success' | 'error'
  meta?: string
}

type RunTemplate = 'standard' | 'research_debate'
type RunPolicyPreset = 'short' | 'balanced' | 'long' | 'custom'
type WorkflowReasoningEffort = api.ReasoningEffort | 'auto'

const PROTOCOL_OPTIONS: Array<{
  id: api.AgentRunProtocolId
  label: string
  description: string
}> = [
  {
    id: 'teacher_student_distill',
    label: 'Teacher Student Distill',
    description: 'Teacher guides students to produce a concise output.',
  },
  {
    id: 'multi_agent_debate',
    label: 'Multi Agent Debate',
    description: 'Multiple agents debate and converge on conclusions.',
  },
  {
    id: 'dr_zero_self_evolve',
    label: 'Dr.Zero Self-Evolve',
    description: 'Proposer generates hard-but-solvable tasks; solver creates evidence-aware rollouts.',
  },
]

const WORKFLOW_REASONING_EFFORT_OPTIONS: Array<{
  value: WorkflowReasoningEffort
  label: string
  description: string
}> = [
  { value: 'auto', label: 'Auto', description: 'Let each model use its default behavior.' },
  { value: 'none', label: 'None', description: 'Disable explicit reasoning effort.' },
  { value: 'minimal', label: 'Minimal', description: 'Keep reasoning very short.' },
  { value: 'low', label: 'Low', description: 'Prefer a lighter reasoning pass.' },
  { value: 'medium', label: 'Medium', description: 'Balanced depth and latency.' },
  { value: 'high', label: 'High', description: 'Spend more effort on reasoning.' },
  { value: 'xhigh', label: 'X-High', description: 'Use the highest supported effort.' },
]

const TERMINAL_RUN_STATUSES = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'completed',
  'done',
  'error',
])

const TIMELINE_ROLE_STYLES: Record<
  WorkflowTimelineItem['role'],
  { badge: string; bubble: string; label: string }
> = {
  system: {
    badge: 'bg-primary-500/15 text-primary-200 ring-1 ring-primary-500/30',
    bubble: 'border-primary-500/20 bg-primary-500/8',
    label: 'System',
  },
  assistant: {
    badge: 'bg-emerald-500/15 text-emerald-200 ring-1 ring-emerald-500/30',
    bubble: 'border-emerald-500/20 bg-emerald-500/8',
    label: 'Workflow',
  },
  operator: {
    badge: 'bg-neutral-700/80 text-foreground ring-1 ring-border',
    bubble: 'border-border bg-surface-layer',
    label: 'Operator',
  },
}

function createTimelineItem(
  item: Omit<WorkflowTimelineItem, 'id' | 'timestamp'> & { timestamp?: string }
): WorkflowTimelineItem {
  return {
    id:
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`,
    timestamp: item.timestamp ?? new Date().toISOString(),
    ...item,
  }
}

function createInitialTimeline(): WorkflowTimelineItem[] {
  return [
    createTimelineItem({
      role: 'system',
      title: 'Workflow console ready',
      body: 'Configure the workspace, then brief the operator console with the task you want this run to execute.',
      status: 'ready',
      meta: 'Local UI timeline',
    }),
    createTimelineItem({
      role: 'assistant',
      title: 'Standing by',
      body: 'I will turn your next operator message into a workflow run and keep the detailed protocol settings compact unless you need them.',
      status: 'ready',
    }),
  ]
}

function readEventString(event: Record<string, unknown>, key: string): string | null {
  const value = event[key]
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function readEventRecord(event: Record<string, unknown>, key: string): Record<string, unknown> | null {
  const value = event[key]
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return null
  }
  return value as Record<string, unknown>
}

function eventTimestamp(event: Record<string, unknown>, fallback: string): string {
  return (
    readEventString(event, 'created_at') ??
    readEventString(event, 'timestamp') ??
    readEventString(event, 'occurred_at') ??
    fallback
  )
}

function buildTimelineItemFromEvent(
  event: Record<string, unknown>,
  fallbackTimestamp: string
): WorkflowTimelineItem | null {
  const type = readEventString(event, 'type') ?? 'event'
  const timestamp = eventTimestamp(event, fallbackTimestamp)

  if (type === 'operator_message' || type === 'guidance') {
    const payload = readEventRecord(event, 'payload')
    const content =
      readEventString(event, 'content') ??
      readEventString(event, 'guidance') ??
      readEventString(payload ?? {}, 'content')
    if (!content) {
      return null
    }
    return createTimelineItem({
      role: 'operator',
      title: type === 'guidance' ? 'Legacy guidance' : 'Operator message',
      body: content,
      timestamp,
      status: 'ready',
      meta:
        readEventString(event, 'workspace_dir') ??
        readEventString(readEventRecord(event, 'metadata') ?? {}, 'workspace_dir') ??
        undefined,
    })
  }

  if (type === 'assistant_message') {
    const content = readEventString(event, 'content')
    if (!content) {
      return null
    }
    return createTimelineItem({
      role: 'assistant',
      title: 'Workflow assistant',
      body: content,
      timestamp,
      status: 'success',
      meta: readEventString(readEventRecord(event, 'metadata') ?? {}, 'source') ?? undefined,
    })
  }

  if (type === 'artifact') {
    const artifactType = readEventString(event, 'artifact_type') ?? 'artifact'
    return createTimelineItem({
      role: 'system',
      title: 'Artifact recorded',
      body: readEventString(event, 'title') ?? `Artifact type: ${artifactType}`,
      timestamp,
      status: 'success',
      meta: artifactType,
    })
  }

  if (type === 'exec_update') {
    return createTimelineItem({
      role: 'system',
      title: 'Execution update',
      body:
        readEventString(event, 'content') ??
        readEventString(event, 'status') ??
        'Shared runtime execution state changed.',
      timestamp,
      status: 'pending',
      meta: readEventString(event, 'request_id') ?? undefined,
    })
  }

  const runLifecycleTypes = new Set([
    'run_created',
    'run_started',
    'run_status',
    'run_scheduled',
    'run_completed',
    'run_failed',
    'run_paused',
    'run_resumed',
    'run_finalized_partial',
  ])

  if (runLifecycleTypes.has(type)) {
    const status = readEventString(event, 'status')
    return createTimelineItem({
      role: 'system',
      title: type.replaceAll('_', ' '),
      body: status ? `Run status: ${status}.` : 'Workflow runtime emitted a status update.',
      timestamp,
      status:
        status === 'failed' || status === 'error'
          ? 'error'
          : status === 'completed' || status === 'succeeded'
            ? 'success'
            : 'pending',
      meta: readEventString(event, 'protocol_id') ?? undefined,
    })
  }

  return createTimelineItem({
    role: 'system',
    title: type.replaceAll('_', ' '),
    body: readEventString(event, 'content') ?? 'Workflow event captured.',
    timestamp,
    status: 'ready',
  })
}

function buildTimelineFromRun(run: api.AgentRunDetail | null): WorkflowTimelineItem[] {
  if (!run) {
    return createInitialTimeline()
  }

  const items = run.events
    .map((event) => buildTimelineItemFromEvent(event, run.updated_at || run.created_at))
    .filter((item): item is WorkflowTimelineItem => item !== null)

  if (items.length > 0) {
    return items
  }

  return [
    createTimelineItem({
      role: 'system',
      title: 'Run loaded',
      body: run.topic || run.title || 'Workflow run loaded.',
      timestamp: run.updated_at || run.created_at,
      status: 'ready',
      meta: run.run_id,
    }),
  ]
}

function createSubagentDraft(defaultModelId: string | null): SubagentDraft {
  return {
    id:
      typeof crypto !== 'undefined' && typeof crypto.randomUUID === 'function'
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random()}`,
    role: '',
    modelId: defaultModelId ?? '',
  }
}

function defaultRolesForProtocol(protocolId: api.AgentRunProtocolId): string[] {
  if (protocolId === 'dr_zero_self_evolve') {
    return ['proposer', 'solver', 'verifier']
  }
  if (protocolId === 'multi_agent_debate') {
    return ['debater_a', 'debater_b', 'judge']
  }
  return ['teacher', 'student']
}

function createProtocolSubagentDrafts(
  protocolId: api.AgentRunProtocolId,
  defaultModelId: string | null,
  count?: number
): SubagentDraft[] {
  const roles = defaultRolesForProtocol(protocolId)
  const targetCount = count ?? roles.length
  return Array.from({ length: Math.max(targetCount, 1) }, (_, index) => ({
    ...createSubagentDraft(defaultModelId),
    role: roles[index] ?? '',
  }))
}

function normalizeSubagentInputs(subagents: SubagentDraft[]): api.AgentRunSubagentInput[] {
  return subagents
    .map((item) => ({
      role: item.role.trim(),
      model_id: item.modelId.trim(),
    }))
    .filter((item) => item.role.length > 0 && item.model_id.length > 0)
}

function buildSelectedModelsRolesPayload(
  subagents: api.AgentRunSubagentInput[],
  extraByRole: Record<string, string> = {}
): Record<string, unknown> {
  const by_role: Record<string, string> = {}

  for (const item of subagents) {
    by_role[item.role] = item.model_id
  }

  Object.entries(extraByRole).forEach(([role, modelId]) => {
    const normalizedModelId = modelId.trim()
    if (role.trim() && normalizedModelId) {
      by_role[role] = normalizedModelId
    }
  })

  return {
    subagents,
    by_role,
    entries: Object.entries(by_role).map(([role, model_id]) => ({ role, model_id })),
  }
}

function buildExecutionPolicyPayload(input: {
  enabled: boolean
  maxExecutionRequests: string
  maxCommandsPerRequest: string
  defaultTimeoutSec: string
  backgroundAllowed: boolean
}): Record<string, unknown> | null {
  if (!input.enabled) {
    return null
  }

  return {
    mode: 'controlled',
    max_execution_requests: parsePositiveInteger(input.maxExecutionRequests, 1),
    max_commands_per_request: parsePositiveInteger(input.maxCommandsPerRequest, 1),
    default_timeout_sec: parsePositiveInteger(input.defaultTimeoutSec, 300),
    background_allowed: input.backgroundAllowed,
  }
}

function formatDateTime(value: string | null): string {
  if (!value) {
    return 'N/A'
  }
  const timestamp = Date.parse(value)
  if (Number.isNaN(timestamp)) {
    return value
  }
  return new Date(timestamp).toLocaleString()
}

function modelOptionLabel(model: api.ModelInfo): string {
  const provider = model.provider ?? model.backendType ?? 'unknown'
  return `${model.label || model.name || model.id} (${provider})`
}

function parsePositiveInteger(value: string, fallback: number): number {
  const numeric = Number.parseInt(value, 10)
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return fallback
  }
  return numeric
}

function defaultScheduleTimezone(): string {
  try {
    return Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC'
  } catch {
    return 'UTC'
  }
}

function sourceModeToEvidenceMode(mode: string): string {
  if (mode === 'local_only') {
    return 'rag'
  }
  if (mode === 'web_first') {
    return 'web'
  }
  return 'hybrid'
}

function statusVariant(status: string): BadgeProps['variant'] {
  const normalized = status.toLowerCase()
  if (
    normalized === 'running' ||
    normalized === 'queued' ||
    normalized === 'pending' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled' ||
    normalized === 'partial'
  ) {
    return 'warning'
  }
  if (normalized === 'succeeded' || normalized === 'completed' || normalized === 'done') {
    return 'success'
  }
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'error'
  }
  return 'neutral'
}

function translateRunStatus(status: string, t: (key: string, values?: Record<string, string | number | boolean | null | undefined>) => string): string {
  const normalized = status.toLowerCase()
  const key = `agentRuns.status.${normalized}`
  const translated = t(key)
  return translated === key ? status : translated
}

function isUnavailableError(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 404 || error.status === 405)
}

function runPolicyPresetValues(preset: RunPolicyPreset): Required<api.AgentRunRunPolicy> {
  if (preset === 'short') {
    return {
      max_wall_clock_sec: 600,
      heartbeat_timeout_sec: 45,
      checkpoint_interval_steps: 1,
      max_subagent_failures_per_role: 1,
      on_budget_exhausted: 'pause',
      on_subagent_disconnect: 'pause',
    }
  }
  if (preset === 'long') {
    return {
      max_wall_clock_sec: 5400,
      heartbeat_timeout_sec: 180,
      checkpoint_interval_steps: 2,
      max_subagent_failures_per_role: 3,
      on_budget_exhausted: 'finalize_partial',
      on_subagent_disconnect: 'retry_then_degrade',
    }
  }
  return {
    max_wall_clock_sec: 1800,
    heartbeat_timeout_sec: 90,
    checkpoint_interval_steps: 1,
    max_subagent_failures_per_role: 2,
    on_budget_exhausted: 'pause',
    on_subagent_disconnect: 'retry_then_degrade',
  }
}

export default function AgentRunsPage() {
  const { t } = useI18n()
  const projects = useProjectStore((state) => state.projects)
  const activeProjectId = useProjectStore((state) => state.activeProjectId)
  const setActiveProjectId = useProjectStore((state) => state.setActiveProjectId)
  const loadProjects = useProjectStore((state) => state.loadProjects)
  const hasLoadedProjects = useProjectStore((state) => state.hasLoadedProjects)
  const isLoadingProjects = useProjectStore((state) => state.isLoadingProjects)
  const projectStoreError = useProjectStore((state) => state.error)
  const [runs, setRuns] = React.useState<api.AgentRunSummary[]>([])
  const [models, setModels] = React.useState<api.ModelInfo[]>([])
  const [loadingRuns, setLoadingRuns] = React.useState(true)
  const [loadingModels, setLoadingModels] = React.useState(true)
  const [refreshingRuns, setRefreshingRuns] = React.useState(false)
  const [createPending, setCreatePending] = React.useState(false)
  const [runError, setRunError] = React.useState<string | null>(null)
  const [modelError, setModelError] = React.useState<string | null>(null)
  const [createError, setCreateError] = React.useState<string | null>(null)

  const [runTemplate, setRunTemplate] = React.useState<RunTemplate>('standard')
  const [protocolId, setProtocolId] = React.useState<api.AgentRunProtocolId>('teacher_student_distill')
  const [title, setTitle] = React.useState('')
  const [topic, setTopic] = React.useState('')
  const [operatorMessage, setOperatorMessage] = React.useState('')
  const [workspaceOverride, setWorkspaceOverride] = React.useState('')
  const [showAdvancedConfig, setShowAdvancedConfig] = React.useState(false)
  const [timeline, setTimeline] = React.useState<WorkflowTimelineItem[]>(() => createInitialTimeline())
  const [activeRunId, setActiveRunId] = React.useState<string | null>(null)
  const [activeRunDetail, setActiveRunDetail] = React.useState<api.AgentRunDetail | null>(null)
  const [messagePending, setMessagePending] = React.useState(false)
  const [reasoningEffort, setReasoningEffort] = React.useState<WorkflowReasoningEffort>('auto')
  const [subagents, setSubagents] = React.useState<SubagentDraft[]>([])
  const [evidenceQueriesText, setEvidenceQueriesText] = React.useState('')
  const [evidenceCollectionEnabled, setEvidenceCollectionEnabled] = React.useState(true)
  const [evidenceCollectionMode, setEvidenceCollectionMode] = React.useState('hybrid')
  const [ragProvider, setRagProvider] = React.useState('memory')
  const [ragMcpServersText, setRagMcpServersText] = React.useState('')
  const [maxResultsPerQuery, setMaxResultsPerQuery] = React.useState('3')
  const [maxFetchPerQuery, setMaxFetchPerQuery] = React.useState('2')
  const [maxContentChars, setMaxContentChars] = React.useState('2000')
  const [smartModelId, setSmartModelId] = React.useState('')
  const [localWorkerModelId, setLocalWorkerModelId] = React.useState('')
  const [researchOutputTargets, setResearchOutputTargets] = React.useState<Array<'research_brief' | 'dataset_package'>>([
    'research_brief',
    'dataset_package',
  ])
  const [researchSourceMode, setResearchSourceMode] = React.useState('hybrid')
  const [citationPolicy, setCitationPolicy] = React.useState('claim_level_required')
  const [localWorkerCount, setLocalWorkerCount] = React.useState('3')
  const [debateRounds, setDebateRounds] = React.useState('2')
  const [controlledExecutionEnabled, setControlledExecutionEnabled] = React.useState(false)
  const [maxExecutionRequests, setMaxExecutionRequests] = React.useState('1')
  const [maxCommandsPerRequest, setMaxCommandsPerRequest] = React.useState('1')
  const [defaultExecutionTimeoutSec, setDefaultExecutionTimeoutSec] = React.useState('300')
  const [controlledExecutionBackgroundAllowed, setControlledExecutionBackgroundAllowed] = React.useState(false)
  const [runPolicyPreset, setRunPolicyPreset] = React.useState<RunPolicyPreset>('balanced')
  const [maxWallClockSec, setMaxWallClockSec] = React.useState('1800')
  const [heartbeatTimeoutSec, setHeartbeatTimeoutSec] = React.useState('90')
  const [checkpointIntervalSteps, setCheckpointIntervalSteps] = React.useState('1')
  const [maxSubagentFailuresPerRole, setMaxSubagentFailuresPerRole] = React.useState('2')
  const [onBudgetExhausted, setOnBudgetExhausted] = React.useState<NonNullable<api.AgentRunRunPolicy['on_budget_exhausted']>>('pause')
  const [onSubagentDisconnect, setOnSubagentDisconnect] = React.useState<NonNullable<api.AgentRunRunPolicy['on_subagent_disconnect']>>('retry_then_degrade')
  const [scheduleEnabled, setScheduleEnabled] = React.useState(false)
  const [scheduleType, setScheduleType] = React.useState<'interval' | 'once' | 'cron'>('interval')
  const [scheduleIntervalSeconds, setScheduleIntervalSeconds] = React.useState('3600')
  const [scheduleRunAt, setScheduleRunAt] = React.useState('')
  const [scheduleCron, setScheduleCron] = React.useState('0 9 * * 1')
  const [scheduleTimezone, setScheduleTimezone] = React.useState(defaultScheduleTimezone)
  const [scheduleStartImmediately, setScheduleStartImmediately] = React.useState(true)
  const [scheduleMaxRuns, setScheduleMaxRuns] = React.useState('')
  const [scheduleAutoPauseOnFailure, setScheduleAutoPauseOnFailure] = React.useState(true)

  const defaultModelId = models[0]?.id ?? null
  const activeProject = React.useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [activeProjectId, projects]
  )
  const activeRunSummary = React.useMemo(
    () => runs.find((run) => run.run_id === activeRunId) ?? null,
    [activeRunId, runs]
  )
  const effectiveWorkspacePath = workspaceOverride.trim() || activeProject?.workspaceDir || ''
  const derivedTopic = React.useMemo(() => {
    const firstLine = operatorMessage
      .split('\n')
      .map((line) => line.trim())
      .find((line) => line.length > 0)
    if (!firstLine) {
      return topic.trim()
    }
    return firstLine.slice(0, 160)
  }, [operatorMessage, topic])
  const sidebarRuns = React.useMemo(() => runs.slice(0, 8), [runs])
  const activeRunsCount = React.useMemo(
    () => runs.filter((run) => !TERMINAL_RUN_STATUSES.has(run.status.toLowerCase())).length,
    [runs]
  )

  const loadRuns = React.useCallback(async (showRefreshing = false) => {
    if (showRefreshing) {
      setRefreshingRuns(true)
    } else {
      setLoadingRuns(true)
    }
    setRunError(null)
    try {
      const data = await api.fetchAgentRuns()
      setRuns(data)
    } catch (error) {
      if (isUnavailableError(error)) {
        setRunError(t('workflows.apiUnavailable'))
      } else {
        const detail = error instanceof Error ? error.message : t('workflows.loadError')
        setRunError(detail)
      }
    } finally {
      setLoadingRuns(false)
      setRefreshingRuns(false)
    }
  }, [t])

  const loadModels = React.useCallback(async () => {
    setLoadingModels(true)
    setModelError(null)
    try {
      const data = await api.fetchModels()
      const deduped: api.ModelInfo[] = []
      const seen = new Set<string>()
      for (const model of data) {
        if (!model.id || seen.has(model.id)) {
          continue
        }
        seen.add(model.id)
        deduped.push(model)
      }
      setModels(deduped)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to load configured models.'
      setModelError(detail)
    } finally {
      setLoadingModels(false)
    }
  }, [])

  React.useEffect(() => {
    void loadRuns()
    void loadModels()
  }, [loadModels, loadRuns])

  React.useEffect(() => {
    if (!hasLoadedProjects) {
      void loadProjects()
    }
  }, [hasLoadedProjects, loadProjects])

  React.useEffect(() => {
    if (runs.length === 0) {
      if (!activeRunId) {
        setActiveRunDetail(null)
        setTimeline(createInitialTimeline())
      }
      return
    }

    if (!activeRunId || !runs.some((run) => run.run_id === activeRunId)) {
      setActiveRunId(runs[0]?.run_id ?? null)
    }
  }, [activeRunId, runs])

  React.useEffect(() => {
    if (!activeRunId) {
      setActiveRunDetail(null)
      setTimeline(createInitialTimeline())
      return
    }

    let cancelled = false
    const loadActiveRun = async () => {
      try {
        const detail = await api.fetchAgentRun(activeRunId)
        if (cancelled) {
          return
        }
        setActiveRunDetail(detail)
        setTimeline(buildTimelineFromRun(detail))
      } catch {
        if (cancelled) {
          return
        }
        setActiveRunDetail(null)
      }
    }

    void loadActiveRun()
    return () => {
      cancelled = true
    }
  }, [activeRunId])

  React.useEffect(() => {
    if (!defaultModelId) {
      return
    }
    setSubagents((current) => {
      if (current.length === 0) {
        return createProtocolSubagentDrafts(protocolId, defaultModelId)
      }
      return current.map((subagent) =>
        subagent.modelId ? subagent : { ...subagent, modelId: defaultModelId }
      )
    })
  }, [defaultModelId, protocolId])

  React.useEffect(() => {
    if (!defaultModelId) {
      return
    }
    setSmartModelId((current) => current || defaultModelId)
    setLocalWorkerModelId((current) => current || defaultModelId)
  }, [defaultModelId])

  const activeProtocol = React.useMemo(
    () => PROTOCOL_OPTIONS.find((option) => option.id === protocolId),
    [protocolId]
  )

  const canCreate =
    models.length > 0 &&
    operatorMessage.trim().length > 0 &&
    (runTemplate === 'research_debate'
      ? smartModelId.trim().length > 0 && localWorkerModelId.trim().length > 0
      : subagents.some((item) => item.role.trim() && item.modelId.trim()))

  const handleAddSubagent = React.useCallback(() => {
    setSubagents((current) => {
      const defaults = defaultRolesForProtocol(protocolId)
      return [
        ...current,
        {
          ...createSubagentDraft(defaultModelId),
          role: defaults[current.length] ?? '',
        },
      ]
    })
  }, [defaultModelId, protocolId])

  const handleRemoveSubagent = React.useCallback((id: string) => {
    setSubagents((current) => current.filter((item) => item.id !== id))
  }, [])

  const handleSubagentChange = React.useCallback(
    <K extends keyof SubagentDraft>(id: string, key: K, value: SubagentDraft[K]) => {
      setSubagents((current) =>
        current.map((item) => (item.id === id ? { ...item, [key]: value } : item))
      )
    },
    []
  )

  const handleProtocolChange = React.useCallback((nextProtocolId: api.AgentRunProtocolId) => {
    setProtocolId(nextProtocolId)
    setSubagents((current) => {
      const currentDefaults = defaultRolesForProtocol(protocolId)
      const hasCustomRoles = current.some((item, index) => {
        const role = item.role.trim()
        if (!role) {
          return false
        }
        return role !== (currentDefaults[index] ?? '')
      })
      if (hasCustomRoles) {
        return current
      }
      const nextDrafts = createProtocolSubagentDrafts(
        nextProtocolId,
        defaultModelId,
        Math.max(current.length, defaultRolesForProtocol(nextProtocolId).length)
      )
      return nextDrafts.map((draft, index) => ({
        ...draft,
        modelId: current[index]?.modelId || draft.modelId,
      }))
    })
  }, [defaultModelId, protocolId])

  const handleTemplateChange = React.useCallback((nextTemplate: RunTemplate) => {
    setRunTemplate(nextTemplate)
    if (nextTemplate === 'research_debate') {
      setProtocolId('multi_agent_debate')
      setSubagents((current) =>
        current.length > 0
          ? current
          : createProtocolSubagentDrafts('multi_agent_debate', defaultModelId)
      )
    }
  }, [defaultModelId])

  const handleRunPolicyPresetChange = React.useCallback((preset: RunPolicyPreset) => {
    setRunPolicyPreset(preset)
    if (preset === 'custom') {
      return
    }
    const values = runPolicyPresetValues(preset)
    setMaxWallClockSec(String(values.max_wall_clock_sec))
    setHeartbeatTimeoutSec(String(values.heartbeat_timeout_sec))
    setCheckpointIntervalSteps(String(values.checkpoint_interval_steps))
    setMaxSubagentFailuresPerRole(String(values.max_subagent_failures_per_role))
    setOnBudgetExhausted(values.on_budget_exhausted)
    setOnSubagentDisconnect(values.on_subagent_disconnect)
  }, [])

  const toggleResearchOutputTarget = React.useCallback(
    (target: 'research_brief' | 'dataset_package') => {
      setResearchOutputTargets((current) => {
        const next = current.includes(target)
          ? current.filter((item) => item !== target)
          : [...current, target]
        return next.length > 0 ? next : [target]
      })
    },
    []
  )

  const handleCreate = React.useCallback(async () => {
    setCreateError(null)
    const normalizedSubagents = normalizeSubagentInputs(subagents)
    const normalizedOperatorMessage = operatorMessage.trim()
    const normalizedTopic = topic.trim() || derivedTopic

    if (!normalizedOperatorMessage) {
      setCreateError('Write an operator message before creating a run.')
      return
    }
    if (normalizedSubagents.length === 0) {
      setCreateError('Add at least one subagent with role and model.')
      return
    }

    const operatorTimelineEntry = createTimelineItem({
      role: 'operator',
      title: title.trim() || normalizedTopic || 'New workflow request',
      body: normalizedOperatorMessage,
      status: 'ready',
      meta: effectiveWorkspacePath ? `Workspace: ${effectiveWorkspacePath}` : undefined,
    })
    const pendingAssistantEntry = createTimelineItem({
      role: 'assistant',
      title: 'Creating run',
      body: scheduleEnabled
        ? 'Saving the workflow and handing execution off to the scheduler.'
        : 'Creating the workflow run and preparing an optimistic start.',
      status: 'pending',
      meta: normalizedTopic || undefined,
    })
    const pendingAssistantId = pendingAssistantEntry.id
    setTimeline((current) => [...current, operatorTimelineEntry, pendingAssistantEntry])

    setCreatePending(true)
    try {
      const evidenceQueries = evidenceQueriesText
        .split('\n')
        .map((item) => item.trim())
        .filter((item) => item.length > 0)
      const ragMcpServers = ragMcpServersText
        .split('\n')
        .map((item) => item.trim())
        .filter((item) => item.length > 0)
      const executionPolicy = buildExecutionPolicyPayload({
        enabled: controlledExecutionEnabled,
        maxExecutionRequests,
        maxCommandsPerRequest,
        defaultTimeoutSec: defaultExecutionTimeoutSec,
        backgroundAllowed: controlledExecutionBackgroundAllowed,
      })
      const fallbackControlledModelId =
        normalizedSubagents[0]?.model_id || defaultModelId?.trim() || ''
      const selectedModelsRoles =
        runTemplate === 'research_debate'
          ? buildSelectedModelsRolesPayload([], {
              debater_a: localWorkerModelId.trim(),
              debater_b: localWorkerModelId.trim(),
              judge: smartModelId.trim(),
              verifier: smartModelId.trim(),
              planner: smartModelId.trim(),
              synthesizer: smartModelId.trim(),
              local_worker: localWorkerModelId.trim(),
              skeptic: localWorkerModelId.trim(),
              ...(controlledExecutionEnabled
                ? {
                    executor: localWorkerModelId.trim() || smartModelId.trim(),
                    controller: smartModelId.trim(),
                    evaluator: smartModelId.trim(),
                  }
                : {}),
            })
          : controlledExecutionEnabled
            ? buildSelectedModelsRolesPayload(normalizedSubagents, {
                planner: fallbackControlledModelId,
                executor: fallbackControlledModelId,
                controller: fallbackControlledModelId,
                evaluator: fallbackControlledModelId,
              })
            : undefined
      const schedule = (() => {
        if (!scheduleEnabled) {
          return {}
        }
        const timezone = scheduleTimezone.trim() || 'UTC'
        if (scheduleType === 'once') {
          if (!scheduleRunAt) {
            throw new Error('Select a run time for one-shot automation.')
          }
          const timestamp = new Date(scheduleRunAt)
          if (Number.isNaN(timestamp.getTime())) {
            throw new Error('One-shot run time is invalid.')
          }
          return {
            enabled: true,
            run_at: timestamp.toISOString(),
            timezone,
            max_runs: parsePositiveInteger(scheduleMaxRuns, 1),
            auto_pause_on_failure: scheduleAutoPauseOnFailure,
          }
        }
        if (scheduleType === 'cron') {
          const cron = scheduleCron.trim()
          if (!cron) {
            throw new Error('Enter a cron expression for recurring automation.')
          }
          return {
            enabled: true,
            cron,
            timezone,
            start_immediately: scheduleStartImmediately,
            max_runs: scheduleMaxRuns.trim() ? parsePositiveInteger(scheduleMaxRuns, 1) : null,
            auto_pause_on_failure: scheduleAutoPauseOnFailure,
          }
        }
        return {
          enabled: true,
          interval_seconds: parsePositiveInteger(scheduleIntervalSeconds, 3600),
          timezone,
          start_immediately: scheduleStartImmediately,
          max_runs: scheduleMaxRuns.trim() ? parsePositiveInteger(scheduleMaxRuns, 1) : null,
          auto_pause_on_failure: scheduleAutoPauseOnFailure,
        }
      })()
      const created = await api.createAgentRun({
        protocol_id: runTemplate === 'research_debate' ? 'multi_agent_debate' : protocolId,
        title: title.trim() || null,
        topic: normalizedTopic || null,
        projectId: activeProject?.id ?? null,
        workspaceDir: effectiveWorkspacePath || null,
        reasoning_effort: reasoningEffort === 'auto' ? null : reasoningEffort,
        subagents: runTemplate === 'research_debate' ? [] : normalizedSubagents,
        selected_models_roles: selectedModelsRoles,
        run_policy: {
          max_wall_clock_sec: parsePositiveInteger(maxWallClockSec, 1800),
          heartbeat_timeout_sec: parsePositiveInteger(heartbeatTimeoutSec, 90),
          checkpoint_interval_steps: parsePositiveInteger(checkpointIntervalSteps, 1),
          max_subagent_failures_per_role: parsePositiveInteger(maxSubagentFailuresPerRole, 2),
          on_budget_exhausted: onBudgetExhausted,
          on_subagent_disconnect: onSubagentDisconnect,
        },
        schedule,
        summary: {
          operator_message: normalizedOperatorMessage,
          ...(activeProject
            ? {
                project: {
                  id: activeProject.id,
                  name: activeProject.name,
                  workspace_dir: activeProject.workspaceDir,
                },
              }
            : {}),
          ...(effectiveWorkspacePath ? { workspace_dir: effectiveWorkspacePath } : {}),
          ...(evidenceQueries.length > 0 ? { evidence_queries: evidenceQueries } : {}),
          ...(protocolId === 'dr_zero_self_evolve' && runTemplate !== 'research_debate'
            ? {
                protocol_config: {
                  iterations: 1,
                  proposal_sample_size: 3,
                  solver_rollouts_per_task: 1,
                  proposer_role_id: 'proposer',
                  solver_role_id: 'solver',
                  verifier_role_id: 'verifier',
                },
              }
            : {}),
          ...(protocolId === 'multi_agent_debate' || runTemplate === 'research_debate'
            ? { protocol_config: { rounds: parsePositiveInteger(debateRounds, 2) } }
            : {}),
          ...(executionPolicy ? { execution_policy: executionPolicy } : {}),
        },
        evaluation_policy: {
          evidence_collection: {
            enabled: evidenceCollectionEnabled,
            mode:
              runTemplate === 'research_debate'
                ? sourceModeToEvidenceMode(researchSourceMode)
                : evidenceCollectionMode,
            rag_provider: ragProvider,
            rag_mcp_servers: ragMcpServers,
            max_results_per_query:
              runTemplate === 'research_debate'
                ? parsePositiveInteger(maxResultsPerQuery, 4)
                : parsePositiveInteger(maxResultsPerQuery, 3),
            max_fetch_per_query: parsePositiveInteger(maxFetchPerQuery, 2),
            max_content_chars: parsePositiveInteger(maxContentChars, 2000),
          },
          ...(runTemplate === 'research_debate'
            ? {
                research: {
                  enabled: true,
                  preset: 'smart_judge_research_debate',
                  output_targets: researchOutputTargets,
                  source_mode: researchSourceMode,
                  citation_policy: citationPolicy,
                  local_worker_count: parsePositiveInteger(localWorkerCount, 3),
                  local_worker_count_max: 6,
                  max_research_queries: Math.max(
                    parsePositiveInteger(maxResultsPerQuery, 4),
                    evidenceQueries.length || 1
                  ),
                  max_sources_per_query: parsePositiveInteger(maxResultsPerQuery, 4),
                  debate_rounds: parsePositiveInteger(debateRounds, 2),
                },
              }
            : {}),
        },
      })
      if (created.run_id) {
        try {
          const appendedDetail = await api.appendAgentRunMessage(created.run_id, {
            role: 'operator',
            content: normalizedOperatorMessage,
            projectId: activeProject?.id ?? null,
            workspaceDir: effectiveWorkspacePath || null,
            metadata: {
              channel: 'workflow-chat',
              topic: normalizedTopic || null,
            },
          })
          setActiveRunId(created.run_id)
          setActiveRunDetail(appendedDetail)
          setTimeline(buildTimelineFromRun(appendedDetail))
          if (!scheduleEnabled) {
            const started = await api.startAgentRun(created.run_id)
            setRuns((current) =>
              [started, ...current.filter((run) => run.run_id !== started.run_id)].sort(
                (left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at)
              )
            )
          } else {
            setRuns((current) =>
              [created, ...current.filter((run) => run.run_id !== created.run_id)].sort(
                (left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at)
              )
            )
          }
          const refreshedDetail = await api.fetchAgentRun(created.run_id)
          setActiveRunDetail(refreshedDetail)
          setTimeline(buildTimelineFromRun(refreshedDetail))
          setTimeline((current) =>
            current.map((item) =>
              item.id === pendingAssistantId
                ? {
                    ...item,
                    title: scheduleEnabled ? 'Run scheduled' : 'Run started',
                    body: scheduleEnabled
                      ? 'The workflow was created and queued with the selected automation schedule.'
                      : 'The workflow was created successfully and an optimistic start signal was sent.',
                    status: 'success',
                    meta: created.run_id,
                  }
                : item
            )
          )
        } catch (startError) {
          if (!isUnavailableError(startError)) {
            const detail = startError instanceof Error
              ? startError.message
              : scheduleEnabled
                ? 'Run was created but scheduler could not be initialized.'
                : 'Run was created but could not be auto-started.'
            setCreateError(detail)
            setTimeline((current) =>
              current.map((item) =>
                item.id === pendingAssistantId
                  ? {
                      ...item,
                      title: 'Run created with warnings',
                      body: detail,
                      status: 'error',
                      meta: created.run_id,
                    }
                  : item
              )
            )
          }
        }
        setOperatorMessage('')
        setTopic('')
        setTitle('')
        setWorkspaceOverride('')
        return
      }
      await loadRuns()
      setTitle('')
      setTopic('')
      setOperatorMessage('')
      setWorkspaceOverride('')
      setReasoningEffort('auto')
      setSubagents(createProtocolSubagentDrafts(protocolId, defaultModelId))
      setEvidenceQueriesText('')
      setRagMcpServersText('')
      setRunTemplate('standard')
      setSmartModelId(defaultModelId ?? '')
      setLocalWorkerModelId(defaultModelId ?? '')
      setResearchOutputTargets(['research_brief', 'dataset_package'])
      setResearchSourceMode('hybrid')
      setCitationPolicy('claim_level_required')
      setLocalWorkerCount('3')
      setDebateRounds('2')
      setControlledExecutionEnabled(false)
      setMaxExecutionRequests('1')
      setMaxCommandsPerRequest('1')
      setDefaultExecutionTimeoutSec('300')
      setControlledExecutionBackgroundAllowed(false)
      setRunPolicyPreset('balanced')
      setMaxWallClockSec('1800')
      setHeartbeatTimeoutSec('90')
      setCheckpointIntervalSteps('1')
      setMaxSubagentFailuresPerRole('2')
      setOnBudgetExhausted('pause')
      setOnSubagentDisconnect('retry_then_degrade')
      setScheduleEnabled(false)
      setScheduleType('interval')
      setScheduleIntervalSeconds('3600')
      setScheduleRunAt('')
      setScheduleCron('0 9 * * 1')
      setScheduleTimezone(defaultScheduleTimezone())
      setScheduleStartImmediately(true)
      setScheduleMaxRuns('')
      setScheduleAutoPauseOnFailure(true)
      setTimeline((current) =>
        current.map((item) =>
          item.id === pendingAssistantId
            ? {
                ...item,
                title: 'Run created',
                body: 'The workflow was created successfully.',
                status: 'success',
              }
            : item
        )
      )
    } catch (error) {
      if (isUnavailableError(error)) {
        setCreateError(t('workflows.apiUnavailable'))
      } else {
        const detail = error instanceof Error ? error.message : t('workflows.createError')
        setCreateError(detail)
        setTimeline((current) =>
          current.map((item) =>
            item.id === pendingAssistantId
              ? {
                  ...item,
                  title: 'Run creation failed',
                  body: detail,
                  status: 'error',
                }
              : item
          )
        )
      }
    } finally {
      setCreatePending(false)
    }
  }, [
    activeProject,
    citationPolicy,
    controlledExecutionBackgroundAllowed,
    controlledExecutionEnabled,
    defaultModelId,
    defaultExecutionTimeoutSec,
    debateRounds,
    derivedTopic,
    evidenceCollectionEnabled,
    evidenceCollectionMode,
    evidenceQueriesText,
    effectiveWorkspacePath,
    localWorkerCount,
    localWorkerModelId,
    loadRuns,
    maxCommandsPerRequest,
    maxExecutionRequests,
    maxContentChars,
    maxFetchPerQuery,
    maxResultsPerQuery,
    maxSubagentFailuresPerRole,
    maxWallClockSec,
    heartbeatTimeoutSec,
    checkpointIntervalSteps,
    onBudgetExhausted,
    onSubagentDisconnect,
    protocolId,
    ragMcpServersText,
    ragProvider,
    reasoningEffort,
    researchOutputTargets,
    researchSourceMode,
    operatorMessage,
    runTemplate,
    scheduleCron,
    scheduleEnabled,
    scheduleIntervalSeconds,
    scheduleMaxRuns,
    scheduleAutoPauseOnFailure,
    scheduleRunAt,
    scheduleStartImmediately,
    scheduleTimezone,
    scheduleType,
    subagents,
    smartModelId,
    t,
    title,
    topic,
  ])

  const activeRunProjectId = activeRunSummary?.project_id ?? null
  const activeRunTopic = activeRunSummary?.topic ?? null
  const activeRunWorkspaceDir = activeRunSummary?.workspace_dir ?? null

  const handleSendMessage = React.useCallback(async () => {
    if (!activeRunId) {
      await handleCreate()
      return
    }

    const content = operatorMessage.trim()
    if (!content) {
      setCreateError('Write an operator message before sending.')
      return
    }

    setCreateError(null)
    setMessagePending(true)
    try {
      const detail = await api.appendAgentRunMessage(activeRunId, {
        role: 'operator',
        content,
        projectId: activeProject?.id ?? activeRunProjectId,
        workspaceDir: effectiveWorkspacePath || activeRunWorkspaceDir,
        metadata: {
          channel: 'workflow-chat',
          topic: derivedTopic || activeRunTopic,
        },
      })
      setActiveRunDetail(detail)
      setTimeline(buildTimelineFromRun(detail))
      setRuns((current) =>
        [detail, ...current.filter((run) => run.run_id !== detail.run_id)].sort(
          (left, right) => Date.parse(right.updated_at) - Date.parse(left.updated_at)
        )
      )
      setOperatorMessage('')
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Unable to append workflow message.'
      setCreateError(detail)
    } finally {
      setMessagePending(false)
    }
  }, [
    activeProject,
    activeRunId,
    activeRunProjectId,
    activeRunTopic,
    activeRunWorkspaceDir,
    derivedTopic,
    effectiveWorkspacePath,
    handleCreate,
    operatorMessage,
  ])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
          <div className="space-y-2">
            <div className="flex flex-wrap items-center gap-2">
              <Badge variant="outline">Chat-first workflow</Badge>
              <Badge variant={activeRunSummary ? statusVariant(activeRunSummary.status) : 'neutral'}>
                {activeRunSummary ? translateRunStatus(activeRunSummary.status, t) : 'No active run'}
              </Badge>
              <Badge variant="outline">{activeRunsCount} active</Badge>
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">{t('workflows.title')}</h1>
              <p className="mt-0.5 text-sm text-muted-foreground">{t('workflows.description')}</p>
            </div>
          </div>
          <div className="flex flex-wrap items-center gap-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setShowAdvancedConfig((current) => !current)}
            >
              <Settings2 className="h-3.5 w-3.5" />
              {showAdvancedConfig ? 'Hide advanced config' : 'Show advanced config'}
            </Button>
            <Button variant="secondary" size="sm" onClick={() => void loadRuns(true)} loading={refreshingRuns}>
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh
            </Button>
          </div>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
          <div className="space-y-6">
            <Card className="overflow-hidden">
              <CardHeader className="border-b border-border/80">
                <CardTitle className="flex items-center gap-2">
                  <MessageSquare className="h-4 w-4" />
                  Workflow Conversation
                </CardTitle>
                <CardDescription>
                  Talk to the workflow main agent here. The right sidebar keeps the run machinery visible without making it the primary surface.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-5 p-5">
                <div className="grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(220px,280px)]">
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Operator message
                      </label>
                      <Textarea
                        value={operatorMessage}
                        onChange={(event) => setOperatorMessage(event.target.value)}
                        placeholder="Describe the task, constraints, and what the workflow should do next."
                        rows={5}
                      />
                      <p className="text-xs text-muted-foreground">
                        Your first message creates the workflow shell and the first run. Later messages continue the active run conversation.
                      </p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Title
                      </label>
                      <Input
                        value={title}
                        onChange={(event) => setTitle(event.target.value)}
                        placeholder="Optional workflow title"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Topic summary
                      </label>
                      <Input
                        value={topic}
                        onChange={(event) => setTopic(event.target.value)}
                        placeholder="Optional summary. Leave blank to derive it from the first operator message."
                      />
                      <p className="text-xs text-muted-foreground">
                        Derived topic: <span className="text-foreground">{derivedTopic || 'Not derived yet'}</span>
                      </p>
                    </div>
                  </div>

                  <div className="space-y-4 rounded-xl border border-border bg-surface-layer p-4">
                    <div className="flex items-center gap-2">
                      <FolderKanban className="h-4 w-4 text-primary-300" />
                      <div>
                        <h3 className="text-sm font-semibold text-foreground">Workspace Control</h3>
                        <p className="text-xs text-muted-foreground">
                          Project + path determine where the workflow will read, write, and execute.
                        </p>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Project
                      </label>
                      <Select
                        value={activeProjectId ?? '__none__'}
                        onValueChange={(value) => setActiveProjectId(value === '__none__' ? null : value)}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Choose a project" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="__none__">No project binding</SelectItem>
                          {projects.map((project) => (
                            <SelectItem key={project.id} value={project.id}>
                              {project.name}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        {isLoadingProjects
                          ? 'Loading projects...'
                          : activeProject
                            ? `Bound project workspace: ${activeProject.workspaceDir}`
                            : 'No project selected. You can still use a direct workspace path override.'}
                      </p>
                      {projectStoreError ? (
                        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                          {projectStoreError}
                        </div>
                      ) : null}
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Workspace path override
                      </label>
                      <Input
                        value={workspaceOverride}
                        onChange={(event) => setWorkspaceOverride(event.target.value)}
                        placeholder="Optional absolute path"
                      />
                      <p className="rounded-md border border-border/80 bg-background px-3 py-2 text-xs text-muted-foreground">
                        Effective workspace:
                        <span className="ml-1 font-medium text-foreground">
                          {effectiveWorkspacePath || 'Not set'}
                        </span>
                      </p>
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Active run target
                      </label>
                      <p className="rounded-md border border-border/80 bg-background px-3 py-2 text-xs text-muted-foreground">
                        {activeRunSummary
                          ? `${activeRunSummary.title || activeRunSummary.topic || activeRunSummary.run_id} (${activeRunSummary.run_id})`
                          : 'No active run selected. Sending a message will create one.'}
                      </p>
                    </div>
                  </div>
                </div>

                <div className="flex flex-wrap items-center gap-3">
                  <Button
                    variant="primary"
                    size="md"
                    onClick={() => void handleSendMessage()}
                    loading={createPending || messagePending}
                    disabled={!operatorMessage.trim() || (!activeRunId && !canCreate)}
                  >
                    <Rocket className="h-4 w-4" />
                    {activeRunId ? 'Send to active run' : 'Create run from message'}
                  </Button>
                  {activeRunSummary ? (
                    <Link
                      href={`/agent-runs/${encodeURIComponent(activeRunSummary.run_id)}`}
                      className="text-sm text-primary-300 underline-offset-4 hover:underline"
                    >
                      Open operator console
                    </Link>
                  ) : null}
                </div>

                {createError ? (
                  <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    {createError}
                  </div>
                ) : null}

                <Separator />

                <div className="space-y-4">
                  <div className="flex items-center justify-between gap-3">
                    <div>
                      <h3 className="text-sm font-semibold text-foreground">Conversation timeline</h3>
                      <p className="text-xs text-muted-foreground">
                        Operator and assistant messages are reconstructed from agent run events.
                      </p>
                    </div>
                    {activeRunDetail ? (
                      <Badge variant="outline">
                        Events: {activeRunDetail.events.length}
                      </Badge>
                    ) : null}
                  </div>

                  <div className="space-y-3">
                    {timeline.map((item) => {
                      const style = TIMELINE_ROLE_STYLES[item.role]
                      return (
                        <div
                          key={item.id}
                          className={cn(
                            'rounded-xl border p-4 shadow-sm',
                            style.bubble,
                            item.status === 'error' && 'border-destructive/40 bg-destructive/10',
                            item.status === 'pending' && 'border-warning/40 bg-warning/10'
                          )}
                        >
                          <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                            <div className="flex items-center gap-2">
                              <span className={cn('rounded-full px-2 py-0.5 text-[11px] font-medium', style.badge)}>
                                {style.label}
                              </span>
                              <span className="text-sm font-semibold text-foreground">{item.title}</span>
                            </div>
                            <span className="text-[11px] text-muted-foreground">
                              {formatDateTime(item.timestamp)}
                            </span>
                          </div>
                          <p className="whitespace-pre-wrap text-sm text-muted-foreground">{item.body}</p>
                          {item.meta ? (
                            <p className="mt-2 text-[11px] uppercase tracking-wide text-muted-foreground">
                              {item.meta}
                            </p>
                          ) : null}
                        </div>
                      )
                    })}
                  </div>
                </div>
              </CardContent>
            </Card>

            {showAdvancedConfig ? (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Settings2 className="h-4 w-4" />
                    Advanced Workflow Configuration
                  </CardTitle>
                  <CardDescription>
                    Protocol, subagents, evidence, execution policy, and automation settings stay available here but no longer dominate the main operator flow.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Template</label>
                <Select value={runTemplate} onValueChange={(value) => handleTemplateChange(value as RunTemplate)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select template" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="standard">Standard Workflow</SelectItem>
                    <SelectItem value="research_debate">Research Debate</SelectItem>
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {runTemplate === 'research_debate'
                    ? 'Uses the multi_agent_debate protocol with Smart Judge research settings.'
                    : 'Build a general-purpose multi-agent workflow.'}
                </p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Protocol</label>
                <Select
                  value={protocolId}
                  onValueChange={(value) => handleProtocolChange(value as api.AgentRunProtocolId)}
                  disabled={runTemplate === 'research_debate'}
                >
                  <SelectTrigger>
                    <SelectValue placeholder="Select protocol" />
                  </SelectTrigger>
                  <SelectContent>
                    {PROTOCOL_OPTIONS.map((option) => (
                      <SelectItem key={option.id} value={option.id}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">{activeProtocol?.description}</p>
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Title</label>
                <Input
                  value={title}
                  onChange={(event) => setTitle(event.target.value)}
                  placeholder="Optional run title"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Topic</label>
                <Input
                  value={topic}
                  onChange={(event) => setTopic(event.target.value)}
                  placeholder="What should agents work on?"
                />
              </div>

              <div className="space-y-2">
                <label className="text-sm font-medium text-foreground">Thinking Level</label>
                <Select value={reasoningEffort} onValueChange={(value) => setReasoningEffort(value as WorkflowReasoningEffort)}>
                  <SelectTrigger>
                    <SelectValue placeholder="Select thinking level" />
                  </SelectTrigger>
                  <SelectContent>
                    {WORKFLOW_REASONING_EFFORT_OPTIONS.map((option) => (
                      <SelectItem key={option.value} value={option.value}>
                        {option.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
                <p className="text-xs text-muted-foreground">
                  {
                    WORKFLOW_REASONING_EFFORT_OPTIONS.find((option) => option.value === reasoningEffort)
                      ?.description
                  } Applies globally to the workflow. Models that do not support it ignore the setting.
                </p>
              </div>

              {runTemplate === 'research_debate' ? (
                <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">Smart Model</label>
                      <Select
                        value={smartModelId}
                        onValueChange={setSmartModelId}
                        disabled={loadingModels || models.length === 0}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select smart model" />
                        </SelectTrigger>
                        <SelectContent>
                          {models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {modelOptionLabel(model)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Used for planner, judge, verifier, and synthesizer roles.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-sm font-medium text-foreground">Local Worker Model</label>
                      <Select
                        value={localWorkerModelId}
                        onValueChange={setLocalWorkerModelId}
                        disabled={loadingModels || models.length === 0}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select worker model" />
                        </SelectTrigger>
                        <SelectContent>
                          {models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {modelOptionLabel(model)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                      <p className="text-xs text-muted-foreground">
                        Used for debate roles, local worker fan-out, and skeptic passes.
                      </p>
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Local Worker Count
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={6}
                        value={localWorkerCount}
                        onChange={(event) => setLocalWorkerCount(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Debate Rounds
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={8}
                        value={debateRounds}
                        onChange={(event) => setDebateRounds(event.target.value)}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium text-foreground">Output Targets</label>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        variant={researchOutputTargets.includes('research_brief') ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => toggleResearchOutputTarget('research_brief')}
                      >
                        Research Brief
                      </Button>
                      <Button
                        type="button"
                        variant={researchOutputTargets.includes('dataset_package') ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => toggleResearchOutputTarget('dataset_package')}
                      >
                        Dataset Package
                      </Button>
                    </div>
                  </div>

                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Source Mode
                      </label>
                      <Select value={researchSourceMode} onValueChange={setResearchSourceMode}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select source mode" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="hybrid">Hybrid</SelectItem>
                          <SelectItem value="local_only">Local Only</SelectItem>
                          <SelectItem value="web_first">Web First</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Citation Strictness
                      </label>
                      <Select value={citationPolicy} onValueChange={setCitationPolicy}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select citation policy" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="claim_level_required">Claim Level Required</SelectItem>
                          <SelectItem value="best_effort">Best Effort</SelectItem>
                          <SelectItem value="strict_fail">Strict Fail</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <label className="text-sm font-medium text-foreground">Subagents</label>
                    <Button type="button" variant="ghost" size="sm" onClick={handleAddSubagent}>
                      <Plus className="h-3.5 w-3.5" />
                      Add
                    </Button>
                  </div>

                  {subagents.length === 0 ? (
                    <div className="rounded-md border border-dashed border-border px-3 py-2 text-xs text-muted-foreground">
                      Add at least one subagent.
                    </div>
                  ) : null}

                  {subagents.map((subagent, index) => (
                    <div key={subagent.id} className="space-y-2 rounded-lg border border-border bg-surface-layer p-3">
                      <div className="flex items-center justify-between">
                        <p className="text-xs font-medium text-muted-foreground">Subagent {index + 1}</p>
                        <Button
                          type="button"
                          variant="ghost"
                          size="icon-sm"
                          onClick={() => handleRemoveSubagent(subagent.id)}
                          title="Remove subagent"
                        >
                          <Trash2 className="h-3.5 w-3.5" />
                        </Button>
                      </div>
                      <Input
                        value={subagent.role}
                        onChange={(event) => handleSubagentChange(subagent.id, 'role', event.target.value)}
                        placeholder="Role (e.g. proposer, critic)"
                      />
                      <Select
                        value={subagent.modelId}
                        onValueChange={(value) => handleSubagentChange(subagent.id, 'modelId', value)}
                        disabled={loadingModels || models.length === 0}
                      >
                        <SelectTrigger>
                          <SelectValue placeholder="Select model" />
                        </SelectTrigger>
                        <SelectContent>
                          {models.map((model) => (
                            <SelectItem key={model.id} value={model.id}>
                              {modelOptionLabel(model)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>
                  ))}
                </div>
              )}

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                <div className="flex items-center justify-between gap-3">
                  <div className="space-y-1">
                    <label className="text-sm font-medium text-foreground">Controlled Execution Capability</label>
                    <p className="text-xs text-muted-foreground">
                      Attach shared planner, executor, controller, and evaluator roles without switching the workflow protocol.
                    </p>
                  </div>
                  <Switch
                    checked={controlledExecutionEnabled}
                    onCheckedChange={(checked) => setControlledExecutionEnabled(Boolean(checked))}
                  />
                </div>

                {controlledExecutionEnabled ? (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Max Requests
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={20}
                        value={maxExecutionRequests}
                        onChange={(event) => setMaxExecutionRequests(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Max Commands / Request
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={5}
                        value={maxCommandsPerRequest}
                        onChange={(event) => setMaxCommandsPerRequest(event.target.value)}
                      />
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Default Timeout (sec)
                      </label>
                      <Input
                        type="number"
                        min={1}
                        max={86400}
                        value={defaultExecutionTimeoutSec}
                        onChange={(event) => setDefaultExecutionTimeoutSec(event.target.value)}
                      />
                    </div>
                    <label className="flex items-center justify-between gap-3 rounded-md border border-border px-3 py-2 text-xs text-muted-foreground">
                      <span>Allow background commands</span>
                      <Switch
                        checked={controlledExecutionBackgroundAllowed}
                        onCheckedChange={(checked) => setControlledExecutionBackgroundAllowed(Boolean(checked))}
                      />
                    </label>
                    <p className="text-xs text-muted-foreground sm:col-span-2">
                      {runTemplate === 'research_debate'
                        ? 'Research Debate keeps the multi_agent_debate protocol and adds execution-specific roles behind the scenes.'
                        : 'Standard workflows reuse the first selected subagent model for execution-specific roles unless the backend overrides them later.'}
                    </p>
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    When disabled, subagents stay in reasoning-only mode and no shared execution policy is attached.
                  </p>
                )}
              </div>

              <div className="space-y-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium text-foreground">Evidence Queries</label>
                    <p className="text-xs text-muted-foreground">
                      One query per line. These are collected before verifier scoring.
                    </p>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={evidenceCollectionEnabled}
                      onChange={(event) => setEvidenceCollectionEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-border"
                    />
                    Enable collection
                  </label>
                </div>
                <Textarea
                  value={evidenceQueriesText}
                  onChange={(event) => setEvidenceQueriesText(event.target.value)}
                  placeholder={'approved deployment note\nsecurity review checklist'}
                  rows={4}
                />
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2 sm:col-span-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Mode
                    </label>
                    <Select value={evidenceCollectionMode} onValueChange={setEvidenceCollectionMode}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select mode" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="hybrid">Hybrid</SelectItem>
                        <SelectItem value="web">Web</SelectItem>
                        <SelectItem value="rag">RAG</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2 sm:col-span-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      RAG Provider
                    </label>
                    <Select value={ragProvider} onValueChange={setRagProvider}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select RAG provider" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="memory">Memory</SelectItem>
                        <SelectItem value="mcp_resource">MCP Resource</SelectItem>
                        <SelectItem value="auto">Auto</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Results / Query
                    </label>
                    <Input
                      type="number"
                      min={1}
                      value={maxResultsPerQuery}
                      onChange={(event) => setMaxResultsPerQuery(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Fetch / Query
                    </label>
                    <Input
                      type="number"
                      min={1}
                      value={maxFetchPerQuery}
                      onChange={(event) => setMaxFetchPerQuery(event.target.value)}
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Content Chars
                    </label>
                    <Input
                      type="number"
                      min={128}
                      value={maxContentChars}
                      onChange={(event) => setMaxContentChars(event.target.value)}
                    />
                  </div>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    MCP Servers
                  </label>
                  <Textarea
                    value={ragMcpServersText}
                    onChange={(event) => setRagMcpServersText(event.target.value)}
                    placeholder={'docs\nknowledge-base'}
                    rows={2}
                  />
                  <p className="text-xs text-muted-foreground">
                    Used when RAG provider is <code>mcp_resource</code> or <code>auto</code>. One server name per line.
                  </p>
                </div>
              </div>

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                <div className="space-y-1">
                  <label className="text-sm font-medium text-foreground">{t('agentRuns.runPolicy.title')}</label>
                  <p className="text-xs text-muted-foreground">{t('agentRuns.runPolicy.description')}</p>
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    {t('agentRuns.runPolicy.preset')}
                  </label>
                  <Select value={runPolicyPreset} onValueChange={(value) => handleRunPolicyPresetChange(value as RunPolicyPreset)}>
                    <SelectTrigger>
                      <SelectValue placeholder={t('agentRuns.runPolicy.presetPlaceholder')} />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="short">{t('agentRuns.runPolicy.short')}</SelectItem>
                      <SelectItem value="balanced">{t('agentRuns.runPolicy.balanced')}</SelectItem>
                      <SelectItem value="long">{t('agentRuns.runPolicy.long')}</SelectItem>
                      <SelectItem value="custom">{t('agentRuns.runPolicy.custom')}</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.maxWallClock')}
                    </label>
                    <Input value={maxWallClockSec} onChange={(event) => setMaxWallClockSec(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.heartbeatTimeout')}
                    </label>
                    <Input value={heartbeatTimeoutSec} onChange={(event) => setHeartbeatTimeoutSec(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.checkpointInterval')}
                    </label>
                    <Input value={checkpointIntervalSteps} onChange={(event) => setCheckpointIntervalSteps(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.maxFailuresPerRole')}
                    </label>
                    <Input value={maxSubagentFailuresPerRole} onChange={(event) => setMaxSubagentFailuresPerRole(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.budgetExhausted')}
                    </label>
                    <Select
                      value={onBudgetExhausted}
                      onValueChange={(value) => setOnBudgetExhausted(value as NonNullable<api.AgentRunRunPolicy['on_budget_exhausted']>)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={t('agentRuns.runPolicy.budgetPlaceholder')} />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="pause">{t('agentRuns.runPolicy.budgetPause')}</SelectItem>
                        <SelectItem value="finalize_partial">{t('agentRuns.runPolicy.budgetFinalizePartial')}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      {t('agentRuns.runPolicy.disconnect')}
                    </label>
                    <Select
                      value={onSubagentDisconnect}
                      onValueChange={(value) => setOnSubagentDisconnect(value as NonNullable<api.AgentRunRunPolicy['on_subagent_disconnect']>)}
                    >
                      <SelectTrigger>
                        <SelectValue placeholder={t('agentRuns.runPolicy.disconnectPlaceholder')} />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="retry_then_degrade">{t('agentRuns.runPolicy.disconnectRetryThenDegrade')}</SelectItem>
                        <SelectItem value="pause">{t('agentRuns.runPolicy.disconnectPause')}</SelectItem>
                        <SelectItem value="fail">{t('agentRuns.runPolicy.disconnectFail')}</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>
                </div>
              </div>

              <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                <div className="flex items-center justify-between gap-3">
                  <div>
                    <label className="text-sm font-medium text-foreground">Automation Schedule</label>
                    <p className="text-xs text-muted-foreground">
                      Let this run execute unattended via the backend scheduler.
                    </p>
                  </div>
                  <label className="flex items-center gap-2 text-xs text-muted-foreground">
                    <input
                      type="checkbox"
                      checked={scheduleEnabled}
                      onChange={(event) => setScheduleEnabled(event.target.checked)}
                      className="h-4 w-4 rounded border-border"
                    />
                    Enable schedule
                  </label>
                </div>
                {scheduleEnabled ? (
                  <div className="grid gap-3 sm:grid-cols-2">
                    <div className="space-y-2 sm:col-span-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Schedule Type
                      </label>
                      <Select value={scheduleType} onValueChange={(value) => setScheduleType(value as 'interval' | 'once' | 'cron')}>
                        <SelectTrigger>
                          <SelectValue placeholder="Select schedule type" />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="interval">Interval</SelectItem>
                          <SelectItem value="once">One Shot</SelectItem>
                          <SelectItem value="cron">Cron</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>

                    {scheduleType === 'interval' ? (
                      <div className="space-y-2">
                        <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Interval Seconds
                        </label>
                        <Input
                          type="number"
                          min={1}
                          value={scheduleIntervalSeconds}
                          onChange={(event) => setScheduleIntervalSeconds(event.target.value)}
                        />
                      </div>
                    ) : null}

                    {scheduleType === 'once' ? (
                      <div className="space-y-2 sm:col-span-2">
                        <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Run At
                        </label>
                        <Input
                          type="datetime-local"
                          value={scheduleRunAt}
                          onChange={(event) => setScheduleRunAt(event.target.value)}
                        />
                      </div>
                    ) : null}

                    {scheduleType === 'cron' ? (
                      <div className="space-y-2 sm:col-span-2">
                        <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Cron
                        </label>
                        <Input
                          value={scheduleCron}
                          onChange={(event) => setScheduleCron(event.target.value)}
                          placeholder="0 9 * * 1"
                        />
                      </div>
                    ) : null}

                    <div className="space-y-2 sm:col-span-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Timezone
                      </label>
                      <Input
                        value={scheduleTimezone}
                        onChange={(event) => setScheduleTimezone(event.target.value)}
                        placeholder="Asia/Taipei"
                      />
                    </div>

                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Max Runs
                      </label>
                      <Input
                        type="number"
                        min={1}
                        value={scheduleMaxRuns}
                        onChange={(event) => setScheduleMaxRuns(event.target.value)}
                        placeholder="Unlimited"
                      />
                    </div>

                    <label className="flex items-center gap-2 text-xs text-muted-foreground sm:col-span-2">
                      <input
                        type="checkbox"
                        checked={scheduleAutoPauseOnFailure}
                        onChange={(event) => setScheduleAutoPauseOnFailure(event.target.checked)}
                        className="h-4 w-4 rounded border-border"
                      />
                      Auto-pause the schedule after a failed execution.
                    </label>

                    {scheduleType !== 'once' ? (
                      <label className="flex items-center gap-2 text-xs text-muted-foreground sm:col-span-2">
                        <input
                          type="checkbox"
                          checked={scheduleStartImmediately}
                          onChange={(event) => setScheduleStartImmediately(event.target.checked)}
                          className="h-4 w-4 rounded border-border"
                        />
                        Trigger immediately after creation, then continue on the selected cadence.
                      </label>
                    ) : null}
                  </div>
                ) : (
                  <p className="text-xs text-muted-foreground">
                    When disabled, the run is created and then started immediately from the UI.
                  </p>
                )}
              </div>

              {modelError ? (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {modelError}
                </div>
              ) : null}
              {!loadingModels && models.length === 0 ? (
                <div className="rounded-md border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning">
                  No configured models found. Add local or remote models in Settings first.
                </div>
              ) : null}
              {createError ? (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {createError}
                </div>
              ) : null}
                </CardContent>
              </Card>
            ) : null}
          </div>

          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <PanelRight className="h-4 w-4" />
                  Run Sidebar
                </CardTitle>
                <CardDescription>
                  Recent runs, active status, artifacts, recovery signals, and operator drill-down links.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-1">
                  <div className="rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Active run
                    </p>
                    <p className="mt-2 text-sm font-semibold text-foreground">
                      {activeRunSummary?.title || activeRunSummary?.topic || activeRunSummary?.run_id || 'None'}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <Badge variant={activeRunSummary ? statusVariant(activeRunSummary.status) : 'neutral'}>
                        {activeRunSummary ? translateRunStatus(activeRunSummary.status, t) : 'Idle'}
                      </Badge>
                      {activeRunSummary?.degraded ? <Badge variant="warning">Degraded</Badge> : null}
                    </div>
                    <p className="mt-3 text-xs text-muted-foreground">
                      Workspace: {activeRunSummary?.workspace_dir || effectiveWorkspacePath || 'Not set'}
                    </p>
                  </div>

                  <div className="rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Recovery
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {activeRunSummary?.recovery_state?.status
                        ? `Recovery state: ${String(activeRunSummary.recovery_state.status)}`
                        : 'No active recovery state.'}
                    </p>
                    {activeRunSummary?.latest_error ? (
                      <p className="mt-2 text-xs text-destructive">{activeRunSummary.latest_error}</p>
                    ) : null}
                  </div>

                  <div className="rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Artifacts
                    </p>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {activeRunSummary ? `${activeRunSummary.artifacts.length} attached` : 'No run selected'}
                    </p>
                  </div>

                  <div className="rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Last update
                    </p>
                    <div className="mt-2 flex items-center gap-2 text-sm text-muted-foreground">
                      <Clock3 className="h-4 w-4" />
                      <span>{formatDateTime(activeRunSummary?.updated_at ?? null)}</span>
                    </div>
                  </div>
                </div>

                <Separator />

                <div className="space-y-3">
                  <div className="flex items-center justify-between gap-3">
                    <h3 className="text-sm font-semibold text-foreground">{t('agentRuns.recentRuns.title')}</h3>
                    <Badge variant="outline">{sidebarRuns.length}</Badge>
                  </div>
                  {runError ? (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                      {runError}
                    </div>
                  ) : loadingRuns ? (
                    <div className="space-y-3">
                      {Array.from({ length: 4 }).map((_, index) => (
                        <div key={index} className="h-20 animate-pulse rounded-lg border border-border bg-surface-layer" />
                      ))}
                    </div>
                  ) : sidebarRuns.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                      {t('agentRuns.recentRuns.empty')}
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {sidebarRuns.map((run) => {
                        const runState = run.status.toLowerCase()
                        const isTerminal = TERMINAL_RUN_STATUSES.has(runState)
                        const isSelected = run.run_id === activeRunId
                        return (
                          <button
                            key={run.run_id || `${run.created_at}-${run.title}`}
                            type="button"
                            onClick={() => setActiveRunId(run.run_id)}
                            className={cn(
                              'block w-full rounded-lg border bg-surface-layer p-4 text-left transition-colors hover:border-primary-500/50',
                              isSelected ? 'border-primary-500/60' : 'border-border'
                            )}
                          >
                            <div className="mb-2 flex items-start justify-between gap-3">
                              <div>
                                <h3 className="text-sm font-semibold text-foreground">
                                  {run.title || run.topic || run.run_id || t('agentRuns.run.untitled')}
                                </h3>
                                <p className="mt-0.5 text-xs text-muted-foreground">{run.protocol_id}</p>
                              </div>
                              <div className="flex flex-wrap items-center gap-2">
                                <Badge variant={statusVariant(run.status)}>{translateRunStatus(run.status, t)}</Badge>
                                {run.degraded ? <Badge variant="warning">{t('agentRuns.badge.degraded')}</Badge> : null}
                              </div>
                            </div>
                            <p className="line-clamp-2 text-sm text-muted-foreground">
                              {run.topic || t('agentRuns.run.noTopic')}
                            </p>
                            <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                              <span>{formatDateTime(run.updated_at)}</span>
                              <span>{isTerminal ? 'Finished' : 'Active'}</span>
                            </div>
                          </button>
                        )
                      })}
                    </div>
                  )}
                </div>

                {activeRunSummary ? (
                  <div className="rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Open full run detail
                    </p>
                    <Link
                      href={`/agent-runs/${encodeURIComponent(activeRunSummary.run_id)}`}
                      className="mt-2 inline-flex text-sm text-primary-300 underline-offset-4 hover:underline"
                    >
                      Inspect run detail, logs, and recovery console
                    </Link>
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Current Workflow Target</CardTitle>
                <CardDescription>
                  The workflow should operate against this project and workspace unless the active run carries a newer override.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-3 text-sm text-muted-foreground">
                <div>
                  <span className="font-medium text-foreground">Project:</span>{' '}
                  {activeProject ? `${activeProject.name} (${activeProject.id})` : 'None'}
                </div>
                <div>
                  <span className="font-medium text-foreground">Workspace:</span>{' '}
                  {effectiveWorkspacePath || activeRunSummary?.workspace_dir || 'Not set'}
                </div>
                <div>
                  <span className="font-medium text-foreground">Protocol:</span>{' '}
                  {runTemplate === 'research_debate' ? 'multi_agent_debate' : protocolId}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
