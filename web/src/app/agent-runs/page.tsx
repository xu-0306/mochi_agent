'use client'

import * as React from 'react'
import Link from 'next/link'
import { useRouter } from 'next/navigation'
import { Plus, RefreshCw, Rocket, Trash2 } from 'lucide-react'
import * as api from '@/lib/api'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/lib/i18n'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Textarea } from '@/components/ui/textarea'

interface SubagentDraft {
  id: string
  role: string
  modelId: string
}

type RunTemplate = 'standard' | 'research_debate'

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
  {
    id: 'controlled_subagent_execution',
    label: 'Controlled Subagent Execution',
    description: 'Subagents propose execution requests; a controller gates runtime execution.',
  },
]

const TERMINAL_RUN_STATUSES = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'completed',
  'done',
  'error',
])

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
  if (protocolId === 'controlled_subagent_execution') {
    return ['planner', 'executor', 'controller', 'evaluator']
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
  if (normalized === 'running' || normalized === 'queued' || normalized === 'pending') {
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

function isUnavailableError(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 404 || error.status === 405)
}

export default function AgentRunsPage() {
  const router = useRouter()
  const { t } = useI18n()
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
    const normalizedSubagents = subagents
      .map((item) => ({
        role: item.role.trim(),
        model_id: item.modelId.trim(),
      }))
      .filter((item) => item.role.length > 0 && item.model_id.length > 0)

    if (normalizedSubagents.length === 0) {
      setCreateError('Add at least one subagent with role and model.')
      return
    }

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
        topic: topic.trim() || null,
        subagents: runTemplate === 'research_debate' ? [] : normalizedSubagents,
        selected_models_roles:
          runTemplate === 'research_debate'
            ? {
                by_role: {
                  debater_a: localWorkerModelId.trim(),
                  debater_b: localWorkerModelId.trim(),
                  judge: smartModelId.trim(),
                  verifier: smartModelId.trim(),
                  planner: smartModelId.trim(),
                  synthesizer: smartModelId.trim(),
                  local_worker: localWorkerModelId.trim(),
                  skeptic: localWorkerModelId.trim(),
                },
              }
            : undefined,
        schedule,
        summary: {
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
          ...(protocolId === 'controlled_subagent_execution' && runTemplate !== 'research_debate'
            ? {
                protocol_config: {
                  max_execution_requests: 5,
                  max_commands_per_request: 1,
                  default_timeout_sec: 300,
                  background_allowed: true,
                  workspace_mode: 'task_sandbox',
                },
              }
            : {}),
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
          if (!scheduleEnabled) {
            await api.startAgentRun(created.run_id)
          }
        } catch (startError) {
          if (!isUnavailableError(startError)) {
            const detail = startError instanceof Error
              ? startError.message
              : scheduleEnabled
                ? 'Run was created but scheduler could not be initialized.'
                : 'Run was created but could not be auto-started.'
            setCreateError(detail)
          }
        }
        router.push(`/agent-runs/${encodeURIComponent(created.run_id)}`)
        return
      }
      await loadRuns()
      setTitle('')
      setTopic('')
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
      setScheduleEnabled(false)
      setScheduleType('interval')
      setScheduleIntervalSeconds('3600')
      setScheduleRunAt('')
      setScheduleCron('0 9 * * 1')
      setScheduleTimezone(defaultScheduleTimezone())
      setScheduleStartImmediately(true)
      setScheduleMaxRuns('')
      setScheduleAutoPauseOnFailure(true)
    } catch (error) {
      if (isUnavailableError(error)) {
        setCreateError(t('workflows.apiUnavailable'))
      } else {
        const detail = error instanceof Error ? error.message : t('workflows.createError')
        setCreateError(detail)
      }
    } finally {
      setCreatePending(false)
    }
  }, [
    citationPolicy,
    defaultModelId,
    debateRounds,
    evidenceCollectionEnabled,
    evidenceCollectionMode,
    evidenceQueriesText,
    localWorkerCount,
    localWorkerModelId,
    loadRuns,
    maxContentChars,
    maxFetchPerQuery,
    maxResultsPerQuery,
    protocolId,
    ragMcpServersText,
    ragProvider,
    researchOutputTargets,
    researchSourceMode,
    router,
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

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-foreground">{t('workflows.title')}</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              {t('workflows.description')}
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={() => void loadRuns(true)} loading={refreshingRuns}>
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="grid gap-6 xl:grid-cols-[420px_minmax(0,1fr)]">
          <Card>
            <CardHeader>
              <CardTitle>{t('workflows.create')}</CardTitle>
              <CardDescription>
                {t('workflows.createDescription')}
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
            <CardFooter className="justify-end">
              <Button variant="primary" size="md" onClick={() => void handleCreate()} loading={createPending} disabled={!canCreate}>
                <Rocket className="h-4 w-4" />
                Create Run
              </Button>
            </CardFooter>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Recent Runs</CardTitle>
              <CardDescription>Open a run to inspect logs, artifacts, and guidance controls.</CardDescription>
            </CardHeader>
            <CardContent>
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
              ) : runs.length === 0 ? (
                <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                  No agent runs yet.
                </div>
              ) : (
                <div className="space-y-3">
                  {runs.map((run) => {
                    const runState = run.status.toLowerCase()
                    const isTerminal = TERMINAL_RUN_STATUSES.has(runState)
                    return (
                      <Link
                        key={run.run_id || `${run.created_at}-${run.title}`}
                        href={`/agent-runs/${encodeURIComponent(run.run_id)}`}
                        className="block rounded-lg border border-border bg-surface-layer p-4 transition-colors hover:border-primary-500/50"
                      >
                        <div className="mb-2 flex items-start justify-between gap-3">
                          <div>
                            <h3 className="text-sm font-semibold text-foreground">
                              {run.title || run.topic || run.run_id || 'Untitled run'}
                            </h3>
                            <p className="mt-0.5 text-xs text-muted-foreground">{run.protocol_id}</p>
                          </div>
                          <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
                        </div>
                        <p className="line-clamp-2 text-sm text-muted-foreground">
                          {run.topic || 'No topic provided.'}
                        </p>
                        <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                          <span>Updated: {formatDateTime(run.updated_at)}</span>
                          <span>{isTerminal ? 'Finished' : 'Active'}</span>
                        </div>
                      </Link>
                    )
                  })}
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
