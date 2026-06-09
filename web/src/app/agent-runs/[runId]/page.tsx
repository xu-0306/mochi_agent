'use client'

import * as React from 'react'
import Link from 'next/link'
import { useParams } from 'next/navigation'
import {
  Archive,
  ArrowLeft,
  Download,
  Pause,
  Play,
  RefreshCw,
  SendHorizontal,
  Square,
} from 'lucide-react'
import * as api from '@/lib/api'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import {
  buildAttemptPackageFallback,
  buildDatasetPackageFallback,
  buildTrainingReadyOnlyDatasetPackage,
} from '@/lib/agent-run-packages'
import { useI18n } from '@/lib/i18n'
import { Textarea } from '@/components/ui/textarea'

const TERMINAL_RUN_STATUSES = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'completed',
  'done',
  'error',
])

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

type TranslateFn = (key: string, values?: Record<string, string | number | boolean | null | undefined>) => string

function translateStatus(status: string, t: TranslateFn): string {
  const normalized = status.toLowerCase()
  const key = `agentRuns.status.${normalized}`
  const translated = t(key)
  return translated === key ? status : translated
}

function translateExecStatus(status: string | null | undefined, t: TranslateFn): string {
  const normalized = status?.trim().toLowerCase()
  if (!normalized) {
    return t('common.unknown')
  }
  const key = `agentRuns.exec.status.${normalized}`
  const translated = t(key)
  return translated === key ? status ?? t('common.unknown') : translated
}

function jsonPreview(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

function eventSummary(event: Record<string, unknown>): string {
  const contentCandidates = [
    event.content,
    event.message,
    event.text,
    event.final_answer,
    event.error,
  ]
  for (const candidate of contentCandidates) {
    if (typeof candidate === 'string' && candidate.trim().length > 0) {
      return candidate
    }
  }
  if (typeof event.payload === 'object' && event.payload !== null) {
    return jsonPreview(event.payload)
  }
  return jsonPreview(event)
}

function eventKind(event: Record<string, unknown>): string {
  if (typeof event.type === 'string' && event.type.length > 0) {
    return event.type
  }
  if (typeof event.phase === 'string' && event.phase.length > 0) {
    return event.phase
  }
  return 'event'
}

function isUnavailableError(error: unknown): boolean {
  return error instanceof api.ApiError && (error.status === 404 || error.status === 405)
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getArtifactAttemptId(artifact: api.AgentRunArtifact): string | null {
  return getString(artifact.metadata.attempt_id)
}

function getEventAttemptId(event: Record<string, unknown>): string | null {
  return getString(event.attempt_id)
}

function getLatestArtifactByType(
  run: api.AgentRunDetail | null,
  artifactType: string,
  attemptId: string | null = null
): api.AgentRunArtifact | null {
  if (!run) {
    return null
  }
  const artifacts = [...run.artifacts].reverse()
  return (
    artifacts.find((item) => {
      if (item.artifact_type !== artifactType) {
        return false
      }
      if (attemptId === null) {
        return true
      }
      return getArtifactAttemptId(item) === attemptId
    }) ?? null
  )
}

function getArtifactContent(
  run: api.AgentRunDetail | null,
  artifactType: string,
  attemptId: string | null = null
): Record<string, unknown> | null {
  const artifact = getLatestArtifactByType(run, artifactType, attemptId)
  if (!artifact || !isRecord(artifact.metadata)) {
    return null
  }
  const content = artifact.metadata.content
  return isRecord(content) ? content : null
}

function getArtifactPayload(
  run: api.AgentRunDetail | null,
  artifactType: string,
  attemptId: string | null = null
): Record<string, unknown> | null {
  const artifact = getLatestArtifactByType(run, artifactType, attemptId)
  if (!artifact || !isRecord(artifact.metadata)) {
    return null
  }
  const content = artifact.metadata.content
  if (isRecord(content)) {
    return content
  }
  const record = artifact.metadata.record
  return isRecord(record) ? record : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter((item): item is string => typeof item === 'string')
}

function getNumberRecord(value: unknown): Array<[string, number]> {
  if (!isRecord(value)) {
    return []
  }
  return Object.entries(value).filter(
    (entry): entry is [string, number] => typeof entry[0] === 'string' && typeof entry[1] === 'number'
  )
}

function getRecordArray(value: unknown): Record<string, unknown>[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value.filter(isRecord)
}

function truncatePreview(value: string, maxChars = 280): string {
  const text = value.trim()
  if (text.length <= maxChars) {
    return text
  }
  return `${text.slice(0, maxChars - 3).trimEnd()}...`
}

function getString(value: unknown): string | null {
  return typeof value === 'string' ? value : null
}

function getNullableString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function describeSchedule(schedule: Record<string, unknown>): string {
  if (schedule.enabled !== true) {
    return 'Disabled'
  }
  if (typeof schedule.interval_seconds === 'number') {
    return `Interval (${schedule.interval_seconds}s)`
  }
  if (typeof schedule.cron === 'string' && schedule.cron.trim().length > 0) {
    return 'Cron'
  }
  if (typeof schedule.run_at === 'string' && schedule.run_at.trim().length > 0) {
    return 'One Shot'
  }
  return 'Enabled'
}

function scheduleAttemptStatusLabel(attempt: Record<string, unknown>): string {
  return getString(attempt.status) ?? 'unknown'
}

function booleanLabel(value: boolean, t: TranslateFn): string {
  return value ? t('agentRuns.exec.boolean.yes') : t('agentRuns.exec.boolean.no')
}

function mergeRecoveryState(
  base: api.AgentRunRecoveryState | null | undefined,
  overlay: api.AgentRunRecoveryState | null | undefined
): api.AgentRunRecoveryState {
  const merged: api.AgentRunRecoveryState = { ...(base ?? {}) }
  for (const [key, value] of Object.entries(overlay ?? {})) {
    if (value === undefined || value === null || value === '') {
      continue
    }
    merged[key] = value
  }
  return merged
}

function recoveryPrompt(
  recoveryState: api.AgentRunRecoveryState,
  runStatus: string,
  degraded: boolean,
  t: TranslateFn
): { operatorMessage: string; resumeHint: string; usedFallback: boolean } {
  const operatorMessage =
    getNullableString(recoveryState.operator_message) ??
    getNullableString(recoveryState.operator_note)
  const resumeHint =
    getNullableString(recoveryState.resume_hint) ??
    getNullableString(recoveryState.suggested_action) ??
    getNullableString(recoveryState.suggested_operator_action)
  const status = (getNullableString(recoveryState.status) ?? runStatus).toLowerCase()

  if (operatorMessage || resumeHint) {
    return {
      operatorMessage: operatorMessage ?? t(`agentRuns.recovery.${status}.message`),
      resumeHint: resumeHint ?? t(`agentRuns.recovery.${status}.resume`),
      usedFallback: false,
    }
  }

  if (status === 'awaiting_resources' || status === 'stalled' || status === 'partial' || status === 'degraded') {
    return {
      operatorMessage: t(`agentRuns.recovery.${status}.message`),
      resumeHint: t(`agentRuns.recovery.${status}.resume`),
      usedFallback: true,
    }
  }

  if (degraded) {
    return {
      operatorMessage: t('agentRuns.recovery.degraded.message'),
      resumeHint: t('agentRuns.recovery.degraded.resume'),
      usedFallback: true,
    }
  }

  return {
    operatorMessage: t('agentRuns.recovery.default.message'),
    resumeHint: t('agentRuns.recovery.default.resume'),
    usedFallback: true,
  }
}

interface RoleOutputEntry {
  roleId: string
  candidateId: string | null
  modelId: string | null
  roundIndex: number | null
  content: string
  timestamp: string | null
}

type RunAction = 'start' | 'pause' | 'resume' | 'cancel' | 'finalize_partial'

export default function AgentRunDetailPage() {
  const params = useParams<{ runId: string }>()
  const { t } = useI18n()
  const routeRunId = Array.isArray(params.runId) ? params.runId[0] : params.runId

  const [run, setRun] = React.useState<api.AgentRunDetail | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [refreshing, setRefreshing] = React.useState(false)
  const [error, setError] = React.useState<string | null>(null)
  const [runHealth, setRunHealth] = React.useState<api.AgentRunHealthSummary | null>(null)
  const [guidanceText, setGuidanceText] = React.useState('')
  const [guidancePending, setGuidancePending] = React.useState(false)
  const [guidanceError, setGuidanceError] = React.useState<string | null>(null)
  const [guidanceSuccess, setGuidanceSuccess] = React.useState<string | null>(null)
  const [runActionPending, setRunActionPending] = React.useState<RunAction | null>(null)
  const [runActionError, setRunActionError] = React.useState<string | null>(null)
  const [runActionSuccess, setRunActionSuccess] = React.useState<string | null>(null)
  const [execSession, setExecSession] = React.useState<api.AgentRunExecSessionPayload | null>(null)
  const [execActionPending, setExecActionPending] = React.useState<string | null>(null)
  const [execActionError, setExecActionError] = React.useState<string | null>(null)
  const [exportError, setExportError] = React.useState<string | null>(null)
  const [roleFilter, setRoleFilter] = React.useState<string>('all')
  const [candidateFilter, setCandidateFilter] = React.useState<string>('all')
  const [attemptScope, setAttemptScope] = React.useState<string>('latest')

  const loadRun = React.useCallback(async (options?: { silent?: boolean }) => {
    if (!routeRunId) {
      setLoading(false)
      setRun(null)
      setRunHealth(null)
      setError('Missing run id.')
      return
    }
    if (options?.silent) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }
    setError(null)
    try {
      const data = await api.fetchAgentRun(routeRunId)
      setRun(data)
      try {
        const health = await api.fetchAgentRunHealth(routeRunId)
        setRunHealth(health)
      } catch (healthError) {
        if (!isUnavailableError(healthError)) {
          console.error(healthError)
        }
        setRunHealth(null)
      }
    } catch (loadError) {
      if (isUnavailableError(loadError)) {
        setError('Agent Run endpoint is unavailable or this run was not found.')
      } else {
        const detail = loadError instanceof Error ? loadError.message : 'Unable to load Agent Run.'
        setError(detail)
      }
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }, [routeRunId])

  React.useEffect(() => {
    void loadRun()
  }, [loadRun])

  React.useEffect(() => {
    if (!run || TERMINAL_RUN_STATUSES.has(run.status.toLowerCase())) {
      return
    }
    const timer = window.setInterval(() => {
      void loadRun({ silent: true })
    }, 4000)
    return () => window.clearInterval(timer)
  }, [loadRun, run])

  const runStatus = run?.status.toLowerCase() ?? 'unknown'
  const isTerminal = run ? TERMINAL_RUN_STATUSES.has(runStatus) : false
  const canStart = runStatus === 'created'
  const canPause = runStatus === 'running'
  const canResume = ['paused', 'awaiting_resources', 'partial', 'stalled'].includes(runStatus)
  const canCancel = !isTerminal
  const effectiveRecoveryState = React.useMemo(
    () => mergeRecoveryState(run?.recovery_state, runHealth?.recovery_state),
    [run?.recovery_state, runHealth?.recovery_state]
  )
  const recentScheduleAttempts = React.useMemo(
    () => getRecordArray(run?.schedule?.recent_attempts),
    [run]
  )
  const latestAttemptId = React.useMemo(
    () => getString(recentScheduleAttempts[0]?.attempt_id),
    [recentScheduleAttempts]
  )
  const selectedAttemptId = React.useMemo(() => {
    if (attemptScope === 'all') {
      return null
    }
    if (attemptScope === 'latest') {
      return latestAttemptId
    }
    return attemptScope
  }, [attemptScope, latestAttemptId])
  const scopedEvents = React.useMemo(() => {
    const events = run?.events ?? []
    if (attemptScope === 'all') {
      return events
    }
    if (!selectedAttemptId) {
      return events.filter((event) => getEventAttemptId(event) === null)
    }
    return events.filter((event) => getEventAttemptId(event) === selectedAttemptId)
  }, [attemptScope, run, selectedAttemptId])
  const roleOutputs = React.useMemo(
    () =>
      scopedEvents
        .filter((event) => event.type === 'role_output' && isRecord(event.payload))
        .map((event) => ({
          roleId: getString((event.payload as Record<string, unknown>).role_id) ?? 'unknown',
          candidateId: getString((event.payload as Record<string, unknown>).candidate_id),
          modelId: getString((event.payload as Record<string, unknown>).model_id),
          roundIndex:
            typeof (event.payload as Record<string, unknown>).round_index === 'number'
              ? ((event.payload as Record<string, unknown>).round_index as number)
              : null,
          content: getString((event.payload as Record<string, unknown>).content) ?? '',
          timestamp: typeof event.timestamp === 'string' ? formatDateTime(event.timestamp) : null,
        })),
    [scopedEvents]
  )
  const roleOptions = React.useMemo(
    () => Array.from(new Set(roleOutputs.map((item) => item.roleId))).sort(),
    [roleOutputs]
  )
  const candidateOptions = React.useMemo(
    () =>
      Array.from(
        new Set(roleOutputs.map((item) => item.candidateId).filter((item): item is string => Boolean(item)))
      ).sort(),
    [roleOutputs]
  )
  const filteredRoleOutputs = React.useMemo(
    () =>
      roleOutputs.filter((item) => {
        if (roleFilter !== 'all' && item.roleId !== roleFilter) {
          return false
        }
        if (candidateFilter !== 'all' && item.candidateId !== candidateFilter) {
          return false
        }
        return true
      }),
    [candidateFilter, roleFilter, roleOutputs]
  )
  const groupedRoleOutputs = React.useMemo(() => {
    const groups = new Map<string, RoleOutputEntry[]>()
    for (const item of filteredRoleOutputs) {
      const key = `${item.roleId}:::${item.candidateId ?? 'uncategorized'}`
      const current = groups.get(key)
      if (current) {
        current.push(item)
      } else {
        groups.set(key, [item])
      }
    }
    return Array.from(groups.entries()).map(([key, items]) => {
      const [groupRoleId, groupCandidateId] = key.split(':::')
      return {
        roleId: groupRoleId,
        candidateId: groupCandidateId === 'uncategorized' ? null : groupCandidateId,
        items,
      }
    })
  }, [filteredRoleOutputs])
  const evidenceSummary = React.useMemo(
    () => getArtifactContent(run, 'evidence_summary', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const researchPlan = React.useMemo(
    () => getArtifactContent(run, 'research_plan', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const sourceQualityTable = React.useMemo(
    () => getArtifactContent(run, 'source_quality_table', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const claimEvidenceMap = React.useMemo(
    () => getArtifactContent(run, 'claim_evidence_map', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const researchBrief = React.useMemo(
    () => getArtifactContent(run, 'research_brief', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const verificationSummary = React.useMemo(
    () => getArtifactContent(run, 'verification_summary', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const datasetRecord = React.useMemo(
    () => getArtifactPayload(run, 'dataset_record', selectedAttemptId),
    [run, selectedAttemptId]
  )
  const evidencePacketIndex = React.useMemo(() => {
    const packets = getRecordArray(evidenceSummary?.evidence_packets)
    const index = new Map<string, Record<string, unknown>>()
    for (const packet of packets) {
      if (typeof packet.evidence_id === 'string' && packet.evidence_id.trim()) {
        index.set(packet.evidence_id, packet)
      }
    }
    return index
  }, [evidenceSummary])
  const filteredArtifacts = React.useMemo(() => {
    if (!run) {
      return []
    }
    if (attemptScope === 'all') {
      return run.artifacts
    }
    if (!selectedAttemptId) {
      return run.artifacts.filter((artifact) => getArtifactAttemptId(artifact) === null)
    }
    return run.artifacts.filter((artifact) => getArtifactAttemptId(artifact) === selectedAttemptId)
  }, [attemptScope, run, selectedAttemptId])
  const attemptPackagePreview = React.useMemo(
    () => (run ? buildAttemptPackageFallback(run, attemptScope, selectedAttemptId) : null),
    [attemptScope, run, selectedAttemptId]
  )
  const datasetPackagePreview = React.useMemo(
    () => (run ? buildDatasetPackageFallback(run) : null),
    [run]
  )
  const detachedExecJobs = React.useMemo(() => {
    const artifactJobs = getRecordArray(getArtifactContent(run, 'detached_exec_jobs', selectedAttemptId)?.items)
    if (artifactJobs.length > 0) {
      return artifactJobs
    }
    return getRecordArray(runHealth?.detached_exec_jobs?.items)
  }, [run, runHealth, selectedAttemptId])
  const subagentHealthSnapshot = React.useMemo(() => {
    const artifactSnapshot = getArtifactContent(run, 'subagent_health_snapshot', selectedAttemptId)
    if (artifactSnapshot && Object.keys(artifactSnapshot).length > 0) {
      return artifactSnapshot
    }
    return isRecord(runHealth?.subagent_health_snapshot) ? runHealth.subagent_health_snapshot : null
  }, [run, runHealth, selectedAttemptId])
  const trainingReadyDatasetPackagePreview = React.useMemo(
    () =>
      datasetPackagePreview
        ? buildTrainingReadyOnlyDatasetPackage(datasetPackagePreview)
        : null,
    [datasetPackagePreview]
  )
  const recoveryStatus = React.useMemo(
    () => getNullableString(effectiveRecoveryState.status),
    [effectiveRecoveryState]
  )
  const recoveryPromptCopy = React.useMemo(
    () => (run ? recoveryPrompt(effectiveRecoveryState, run.status, run.degraded, t) : null),
    [effectiveRecoveryState, run, t]
  )
  const canFinalizePartial = React.useMemo(() => {
    const explicit = runHealth?.recovery_state?.finalize_partial_ready
    if (typeof explicit === 'boolean') {
      return explicit
    }
    const status = (getNullableString(effectiveRecoveryState.status) ?? runStatus).toLowerCase()
    return status === 'awaiting_resources' || status === 'stalled'
  }, [effectiveRecoveryState.status, runHealth?.recovery_state?.finalize_partial_ready, runStatus])
  const finalizePartialReason = React.useMemo(
    () =>
      getNullableString(effectiveRecoveryState.finalize_partial_reason) ??
      (canFinalizePartial ? t('agentRuns.recovery.finalizePartialAvailable') : null),
    [canFinalizePartial, effectiveRecoveryState.finalize_partial_reason, t]
  )
  const execJobRows = React.useMemo(() => {
    const byId = new Map<string, Record<string, unknown>>()
    for (const job of detachedExecJobs) {
      const sessionId = getNullableString(job.session_id)
      if (sessionId) {
        byId.set(sessionId, job)
      }
    }
    if (execSession?.session_id && !byId.has(execSession.session_id)) {
      byId.set(execSession.session_id, {
        session_id: execSession.session_id,
        status: execSession.lease.status ?? execSession.live_status,
        command: execSession.lease.command,
        reattach_supported: execSession.lease.reattach_supported,
        lease_owner: execSession.lease.lease_owner,
        workdir: execSession.lease.workdir,
        log_path: execSession.lease.log_path,
        approval_state: execSession.lease.approval_state,
        timeout: execSession.lease.timeout,
        pid: execSession.lease.pid,
        background: execSession.lease.background,
      })
    }
    return Array.from(byId.values())
  }, [detachedExecJobs, execSession])

  React.useEffect(() => {
    if (attemptScope === 'latest' || attemptScope === 'all') {
      return
    }
    if (!recentScheduleAttempts.some((attempt) => getString(attempt.attempt_id) === attemptScope)) {
      setAttemptScope(latestAttemptId ? 'latest' : 'all')
    }
  }, [attemptScope, latestAttemptId, recentScheduleAttempts])

  const handleRunAction = React.useCallback(async (action: RunAction) => {
    if (!run) {
      return
    }

    setRunActionPending(action)
    setRunActionError(null)
    setRunActionSuccess(null)

    try {
      let updatedSummary: api.AgentRunSummary
      if (action === 'start') {
        updatedSummary = await api.startAgentRun(run.run_id)
      } else if (action === 'pause') {
        updatedSummary = await api.pauseAgentRun(run.run_id)
      } else if (action === 'resume') {
        updatedSummary = await api.resumeAgentRun(run.run_id)
      } else if (action === 'finalize_partial') {
        updatedSummary = await api.finalizeAgentRunPartial(run.run_id)
      } else {
        updatedSummary = await api.cancelAgentRun(run.run_id)
      }

      setRun((current) => (current ? { ...current, ...updatedSummary } : current))
      setRunActionSuccess(
        action === 'finalize_partial'
          ? 'Run finalized as partial.'
          : `Run ${action} request accepted.`
      )
      await loadRun({ silent: true })
    } catch (actionError) {
      if (isUnavailableError(actionError)) {
        setRunActionError('Run control endpoint is not available on this backend yet.')
      } else {
        const detail =
          actionError instanceof Error
            ? actionError.message
            : action === 'finalize_partial'
              ? 'Unable to finalize run as partial.'
              : `Unable to ${action} run.`
        setRunActionError(detail)
      }
    } finally {
      setRunActionPending(null)
    }
  }, [loadRun, run])

  const handleSubmitGuidance = React.useCallback(async () => {
    if (!run) {
      return
    }
    const guidance = guidanceText.trim()
    if (!guidance) {
      return
    }
    setGuidancePending(true)
    setGuidanceError(null)
    setGuidanceSuccess(null)
    try {
      const updated = await api.appendAgentRunGuidance(run.run_id, { guidance })
      setRun(updated)
      setGuidanceText('')
      setGuidanceSuccess('Guidance submitted.')
    } catch (submitError) {
      if (isUnavailableError(submitError)) {
        setGuidanceError('Guidance endpoint is not available on this backend yet.')
      } else {
        const detail = submitError instanceof Error ? submitError.message : 'Unable to submit guidance.'
        setGuidanceError(detail)
      }
    } finally {
      setGuidancePending(false)
    }
  }, [guidanceText, run])

  const handleExecAction = React.useCallback(
    async (action: 'poll' | 'reattach' | 'stop', sessionId: string) => {
      if (!run) {
        return
      }
      setExecActionPending(`${action}:${sessionId}`)
      setExecActionError(null)
      try {
        const payload =
          action === 'poll'
            ? await api.fetchAgentRunExecSession(run.run_id, sessionId, { yield_time_ms: 50 })
            : action === 'reattach'
              ? await api.reattachAgentRunExecSession(run.run_id, sessionId, { yield_time_ms: 50 })
              : await api.stopAgentRunExecSession(run.run_id, sessionId)
        setExecSession(payload)
        await loadRun({ silent: true })
      } catch (actionError) {
        const detail = actionError instanceof Error ? actionError.message : `Unable to ${action} exec session.`
        setExecActionError(detail)
      } finally {
        setExecActionPending(null)
      }
    },
    [loadRun, run]
  )

  const handleExportJson = React.useCallback(
    (payload: unknown, suffix: string) => {
      if (!payload || !run) {
        return
      }
      const blob = new Blob([`${JSON.stringify(payload, null, 2)}\n`], {
        type: 'application/json',
      })
      const url = window.URL.createObjectURL(blob)
      const anchor = document.createElement('a')
      anchor.href = url
      anchor.download = `${run.run_id}-${suffix}.json`
      document.body.appendChild(anchor)
      anchor.click()
      anchor.remove()
      window.URL.revokeObjectURL(url)
    },
    [run]
  )

  const handleExportAttemptBundle = React.useCallback(async () => {
    if (!run) {
      return
    }
    setExportError(null)
    try {
      const payload =
        attemptScope !== 'all' && selectedAttemptId
          ? await api.fetchAgentRunAttemptPackage(run.run_id, selectedAttemptId).catch((error) => {
              if (!isUnavailableError(error)) {
                throw error
              }
              return buildAttemptPackageFallback(run, attemptScope, selectedAttemptId)
            })
          : buildAttemptPackageFallback(run, attemptScope, selectedAttemptId)
      const suffix = selectedAttemptId
        ? `attempt-${selectedAttemptId}`
        : attemptScope === 'all'
          ? 'all-attempts'
          : 'latest-attempt'
      handleExportJson(payload, `${suffix}-bundle`)
    } catch (exportActionError) {
      const detail = exportActionError instanceof Error ? exportActionError.message : 'Unable to export Attempt Package.'
      setExportError(detail)
    }
  }, [attemptScope, handleExportJson, run, selectedAttemptId])

  const handleExportDatasetPackage = React.useCallback(async () => {
    if (!run) {
      return
    }
    setExportError(null)
    try {
      const payload = await api.fetchAgentRunDatasetPackage(run.run_id).catch((error) => {
        if (!isUnavailableError(error)) {
          throw error
        }
        return buildDatasetPackageFallback(run)
      })
      handleExportJson(payload, 'dataset-package')
    } catch (exportActionError) {
      const detail = exportActionError instanceof Error ? exportActionError.message : 'Unable to export Dataset Package.'
      setExportError(detail)
    }
  }, [handleExportJson, run])

  const handleExportTrainingReadyDatasetPackage = React.useCallback(async () => {
    if (!run) {
      return
    }
    setExportError(null)
    try {
      const payload = await api.fetchAgentRunDatasetPackage(run.run_id).catch((error) => {
        if (!isUnavailableError(error)) {
          throw error
        }
        return buildDatasetPackageFallback(run)
      })
      const filtered = buildTrainingReadyOnlyDatasetPackage(payload as unknown as Record<string, unknown>)
      handleExportJson(filtered, 'dataset-package-training-ready')
    } catch (exportActionError) {
      const detail = exportActionError instanceof Error ? exportActionError.message : 'Unable to export training-ready dataset package.'
      setExportError(detail)
    }
  }, [handleExportJson, run])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex items-start justify-between gap-4">
          <div>
            <Link href="/agent-runs" className="mb-2 inline-flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground">
              <ArrowLeft className="h-3.5 w-3.5" />
              {t('workflows.backToList')}
            </Link>
            <h1 className="text-xl font-bold text-foreground">Run {run?.run_id || routeRunId}</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              Status, metadata, events, artifacts, and live guidance.
            </p>
          </div>
          <Button variant="secondary" size="sm" onClick={() => void loadRun({ silent: true })} loading={refreshing}>
            <RefreshCw className="h-3.5 w-3.5" />
            Refresh
          </Button>
        </div>
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        {error ? (
          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
            {error}
          </div>
        ) : loading ? (
          <div className="space-y-4">
            {Array.from({ length: 4 }).map((_, index) => (
              <div key={index} className="h-24 animate-pulse rounded-lg border border-border bg-surface-layer" />
            ))}
          </div>
        ) : run ? (
          <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_360px]">
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle>{t('agentRuns.recovery.title')}</CardTitle>
                  <CardDescription>{t('agentRuns.recovery.description')}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="flex flex-wrap items-center gap-2">
                    <Badge variant={statusVariant(run.status)}>{translateStatus(run.status, t)}</Badge>
                    {run.degraded ? <Badge variant="warning">{t('agentRuns.badge.degraded')}</Badge> : null}
                    {recoveryStatus ? (
                      <Badge variant="outline">{translateStatus(recoveryStatus, t)}</Badge>
                    ) : null}
                    {canFinalizePartial ? (
                      <Badge variant="outline">{t('agentRuns.badge.finalizePartialReady')}</Badge>
                    ) : null}
                  </div>
                  {recoveryPromptCopy ? (
                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-lg border border-warning/30 bg-warning/10 p-3 text-sm text-foreground">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          {t('agentRuns.recovery.operatorTitle')}
                        </p>
                        <p className="mt-2 whitespace-pre-wrap">{recoveryPromptCopy.operatorMessage}</p>
                        {recoveryPromptCopy.usedFallback ? (
                          <p className="mt-2 text-xs text-muted-foreground">{t('agentRuns.recovery.operatorFallback')}</p>
                        ) : null}
                      </div>
                      <div className="rounded-lg border border-border bg-surface-layer p-3 text-sm text-foreground">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          {t('agentRuns.recovery.resumeTitle')}
                        </p>
                        <p className="mt-2 whitespace-pre-wrap">{recoveryPromptCopy.resumeHint}</p>
                      </div>
                    </div>
                  ) : null}
                  {run.latest_error ? (
                    <div className="rounded-lg border border-destructive/30 bg-destructive/10 p-3 text-sm text-destructive">
                      <p className="text-xs font-medium uppercase tracking-wide">{t('agentRuns.recovery.latestError')}</p>
                      <p className="mt-2 whitespace-pre-wrap">{run.latest_error}</p>
                    </div>
                  ) : null}
                  {getNullableString(effectiveRecoveryState.suggested_action) || getNullableString(effectiveRecoveryState.suggested_operator_action) ? (
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-sm text-foreground">
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {t('agentRuns.recovery.suggestedAction')}
                      </p>
                      <p className="mt-2 whitespace-pre-wrap">
                        {getNullableString(effectiveRecoveryState.suggested_action) ?? getNullableString(effectiveRecoveryState.suggested_operator_action)}
                      </p>
                    </div>
                  ) : null}
                  {finalizePartialReason ? (
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-sm text-foreground">
                      <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        {t('agentRuns.recovery.finalizePartialReason')}
                      </p>
                      <p className="mt-2 whitespace-pre-wrap">{finalizePartialReason}</p>
                    </div>
                  ) : null}
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      <p className="font-medium uppercase tracking-wide">{t('agentRuns.recovery.policyTitle')}</p>
                      <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px] text-foreground">
                        {jsonPreview(run.run_policy)}
                      </pre>
                    </div>
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      <p className="font-medium uppercase tracking-wide">{t('agentRuns.recovery.stateTitle')}</p>
                      <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px] text-foreground">
                        {jsonPreview(effectiveRecoveryState)}
                      </pre>
                    </div>
                  </div>
                  {subagentHealthSnapshot ? (
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      <p className="font-medium uppercase tracking-wide">{t('agentRuns.recovery.healthTitle')}</p>
                      <pre className="mt-2 overflow-x-auto whitespace-pre-wrap text-[11px] text-foreground">
                        {jsonPreview(subagentHealthSnapshot)}
                      </pre>
                    </div>
                  ) : (
                    <div className="rounded-lg border border-dashed border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      {t('agentRuns.recovery.emptyHealth')}
                    </div>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>{t('agentRuns.exec.title')}</CardTitle>
                  <CardDescription>{t('agentRuns.exec.description')}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  {execJobRows.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-6 text-sm text-muted-foreground">
                      {runStatus === 'stalled' ? t('agentRuns.exec.emptyStalled') : t('agentRuns.exec.empty')}
                    </div>
                  ) : (
                    execJobRows.map((job) => {
                      const sessionId = getNullableString(job.session_id) ?? ''
                      const isSelectedSession = execSession?.session_id === sessionId
                      const jobStatus = getNullableString(job.status)
                      const jobCommand = getNullableString(job.command)
                      const reattachSupported =
                        getBoolean(job.reattach_supported) ??
                        (isSelectedSession ? execSession?.lease.reattach_supported ?? false : false)
                      return (
                        <div key={sessionId} className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant="outline">{sessionId || t('common.unknown')}</Badge>
                            <Badge variant="neutral">{translateExecStatus(jobStatus, t)}</Badge>
                            {isSelectedSession ? (
                              <Badge variant="outline">
                                {t('agentRuns.exec.liveStatus')}: {translateExecStatus(execSession?.live_status, t)}
                              </Badge>
                            ) : null}
                            {getBoolean(job.background) ? <Badge variant="outline">{t('agentRuns.exec.background')}</Badge> : null}
                          </div>
                          <div className="mt-3 grid gap-3 md:grid-cols-2">
                            <div className="space-y-1">
                              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.command')}</p>
                              <p className="font-mono text-[11px] text-foreground">
                                {jobCommand ? truncatePreview(jobCommand, 180) : t('agentRuns.exec.noCommand')}
                              </p>
                            </div>
                            <div>
                              <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.reattachSupported')}</p>
                              <p className="mt-1 text-foreground">{booleanLabel(reattachSupported, t)}</p>
                            </div>
                            {getNullableString(job.lease_owner) ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.leaseOwner')}</p>
                                <p className="mt-1 text-foreground">{getNullableString(job.lease_owner)}</p>
                              </div>
                            ) : null}
                            {getNullableString(job.workdir) ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.workdir')}</p>
                                <p className="mt-1 break-all text-foreground">{getNullableString(job.workdir)}</p>
                              </div>
                            ) : null}
                            {getNullableString(job.log_path) ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.logPath')}</p>
                                <p className="mt-1 break-all text-foreground">{getNullableString(job.log_path)}</p>
                              </div>
                            ) : null}
                            {getNullableString(job.approval_state) ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.approval')}</p>
                                <p className="mt-1 text-foreground">{getNullableString(job.approval_state)}</p>
                              </div>
                            ) : null}
                            {getNumber(job.timeout) !== null ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.timeout')}</p>
                                <p className="mt-1 text-foreground">{getNumber(job.timeout)}s</p>
                              </div>
                            ) : null}
                            {getNumber(job.pid) !== null ? (
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.pid')}</p>
                                <p className="mt-1 text-foreground">{getNumber(job.pid)}</p>
                              </div>
                            ) : null}
                          </div>
                          <div className="mt-3 flex flex-wrap gap-2">
                            <Button
                              variant="secondary"
                              size="sm"
                              disabled={!sessionId}
                              loading={execActionPending === `poll:${sessionId}`}
                              onClick={() => void handleExecAction('poll', sessionId)}
                            >
                              {t('agentRuns.exec.action.poll')}
                            </Button>
                            <Button
                              variant="secondary"
                              size="sm"
                              disabled={!sessionId || !reattachSupported}
                              loading={execActionPending === `reattach:${sessionId}`}
                              onClick={() => void handleExecAction('reattach', sessionId)}
                            >
                              {reattachSupported ? t('agentRuns.exec.action.reattach') : t('agentRuns.exec.action.unavailable')}
                            </Button>
                            <Button
                              variant="secondary"
                              size="sm"
                              disabled={!sessionId}
                              loading={execActionPending === `stop:${sessionId}`}
                              onClick={() => void handleExecAction('stop', sessionId)}
                            >
                              {t('agentRuns.exec.action.stop')}
                            </Button>
                          </div>
                          {isSelectedSession ? (
                            <div className="mt-3 space-y-3 rounded-lg border border-border bg-canvas p-3">
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.snapshotTitle')}</p>
                                <p className="mt-1 text-xs text-muted-foreground">{t('agentRuns.exec.snapshotDescription')}</p>
                              </div>
                              <div className="grid gap-3 md:grid-cols-2">
                                <div>
                                  <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.status')}</p>
                                  <p className="mt-1 text-foreground">{translateExecStatus(execSession.session?.status ?? execSession.live_status, t)}</p>
                                </div>
                                {execSession.stop_status ? (
                                  <div>
                                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.stopStatus')}</p>
                                    <p className="mt-1 text-foreground">{execSession.stop_status}</p>
                                  </div>
                                ) : null}
                                {execSession.reattach_status ? (
                                  <div>
                                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.reattachStatus')}</p>
                                    <p className="mt-1 text-foreground">{execSession.reattach_status}</p>
                                  </div>
                                ) : null}
                                {execSession.session?.approval_state ? (
                                  <div>
                                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.approval')}</p>
                                    <p className="mt-1 text-foreground">{execSession.session.approval_state}</p>
                                  </div>
                                ) : null}
                              </div>
                              {execSession.session ? (
                                <div className="grid gap-3 md:grid-cols-2">
                                  <div>
                                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.stdout')}</p>
                                    <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-surface-layer p-2 text-[11px] text-foreground">
                                      {execSession.session.stdout || t('agentRuns.exec.stdoutEmpty')}
                                    </pre>
                                  </div>
                                  <div>
                                    <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.stderr')}</p>
                                    <pre className="mt-1 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-surface-layer p-2 text-[11px] text-foreground">
                                      {execSession.session.stderr || t('agentRuns.exec.stderrEmpty')}
                                    </pre>
                                  </div>
                                </div>
                              ) : (
                                <div className="rounded-lg border border-dashed border-border bg-surface-layer p-3 text-sm text-muted-foreground">
                                  {t('agentRuns.exec.noSnapshot')}
                                </div>
                              )}
                              <div>
                                <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{t('agentRuns.exec.rawPayload')}</p>
                                <pre className="mt-1 overflow-x-auto whitespace-pre-wrap text-[11px] text-foreground">
                                  {jsonPreview(execSession)}
                                </pre>
                              </div>
                            </div>
                          ) : null}
                        </div>
                      )
                    })
                  )}
                  {execActionError ? (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                      {execActionError}
                    </div>
                  ) : null}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Exports</CardTitle>
                  <CardDescription>Download structured packages for replay, evaluation, and training-set review.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div className="space-y-2 rounded-lg border border-border bg-surface-layer p-3">
                    <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Attempt Scope
                    </p>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant={attemptScope === 'latest' ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => setAttemptScope('latest')}
                      >
                        Latest
                      </Button>
                      <Button
                        variant={attemptScope === 'all' ? 'secondary' : 'ghost'}
                        size="sm"
                        onClick={() => setAttemptScope('all')}
                      >
                        All
                      </Button>
                      {recentScheduleAttempts.map((attempt, index) => {
                        const attemptId = getString(attempt.attempt_id)
                        if (!attemptId) {
                          return null
                        }
                        return (
                          <Button
                            key={attemptId}
                            variant={attemptScope === attemptId ? 'secondary' : 'ghost'}
                            size="sm"
                            onClick={() => setAttemptScope(attemptId)}
                          >
                            Attempt {getNumber(attempt.attempt_number) ?? index + 1}
                          </Button>
                        )
                      })}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {attemptScope === 'all'
                        ? 'Showing artifacts across all attempts.'
                        : selectedAttemptId
                          ? `Scoped to attempt ${selectedAttemptId}.`
                          : 'Using the latest available run payload.'}
                    </p>
                  </div>
                  <div className="grid gap-3 md:grid-cols-2">
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      <p className="font-medium uppercase tracking-wide">Attempt Package</p>
                      <p className="mt-2">
                        <span className="text-foreground">manifest:</span>{' '}
                        {getString(attemptPackagePreview?.manifest_version) ?? 'N/A'}
                      </p>
                      <p>
                        <span className="text-foreground">type:</span>{' '}
                        {getString(attemptPackagePreview?.package_type) ?? 'N/A'}
                      </p>
                      <p>
                        <span className="text-foreground">records:</span>{' '}
                        {getNumber(attemptPackagePreview?.artifact_count) ?? 0} artifacts /{' '}
                        {getRecordArray(attemptPackagePreview?.dataset_records).length} dataset records
                      </p>
                      <p>
                        <span className="text-foreground">replay ready:</span>{' '}
                        {attemptPackagePreview?.replay_ready === true ? 'Yes' : 'No'}
                      </p>
                    </div>
                    <div className="rounded-lg border border-border bg-surface-layer p-3 text-xs text-muted-foreground">
                      <p className="font-medium uppercase tracking-wide">Dataset Package</p>
                      <p className="mt-2">
                        <span className="text-foreground">manifest:</span>{' '}
                        {getString(datasetPackagePreview?.manifest_version) ?? 'N/A'}
                      </p>
                      <p>
                        <span className="text-foreground">type:</span>{' '}
                        {getString(datasetPackagePreview?.package_type) ?? 'N/A'}
                      </p>
                      <p>
                        <span className="text-foreground">record count:</span>{' '}
                        {getNumber(datasetPackagePreview?.dataset_record_count) ?? 0}
                      </p>
                      <p>
                        <span className="text-foreground">training-ready:</span>{' '}
                        {getNumber(datasetPackagePreview?.training_ready_count) ?? 0}
                      </p>
                    </div>
                  </div>
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!run}
                      onClick={() => void handleExportAttemptBundle()}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Attempt Bundle
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!run}
                      onClick={() => void handleExportDatasetPackage()}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Dataset Package
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!trainingReadyDatasetPackagePreview || (getNumber(trainingReadyDatasetPackagePreview.dataset_record_count) ?? 0) === 0}
                      onClick={() => void handleExportTrainingReadyDatasetPackage()}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Training-Ready Only
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!researchPlan}
                      onClick={() => handleExportJson(researchPlan, 'research-plan')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Research Plan
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!sourceQualityTable}
                      onClick={() => handleExportJson(sourceQualityTable, 'source-quality-table')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Sources
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!claimEvidenceMap}
                      onClick={() => handleExportJson(claimEvidenceMap, 'claim-evidence-map')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Claims
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!researchBrief}
                      onClick={() => handleExportJson(researchBrief, 'research-brief')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Research Brief
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!datasetRecord}
                      onClick={() => handleExportJson(datasetRecord, 'dataset-record')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Dataset Record
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!evidenceSummary}
                      onClick={() => handleExportJson(evidenceSummary, 'evidence-summary')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Evidence Summary
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!verificationSummary}
                      onClick={() => handleExportJson(verificationSummary, 'verification-summary')}
                    >
                      <Download className="h-3.5 w-3.5" />
                      Verification Summary
                    </Button>
                  </div>
                  {exportError ? (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                      {exportError}
                    </div>
                  ) : null}
                  <p className="text-xs text-muted-foreground">
                    Attempt and dataset package downloads prefer backend package endpoints and fall back to the current run payload when the backend surface is unavailable.
                  </p>
                  <p className="text-xs text-muted-foreground">
                    `Training-Ready Only` exports the dataset package schema with non-ready records removed, not a separate schema.
                  </p>
                </CardContent>
              </Card>

              {roleOutputs.length > 0 ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Role Outputs</CardTitle>
                    <CardDescription>Structured view of subagent conversation turns, with filters and grouped threads.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                      <div>
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Filter By Role</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Button
                            variant={roleFilter === 'all' ? 'secondary' : 'ghost'}
                            size="sm"
                            onClick={() => setRoleFilter('all')}
                          >
                            All Roles
                          </Button>
                          {roleOptions.map((option) => (
                            <Button
                              key={option}
                              variant={roleFilter === option ? 'secondary' : 'ghost'}
                              size="sm"
                              onClick={() => setRoleFilter(option)}
                            >
                              {option}
                            </Button>
                          ))}
                        </div>
                      </div>
                      <div>
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Filter By Candidate</p>
                        <div className="mt-2 flex flex-wrap gap-2">
                          <Button
                            variant={candidateFilter === 'all' ? 'secondary' : 'ghost'}
                            size="sm"
                            onClick={() => setCandidateFilter('all')}
                          >
                            All Candidates
                          </Button>
                          {candidateOptions.map((option) => (
                            <Button
                              key={option}
                              variant={candidateFilter === option ? 'secondary' : 'ghost'}
                              size="sm"
                              onClick={() => setCandidateFilter(option)}
                            >
                              {option}
                            </Button>
                          ))}
                        </div>
                      </div>
                    </div>

                    {filteredRoleOutputs.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                        No role outputs match the current filters.
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {groupedRoleOutputs.map((group, groupIndex) => (
                          <div key={`${group.roleId}-${group.candidateId ?? 'none'}-${groupIndex}`} className="space-y-3 rounded-lg border border-border bg-surface-layer p-3">
                            <div className="flex flex-wrap items-center gap-2">
                              <Badge variant="outline">{group.roleId}</Badge>
                              {group.candidateId ? (
                                <Badge variant="outline">candidate: {group.candidateId}</Badge>
                              ) : null}
                              <Badge variant="neutral">{group.items.length} turns</Badge>
                            </div>
                            <div className="space-y-3">
                              {group.items.map((item, index) => (
                                <div key={`${item.roleId}-${index}-${item.timestamp ?? 'untimed'}`} className="rounded-lg border border-border bg-canvas p-3">
                                  <div className="flex flex-wrap items-start justify-between gap-2">
                                    <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                                      {item.roundIndex !== null ? <span>round: {item.roundIndex}</span> : null}
                                      {item.modelId ? <span>model: {item.modelId}</span> : null}
                                    </div>
                                    {item.timestamp ? (
                                      <span className="text-xs text-muted-foreground">{item.timestamp}</span>
                                    ) : null}
                                  </div>
                                  <pre className="mt-2 whitespace-pre-wrap break-words text-xs text-foreground">
                                    {item.content}
                                  </pre>
                                </div>
                              ))}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              ) : null}

              <Card>
                <CardHeader>
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <CardTitle>{run.title || run.topic || 'Untitled run'}</CardTitle>
                    <Badge variant={statusVariant(run.status)}>{run.status}</Badge>
                  </div>
                  <CardDescription>{run.protocol_id}</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-wrap gap-2">
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!canStart || runActionPending !== null}
                      loading={runActionPending === 'start'}
                      onClick={() => void handleRunAction('start')}
                    >
                      <Play className="h-3.5 w-3.5" />
                      Start
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!canPause || runActionPending !== null}
                      loading={runActionPending === 'pause'}
                      onClick={() => void handleRunAction('pause')}
                    >
                      <Pause className="h-3.5 w-3.5" />
                      Pause
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!canResume || runActionPending !== null}
                      loading={runActionPending === 'resume'}
                      onClick={() => void handleRunAction('resume')}
                    >
                      <Play className="h-3.5 w-3.5" />
                      Resume
                    </Button>
                    <Button
                      variant="secondary"
                      size="sm"
                      disabled={!canFinalizePartial || runActionPending !== null}
                      loading={runActionPending === 'finalize_partial'}
                      onClick={() => void handleRunAction('finalize_partial')}
                    >
                      <Archive className="h-3.5 w-3.5" />
                      {t('agentRuns.actions.finalizePartial')}
                    </Button>
                    <Button
                      variant="destructive"
                      size="sm"
                      disabled={!canCancel || runActionPending !== null}
                      loading={runActionPending === 'cancel'}
                      onClick={() => void handleRunAction('cancel')}
                    >
                      <Square className="h-3.5 w-3.5" />
                      Cancel
                    </Button>
                  </div>

                  {runActionError ? (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                      {runActionError}
                    </div>
                  ) : null}
                  {runActionSuccess ? (
                    <div className="rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success">
                      {runActionSuccess}
                    </div>
                  ) : null}

                  <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                    <p><span className="text-foreground">Topic:</span> {run.topic || 'N/A'}</p>
                    <p><span className="text-foreground">Created:</span> {formatDateTime(run.created_at)}</p>
                    <p><span className="text-foreground">Started:</span> {formatDateTime(run.started_at)}</p>
                    <p><span className="text-foreground">Finished:</span> {formatDateTime(run.finished_at)}</p>
                    <p className="sm:col-span-2">
                      <span className="text-foreground">Updated:</span> {formatDateTime(run.updated_at)}
                    </p>
                    {run.latest_error ? (
                      <p className="sm:col-span-2 text-destructive">
                        <span className="text-foreground">Latest error:</span> {run.latest_error}
                      </p>
                    ) : null}
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Event Stream</CardTitle>
                  <CardDescription>Conversation log and orchestration events.</CardDescription>
                </CardHeader>
                <CardContent>
                  {run.events.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                      No events yet.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {run.events.map((event, index) => {
                        const kind = eventKind(event)
                        const timestamp =
                          typeof event.timestamp === 'string' ? formatDateTime(event.timestamp) : null
                        return (
                          <div key={`${kind}-${index}`} className="rounded-lg border border-border bg-surface-layer p-3">
                            <div className="mb-2 flex items-center justify-between gap-2">
                              <Badge variant="outline">{kind}</Badge>
                              {timestamp ? (
                                <span className="text-xs text-muted-foreground">{timestamp}</span>
                              ) : null}
                            </div>
                            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words text-xs text-foreground">
                              {eventSummary(event)}
                            </pre>
                          </div>
                        )
                      })}
                    </div>
                  )}
                </CardContent>
              </Card>

              {researchPlan ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Plan</CardTitle>
                    <CardDescription>Research plan, subquestions, and generated evidence queries.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Status:</span> {getString(researchPlan.status) ?? 'N/A'}</p>
                      <p><span className="text-foreground">Planner model:</span> {getString(researchPlan.planner_model_id) ?? 'N/A'}</p>
                    </div>
                    {getStringArray(researchPlan.output_targets).length > 0 ? (
                      <div className="flex flex-wrap gap-2">
                        {getStringArray(researchPlan.output_targets).map((target) => (
                          <Badge key={target} variant="outline">{target}</Badge>
                        ))}
                      </div>
                    ) : null}
                    {[
                      ['Subquestions', getStringArray(researchPlan.subquestions)],
                      ['Evidence Requirements', getStringArray(researchPlan.evidence_requirements)],
                      ['Exclusion Rules', getStringArray(researchPlan.exclusion_rules)],
                      ['Evidence Queries', getStringArray(researchPlan.evidence_queries)],
                    ].map(([label, values]) =>
                      Array.isArray(values) && values.length > 0 ? (
                        <div key={label as string} className="space-y-2">
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">{label as string}</p>
                          <div className="space-y-2">
                            {values.map((value) => (
                              <div key={value} className="rounded border border-border bg-surface-layer px-3 py-2 text-sm text-foreground">
                                {value}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null
                    )}
                  </CardContent>
                </Card>
              ) : null}

              {sourceQualityTable ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Sources</CardTitle>
                    <CardDescription>Normalized source quality records for the current attempt scope.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Source count:</span> {getNumber(sourceQualityTable.source_count) ?? 0}</p>
                      <p><span className="text-foreground">Quality tiers:</span> {jsonPreview(sourceQualityTable.tier_counts ?? {})}</p>
                    </div>
                    <div className="space-y-3">
                      {getRecordArray(sourceQualityTable.sources).map((item, index) => (
                        <div key={`${getString(item.evidence_id) ?? 'source'}-${index}`} className="rounded-lg border border-border bg-surface-layer p-3 text-sm">
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div>
                              <p className="font-medium text-foreground">
                                {getString(item.title) ?? getString(item.evidence_id) ?? `Source ${index + 1}`}
                              </p>
                              <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                                {getString(item.source_type) ? <span>type: {getString(item.source_type)}</span> : null}
                                {getString(item.provider) ? <span>provider: {getString(item.provider)}</span> : null}
                                {getString(item.query) ? <span>query: {getString(item.query)}</span> : null}
                              </div>
                            </div>
                            <Badge variant="outline">
                              {getString(item.source_quality_tier) ?? 'unknown'}
                            </Badge>
                          </div>
                          {getString(item.quality_rationale) ? (
                            <p className="mt-2 text-muted-foreground">{getString(item.quality_rationale)}</p>
                          ) : null}
                          {getString(item.url) ? (
                            <p className="mt-2 break-all text-xs text-muted-foreground">{getString(item.url)}</p>
                          ) : null}
                          {getString(item.content_preview) ? (
                            <pre className="mt-2 whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                              {getString(item.content_preview)}
                            </pre>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ) : null}

              {claimEvidenceMap ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Claims</CardTitle>
                    <CardDescription>Claim-level support status, confidence, and citation coverage.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Citation policy:</span> {getString(claimEvidenceMap.citation_policy) ?? 'N/A'}</p>
                      <p><span className="text-foreground">Counts:</span> {jsonPreview(claimEvidenceMap.counts ?? {})}</p>
                    </div>
                    <div className="space-y-3">
                      {getRecordArray(claimEvidenceMap.claims).map((item, index) => (
                        <div key={`${getString(item.claim_id) ?? 'claim'}-${index}`} className="rounded-lg border border-border bg-surface-layer p-3 text-sm">
                          <div className="flex flex-wrap items-start justify-between gap-2">
                            <div>
                              <p className="font-medium text-foreground">
                                {getString(item.claim) ?? `Claim ${index + 1}`}
                              </p>
                              <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                                {getString(item.speaker_role_id) ? <span>role: {getString(item.speaker_role_id)}</span> : null}
                                {getNumber(item.round_index) !== null ? <span>round: {getNumber(item.round_index)}</span> : null}
                              </div>
                            </div>
                            <div className="flex flex-wrap gap-2">
                              <Badge variant="outline">{getString(item.support_status) ?? 'unknown'}</Badge>
                              <Badge variant="outline">
                                confidence: {getNumber(item.confidence) ?? 0}
                              </Badge>
                            </div>
                          </div>
                          {getString(item.rationale) ? (
                            <p className="mt-2 text-muted-foreground">{getString(item.rationale)}</p>
                          ) : null}
                          <div className="mt-3 flex flex-wrap gap-2">
                            {getStringArray(item.citation_refs).map((citation) => (
                              <Badge key={citation} variant="outline">{citation}</Badge>
                            ))}
                            {getString(item.source_quality_tier) ? (
                              <Badge variant="outline">tier: {getString(item.source_quality_tier)}</Badge>
                            ) : null}
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              ) : null}

              {researchBrief ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Report</CardTitle>
                    <CardDescription>Rendered research brief for the selected candidate.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Status:</span> {getString(researchBrief.status) ?? 'N/A'}</p>
                      <p><span className="text-foreground">Synthesizer model:</span> {getString(researchBrief.synthesizer_model_id) ?? 'N/A'}</p>
                    </div>
                    {getString(researchBrief.markdown) ? (
                      <pre className="whitespace-pre-wrap break-words rounded border border-border bg-canvas p-3 text-sm text-foreground">
                        {getString(researchBrief.markdown)}
                      </pre>
                    ) : null}
                  </CardContent>
                </Card>
              ) : null}

              <Card>
                <CardHeader>
                  <CardTitle>Artifacts</CardTitle>
                  <CardDescription>Outputs attached by the run, scoped to the current attempt filter.</CardDescription>
                </CardHeader>
                <CardContent>
                  {filteredArtifacts.length === 0 ? (
                    <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                      No artifacts match the current attempt scope.
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {filteredArtifacts.map((artifact, index) => (
                        <div
                          key={artifact.artifact_id || `${artifact.artifact_type}-${index}`}
                          className="rounded-lg border border-border bg-surface-layer p-3"
                        >
                          <div className="mb-1 flex items-center justify-between gap-2">
                            <div>
                              <p className="text-sm font-medium text-foreground">
                                {artifact.title || artifact.artifact_type}
                              </p>
                              {getString(artifact.metadata.attempt_id) ? (
                                <p className="text-xs text-muted-foreground">
                                  attempt: {getString(artifact.metadata.attempt_id)}
                                </p>
                              ) : null}
                            </div>
                            <div className="flex flex-wrap gap-2">
                              {getString(artifact.metadata.attempt_id) ? (
                                <Badge variant="neutral">attempt-linked</Badge>
                              ) : null}
                              <Badge variant="outline">{artifact.artifact_type}</Badge>
                            </div>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            URI: {artifact.uri || 'N/A'}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            MIME: {artifact.mime_type || 'N/A'} | Size: {artifact.size_bytes ?? 'N/A'}
                          </p>
                          {Object.keys(artifact.metadata).length > 0 ? (
                            <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                              {jsonPreview(artifact.metadata)}
                            </pre>
                          ) : null}
                        </div>
                      ))}
                    </div>
                  )}
                </CardContent>
              </Card>
            </div>

            <div className="space-y-6">
              {evidenceSummary ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Evidence Summary</CardTitle>
                    <CardDescription>Structured retrieval summary for this run.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Mode:</span> {typeof evidenceSummary.mode === 'string' ? evidenceSummary.mode : 'N/A'}</p>
                      <p><span className="text-foreground">RAG provider:</span> {typeof evidenceSummary.rag_provider === 'string' ? evidenceSummary.rag_provider : 'N/A'}</p>
                      <p><span className="text-foreground">Queries:</span> {typeof evidenceSummary.query_count === 'number' ? evidenceSummary.query_count : 'N/A'}</p>
                      <p><span className="text-foreground">Collected packets:</span> {typeof evidenceSummary.collected_packet_count === 'number' ? evidenceSummary.collected_packet_count : 'N/A'}</p>
                      <p><span className="text-foreground">Provided packets:</span> {typeof evidenceSummary.provided_packet_count === 'number' ? evidenceSummary.provided_packet_count : 'N/A'}</p>
                      <p><span className="text-foreground">Total packets:</span> {typeof evidenceSummary.total_packet_count === 'number' ? evidenceSummary.total_packet_count : 'N/A'}</p>
                    </div>

                    {getNumberRecord(evidenceSummary.retrieval_path_counts).length > 0 ? (
                      <div className="space-y-2">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Retrieval Paths</p>
                        <div className="flex flex-wrap gap-2">
                          {getNumberRecord(evidenceSummary.retrieval_path_counts).map(([name, count]) => (
                            <Badge key={name} variant="outline">
                              {name}: {count}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {getNumberRecord(evidenceSummary.provider_counts).length > 0 ? (
                      <div className="space-y-2">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Providers</p>
                        <div className="flex flex-wrap gap-2">
                          {getNumberRecord(evidenceSummary.provider_counts).map(([name, count]) => (
                            <Badge key={name} variant="outline">
                              {name}: {count}
                            </Badge>
                          ))}
                        </div>
                      </div>
                    ) : null}

                    {Array.isArray(evidenceSummary.queries) && evidenceSummary.queries.length > 0 ? (
                      <div className="space-y-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Query Breakdown</p>
                        {evidenceSummary.queries.map((item, index) => {
                          const query = isRecord(item) ? item : {}
                          return (
                            <div key={index} className="rounded-lg border border-border bg-surface-layer p-3 text-sm">
                              <p className="font-medium text-foreground">
                                {typeof query.query === 'string' ? query.query : `Query ${index + 1}`}
                              </p>
                              <div className="mt-1 flex flex-wrap gap-2 text-xs text-muted-foreground">
                                {typeof query.provider === 'string' ? <span>provider: {query.provider}</span> : null}
                                {typeof query.packet_count === 'number' ? <span>packets: {query.packet_count}</span> : null}
                                {typeof query.error === 'string' && query.error ? (
                                  <span className="text-destructive">error: {query.error}</span>
                                ) : null}
                              </div>
                            </div>
                          )
                        })}
                      </div>
                    ) : null}

                    {getRecordArray(evidenceSummary.evidence_packets).length > 0 ? (
                      <div className="space-y-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Evidence Packets</p>
                        {getRecordArray(evidenceSummary.evidence_packets).map((item, index) => {
                          const content = typeof item.content === 'string' ? item.content : ''
                          return (
                            <div key={`${item.evidence_id ?? 'packet'}-${index}`} className="rounded-lg border border-border bg-surface-layer p-3 text-sm">
                              <div className="flex flex-wrap items-start justify-between gap-2">
                                <div className="space-y-1">
                                  <p className="font-medium text-foreground">
                                    {typeof item.title === 'string' && item.title.trim()
                                      ? item.title
                                      : typeof item.evidence_id === 'string'
                                        ? item.evidence_id
                                        : `Packet ${index + 1}`}
                                  </p>
                                  <div className="flex flex-wrap gap-2 text-xs text-muted-foreground">
                                    {typeof item.source_type === 'string' ? <span>source: {item.source_type}</span> : null}
                                    {typeof item.provider === 'string' ? <span>provider: {item.provider}</span> : null}
                                    {typeof item.query === 'string' ? <span>query: {item.query}</span> : null}
                                    {typeof item.rank === 'number' ? <span>rank: {item.rank}</span> : null}
                                  </div>
                                </div>
                                {typeof item.source_type === 'string' ? (
                                  <Badge variant="outline">{item.source_type}</Badge>
                                ) : null}
                              </div>

                              {typeof item.url === 'string' && item.url.trim() ? (
                                <p className="mt-2 break-all text-xs text-muted-foreground">
                                  {item.url}
                                </p>
                              ) : null}

                              {content ? (
                                <pre className="mt-2 whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                                  {truncatePreview(content)}
                                </pre>
                              ) : null}
                            </div>
                          )
                        })}
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              ) : null}

              {verificationSummary ? (
                <Card>
                  <CardHeader>
                    <CardTitle>Verification Summary</CardTitle>
                    <CardDescription>Verifier output and evidence gating results.</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-2 text-sm text-muted-foreground sm:grid-cols-2">
                      <p><span className="text-foreground">Verifier model:</span> {typeof verificationSummary.verifier_model_id === 'string' ? verificationSummary.verifier_model_id : 'N/A'}</p>
                      <p><span className="text-foreground">Evidence packets:</span> {typeof verificationSummary.evidence_packet_count === 'number' ? verificationSummary.evidence_packet_count : 'N/A'}</p>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {getStringArray(verificationSummary.verified_candidate_ids).map((candidateId) => (
                        <Badge key={`verified-${candidateId}`} variant="success">
                          verified: {candidateId}
                        </Badge>
                      ))}
                      {getStringArray(verificationSummary.failed_candidate_ids).map((candidateId) => (
                        <Badge key={`failed-${candidateId}`} variant="error">
                          failed: {candidateId}
                        </Badge>
                      ))}
                      {getStringArray(verificationSummary.skipped_candidate_ids).map((candidateId) => (
                        <Badge key={`skipped-${candidateId}`} variant="warning">
                          skipped: {candidateId}
                        </Badge>
                      ))}
                    </div>

                    {Array.isArray(verificationSummary.verifications) && verificationSummary.verifications.length > 0 ? (
                      <div className="space-y-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Candidate Checks</p>
                        {verificationSummary.verifications.map((item, index) => {
                          const verification = isRecord(item) ? item : {}
                          const issues = getStringArray(verification.issues)
                          const citations = getRecordArray(verification.citations)
                          return (
                            <div key={index} className="rounded-lg border border-border bg-surface-layer p-3 text-sm">
                              <div className="flex items-center justify-between gap-2">
                                <p className="font-medium text-foreground">
                                  {typeof verification.candidate_id === 'string' ? verification.candidate_id : `candidate-${index + 1}`}
                                </p>
                                <Badge variant="outline">
                                  {typeof verification.status === 'string' ? verification.status : 'unknown'}
                                </Badge>
                              </div>
                              {typeof verification.rationale === 'string' && verification.rationale ? (
                                <p className="mt-2 text-muted-foreground">{verification.rationale}</p>
                              ) : null}
                              {issues.length > 0 ? (
                                <div className="mt-2 flex flex-wrap gap-2">
                                  {issues.map((issue) => (
                                    <Badge key={issue} variant="outline">
                                      {issue}
                                    </Badge>
                                  ))}
                                </div>
                              ) : null}

                              {citations.length > 0 ? (
                                <div className="mt-3 space-y-2">
                                  <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Citations</p>
                                  {citations.map((citation, citationIndex) => {
                                    const evidenceId = getString(citation.evidence_id)
                                    const linkedPacket = evidenceId ? evidencePacketIndex.get(evidenceId) ?? null : null
                                    return (
                                      <div
                                        key={`${evidenceId ?? 'citation'}-${citationIndex}`}
                                        className="rounded border border-border bg-canvas p-2 text-xs"
                                      >
                                        <div className="flex flex-wrap gap-2 text-muted-foreground">
                                          {evidenceId ? <span>evidence: {evidenceId}</span> : null}
                                          {linkedPacket && typeof linkedPacket.source_type === 'string' ? (
                                            <span>source: {linkedPacket.source_type}</span>
                                          ) : null}
                                          {linkedPacket && typeof linkedPacket.provider === 'string' ? (
                                            <span>provider: {linkedPacket.provider}</span>
                                          ) : null}
                                        </div>
                                        {typeof citation.summary === 'string' && citation.summary ? (
                                          <p className="mt-1 text-foreground">{citation.summary}</p>
                                        ) : null}
                                        {linkedPacket ? (
                                          <div className="mt-2 rounded border border-border bg-surface-layer p-2">
                                            <p className="font-medium text-foreground">
                                              {getString(linkedPacket.title) ?? evidenceId ?? 'Linked evidence'}
                                            </p>
                                            {typeof linkedPacket.url === 'string' && linkedPacket.url.trim() ? (
                                              <p className="mt-1 break-all text-muted-foreground">{linkedPacket.url}</p>
                                            ) : null}
                                            {typeof linkedPacket.content === 'string' && linkedPacket.content.trim() ? (
                                              <pre className="mt-2 whitespace-pre-wrap break-words text-foreground">
                                                {truncatePreview(linkedPacket.content)}
                                              </pre>
                                            ) : null}
                                          </div>
                                        ) : null}
                                      </div>
                                    )
                                  })}
                                </div>
                              ) : null}
                            </div>
                          )
                        })}
                      </div>
                    ) : null}
                  </CardContent>
                </Card>
              ) : null}

              <Card>
                <CardHeader>
                  <CardTitle>Submit Guidance</CardTitle>
                  <CardDescription>
                    Send additional instructions while the run is active.
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <Textarea
                    value={guidanceText}
                    onChange={(event) => setGuidanceText(event.target.value)}
                    placeholder={isTerminal ? 'Run is finished. Guidance is disabled.' : 'Refine direction, constraints, or priorities...'}
                    minRows={4}
                    disabled={isTerminal || guidancePending}
                  />
                  {guidanceError ? (
                    <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                      {guidanceError}
                    </div>
                  ) : null}
                  {guidanceSuccess ? (
                    <div className="rounded-md border border-success/30 bg-success/10 px-3 py-2 text-xs text-success">
                      {guidanceSuccess}
                    </div>
                  ) : null}
                  <Button
                    variant="primary"
                    size="md"
                    onClick={() => void handleSubmitGuidance()}
                    loading={guidancePending}
                    disabled={isTerminal || guidanceText.trim().length === 0}
                    className="w-full"
                  >
                    <SendHorizontal className="h-4 w-4" />
                    Send Guidance
                  </Button>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Schedule</CardTitle>
                  <CardDescription>Automation status and next trigger metadata.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3 text-sm text-muted-foreground">
                  {Object.keys(run.schedule).length === 0 ? (
                    <p>No automation schedule configured for this run.</p>
                  ) : (
                    <>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Mode</p>
                          <p className="mt-1 text-foreground">{describeSchedule(run.schedule)}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Status</p>
                          <p className="mt-1 text-foreground">{getString(run.schedule.schedule_status) ?? 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Next Run</p>
                          <p className="mt-1 text-foreground">{formatDateTime(getString(run.schedule.next_run_at))}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Last Scheduled</p>
                          <p className="mt-1 text-foreground">{formatDateTime(getString(run.schedule.last_scheduled_at))}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Last Completed</p>
                          <p className="mt-1 text-foreground">{formatDateTime(getString(run.schedule.last_completed_at))}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Completion Status</p>
                          <p className="mt-1 text-foreground">{getString(run.schedule.last_completion_status) ?? 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Health</p>
                          <p className="mt-1 text-foreground">{getString(run.schedule.health_status) ?? 'N/A'}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Failure Streak</p>
                          <p className="mt-1 text-foreground">{getNumber(run.schedule.failure_streak) ?? 0}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Timezone</p>
                          <p className="mt-1 text-foreground">{getString(run.schedule.timezone) ?? 'UTC'}</p>
                        </div>
                        <div>
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Start Immediately</p>
                          <p className="mt-1 text-foreground">{run.schedule.start_immediately === true ? 'Yes' : 'No'}</p>
                        </div>
                      </div>
                      {getNumber(run.schedule.max_runs) !== null ? (
                        <p><span className="text-foreground">Max runs:</span> {getNumber(run.schedule.max_runs)}</p>
                      ) : null}
                      <p>
                        <span className="text-foreground">Auto pause on failure:</span>{' '}
                        {run.schedule.auto_pause_on_failure === true ? 'Yes' : 'No'}
                      </p>
                      <p>
                        <span className="text-foreground">Attempt count:</span>{' '}
                        {getNumber(run.schedule.attempt_count) ?? 0}
                      </p>
                      {getNumber(run.schedule.interval_seconds) !== null ? (
                        <p><span className="text-foreground">Interval:</span> {getNumber(run.schedule.interval_seconds)} seconds</p>
                      ) : null}
                      {getString(run.schedule.cron) ? (
                        <p><span className="text-foreground">Cron:</span> {getString(run.schedule.cron)}</p>
                      ) : null}
                      {getString(run.schedule.run_at) ? (
                        <p><span className="text-foreground">Run at:</span> {formatDateTime(getString(run.schedule.run_at))}</p>
                      ) : null}
                      {recentScheduleAttempts.length > 0 ? (
                        <div className="space-y-2">
                          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                            Recent Attempts
                          </p>
                          <div className="space-y-2">
                            {recentScheduleAttempts.map((attempt, index) => (
                              <div
                                key={`${getNumber(attempt.attempt_number) ?? index}-${getString(attempt.completed_at) ?? index}`}
                                className="rounded-md border border-border bg-canvas p-3"
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div>
                                    <p className="text-sm font-medium text-foreground">
                                      Attempt {getNumber(attempt.attempt_number) ?? index + 1}
                                    </p>
                                    {getString(attempt.attempt_id) ? (
                                      <p className="text-xs text-muted-foreground">
                                        ID: {getString(attempt.attempt_id)}
                                      </p>
                                    ) : null}
                                    <p className="text-xs text-muted-foreground">
                                      Started: {formatDateTime(getString(attempt.started_at))}
                                    </p>
                                    <p className="text-xs text-muted-foreground">
                                      Completed: {formatDateTime(getString(attempt.completed_at))}
                                    </p>
                                  </div>
                                  <Badge variant={statusVariant(scheduleAttemptStatusLabel(attempt))}>
                                    {scheduleAttemptStatusLabel(attempt)}
                                  </Badge>
                                </div>
                                {getString(attempt.selected_candidate_id) ? (
                                  <p className="mt-2 text-xs text-muted-foreground">
                                    <span className="text-foreground">Selected candidate:</span>{' '}
                                    {getString(attempt.selected_candidate_id)}
                                  </p>
                                ) : null}
                                {getString(attempt.final_answer_preview) ? (
                                  <p className="mt-2 whitespace-pre-wrap text-xs text-muted-foreground">
                                    <span className="text-foreground">Answer preview:</span>{' '}
                                    {getString(attempt.final_answer_preview)}
                                  </p>
                                ) : null}
                                {getString(attempt.latest_error) ? (
                                  <p className="mt-2 whitespace-pre-wrap text-xs text-destructive">
                                    {getString(attempt.latest_error)}
                                  </p>
                                ) : null}
                              </div>
                            ))}
                          </div>
                        </div>
                      ) : null}
                    </>
                  )}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Run Metadata</CardTitle>
                  <CardDescription>Raw protocol metadata from backend.</CardDescription>
                </CardHeader>
                <CardContent className="space-y-3">
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">selected_models_roles</p>
                    <pre className="max-h-52 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                      {jsonPreview(run.selected_models_roles)}
                    </pre>
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">evaluation_policy</p>
                    <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                      {jsonPreview(run.evaluation_policy)}
                    </pre>
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">schedule</p>
                    <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                      {jsonPreview(run.schedule)}
                    </pre>
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">summary</p>
                    <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                      {jsonPreview(run.summary)}
                    </pre>
                  </div>
                  <div>
                    <p className="mb-1 text-xs font-medium uppercase tracking-wide text-muted-foreground">evidence_status</p>
                    <pre className="max-h-40 overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-canvas p-2 text-xs text-foreground">
                      {jsonPreview(run.evidence_status)}
                    </pre>
                  </div>
                </CardContent>
              </Card>
            </div>
          </div>
        ) : (
          <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-sm text-muted-foreground">
            Run not found.
          </div>
        )}
      </div>
    </div>
  )
}
