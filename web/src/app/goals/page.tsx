'use client'

import * as React from 'react'
import Link from 'next/link'
import {
  AlertTriangle,
  BrainCircuit,
  Clock3,
  Database,
  Pause,
  Play,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldAlert,
  Square,
  Waypoints,
  Workflow,
} from 'lucide-react'
import * as api from '@/lib/api'
import { Badge, type BadgeProps } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Separator } from '@/components/ui/separator'
import { Textarea } from '@/components/ui/textarea'
import { useProjectStore } from '@/lib/stores/project-store'
import { cn } from '@/lib/utils'

const GOAL_PROTOCOL_OPTIONS = [
  { value: 'none', label: 'Unspecified protocol' },
  { value: 'teacher_student_distill', label: 'Teacher Student Distill' },
  { value: 'multi_agent_debate', label: 'Multi Agent Debate' },
  { value: 'dr_zero_self_evolve', label: 'Dr.Zero Self-Evolve' },
] as const

type GoalAction = 'start' | 'pause' | 'resume' | 'refresh' | 'finalize_partial' | 'cancel'

function formatDateTime(value: string | null | undefined): string {
  if (!value) {
    return 'N/A'
  }
  const timestamp = Date.parse(value)
  if (Number.isNaN(timestamp)) {
    return value
  }
  return new Date(timestamp).toLocaleString()
}

function formatDurationSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'N/A'
  }
  if (value < 60) {
    return `${Math.round(value)}s`
  }
  if (value < 3600) {
    return `${(value / 60).toFixed(1)}m`
  }
  if (value < 86400) {
    return `${(value / 3600).toFixed(1)}h`
  }
  return `${(value / 86400).toFixed(1)}d`
}

function parseCsvList(value: string): string[] {
  return value
    .split(',')
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
}

function parseOptionalInteger(value: string): number | undefined {
  const parsed = Number.parseInt(value, 10)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return undefined
  }
  return parsed
}

function parseOptionalNumber(value: string): number | undefined {
  const parsed = Number.parseFloat(value)
  if (!Number.isFinite(parsed) || parsed <= 0) {
    return undefined
  }
  return parsed
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function getBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value : null
}

function getNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function getRecordArray(value: unknown): Array<Record<string, unknown>> {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function getStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value
        .map((item) => (typeof item === 'string' ? item.trim() : ''))
        .filter((item) => item.length > 0)
    : []
}

function previewText(value: string, max = 120): string {
  const trimmed = value.trim()
  if (trimmed.length <= max) {
    return trimmed
  }
  return `${trimmed.slice(0, max)}...`
}

function compareIsoDesc(left: string | null | undefined, right: string | null | undefined): number {
  const leftTime = left ? Date.parse(left) : Number.NaN
  const rightTime = right ? Date.parse(right) : Number.NaN
  const normalizedLeft = Number.isNaN(leftTime) ? 0 : leftTime
  const normalizedRight = Number.isNaN(rightTime) ? 0 : rightTime
  return normalizedRight - normalizedLeft
}

function statusVariant(status: string): BadgeProps['variant'] {
  const normalized = status.toLowerCase()
  if (
    normalized === 'running' ||
    normalized === 'queued' ||
    normalized === 'waiting_approval' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled'
  ) {
    return 'warning'
  }
  if (normalized === 'completed' || normalized === 'succeeded') {
    return 'success'
  }
  if (normalized === 'failed' || normalized === 'cancelled') {
    return 'error'
  }
  return 'neutral'
}

function JsonPreview({
  value,
  emptyLabel = 'No data',
}: {
  value: unknown
  emptyLabel?: string
}) {
  const formatted = React.useMemo(() => {
    if (!value) {
      return null
    }
    if (typeof value === 'object' && !Array.isArray(value) && Object.keys(value as Record<string, unknown>).length === 0) {
      return null
    }
    if (Array.isArray(value) && value.length === 0) {
      return null
    }
    try {
      return JSON.stringify(value, null, 2)
    } catch {
      return String(value)
    }
  }, [value])

  return formatted ? (
    <pre className="overflow-x-auto rounded-md border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
      {formatted}
    </pre>
  ) : (
    <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
      {emptyLabel}
    </div>
  )
}

function LabeledValue({
  label,
  value,
}: {
  label: string
  value: React.ReactNode
}) {
  return (
    <div className="space-y-1">
      <p className="text-[11px] font-medium uppercase tracking-wide text-muted-foreground">{label}</p>
      <div className="text-sm text-foreground">{value}</div>
    </div>
  )
}

function GoalStatePanel({
  goal,
  goalHealth,
}: {
  goal: api.GoalSummary | null | undefined
  goalHealth: api.GoalHealthSummary | null | undefined
}) {
  if (!goal) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No goal is selected.
      </div>
    )
  }

  const recommendedAction = isRecord(goalHealth?.recommended_next_action)
    ? getString(goalHealth?.recommended_next_action.action)
    : null
  const openFindingCount = goalHealth?.open_findings.length ?? 0

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Goal id" value={goal.goal_id} />
        <LabeledValue
          label="Status"
          value={<Badge variant={statusVariant(goal.status)}>{goal.status}</Badge>}
        />
        <LabeledValue label="Protocol" value={goal.protocol_id || 'Unspecified'} />
        <LabeledValue label="Type" value={goal.goal_type || 'Unspecified'} />
        <LabeledValue label="Project" value={goal.project_id || 'None'} />
        <LabeledValue label="Workspace" value={goal.workspace_dir || 'Not set'} />
        <LabeledValue label="Attempts" value={goal.attempts.length} />
        <LabeledValue
          label="Current attempt"
          value={goal.current_attempt_id || goalHealth?.current_attempt_id || 'None'}
        />
        <LabeledValue label="Recommended action" value={recommendedAction || 'Inspect'} />
        <LabeledValue label="Open findings" value={openFindingCount} />
        <LabeledValue label="Created" value={formatDateTime(goal.created_at)} />
        <LabeledValue label="Updated" value={formatDateTime(goal.updated_at)} />
        <LabeledValue label="Started" value={formatDateTime(goal.started_at)} />
        <LabeledValue label="Finished" value={formatDateTime(goal.finished_at)} />
        <LabeledValue label="Runtime owner" value={goalHealth?.runtime_owner_id || 'N/A'} />
      </div>
      <JsonPreview value={goal.summary} emptyLabel="No goal summary." />
    </div>
  )
}

function RuntimeBudgetPanel({
  runtimeBudget,
}: {
  runtimeBudget: Record<string, unknown> | null | undefined
}) {
  if (!runtimeBudget || Object.keys(runtimeBudget).length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No runtime budget reported.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue
          label="Status"
          value={
            <Badge variant={statusVariant(getString(runtimeBudget.status) || 'unknown')}>
              {getString(runtimeBudget.status) || 'unknown'}
            </Badge>
          }
        />
        <LabeledValue label="Mode" value={getString(runtimeBudget.runtime_mode) || 'N/A'} />
        <LabeledValue
          label="Requested duration"
          value={
            getNumber(runtimeBudget.requested_duration_sec) !== null
              ? formatDurationSeconds(getNumber(runtimeBudget.requested_duration_sec))
              : getString(runtimeBudget.requested_duration_text) || 'N/A'
          }
        />
        <LabeledValue label="Started" value={formatDateTime(getString(runtimeBudget.started_at))} />
        <LabeledValue label="Soft deadline" value={formatDateTime(getString(runtimeBudget.soft_deadline_at))} />
        <LabeledValue label="Hard stop" value={formatDateTime(getString(runtimeBudget.hard_stop_at))} />
        <LabeledValue label="Elapsed" value={formatDurationSeconds(getNumber(runtimeBudget.elapsed_sec))} />
        <LabeledValue label="Remaining" value={formatDurationSeconds(getNumber(runtimeBudget.remaining_sec))} />
        <LabeledValue label="Retry limit" value={getNumber(runtimeBudget.max_attempt_retries) ?? 'N/A'} />
        <LabeledValue label="Attempts used" value={getNumber(runtimeBudget.attempt_count) ?? 'N/A'} />
        <LabeledValue
          label="Retry limit reached"
          value={getBoolean(runtimeBudget.retry_limit_reached) ? 'Yes' : 'No'}
        />
        <LabeledValue
          label="Hard stop reached"
          value={getBoolean(runtimeBudget.hard_stop_reached) ? 'Yes' : 'No'}
        />
      </div>
      <JsonPreview value={runtimeBudget} emptyLabel="No runtime budget reported." />
    </div>
  )
}

function ApprovalStatePanel({
  approvalState,
  approvals,
  loadingApprovals,
  approvalError,
  resolvingApprovalKey,
  onResolveApproval,
}: {
  approvalState: Record<string, unknown> | null | undefined
  approvals: api.ApprovalSummary[]
  loadingApprovals: boolean
  approvalError: string | null
  resolvingApprovalKey: string | null
  onResolveApproval: (
    approvalId: string,
    decision: 'approve_once' | 'reject'
  ) => void | Promise<void>
}) {
  if (!approvalState || Object.keys(approvalState).length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No approval wait is active.
      </div>
    )
  }

  const pendingCount = getNumber(approvalState.pending_count) ?? approvals.length
  const approvalIds = getStringArray(approvalState.approval_ids)
  const toolNames = getStringArray(approvalState.tool_names)
  const pendingApprovals = getRecordArray(approvalState.pending_approvals)

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue
          label="Status"
          value={
            <Badge variant={statusVariant(getString(approvalState.status) || 'unknown')}>
              {getString(approvalState.status) || 'unknown'}
            </Badge>
          }
        />
        <LabeledValue label="Pending count" value={pendingCount} />
        <LabeledValue label="Tools" value={toolNames.join(', ') || 'N/A'} />
        <LabeledValue
          label="Wait started"
          value={formatDateTime(getString(approvalState.approval_wait_started_at))}
        />
        <LabeledValue
          label="Wait elapsed"
          value={formatDurationSeconds(getNumber(approvalState.approval_wait_elapsed_sec))}
        />
        <LabeledValue
          label="Wait timeout"
          value={formatDurationSeconds(getNumber(approvalState.approval_wait_timeout_sec))}
        />
        <LabeledValue
          label="Wait expires"
          value={formatDateTime(getString(approvalState.approval_wait_expires_at))}
        />
        <LabeledValue label="Approval ids" value={approvalIds.join(', ') || 'N/A'} />
      </div>

      {approvalError ? (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {approvalError}
        </div>
      ) : null}

      {loadingApprovals ? (
        <div className="rounded-md border border-border bg-surface-layer px-3 py-4 text-xs text-muted-foreground">
          Loading pending approvals...
        </div>
      ) : approvals.length > 0 ? (
        <div className="space-y-3">
          {approvals.map((approval) => (
            <div key={approval.approval_id} className="rounded-lg border border-border bg-surface-layer p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <Badge variant={statusVariant(approval.status)}>{approval.status}</Badge>
                    <span className="text-sm font-semibold text-foreground">{approval.tool_name}</span>
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {approval.command
                      ? previewText(approval.command, 140)
                      : approval.reason || approval.policy_reason || approval.approval_id}
                  </p>
                  <p className="text-xs text-muted-foreground">
                    {approval.workdir ? `${approval.shell || 'shell'} | ${approval.workdir}` : approval.shell || 'No shell'}
                  </p>
                </div>
                <div className="flex gap-2">
                  {approval.allowed_decisions.includes('approve_once') ? (
                    <Button
                      size="sm"
                      variant="secondary"
                      onClick={() => void onResolveApproval(approval.approval_id, 'approve_once')}
                      loading={resolvingApprovalKey === `${approval.approval_id}:approve_once`}
                    >
                      Approve once
                    </Button>
                  ) : null}
                  {approval.allowed_decisions.includes('reject') ? (
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={() => void onResolveApproval(approval.approval_id, 'reject')}
                      loading={resolvingApprovalKey === `${approval.approval_id}:reject`}
                    >
                      Reject
                    </Button>
                  ) : null}
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : pendingApprovals.length > 0 ? (
        <div className="space-y-3">
          {pendingApprovals.map((approval, index) => (
            <div key={`${getString(approval.approval_id) || 'pending'}-${index}`} className="rounded-lg border border-border bg-surface-layer p-4">
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    {getString(approval.tool_name) || 'Pending approval'}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {getString(approval.reason) || getString(approval.request_id) || 'Approval metadata only'}
                  </p>
                </div>
                <Badge variant="warning">{getString(approval.status) || 'pending'}</Badge>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
          Approval wait is active, but no pending approval payload is currently available on this page.
        </div>
      )}

      <JsonPreview value={approvalState} emptyLabel="No approval wait is active." />
    </div>
  )
}

function CurrentGenerationPanel({
  generation,
}: {
  generation: api.GoalWorkerGenerationSummary | null | undefined
}) {
  if (!generation) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No worker generation has been opened yet.
      </div>
    )
  }

  const finishReasonEntries = Object.entries(generation.observed_finish_reason_counts ?? {})

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue
          label="Generation"
          value={`#${generation.generation_index ?? generation.generation_id}`}
        />
        <LabeledValue
          label="Status"
          value={
            <Badge variant={statusVariant(generation.status || 'unknown')}>
              {generation.status || 'unknown'}
            </Badge>
          }
        />
        <LabeledValue label="Rollover" value={generation.rollover_reason || 'Initial worker'} />
        <LabeledValue label="Elapsed" value={formatDurationSeconds(generation.elapsed_sec)} />
        <LabeledValue
          label="Refresh interval"
          value={formatDurationSeconds(generation.generation_refresh_interval_sec)}
        />
        <LabeledValue
          label="Refresh due"
          value={
            generation.refresh_due === undefined ? 'Unknown' : generation.refresh_due ? 'Due' : 'No'
          }
        />
        <LabeledValue
          label="Refresh overdue"
          value={
            generation.refresh_overdue_sec !== undefined
              ? formatDurationSeconds(generation.refresh_overdue_sec)
              : 'None'
          }
        />
        <LabeledValue label="Started" value={formatDateTime(generation.started_at)} />
        <LabeledValue label="Finished" value={formatDateTime(generation.finished_at)} />
        <LabeledValue label="Run id" value={generation.agent_run_id || 'None'} />
        <LabeledValue label="Parent" value={generation.parent_generation_id ?? 'None'} />
        <LabeledValue
          label="Resume snapshot"
          value={generation.resume_source_snapshot_id ?? 'None'}
        />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-4">
          <div>
            <p className="text-sm font-semibold text-foreground">Context handoff</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Generation-local context pressure projected from debate snapshots.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <LabeledValue
              label="Usage ratio"
              value={generation.usage_ratio !== undefined ? generation.usage_ratio.toFixed(2) : 'N/A'}
            />
            <LabeledValue
              label="Threshold"
              value={
                generation.context_handoff_threshold !== undefined
                  ? generation.context_handoff_threshold.toFixed(2)
                  : 'N/A'
              }
            />
            <LabeledValue
              label="Handoff due"
              value={
                generation.context_handoff_due === undefined
                  ? 'Unknown'
                  : generation.context_handoff_due
                    ? 'Due'
                    : 'No'
              }
            />
            <LabeledValue
              label="Over threshold"
              value={
                generation.context_handoff_over_threshold !== undefined
                  ? generation.context_handoff_over_threshold.toFixed(2)
                  : 'N/A'
              }
            />
            <LabeledValue label="Role" value={generation.role_id || 'N/A'} />
            <LabeledValue label="Stage" value={generation.stage || 'N/A'} />
            <LabeledValue label="Compaction" value={generation.compaction_level || 'N/A'} />
            <LabeledValue label="Largest section" value={generation.largest_section || 'N/A'} />
            <LabeledValue
              label="Prompt tokens"
              value={generation.estimated_prompt_tokens ?? 'N/A'}
            />
            <LabeledValue
              label="Reserved output"
              value={generation.reserved_output_tokens ?? 'N/A'}
            />
            <LabeledValue label="Max input" value={generation.max_input_tokens ?? 'N/A'} />
            <LabeledValue
              label="Flags"
              value={[
                generation.truncated ? 'truncated' : null,
                generation.used_chunking ? 'chunking' : null,
                generation.overflow ? 'overflow' : null,
              ]
                .filter(Boolean)
                .join(', ') || 'None'}
            />
          </div>
        </div>

        <div className="space-y-3 rounded-lg border border-border bg-surface-layer p-4">
          <div>
            <p className="text-sm font-semibold text-foreground">Live subagent runtime</p>
            <p className="mt-1 text-xs text-muted-foreground">
              Generation-scoped invocation, token, and runtime telemetry from live snapshots.
            </p>
          </div>
          <div className="grid gap-3 sm:grid-cols-2">
            <LabeledValue
              label="Invocations"
              value={generation.subagent_invocation_count ?? 'N/A'}
            />
            <LabeledValue
              label="Completed"
              value={generation.subagent_completed_invocation_count ?? 'N/A'}
            />
            <LabeledValue
              label="Token tracked"
              value={generation.subagent_token_tracked_invocation_count ?? 'N/A'}
            />
            <LabeledValue
              label="Token threshold"
              value={generation.generation_token_refresh_threshold ?? 'N/A'}
            />
            <LabeledValue
              label="Token refresh due"
              value={
                generation.token_refresh_due === undefined
                  ? 'Unknown'
                  : generation.token_refresh_due
                    ? 'Due'
                    : 'No'
              }
            />
            <LabeledValue
              label="Over threshold"
              value={generation.token_refresh_over_threshold ?? 'N/A'}
            />
            <LabeledValue
              label="Snapshot"
              value={formatDateTime(generation.last_subagent_runtime_snapshot_at)}
            />
            <LabeledValue
              label="Input tokens"
              value={generation.observed_input_tokens ?? 'N/A'}
            />
            <LabeledValue
              label="Output tokens"
              value={generation.observed_output_tokens ?? 'N/A'}
            />
            <LabeledValue
              label="Total tokens"
              value={generation.observed_total_tokens ?? 'N/A'}
            />
            <LabeledValue
              label="Runtime"
              value={
                generation.observed_generation_time_ms !== undefined
                  ? `${generation.observed_generation_time_ms.toFixed(1)}ms`
                  : 'N/A'
              }
            />
          </div>
          {finishReasonEntries.length > 0 ? (
            <div className="space-y-2">
              <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Finish reasons
              </p>
              <div className="flex flex-wrap gap-2">
                {finishReasonEntries.map(([reason, count]) => (
                  <Badge key={reason} variant="outline">
                    {reason}: {count}
                  </Badge>
                ))}
              </div>
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
              No live subagent runtime telemetry has been persisted for this generation yet.
            </div>
          )}
        </div>
      </div>

      <JsonPreview value={generation.metadata} emptyLabel="No generation metadata." />
    </div>
  )
}

function LinkedRunPanel({
  linkedRun,
}: {
  linkedRun: Record<string, unknown> | null | undefined
}) {
  if (!linkedRun || Object.keys(linkedRun).length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No linked agent run.
      </div>
    )
  }

  const recoveryState = isRecord(linkedRun?.recovery_state) ? linkedRun.recovery_state : {}
  const approvalState = isRecord(linkedRun?.approval_state) ? linkedRun.approval_state : {}

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Run id" value={getString(linkedRun.run_id) || 'N/A'} />
        <LabeledValue
          label="Status"
          value={
            <Badge variant={statusVariant(getString(linkedRun.status) || 'unknown')}>
              {getString(linkedRun.status) || 'unknown'}
            </Badge>
          }
        />
        <LabeledValue label="Approval" value={getString(approvalState.status) || 'N/A'} />
        <LabeledValue label="Degraded" value={getBoolean(linkedRun.degraded) ? 'Yes' : 'No'} />
        <LabeledValue label="Recovery status" value={getString(recoveryState.status) || 'N/A'} />
        <LabeledValue label="Recovery action" value={getString(recoveryState.action) || 'N/A'} />
        <LabeledValue label="Stage" value={getString(recoveryState.stage) || 'N/A'} />
        <LabeledValue
          label="Checkpoint stage"
          value={getString(isRecord(linkedRun.checkpoint) ? linkedRun.checkpoint.stage : null) || 'N/A'}
        />
      </div>
      <JsonPreview value={linkedRun} emptyLabel="No linked agent run." />
    </div>
  )
}

function LeasePanel({
  lease,
}: {
  lease: Record<string, unknown> | null | undefined
}) {
  if (!lease || Object.keys(lease).length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No active supervisor lease.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Owner" value={getString(lease.owner_id) || 'N/A'} />
        <LabeledValue label="Owned by runtime" value={getBoolean(lease.owned_by_runtime) ? 'Yes' : 'No'} />
        <LabeledValue label="Stale" value={getBoolean(lease.stale) ? 'Yes' : 'No'} />
        <LabeledValue label="Takeovers" value={getNumber(lease.takeover_count) ?? 0} />
        <LabeledValue label="Acquired" value={formatDateTime(getString(lease.acquired_at))} />
        <LabeledValue label="Heartbeat" value={formatDateTime(getString(lease.heartbeat_at))} />
        <LabeledValue label="Expires" value={formatDateTime(getString(lease.expires_at))} />
        <LabeledValue label="Updated" value={formatDateTime(getString(lease.updated_at))} />
      </div>
      <JsonPreview value={lease.metadata} emptyLabel="No lease metadata." />
      <JsonPreview value={lease} emptyLabel="No active supervisor lease." />
    </div>
  )
}

function CurrentAttemptPanel({
  attempt,
}: {
  attempt: api.GoalAttemptSummary | null | undefined
}) {
  if (!attempt) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No current attempt.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Attempt id" value={attempt.attempt_id} />
        <LabeledValue label="Index" value={attempt.attempt_index} />
        <LabeledValue
          label="Status"
          value={<Badge variant={statusVariant(attempt.status)}>{attempt.status}</Badge>}
        />
        <LabeledValue label="Trigger" value={attempt.trigger || 'N/A'} />
        <LabeledValue label="Run id" value={attempt.agent_run_id || 'N/A'} />
        <LabeledValue label="Created" value={formatDateTime(attempt.created_at)} />
        <LabeledValue label="Started" value={formatDateTime(attempt.started_at)} />
        <LabeledValue label="Finished" value={formatDateTime(attempt.finished_at)} />
      </div>
      {attempt.latest_error ? (
        <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          {attempt.latest_error}
        </div>
      ) : null}
      <JsonPreview value={attempt.summary} emptyLabel="No attempt summary." />
    </div>
  )
}

function FindingTimelinePanel({
  findings,
}: {
  findings: api.GoalAuditFinding[]
}) {
  if (findings.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No finding history yet.
      </div>
    )
  }

  return (
    <div className="space-y-3">
      {findings.map((finding) => (
        <div key={finding.finding_id} className="rounded-lg border border-border bg-surface-layer p-4">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="space-y-1">
              <div className="flex flex-wrap items-center gap-2">
                <Badge variant={finding.severity === 'error' ? 'error' : finding.status === 'resolved' ? 'success' : 'warning'}>
                  {finding.status}
                </Badge>
                <span className="text-sm font-semibold text-foreground">{finding.finding_code}</span>
              </div>
              <p className="text-sm text-muted-foreground">{finding.summary || 'No summary provided.'}</p>
            </div>
            <Badge variant="outline">{formatDateTime(finding.created_at)}</Badge>
          </div>
          <div className="mt-3 grid gap-3 sm:grid-cols-3">
            <LabeledValue label="Resolved" value={formatDateTime(finding.resolved_at)} />
            <LabeledValue label="Closed" value={formatDateTime(finding.closed_at)} />
            <LabeledValue label="Updated" value={formatDateTime(finding.updated_at)} />
          </div>
          <div className="mt-3">
            <JsonPreview value={finding.details} emptyLabel="No structured details." />
          </div>
        </div>
      ))}
    </div>
  )
}

function SummaryKeyList({
  items,
}: {
  items: Array<{ label: string; value: React.ReactNode }>
}) {
  if (items.length === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No summary data.
      </div>
    )
  }

  return (
    <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
      {items.map((item) => (
        <LabeledValue key={item.label} label={item.label} value={item.value} />
      ))}
    </div>
  )
}

function CheckpointSummaryPanel({
  checkpoint,
}: {
  checkpoint: api.GoalCheckpointRecord | null | undefined
}) {
  if (!checkpoint) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No persisted checkpoints yet.
      </div>
    )
  }

  const payload = isRecord(checkpoint.payload) ? checkpoint.payload : {}
  const recoveryState = isRecord(payload.recovery_state) ? payload.recovery_state : {}
  const roleTaskSummary = isRecord(recoveryState.role_task_summary) ? recoveryState.role_task_summary : {}
  const promotedArtifacts = getRecordArray(payload.promoted_artifacts)

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Checkpoint" value={checkpoint.checkpoint_index ?? checkpoint.checkpoint_id} />
        <LabeledValue label="Stage" value={checkpoint.stage || 'N/A'} />
        <LabeledValue label="Source" value={checkpoint.source || 'N/A'} />
        <LabeledValue label="Captured" value={formatDateTime(checkpoint.captured_at)} />
        <LabeledValue label="Candidate" value={getString(recoveryState.selected_candidate_id) || 'N/A'} />
        <LabeledValue label="Candidate count" value={getNumber(recoveryState.candidate_count) ?? 'N/A'} />
        <LabeledValue label="Unfinished" value={getStringArray(recoveryState.unfinished_steps).length} />
        <LabeledValue label="Promoted artifacts" value={promotedArtifacts.length} />
      </div>
      <SummaryKeyList
        items={[
          { label: 'Tracked roles', value: getNumber(roleTaskSummary.tracked_role_count) ?? 'N/A' },
          { label: 'Reassigned roles', value: getNumber(roleTaskSummary.reassigned_role_count) ?? 'N/A' },
          { label: 'Role task entries', value: getRecordArray(roleTaskSummary.roles).length },
        ]}
      />
      <JsonPreview value={checkpoint.payload} emptyLabel="No checkpoint payload." />
    </div>
  )
}

function MemorySnapshotSummaryPanel({
  snapshot,
}: {
  snapshot: api.GoalMemorySnapshotRecord | null | undefined
}) {
  if (!snapshot) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No persisted memory snapshots yet.
      </div>
    )
  }

  const payload = isRecord(snapshot.snapshot) ? snapshot.snapshot : {}
  const roleTaskSummary = isRecord(payload.role_task_summary) ? payload.role_task_summary : {}

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Snapshot" value={snapshot.snapshot_id} />
        <LabeledValue label="Kind" value={snapshot.snapshot_kind || 'N/A'} />
        <LabeledValue label="Captured" value={formatDateTime(snapshot.captured_at)} />
        <LabeledValue label="Checkpoint" value={snapshot.checkpoint_id ?? 'N/A'} />
        <LabeledValue label="Candidate" value={getString(payload.selected_candidate_id) || 'N/A'} />
        <LabeledValue label="Candidate count" value={getNumber(payload.candidate_count) ?? 'N/A'} />
        <LabeledValue label="Pending approvals" value={getStringArray(payload.pending_approval_ids).length} />
        <LabeledValue label="Pending actions" value={getStringArray(payload.pending_actions).length} />
      </div>
      <SummaryKeyList
        items={[
          { label: 'Accepted facts', value: getStringArray(payload.accepted_facts).length },
          { label: 'Rejected paths', value: getStringArray(payload.rejected_paths).length },
          { label: 'Unfinished steps', value: getStringArray(payload.unfinished_steps).length },
          { label: 'Tracked roles', value: getNumber(roleTaskSummary.tracked_role_count) ?? 'N/A' },
          { label: 'Reassigned roles', value: getNumber(roleTaskSummary.reassigned_role_count) ?? 'N/A' },
          { label: 'Role task entries', value: getRecordArray(roleTaskSummary.roles).length },
        ]}
      />
      <JsonPreview value={snapshot.snapshot} emptyLabel="No memory snapshot content." />
    </div>
  )
}

function RecommendedActionPanel({
  recommendedAction,
  canFinalizeLinkedRunPartial,
  onRunGoalAction,
  onJumpToSection,
  actionPending,
}: {
  recommendedAction: Record<string, unknown> | null | undefined
  canFinalizeLinkedRunPartial: boolean
  onRunGoalAction: (action: GoalAction) => void | Promise<void>
  onJumpToSection: (section: 'runtime-budget' | 'approval' | 'collector' | 'checkpoints' | 'linked-run') => void
  actionPending: GoalAction | null
}) {
  if (!recommendedAction || Object.keys(recommendedAction).length === 0) {
    return null
  }

  const action = getString(recommendedAction.action) || 'inspect'
  const summary = getString(recommendedAction.summary) || 'Inspect goal health.'
  const blocking = getBoolean(recommendedAction.blocking)

  let primaryAction: React.ReactNode = null
  if (action === 'refresh_worker_generation') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => void onRunGoalAction('refresh')} loading={actionPending === 'refresh'}>
        <RefreshCw className="h-3.5 w-3.5" />
        Refresh worker
      </Button>
    )
  } else if (action === 'resolve_approval') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => onJumpToSection('approval')}>
        <ShieldAlert className="h-3.5 w-3.5" />
        Review approvals
      </Button>
    )
  } else if (action === 'resume_goal') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => void onRunGoalAction('resume')} loading={actionPending === 'resume'}>
        <RotateCcw className="h-3.5 w-3.5" />
        Resume
      </Button>
    )
  } else if (action === 'inspect_collector_shards') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => onJumpToSection('collector')}>
        <Database className="h-3.5 w-3.5" />
        Inspect shards
      </Button>
    )
  } else if (action === 'inspect_runtime_budget') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => onJumpToSection('runtime-budget')}>
        <Clock3 className="h-3.5 w-3.5" />
        Inspect runtime budget
      </Button>
    )
  } else if (action === 'capture_checkpoint') {
    primaryAction = canFinalizeLinkedRunPartial ? (
      <Button
        size="sm"
        variant="secondary"
        onClick={() => void onRunGoalAction('finalize_partial')}
        loading={actionPending === 'finalize_partial'}
      >
        <Database className="h-3.5 w-3.5" />
        Finalize partial
      </Button>
    ) : (
      <Button size="sm" variant="secondary" onClick={() => onJumpToSection('checkpoints')}>
        <Database className="h-3.5 w-3.5" />
        Inspect checkpoints
      </Button>
    )
  } else if (action !== 'monitor') {
    primaryAction = (
      <Button size="sm" variant="secondary" onClick={() => onJumpToSection('linked-run')}>
        <Waypoints className="h-3.5 w-3.5" />
        Inspect linked run
      </Button>
    )
  }

  return (
    <div className="rounded-lg border border-border bg-surface-layer p-4">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Badge variant={blocking ? 'warning' : 'outline'}>{action}</Badge>
            {getString(recommendedAction.finding_code) ? (
              <Badge variant="outline">{getString(recommendedAction.finding_code)}</Badge>
            ) : null}
          </div>
          <p className="text-sm text-foreground">{summary}</p>
        </div>
        <div className="flex flex-wrap gap-2">{primaryAction}</div>
      </div>
    </div>
  )
}

function CollectorStatePanel({
  collectorState,
  onRetryFailedShard,
  retryingShardId,
}: {
  collectorState: Record<string, unknown> | null | undefined
  onRetryFailedShard?: (shardId: string) => void | Promise<void>
  retryingShardId?: string | null
}) {
  const shardCount = getNumber(collectorState?.shard_count) ?? 0
  const activeShardCount = getNumber(collectorState?.active_shard_count) ?? 0
  const completedShardCount = getNumber(collectorState?.completed_shard_count) ?? 0
  const stalledShardCount = getNumber(collectorState?.stalled_shard_count) ?? 0
  const stallTimeoutSec = getNumber(collectorState?.stall_timeout_sec)
  const latestActivityAt = getString(collectorState?.latest_activity_at)
  const shards = getRecordArray(collectorState?.shards)
  const failedShards = shards.filter((shard) => {
    const status = getString(shard.status)?.toLowerCase()
    return status === 'failed' || status === 'error'
  })
  const stalledShards = getRecordArray(collectorState?.stalled_shards)

  if (!collectorState || shardCount === 0) {
    return (
      <div className="rounded-md border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
        No collector shard state has been reported for this goal.
      </div>
    )
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <LabeledValue label="Shards" value={shardCount} />
        <LabeledValue label="Active" value={activeShardCount} />
        <LabeledValue label="Completed" value={completedShardCount} />
        <LabeledValue label="Stalled" value={stalledShardCount} />
        <LabeledValue label="Latest activity" value={formatDateTime(latestActivityAt)} />
        <LabeledValue
          label="Stall timeout"
          value={stallTimeoutSec !== null ? `${stallTimeoutSec}s` : 'Not configured'}
        />
      </div>

      {failedShards.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Failed shards</p>
          <div className="space-y-2">
            {failedShards.map((shard, index) => {
              const shardId = getString(shard.shard_id) || `failed-${index}`
              return (
                <div key={`${shardId}-${index}`} className="rounded-md border border-error/30 bg-error/5 px-3 py-3">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <p className="text-sm font-semibold text-foreground">
                        {getString(shard.shard_id) || 'Unknown shard'}
                      </p>
                      <p className="mt-1 text-xs text-muted-foreground">
                        {getString(shard.adapter_name) || 'Unknown adapter'}
                        {getString(shard.cursor) ? ` | cursor=${getString(shard.cursor)}` : ''}
                      </p>
                    </div>
                    <div className="flex items-center gap-2">
                      <Badge variant="error">{getString(shard.status) || 'failed'}</Badge>
                      {onRetryFailedShard ? (
                        <Button
                          size="sm"
                          variant="outline"
                          onClick={() => void onRetryFailedShard(shardId)}
                          loading={retryingShardId === shardId}
                          disabled={Boolean(retryingShardId && retryingShardId !== shardId)}
                        >
                          <RotateCcw className="mr-2 h-4 w-4" />
                          Retry
                        </Button>
                      ) : null}
                    </div>
                  </div>
                  <div className="mt-2 grid gap-2 sm:grid-cols-2">
                    <LabeledValue label="Source" value={getString(shard.source_id) || getString(shard.source_url) || 'Unknown'} />
                    <LabeledValue label="Progress" value={getString(shard.cursor) || 'N/A'} />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      ) : null}

      {stalledShards.length > 0 ? (
        <div className="space-y-2">
          <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Stalled shards</p>
          <div className="space-y-2">
            {stalledShards.map((shard, index) => (
              <div key={`${getString(shard.shard_id) || 'stalled'}-${index}`} className="rounded-md border border-warning/30 bg-warning/5 px-3 py-3">
                <div className="flex flex-wrap items-start justify-between gap-2">
                  <div>
                    <p className="text-sm font-semibold text-foreground">
                      {getString(shard.shard_id) || 'Unknown shard'}
                    </p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {getString(shard.adapter_name) || 'Unknown adapter'}
                      {getString(shard.cursor) ? ` | cursor=${getString(shard.cursor)}` : ''}
                    </p>
                  </div>
                  <Badge variant="warning">{getString(shard.status) || 'stalled'}</Badge>
                </div>
                <div className="mt-2 grid gap-2 sm:grid-cols-2">
                  <LabeledValue label="Age" value={`${getNumber(shard.stalled_age_sec) ?? 0}s`} />
                  <LabeledValue label="Source" value={getString(shard.source_id) || getString(shard.source_url) || 'Unknown'} />
                </div>
              </div>
            ))}
          </div>
        </div>
      ) : null}

      <div className="space-y-2">
        <p className="text-xs font-medium uppercase tracking-wide text-muted-foreground">Recent shard offsets</p>
        <div className="space-y-2">
          {shards.slice(0, 6).map((shard, index) => (
            <div key={`${getString(shard.shard_id) || 'shard'}-${index}`} className="rounded-md border border-border bg-surface-layer px-3 py-3">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <p className="text-sm font-semibold text-foreground">
                    {getString(shard.shard_id) || 'Unknown shard'}
                  </p>
                  <p className="mt-1 text-xs text-muted-foreground">
                    {getString(shard.adapter_name) || 'Unknown adapter'}
                    {getString(shard.source_id) ? ` | ${getString(shard.source_id)}` : ''}
                  </p>
                </div>
                <Badge variant="outline">{getString(shard.status) || 'unknown'}</Badge>
              </div>
              <div className="mt-2 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                <LabeledValue label="Cursor" value={getString(shard.cursor) || 'N/A'} />
                <LabeledValue label="Collected" value={getNumber(shard.items_collected) ?? 'N/A'} />
                <LabeledValue label="Emitted" value={getNumber(shard.items_emitted) ?? 'N/A'} />
                <LabeledValue label="Last activity" value={formatDateTime(getString(shard.last_activity_at))} />
              </div>
            </div>
          ))}
        </div>
      </div>

      <JsonPreview value={collectorState} emptyLabel="No collector shard state has been reported for this goal." />
    </div>
  )
}

export default function GoalsPage() {
  const projects = useProjectStore((state) => state.projects)
  const activeProjectId = useProjectStore((state) => state.activeProjectId)
  const loadProjects = useProjectStore((state) => state.loadProjects)
  const hasLoadedProjects = useProjectStore((state) => state.hasLoadedProjects)
  const isLoadingProjects = useProjectStore((state) => state.isLoadingProjects)

  const [goals, setGoals] = React.useState<api.GoalSummary[]>([])
  const [selectedGoalId, setSelectedGoalId] = React.useState<string | null>(null)
  const [selectedGoal, setSelectedGoal] = React.useState<api.GoalSummary | null>(null)
  const [goalHealth, setGoalHealth] = React.useState<api.GoalHealthSummary | null>(null)
  const [checkpoints, setCheckpoints] = React.useState<api.GoalCheckpointRecord[]>([])
  const [snapshots, setSnapshots] = React.useState<api.GoalMemorySnapshotRecord[]>([])
  const [findings, setFindings] = React.useState<api.GoalAuditFinding[]>([])
  const [findingHistory, setFindingHistory] = React.useState<api.GoalAuditFinding[]>([])
  const [goalApprovals, setGoalApprovals] = React.useState<api.ApprovalSummary[]>([])
  const [operatorAuditLog, setOperatorAuditLog] = React.useState<api.GoalOperatorAuditEntry[]>([])
  const [estopControls, setEstopControls] = React.useState<api.GoalOperatorControls | null>(null)

  const [loadingGoals, setLoadingGoals] = React.useState(true)
  const [loadingDetail, setLoadingDetail] = React.useState(false)
  const [refreshingGoals, setRefreshingGoals] = React.useState(false)
  const [goalError, setGoalError] = React.useState<string | null>(null)
  const [detailError, setDetailError] = React.useState<string | null>(null)
  const [createError, setCreateError] = React.useState<string | null>(null)
  const [actionError, setActionError] = React.useState<string | null>(null)
  const [approvalError, setApprovalError] = React.useState<string | null>(null)
  const [savingControls, setSavingControls] = React.useState(false)
  const [controlsMessage, setControlsMessage] = React.useState<string | null>(null)
  const [actionPending, setActionPending] = React.useState<GoalAction | null>(null)
  const [retryingShardId, setRetryingShardId] = React.useState<string | null>(null)
  const [findingPendingId, setFindingPendingId] = React.useState<number | null>(null)
  const [loadingGoalApprovals, setLoadingGoalApprovals] = React.useState(false)
  const [resolvingApprovalKey, setResolvingApprovalKey] = React.useState<string | null>(null)

  const [goalSearch, setGoalSearch] = React.useState('')
  const [titleDraft, setTitleDraft] = React.useState('')
  const [objectiveDraft, setObjectiveDraft] = React.useState('')
  const [workspaceDraft, setWorkspaceDraft] = React.useState('')
  const [protocolDraft, setProtocolDraft] = React.useState<string>('none')
  const [requestedDurationDraft, setRequestedDurationDraft] = React.useState('')
  const [generationRefreshDraft, setGenerationRefreshDraft] = React.useState('')
  const [checkpointIntervalDraft, setCheckpointIntervalDraft] = React.useState('')
  const [contextThresholdDraft, setContextThresholdDraft] = React.useState('')
  const [allowedToolsDraft, setAllowedToolsDraft] = React.useState('')
  const [creatingGoal, setCreatingGoal] = React.useState(false)

  const [stopAllGoals, setStopAllGoals] = React.useState(false)
  const [blockNetworkUsage, setBlockNetworkUsage] = React.useState(false)
  const [blockedToolsDraft, setBlockedToolsDraft] = React.useState('')
  const [blockedDomainsDraft, setBlockedDomainsDraft] = React.useState('')
  const [controlsReasonDraft, setControlsReasonDraft] = React.useState('')

  const runtimeBudgetSectionRef = React.useRef<HTMLDivElement | null>(null)
  const approvalSectionRef = React.useRef<HTMLDivElement | null>(null)
  const collectorSectionRef = React.useRef<HTMLDivElement | null>(null)
  const checkpointsSectionRef = React.useRef<HTMLDivElement | null>(null)
  const linkedRunSectionRef = React.useRef<HTMLDivElement | null>(null)

  const activeProject = React.useMemo(
    () => projects.find((project) => project.id === activeProjectId) ?? null,
    [activeProjectId, projects]
  )

  React.useEffect(() => {
    if (!hasLoadedProjects && !isLoadingProjects) {
      void loadProjects()
    }
  }, [hasLoadedProjects, isLoadingProjects, loadProjects])

  const loadGoals = React.useCallback(async () => {
    setGoalError(null)

    try {
      const [goalItems, controls, auditLog] = await Promise.all([
        api.fetchGoals(),
        api.fetchGoalEstop(),
        api.fetchGoalOperatorAuditLog({ limit: 8 }),
      ])
      setGoals(goalItems)
      setEstopControls(controls)
      setOperatorAuditLog(auditLog)
      setSelectedGoalId((current) => {
        if (current && goalItems.some((goal) => goal.goal_id === current)) {
          return current
        }
        return goalItems[0]?.goal_id ?? null
      })
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to load goals.'
      setGoalError(detail)
    } finally {
      setLoadingGoals(false)
      setRefreshingGoals(false)
    }
  }, [])

  const loadGoalApprovals = React.useCallback(async (approvalIds: string[]) => {
    if (approvalIds.length === 0) {
      setGoalApprovals([])
      setApprovalError(null)
      return
    }
    setLoadingGoalApprovals(true)
    setApprovalError(null)
    try {
      const approvals = await api.fetchApprovals()
      const approvalIdSet = new Set(approvalIds)
      setGoalApprovals(approvals.filter((approval) => approvalIdSet.has(approval.approval_id)))
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to load pending approvals.'
      setApprovalError(detail)
      setGoalApprovals([])
    } finally {
      setLoadingGoalApprovals(false)
    }
  }, [])

  const loadSelectedGoal = React.useCallback(async (goalId: string) => {
    setLoadingDetail(true)
    setDetailError(null)

    try {
      const [goal, health, checkpointItems, snapshotItems, findingItems, resolvedFindingItems, closedFindingItems, goalAuditItems] = await Promise.all([
        api.fetchGoal(goalId),
        api.fetchGoalHealth(goalId),
        api.fetchGoalCheckpoints(goalId, { limit: 5 }),
        api.fetchGoalMemorySnapshots(goalId, { limit: 5 }),
        api.fetchGoalAuditFindings(goalId, { status: 'open' }),
        api.fetchGoalAuditFindings(goalId, { status: 'resolved' }),
        api.fetchGoalAuditFindings(goalId, { status: 'closed' }),
        api.fetchGoalOperatorAuditLog({ goalId, limit: 12 }),
      ])
      setSelectedGoal(goal)
      setGoalHealth(health)
      setCheckpoints(checkpointItems)
      setSnapshots(snapshotItems)
      setFindings(findingItems)
      const mergedFindingHistory = [...findingItems, ...resolvedFindingItems, ...closedFindingItems]
        .sort((left, right) => compareIsoDesc(left.updated_at, right.updated_at))
        .filter(
          (item, index, items) => items.findIndex((candidate) => candidate.finding_id === item.finding_id) === index
        )
      setFindingHistory(mergedFindingHistory)
      setOperatorAuditLog(goalAuditItems)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to load goal detail.'
      setDetailError(detail)
      setSelectedGoal(null)
      setGoalHealth(null)
      setCheckpoints([])
      setSnapshots([])
      setFindings([])
      setFindingHistory([])
      setOperatorAuditLog([])
    } finally {
      setLoadingDetail(false)
    }
  }, [])

  React.useEffect(() => {
    void loadGoals()
  }, [loadGoals])

  React.useEffect(() => {
    if (!selectedGoalId) {
      setSelectedGoal(null)
      setGoalHealth(null)
      setCheckpoints([])
      setSnapshots([])
      setFindings([])
      setFindingHistory([])
      setOperatorAuditLog([])
      setGoalApprovals([])
      return
    }
    void loadSelectedGoal(selectedGoalId)
  }, [loadSelectedGoal, selectedGoalId])

  React.useEffect(() => {
    if (!estopControls) {
      return
    }
    setStopAllGoals(estopControls.stop_all_goals)
    setBlockNetworkUsage(estopControls.block_network_usage)
    setBlockedToolsDraft(estopControls.blocked_tools.join(', '))
    setBlockedDomainsDraft(estopControls.blocked_domains.join(', '))
    setControlsReasonDraft(
      typeof estopControls.metadata.reason === 'string' ? estopControls.metadata.reason : ''
    )
  }, [estopControls])

  const currentApprovalIds = React.useMemo(
    () => getStringArray(goalHealth?.approval_state?.approval_ids),
    [goalHealth?.approval_state]
  )

  React.useEffect(() => {
    if (currentApprovalIds.length === 0) {
      setGoalApprovals([])
      setApprovalError(null)
      setLoadingGoalApprovals(false)
      return
    }
    void loadGoalApprovals(currentApprovalIds)
  }, [currentApprovalIds, loadGoalApprovals])

  const refreshEverything = async () => {
    setRefreshingGoals(true)
    await loadGoals()
    if (selectedGoalId) {
      await loadSelectedGoal(selectedGoalId)
    }
  }

  const runGoalAction = async (action: GoalAction) => {
    if (!selectedGoalId) {
      return
    }
    setActionPending(action)
    setActionError(null)

    try {
      if (action === 'start') {
        await api.startGoal(selectedGoalId)
      } else if (action === 'pause') {
        await api.pauseGoal(selectedGoalId)
      } else if (action === 'resume') {
        await api.resumeGoal(selectedGoalId, { strategy: 'restart_attempt' })
      } else if (action === 'refresh') {
        await api.refreshGoal(selectedGoalId, { strategy: 'restart_attempt' })
      } else if (action === 'finalize_partial') {
        const linkedRunId = getString(goalHealth?.linked_agent_run?.run_id)
        if (!linkedRunId) {
          throw new Error('No linked agent run is available to finalize as partial.')
        }
        await api.finalizeAgentRunPartial(linkedRunId)
      } else if (action === 'cancel') {
        await api.cancelGoal(selectedGoalId)
      }

      await loadGoals()
      await loadSelectedGoal(selectedGoalId)
    } catch (error) {
      const detail = error instanceof Error ? error.message : `Failed to ${action} goal.`
      setActionError(detail)
    } finally {
      setActionPending(null)
    }
  }

  const retryFailedShard = async (shardId: string) => {
    if (!selectedGoalId) {
      return
    }
    setRetryingShardId(shardId)
    setActionError(null)

    try {
      await api.retryGoalFailedShard(selectedGoalId, {
        shard_id: shardId,
        strategy: 'continue_from_checkpoint',
      })
      await loadGoals()
      await loadSelectedGoal(selectedGoalId)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to retry collector shard.'
      setActionError(detail)
    } finally {
      setRetryingShardId(null)
    }
  }

  const handleResolveFinding = async (findingId: number, mode: 'resolve' | 'close') => {
    if (!selectedGoalId) {
      return
    }
    setFindingPendingId(findingId)
    setActionError(null)

    try {
      if (mode === 'resolve') {
        await api.resolveGoalAuditFinding(selectedGoalId, findingId)
      } else {
        await api.closeGoalAuditFinding(selectedGoalId, findingId)
      }
      await loadSelectedGoal(selectedGoalId)
      await loadGoals()
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to update goal finding.'
      setActionError(detail)
    } finally {
      setFindingPendingId(null)
    }
  }

  const handleResolveApproval = async (
    approvalId: string,
    decision: 'approve_once' | 'reject'
  ) => {
    setResolvingApprovalKey(`${approvalId}:${decision}`)
    setActionError(null)
    setApprovalError(null)
    try {
      await api.resolveApproval(approvalId, { decision })
      await loadGoals()
      if (selectedGoalId) {
        await loadSelectedGoal(selectedGoalId)
      }
      if (currentApprovalIds.length > 0) {
        await loadGoalApprovals(currentApprovalIds)
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to resolve approval.'
      setApprovalError(detail)
    } finally {
      setResolvingApprovalKey(null)
    }
  }

  const jumpToSection = React.useCallback(
    (section: 'runtime-budget' | 'approval' | 'collector' | 'checkpoints' | 'linked-run') => {
      const target =
        section === 'runtime-budget'
          ? runtimeBudgetSectionRef.current
          : section === 'approval'
            ? approvalSectionRef.current
            : section === 'collector'
              ? collectorSectionRef.current
              : section === 'checkpoints'
                ? checkpointsSectionRef.current
                : linkedRunSectionRef.current
      target?.scrollIntoView({ behavior: 'smooth', block: 'start' })
    },
    []
  )

  const handleCreateGoal = async () => {
    const objective = objectiveDraft.trim()
    if (!objective) {
      setCreateError('Objective is required.')
      return
    }

    setCreatingGoal(true)
    setCreateError(null)

    const runPolicy: Record<string, unknown> = {}
    if (requestedDurationDraft.trim()) {
      runPolicy.requested_duration_text = requestedDurationDraft.trim()
    }
    const generationRefresh = parseOptionalInteger(generationRefreshDraft)
    if (generationRefresh !== undefined) {
      runPolicy.generation_refresh_interval_sec = generationRefresh
    }
    const checkpointInterval = parseOptionalInteger(checkpointIntervalDraft)
    if (checkpointInterval !== undefined) {
      runPolicy.checkpoint_interval_sec = checkpointInterval
    }
    const contextThreshold = parseOptionalNumber(contextThresholdDraft)
    if (contextThreshold !== undefined) {
      runPolicy.context_handoff_threshold = contextThreshold
    }

    const allowedTools = parseCsvList(allowedToolsDraft)

    try {
      const created = await api.createGoal({
        objective,
        title: titleDraft.trim() || null,
        protocol_id: protocolDraft === 'none' ? null : protocolDraft,
        projectId: activeProjectId,
        workspaceDir: workspaceDraft.trim() || activeProject?.workspaceDir || null,
        run_policy: runPolicy,
        capability_policy: allowedTools.length > 0 ? { allowed_tools: allowedTools } : {},
      })
      setTitleDraft('')
      setObjectiveDraft('')
      setWorkspaceDraft('')
      setRequestedDurationDraft('')
      setGenerationRefreshDraft('')
      setCheckpointIntervalDraft('')
      setContextThresholdDraft('')
      setAllowedToolsDraft('')
      setSelectedGoalId(created.goal_id)
      await loadGoals()
      await loadSelectedGoal(created.goal_id)
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to create goal.'
      setCreateError(detail)
    } finally {
      setCreatingGoal(false)
    }
  }

  const handleSaveControls = async () => {
    setSavingControls(true)
    setControlsMessage(null)

    try {
      const updated = await api.updateGoalEstop({
        stop_all_goals: stopAllGoals,
        block_network_usage: blockNetworkUsage,
        blocked_tools: parseCsvList(blockedToolsDraft),
        blocked_domains: parseCsvList(blockedDomainsDraft),
        reason: controlsReasonDraft.trim() || null,
      })
      setEstopControls(updated)
      setControlsMessage('Operator controls saved.')
      await loadGoals()
      if (selectedGoalId) {
        await loadSelectedGoal(selectedGoalId)
      }
    } catch (error) {
      const detail = error instanceof Error ? error.message : 'Failed to save operator controls.'
      setControlsMessage(detail)
    } finally {
      setSavingControls(false)
    }
  }

  const filteredGoals = React.useMemo(() => {
    const query = goalSearch.trim().toLowerCase()
    if (!query) {
      return goals
    }
    return goals.filter((goal) => {
      const text = [
        goal.goal_id,
        goal.title,
        goal.objective,
        goal.topic,
        goal.protocol_id,
        goal.workspace_dir,
      ]
        .filter((value): value is string => typeof value === 'string' && value.length > 0)
        .join(' ')
        .toLowerCase()
      return text.includes(query)
    })
  }, [goalSearch, goals])

  const activeGoalCount = goals.filter((goal) =>
    ['created', 'queued', 'running', 'waiting_approval', 'awaiting_resources', 'stalled', 'paused'].includes(goal.status)
  ).length
  const waitingApprovalCount = goals.filter((goal) => goal.status === 'waiting_approval').length
  const blockedCount = goals.filter((goal) =>
    ['stalled', 'awaiting_resources', 'failed'].includes(goal.status)
  ).length

  const currentAttempt =
    selectedGoal?.attempts.find((attempt) => attempt.attempt_id === selectedGoal.current_attempt_id) ??
    selectedGoal?.attempts[selectedGoal.attempts.length - 1] ??
    null
  const linkedGoalRunId = React.useMemo(
    () => getString(goalHealth?.linked_agent_run?.run_id),
    [goalHealth?.linked_agent_run]
  )
  const canFinalizeLinkedRunPartial = React.useMemo(() => {
    const recoveryState = isRecord(goalHealth?.linked_agent_run?.recovery_state)
      ? goalHealth.linked_agent_run.recovery_state
      : {}
    const explicit = recoveryState.finalize_partial_ready
    if (typeof explicit === 'boolean') {
      return explicit
    }
    const status =
      (getString(recoveryState.status) ?? getString(goalHealth?.linked_agent_run?.status) ?? '').toLowerCase()
    return status === 'awaiting_resources' || status === 'stalled'
  }, [goalHealth?.linked_agent_run])

  return (
    <div className="flex h-full flex-col overflow-hidden">
      <header className="shrink-0 border-b border-border px-6 pb-4 pt-5">
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-foreground">Goals</h1>
            <p className="mt-0.5 text-sm text-muted-foreground">
              Operator surface for long-running goals, runtime budgets, worker generations, audit findings, and emergency stop.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-right">
              <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Loaded goals</p>
              <p className="text-sm font-semibold text-foreground">{goals.length}</p>
            </div>
            <Button variant="secondary" size="sm" onClick={() => void refreshEverything()} loading={refreshingGoals}>
              <RefreshCw className="h-3.5 w-3.5" />
              Refresh
            </Button>
          </div>
        </div>

        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-surface-layer px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Active</p>
            <p className="mt-1 text-lg font-semibold text-foreground">{activeGoalCount}</p>
          </div>
          <div className="rounded-lg border border-border bg-surface-layer px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Waiting approval</p>
            <p className="mt-1 text-lg font-semibold text-foreground">{waitingApprovalCount}</p>
          </div>
          <div className="rounded-lg border border-border bg-surface-layer px-4 py-3">
            <p className="text-[11px] uppercase tracking-wide text-muted-foreground">Blocked or degraded</p>
            <p className="mt-1 text-lg font-semibold text-foreground">{blockedCount}</p>
          </div>
        </div>

        {goalError ? (
          <div className="mt-4 rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {goalError}
          </div>
        ) : null}
      </header>

      <div className="flex-1 overflow-y-auto px-6 py-5">
        <div className="grid gap-6 xl:grid-cols-[360px_minmax(0,1fr)]">
          <div className="space-y-6">
            <Card>
              <CardHeader>
                <CardTitle>Create Goal</CardTitle>
                <CardDescription>
                  Create the durable goal first, then start or resume it explicitly from the operator console.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Title
                  </label>
                  <Input value={titleDraft} onChange={(event) => setTitleDraft(event.target.value)} placeholder="Forum corpus collector" />
                </div>
                <div className="space-y-2">
                  <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                    Objective
                  </label>
                  <Textarea
                    value={objectiveDraft}
                    onChange={(event) => setObjectiveDraft(event.target.value)}
                    placeholder="Collect a resumable high-quality dialogue corpus, checkpoint every hour, and survive runtime restarts."
                    rows={6}
                  />
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Protocol
                    </label>
                    <Select value={protocolDraft} onValueChange={setProtocolDraft}>
                      <SelectTrigger>
                        <SelectValue placeholder="Select protocol" />
                      </SelectTrigger>
                      <SelectContent>
                        {GOAL_PROTOCOL_OPTIONS.map((option) => (
                          <SelectItem key={option.value} value={option.value}>
                            {option.label}
                          </SelectItem>
                        ))}
                      </SelectContent>
                    </Select>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Workspace override
                    </label>
                    <Input
                      value={workspaceDraft}
                      onChange={(event) => setWorkspaceDraft(event.target.value)}
                      placeholder={activeProject?.workspaceDir || 'Use active project workspace'}
                    />
                  </div>
                </div>
                <div className="grid gap-3 sm:grid-cols-2">
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Requested duration
                    </label>
                    <Input
                      value={requestedDurationDraft}
                      onChange={(event) => setRequestedDurationDraft(event.target.value)}
                      placeholder="Run for 12 hours"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Allowed tools
                    </label>
                    <Input
                      value={allowedToolsDraft}
                      onChange={(event) => setAllowedToolsDraft(event.target.value)}
                      placeholder="web_search, web_fetch, web_crawl"
                    />
                  </div>
                </div>
                <div className="grid gap-3 sm:grid-cols-3">
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Generation refresh sec
                    </label>
                    <Input value={generationRefreshDraft} onChange={(event) => setGenerationRefreshDraft(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Checkpoint cadence sec
                    </label>
                    <Input value={checkpointIntervalDraft} onChange={(event) => setCheckpointIntervalDraft(event.target.value)} />
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                      Context threshold
                    </label>
                    <Input value={contextThresholdDraft} onChange={(event) => setContextThresholdDraft(event.target.value)} placeholder="0.85" />
                  </div>
                </div>
                {createError ? (
                  <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                    {createError}
                  </div>
                ) : null}
                <Button className="w-full" onClick={() => void handleCreateGoal()} loading={creatingGoal}>
                  <Workflow className="h-4 w-4" />
                  Create goal
                </Button>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Goal List</CardTitle>
                <CardDescription>
                  Search by goal id, title, objective, protocol, or workspace.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <Input
                  value={goalSearch}
                  onChange={(event) => setGoalSearch(event.target.value)}
                  placeholder="Search goals"
                  leftIcon={<Search className="h-3.5 w-3.5" />}
                  className="pl-8"
                />
                {loadingGoals ? (
                  <div className="space-y-3">
                    {Array.from({ length: 4 }).map((_, index) => (
                      <div key={index} className="h-24 animate-pulse rounded-lg border border-border bg-surface-layer" />
                    ))}
                  </div>
                ) : filteredGoals.length === 0 ? (
                  <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                    No goals match the current filter.
                  </div>
                ) : (
                  <div className="space-y-3">
                    {filteredGoals.map((goal) => {
                      const isSelected = goal.goal_id === selectedGoalId
                      return (
                        <button
                          key={goal.goal_id}
                          type="button"
                          onClick={() => setSelectedGoalId(goal.goal_id)}
                          className={cn(
                            'block w-full rounded-lg border bg-surface-layer p-4 text-left transition-colors hover:border-primary-500/50',
                            isSelected ? 'border-primary-500/60' : 'border-border'
                          )}
                        >
                          <div className="mb-2 flex items-start justify-between gap-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">
                                {goal.title || goal.topic || goal.goal_id}
                              </p>
                              <p className="mt-0.5 text-xs text-muted-foreground">
                                {goal.protocol_id || 'No protocol'}{goal.workspace_dir ? ` • ${goal.workspace_dir}` : ''}
                              </p>
                            </div>
                            <Badge variant={statusVariant(goal.status)}>{goal.status}</Badge>
                          </div>
                          <p className="line-clamp-3 text-sm text-muted-foreground">{goal.objective}</p>
                          <div className="mt-3 flex items-center justify-between text-xs text-muted-foreground">
                            <span>{goal.attempts.length} attempts</span>
                            <span>{formatDateTime(goal.updated_at)}</span>
                          </div>
                        </button>
                      )
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          </div>

          <div className="space-y-6">
            {!selectedGoalId ? (
              <div className="flex h-80 items-center justify-center rounded-lg border border-dashed border-border bg-surface-layer text-center">
                <div>
                  <p className="text-base font-semibold text-foreground">No goal selected</p>
                  <p className="mt-1 text-sm text-muted-foreground">
                    Create a goal or choose one from the list to inspect runtime health and operator controls.
                  </p>
                </div>
              </div>
            ) : (
              <>
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Waypoints className="h-4 w-4" />
                      Goal Overview
                    </CardTitle>
                    <CardDescription>
                      Durable goal state, attempt lineage, runtime budget, and recommended next action.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    {detailError ? (
                      <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                        {detailError}
                      </div>
                    ) : null}
                    {actionError ? (
                      <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                        {actionError}
                      </div>
                    ) : null}
                    {loadingDetail ? (
                      <div className="space-y-3">
                        <div className="h-24 animate-pulse rounded-lg border border-border bg-surface-layer" />
                        <div className="h-48 animate-pulse rounded-lg border border-border bg-surface-layer" />
                      </div>
                    ) : selectedGoal ? (
                      <>
                        <div className="flex flex-wrap items-start justify-between gap-4">
                          <div className="space-y-2">
                            <div className="flex items-center gap-2">
                              <h2 className="text-lg font-semibold text-foreground">
                                {selectedGoal.title || selectedGoal.topic || selectedGoal.goal_id}
                              </h2>
                              <Badge variant={statusVariant(selectedGoal.status)}>{selectedGoal.status}</Badge>
                            </div>
                            <p className="max-w-4xl text-sm text-muted-foreground">{selectedGoal.objective}</p>
                          </div>
                          <div className="flex flex-wrap gap-2">
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => void runGoalAction('start')}
                              loading={actionPending === 'start'}
                            >
                              <Play className="h-3.5 w-3.5" />
                              Start
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => void runGoalAction('pause')}
                              loading={actionPending === 'pause'}
                            >
                              <Pause className="h-3.5 w-3.5" />
                              Pause
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => void runGoalAction('resume')}
                              loading={actionPending === 'resume'}
                            >
                              <RotateCcw className="h-3.5 w-3.5" />
                              Resume
                            </Button>
                            <Button
                              size="sm"
                              variant="secondary"
                              onClick={() => void runGoalAction('refresh')}
                              loading={actionPending === 'refresh'}
                            >
                              <RefreshCw className="h-3.5 w-3.5" />
                              Refresh worker
                            </Button>
                            {linkedGoalRunId && canFinalizeLinkedRunPartial ? (
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => void runGoalAction('finalize_partial')}
                                loading={actionPending === 'finalize_partial'}
                              >
                                <Database className="h-3.5 w-3.5" />
                                Finalize partial
                              </Button>
                            ) : null}
                            <Button
                              size="sm"
                              variant="destructive"
                              onClick={() => void runGoalAction('cancel')}
                              loading={actionPending === 'cancel'}
                            >
                              <Square className="h-3.5 w-3.5" />
                              Cancel
                            </Button>
                          </div>
                        </div>

                        <RecommendedActionPanel
                          recommendedAction={goalHealth?.recommended_next_action}
                          canFinalizeLinkedRunPartial={canFinalizeLinkedRunPartial}
                          onRunGoalAction={runGoalAction}
                          onJumpToSection={jumpToSection}
                          actionPending={actionPending}
                        />

                        {selectedGoal.latest_error ? (
                          <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-sm text-destructive">
                            {selectedGoal.latest_error}
                          </div>
                        ) : null}

                        <Separator />

                        <div className="space-y-3">
                          <div>
                            <p className="text-sm font-semibold text-foreground">Goal state</p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Durable goal identity, lifecycle state, and summary metadata.
                            </p>
                          </div>
                          <GoalStatePanel goal={selectedGoal} goalHealth={goalHealth} />
                        </div>

                        <div ref={runtimeBudgetSectionRef} className="grid gap-6 xl:grid-cols-2">
                          <div className="space-y-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">Runtime budget</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Normalized duration contract and budget exhaustion state.
                              </p>
                            </div>
                            <RuntimeBudgetPanel runtimeBudget={goalHealth?.runtime_budget} />
                          </div>
                          <div ref={approvalSectionRef} className="space-y-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">Approval state</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Goal-level approval projection and wait telemetry.
                              </p>
                            </div>
                            <ApprovalStatePanel
                              approvalState={goalHealth?.approval_state}
                              approvals={goalApprovals}
                              loadingApprovals={loadingGoalApprovals}
                              approvalError={approvalError}
                              resolvingApprovalKey={resolvingApprovalKey}
                              onResolveApproval={handleResolveApproval}
                            />
                          </div>
                        </div>

                        <div className="grid gap-6 xl:grid-cols-3">
                          <div className="space-y-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">Current generation</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Worker-generation state, rollover reason, and refresh telemetry.
                              </p>
                            </div>
                            <CurrentGenerationPanel generation={goalHealth?.current_generation} />
                          </div>
                          <div ref={linkedRunSectionRef} className="space-y-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">Linked run</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                The execution primitive below this goal supervisor.
                              </p>
                            </div>
                            <LinkedRunPanel linkedRun={goalHealth?.linked_agent_run} />
                          </div>
                          <div className="space-y-3">
                            <div>
                              <p className="text-sm font-semibold text-foreground">Supervisor lease</p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                Runtime ownership, heartbeat, and takeover state for this goal.
                              </p>
                            </div>
                            <LeasePanel lease={goalHealth?.lease} />
                          </div>
                        </div>

                        <div ref={collectorSectionRef} className="space-y-3">
                          <div>
                            <p className="text-sm font-semibold text-foreground">Collector shards</p>
                            <p className="mt-1 text-xs text-muted-foreground">
                              Shard progress, stall detection, and persisted collector offsets projected from the linked run.
                            </p>
                          </div>
                          <CollectorStatePanel
                            collectorState={goalHealth?.collector_state}
                            onRetryFailedShard={retryFailedShard}
                            retryingShardId={retryingShardId}
                          />
                        </div>

                        {currentAttempt ? (
                          <div className="space-y-3">
                            <div className="flex items-center justify-between gap-3">
                              <div>
                                <p className="text-sm font-semibold text-foreground">Current attempt</p>
                                <p className="mt-1 text-xs text-muted-foreground">
                                  Goal attempt state and linked run lineage.
                                </p>
                              </div>
                              {currentAttempt.agent_run_id ? (
                                <Link
                                  href={`/agent-runs/${encodeURIComponent(currentAttempt.agent_run_id)}`}
                                  className="text-sm text-primary-300 underline-offset-4 hover:underline"
                                >
                                  Open linked run
                                </Link>
                              ) : null}
                            </div>
                            <CurrentAttemptPanel attempt={currentAttempt} />
                          </div>
                        ) : null}
                      </>
                    ) : null}
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <AlertTriangle className="h-4 w-4" />
                      Audit Findings
                    </CardTitle>
                    <CardDescription>
                      Open findings from supervisor reconciliation, approval waits, checkpoint drift, and refresh pressure.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {findings.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                        No open findings for this goal.
                      </div>
                    ) : (
                      findings.map((finding) => (
                        <div key={finding.finding_id} className="rounded-lg border border-border bg-surface-layer p-4">
                          <div className="flex flex-wrap items-start justify-between gap-3">
                            <div>
                              <div className="flex items-center gap-2">
                                <Badge variant={finding.severity === 'error' ? 'error' : 'warning'}>{finding.severity}</Badge>
                                <span className="text-sm font-semibold text-foreground">{finding.finding_code}</span>
                              </div>
                              <p className="mt-2 text-sm text-muted-foreground">
                                {finding.summary || 'No summary provided.'}
                              </p>
                              <p className="mt-2 text-xs text-muted-foreground">
                                Opened {formatDateTime(finding.created_at)}
                              </p>
                            </div>
                            <div className="flex gap-2">
                              <Button
                                size="sm"
                                variant="secondary"
                                onClick={() => void handleResolveFinding(finding.finding_id, 'resolve')}
                                loading={findingPendingId === finding.finding_id}
                              >
                                Resolve
                              </Button>
                              <Button
                                size="sm"
                                variant="ghost"
                                onClick={() => void handleResolveFinding(finding.finding_id, 'close')}
                                loading={findingPendingId === finding.finding_id}
                              >
                                Close
                              </Button>
                            </div>
                          </div>
                          <div className="mt-3">
                            <JsonPreview value={finding.details} emptyLabel="No structured details." />
                          </div>
                        </div>
                      ))
                    )}
                    <Separator />
                    <div className="space-y-3">
                      <div>
                        <p className="text-sm font-semibold text-foreground">Finding history</p>
                        <p className="mt-1 text-xs text-muted-foreground">
                          Recent open, resolved, and closed findings for this goal.
                        </p>
                      </div>
                      <FindingTimelinePanel findings={findingHistory} />
                    </div>
                  </CardContent>
                </Card>

                <div ref={checkpointsSectionRef} className="grid gap-6 xl:grid-cols-2">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Database className="h-4 w-4" />
                        Checkpoints
                      </CardTitle>
                      <CardDescription>
                        Persisted stage, candidate, and role-task progress snapshots.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {checkpoints.length === 0 ? (
                        <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                          No persisted checkpoints yet.
                        </div>
                      ) : (
                        checkpoints.map((checkpoint) => (
                          <div key={checkpoint.checkpoint_id} className="space-y-2 rounded-lg border border-border bg-surface-layer p-4">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div>
                                <p className="text-sm font-semibold text-foreground">
                                  Checkpoint #{checkpoint.checkpoint_index ?? checkpoint.checkpoint_id}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {checkpoint.stage || 'No stage'} • {checkpoint.source || 'Unknown source'}
                                </p>
                              </div>
                              <Badge variant="outline">{formatDateTime(checkpoint.captured_at)}</Badge>
                            </div>
                            <CheckpointSummaryPanel checkpoint={checkpoint} />
                          </div>
                        ))
                      )}
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <BrainCircuit className="h-4 w-4" />
                        Memory Snapshots
                      </CardTitle>
                      <CardDescription>
                        Compact handoff state for restart-safe worker refresh and resume.
                      </CardDescription>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      {snapshots.length === 0 ? (
                        <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                          No persisted memory snapshots yet.
                        </div>
                      ) : (
                        snapshots.map((snapshot) => (
                          <div key={snapshot.snapshot_id} className="space-y-2 rounded-lg border border-border bg-surface-layer p-4">
                            <div className="flex flex-wrap items-center justify-between gap-2">
                              <div>
                                <p className="text-sm font-semibold text-foreground">
                                  Snapshot #{snapshot.snapshot_id}
                                </p>
                                <p className="text-xs text-muted-foreground">
                                  {snapshot.snapshot_kind || 'Unknown kind'}
                                </p>
                              </div>
                              <Badge variant="outline">{formatDateTime(snapshot.captured_at)}</Badge>
                            </div>
                            <MemorySnapshotSummaryPanel snapshot={snapshot} />
                          </div>
                        ))
                      )}
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <ShieldAlert className="h-4 w-4" />
                      Operator Controls
                    </CardTitle>
                    <CardDescription>
                      Persistent estop, tool blocking, domain blocking, and network restrictions for all goals.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-3 sm:grid-cols-2">
                      <label className="flex items-center gap-2 rounded-lg border border-border bg-surface-layer px-3 py-3 text-sm text-foreground">
                        <input
                          type="checkbox"
                          checked={stopAllGoals}
                          onChange={(event) => setStopAllGoals(event.target.checked)}
                          className="h-4 w-4 rounded border-border"
                        />
                        Stop all goals
                      </label>
                      <label className="flex items-center gap-2 rounded-lg border border-border bg-surface-layer px-3 py-3 text-sm text-foreground">
                        <input
                          type="checkbox"
                          checked={blockNetworkUsage}
                          onChange={(event) => setBlockNetworkUsage(event.target.checked)}
                          className="h-4 w-4 rounded border-border"
                        />
                        Block network usage
                      </label>
                    </div>
                    <div className="grid gap-3 sm:grid-cols-2">
                      <div className="space-y-2">
                        <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Blocked tools
                        </label>
                        <Input
                          value={blockedToolsDraft}
                          onChange={(event) => setBlockedToolsDraft(event.target.value)}
                          placeholder="web_search, web_fetch"
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                          Blocked domains
                        </label>
                        <Input
                          value={blockedDomainsDraft}
                          onChange={(event) => setBlockedDomainsDraft(event.target.value)}
                          placeholder="example.com, news.ycombinator.com"
                        />
                      </div>
                    </div>
                    <div className="space-y-2">
                      <label className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                        Reason
                      </label>
                      <Textarea
                        rows={3}
                        value={controlsReasonDraft}
                        onChange={(event) => setControlsReasonDraft(event.target.value)}
                        placeholder="Maintenance window, source policy issue, or temporary network freeze."
                      />
                    </div>
                    {controlsMessage ? (
                      <div className="rounded-md border border-border bg-surface-layer px-3 py-2 text-xs text-muted-foreground">
                        {controlsMessage}
                      </div>
                    ) : null}
                    <Button onClick={() => void handleSaveControls()} loading={savingControls}>
                      <ShieldAlert className="h-4 w-4" />
                      Save operator controls
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock3 className="h-4 w-4" />
                      Goal Audit Timeline
                    </CardTitle>
                    <CardDescription>
                      Goal-specific operator actions plus relevant global operator-control changes.
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {operatorAuditLog.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border bg-surface-layer px-4 py-8 text-center text-sm text-muted-foreground">
                        No operator audit entries recorded yet.
                      </div>
                    ) : (
                      operatorAuditLog.map((entry) => (
                        <div key={entry.audit_id} className="rounded-lg border border-border bg-surface-layer p-4">
                          <div className="flex flex-wrap items-center justify-between gap-2">
                            <div>
                              <p className="text-sm font-semibold text-foreground">
                                {entry.summary || entry.action || entry.event_type || `Audit #${entry.audit_id}`}
                              </p>
                              <p className="mt-1 text-xs text-muted-foreground">
                                {entry.event_type || 'unknown'} • {entry.subject_type || 'unknown'} • {formatDateTime(entry.created_at)}
                              </p>
                            </div>
                            <Badge variant="outline">{entry.audit_id}</Badge>
                          </div>
                          <div className="mt-3">
                            <JsonPreview value={entry.details} emptyLabel="No structured details." />
                          </div>
                        </div>
                      ))
                    )}
                  </CardContent>
                </Card>
              </>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
