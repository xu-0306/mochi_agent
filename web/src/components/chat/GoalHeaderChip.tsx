'use client'

import {
  CheckCircle2,
  ExternalLink,
  Loader2,
  Pause,
  Play,
  ShieldAlert,
  Square,
  Target,
  X,
  Workflow,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ApprovalSummary } from '@/lib/api'
import { getFileName } from '@/lib/file-change-preview'
import { cn } from '@/lib/utils'

export type GoalHeaderDisplayState = 'active' | 'blocked' | 'completed'

export interface GoalHeaderChipView {
  title: string
  goalId: string | null
  status: string
  executionMode: 'single_agent' | 'workflow'
  protocolId: string | null
  modelCount: number
  runtimeMode: string | null
  pendingApprovalCount: number
  displayState: GoalHeaderDisplayState
}

export interface GoalDrawerBlockerView {
  summary: string | null
  recommendedAction: string | null
  latestError: string | null
  approvalIds: string[]
  approvalToolNames: string[]
  blockedTools: string[]
  blockedDomains: string[]
  blockNetworkUsage: boolean
}

interface GoalHeaderChipProps {
  goal: GoalHeaderChipView
  open: boolean
  onClick: () => void
}

interface GoalDrawerContentProps {
  goal: GoalHeaderChipView
  busyAction?: 'status' | 'pause' | 'resume' | 'stop' | null
  blocker?: GoalDrawerBlockerView | null
  approvals?: ApprovalSummary[]
  approvalLoading?: boolean
  approvalError?: string | null
  resolvingApprovalKey?: string | null
  onRefresh: () => void
  onPause?: () => void
  onResume?: () => void
  onStop?: () => void
  onResolveApproval?: (
    approvalId: string,
    decision: 'approve_once' | 'reject'
  ) => void | Promise<void>
  onOpenConsole: () => void
  onClose: () => void
}

function formatEnumLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

function previewText(value: string, limit = 140): string {
  if (value.length <= limit) {
    return value
  }
  return `${value.slice(0, Math.max(0, limit - 3))}...`
}

function executionModeLabel(mode: GoalHeaderChipView['executionMode']): string {
  return mode === 'single_agent' ? 'Single agent' : 'Workflow'
}

function displayStateTone(state: GoalHeaderDisplayState, open: boolean): string {
  if (state === 'completed') {
    return open
      ? 'border-emerald-400/40 bg-emerald-500/15 text-emerald-100'
      : 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  if (state === 'blocked') {
    return open
      ? 'border-warning/40 bg-warning/15 text-warning-foreground'
      : 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  return open
    ? 'border-primary-400/40 bg-primary-500/15 text-primary-50'
    : 'border-primary-400/30 bg-primary-500/10 text-primary-200'
}

function statusBadgeVariant(state: GoalHeaderDisplayState): 'primary' | 'warning' | 'success' {
  if (state === 'completed') {
    return 'success'
  }
  if (state === 'blocked') {
    return 'warning'
  }
  return 'primary'
}

function displayStateLabel(state: GoalHeaderDisplayState): string {
  if (state === 'completed') {
    return 'Completed goal'
  }
  if (state === 'blocked') {
    return 'Goal needs attention'
  }
  return 'Active goal'
}

function canPauseGoal(status: string): boolean {
  const normalized = status.toLowerCase()
  return (
    normalized === 'running' ||
    normalized === 'active' ||
    normalized === 'started' ||
    normalized === 'in_progress'
  )
}

function canResumeGoal(status: string): boolean {
  const normalized = status.toLowerCase()
  return (
    normalized === 'paused' ||
    normalized === 'blocked' ||
    normalized === 'awaiting_approval' ||
    normalized === 'waiting_approval' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled'
  )
}

function formatApprovalScope(approval: ApprovalSummary): string {
  return approval.approval_scope ? formatEnumLabel(approval.approval_scope) : 'operator'
}

function countApprovalFiles(approval: ApprovalSummary): number {
  return approval.file_change_groups.reduce((total, group) => total + group.files.length, 0)
}

function approvalSummaryBadges(approval: ApprovalSummary): string[] {
  const badges: string[] = []
  const changedFiles = countApprovalFiles(approval)
  if (changedFiles > 0) {
    badges.push(`${changedFiles} file${changedFiles > 1 ? 's' : ''}`)
  }
  if (approval.patch_validation_supported) {
    badges.push('patch validation')
  }
  if (approval.replay_safe) {
    badges.push('replay safe')
  }
  return badges
}

export function GoalHeaderChip({ goal, open, onClick }: GoalHeaderChipProps) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex min-w-0 max-w-[11rem] items-center gap-2 rounded-full border px-2.5 py-1.5 text-left transition-all duration-150 ease-out-smooth hover:translate-y-[-1px] sm:max-w-[18rem] sm:px-3',
        displayStateTone(goal.displayState, open)
      )}
      aria-label={`${displayStateLabel(goal.displayState)}: ${goal.title}`}
      title={goal.title}
    >
      {goal.displayState === 'completed' ? (
        <CheckCircle2 className="h-3.5 w-3.5 shrink-0" />
      ) : goal.displayState === 'blocked' ? (
        <ShieldAlert className="h-3.5 w-3.5 shrink-0" />
      ) : (
        <Target className="h-3.5 w-3.5 shrink-0" />
      )}
      <span className="min-w-0 flex-1">
        <span className="block truncate text-xs font-semibold">{goal.title}</span>
        <span className="hidden truncate text-[11px] opacity-80 sm:block">
          {executionModeLabel(goal.executionMode)}
          {goal.protocolId ? ` · ${goal.protocolId}` : ''}
        </span>
      </span>
      {goal.pendingApprovalCount > 0 ? (
        <span className="inline-flex h-5 min-w-5 shrink-0 items-center justify-center rounded-full bg-destructive px-1 text-[10px] font-semibold text-destructive-foreground">
          {goal.pendingApprovalCount}
        </span>
      ) : null}
    </button>
  )
}

export function GoalDrawerContent({
  goal,
  busyAction = null,
  blocker = null,
  approvals = [],
  approvalLoading = false,
  approvalError = null,
  resolvingApprovalKey = null,
  onRefresh,
  onPause,
  onResume,
  onStop,
  onResolveApproval,
  onOpenConsole,
  onClose,
}: GoalDrawerContentProps) {
  const pauseAvailable = Boolean(goal.goalId && onPause && canPauseGoal(goal.status))
  const resumeAvailable = Boolean(
    goal.goalId &&
      onResume &&
      canResumeGoal(goal.status) &&
      goal.pendingApprovalCount === 0
  )
  const stopAvailable = Boolean(goal.goalId && onStop && goal.displayState !== 'completed')
  const showBlockedDetails = goal.displayState === 'blocked' || blocker !== null
  const hasApprovalMetadata = Boolean(
    blocker &&
      (blocker.approvalIds.length > 0 ||
        blocker.approvalToolNames.length > 0 ||
        approvals.length > 0)
  )

  return (
    <div className="flex h-full flex-col bg-surface-layer/95">
      <div className="flex items-start justify-between gap-3 border-b border-border px-4 py-4">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {displayStateLabel(goal.displayState)}
          </p>
          <h2 className="mt-1 text-sm font-semibold text-foreground">{goal.title}</h2>
        </div>
        <Button type="button" variant="ghost" size="icon-sm" onClick={onClose} title="Close goal drawer">
          <X className="h-3.5 w-3.5" />
        </Button>
      </div>

      <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
        <div className="flex flex-wrap items-center gap-2">
          <Badge variant={statusBadgeVariant(goal.displayState)} className="capitalize">
            {formatEnumLabel(goal.status)}
          </Badge>
          <Badge variant="outline">{executionModeLabel(goal.executionMode)}</Badge>
          {goal.protocolId ? <Badge variant="outline">{goal.protocolId}</Badge> : null}
          {goal.modelCount > 0 ? (
            <Badge variant="outline">
              {goal.modelCount} model{goal.modelCount > 1 ? 's' : ''}
            </Badge>
          ) : null}
          {goal.pendingApprovalCount > 0 ? (
            <Badge variant="error">
              {goal.pendingApprovalCount} approval{goal.pendingApprovalCount > 1 ? 's' : ''}
            </Badge>
          ) : null}
        </div>

        {goal.runtimeMode ? (
          <div className="rounded-2xl border border-border bg-elevated-layer/80 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
              Runtime mode
            </p>
            <p className="mt-1 text-sm leading-6 text-foreground">{goal.runtimeMode}</p>
          </div>
        ) : null}

        {approvalError ? (
          <div className="rounded-2xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
            {approvalError}
          </div>
        ) : null}

        <div className="rounded-2xl border border-border bg-elevated-layer/80 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Goal summary
          </p>
          <div className="mt-3 grid gap-3 text-sm sm:grid-cols-2">
            <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Target className="h-3.5 w-3.5" />
                Objective
              </div>
              <p className="mt-2 whitespace-pre-wrap break-words text-foreground">{goal.title}</p>
            </div>
            <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
              <div className="flex items-center gap-2 text-xs font-medium text-muted-foreground">
                <Workflow className="h-3.5 w-3.5" />
                Goal ID
              </div>
              <p className="mt-2 break-all font-mono text-xs text-foreground">
                {goal.goalId ?? 'Not started'}
              </p>
            </div>
          </div>
        </div>

        {showBlockedDetails ? (
          <div className="rounded-2xl border border-warning/30 bg-warning/10 p-3">
            <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-warning-foreground/80">
              Blocked status
            </p>
            <p className="mt-1 text-sm leading-6 text-foreground">
              {blocker?.summary ?? blocker?.latestError ?? 'This goal needs operator attention before it can continue.'}
            </p>
            {blocker?.recommendedAction ? (
              <p className="mt-2 text-xs text-muted-foreground">
                Recommended action: {formatEnumLabel(blocker.recommendedAction)}
              </p>
            ) : null}
            {blocker?.latestError ? (
              <p className="mt-2 rounded-xl border border-border/70 bg-surface-layer/70 px-3 py-2 font-mono text-[11px] leading-5 text-foreground/85">
                {blocker.latestError}
              </p>
            ) : null}
            {(blocker?.blockNetworkUsage || (blocker?.blockedTools.length ?? 0) > 0 || (blocker?.blockedDomains.length ?? 0) > 0) ? (
              <div className="mt-3 grid gap-3 text-xs sm:grid-cols-2">
                <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
                  <p className="font-medium text-muted-foreground">Operator controls</p>
                  <div className="mt-2 space-y-1.5 text-foreground">
                    <p>Network: {blocker?.blockNetworkUsage ? 'Blocked' : 'Allowed'}</p>
                    <p>
                      Tools: {(blocker?.blockedTools.length ?? 0) > 0 ? blocker?.blockedTools.join(', ') : 'No tool blocks'}
                    </p>
                    <p>
                      Domains: {(blocker?.blockedDomains.length ?? 0) > 0 ? blocker?.blockedDomains.join(', ') : 'No domain blocks'}
                    </p>
                  </div>
                </div>
                <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
                  <p className="font-medium text-muted-foreground">Approval wait</p>
                  <div className="mt-2 space-y-1.5 text-foreground">
                    <p>Pending approvals: {blocker?.approvalIds.length ?? 0}</p>
                    <p>
                      Tools: {(blocker?.approvalToolNames.length ?? 0) > 0 ? blocker?.approvalToolNames.join(', ') : 'Not reported'}
                    </p>
                  </div>
                </div>
              </div>
            ) : null}
          </div>
        ) : null}

        {hasApprovalMetadata ? (
          <div className="rounded-2xl border border-border bg-elevated-layer/80 p-3">
            <div className="flex items-center justify-between gap-3">
              <div>
                <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                  Pending approvals
                </p>
                <p className="mt-1 text-xs text-muted-foreground">
                  Resolve the pending approval here, or open the full Goal Console for deeper review.
                </p>
              </div>
              {goal.pendingApprovalCount > 0 ? (
                <Badge variant="error">
                  {goal.pendingApprovalCount} approval{goal.pendingApprovalCount > 1 ? 's' : ''}
                </Badge>
              ) : null}
            </div>

            {approvalLoading ? (
              <div className="mt-3 rounded-xl border border-border bg-surface-layer/70 px-3 py-4 text-xs text-muted-foreground">
                Loading pending approvals...
              </div>
            ) : approvals.length > 0 ? (
              <div className="mt-3 space-y-3">
                {approvals.map((approval) => {
                  const previewFiles = approval.file_change_groups.flatMap((group) => group.files).slice(0, 3)
                  const extraFileCount = Math.max(0, countApprovalFiles(approval) - previewFiles.length)

                  return (
                    <div key={approval.approval_id} className="rounded-xl border border-border bg-surface-layer/70 p-3">
                      <div className="flex flex-wrap items-start justify-between gap-3">
                        <div className="min-w-0 flex-1 space-y-2">
                          <div className="flex flex-wrap items-center gap-2">
                            <Badge variant="warning">{approval.status}</Badge>
                            <span className="text-sm font-semibold text-foreground">{approval.tool_name}</span>
                          </div>
                          <p className="text-xs text-muted-foreground">
                            {approval.command
                              ? previewText(approval.command, 180)
                              : approval.reason || approval.policy_reason || approval.approval_id}
                          </p>
                          <p className="text-xs text-muted-foreground">
                            {approval.workdir
                              ? `${approval.shell || 'shell'} | ${approval.workdir}`
                              : approval.shell || 'No shell'}
                          </p>
                          <div className="flex flex-wrap gap-2">
                            <Badge variant="outline">{formatApprovalScope(approval)}</Badge>
                            {approvalSummaryBadges(approval).map((badge) => (
                              <Badge key={`${approval.approval_id}-${badge}`} variant="outline">
                                {badge}
                              </Badge>
                            ))}
                          </div>
                          {previewFiles.length > 0 ? (
                            <div className="rounded-xl border border-border bg-elevated-layer/60 p-2.5">
                              <p className="text-[11px] font-medium text-muted-foreground">
                                {approval.file_change_groups[0]?.title ?? 'Pending file review'}
                              </p>
                              <div className="mt-2 space-y-2">
                                {previewFiles.map((file) => (
                                  <div
                                    key={`${approval.approval_id}-${file.filePath}`}
                                    className="flex items-center justify-between gap-3 rounded-lg border border-border/70 bg-surface-layer/60 px-2.5 py-2"
                                  >
                                    <div className="min-w-0">
                                      <p className="truncate text-xs font-medium text-foreground">
                                        {getFileName(file.displayPath)}
                                      </p>
                                      <p className="truncate text-[11px] text-muted-foreground">
                                        {file.displayPath}
                                      </p>
                                    </div>
                                    <div className="shrink-0 text-[11px] text-muted-foreground">
                                      +{file.additions} -{file.deletions}
                                    </div>
                                  </div>
                                ))}
                              </div>
                              {extraFileCount > 0 ? (
                                <p className="mt-2 text-[11px] text-muted-foreground">
                                  +{extraFileCount} more file{extraFileCount > 1 ? 's' : ''}
                                </p>
                              ) : null}
                            </div>
                          ) : null}
                        </div>
                        <div className="flex flex-wrap gap-2">
                          {approval.allowed_decisions.includes('approve_once') && onResolveApproval ? (
                            <Button
                              type="button"
                              size="sm"
                              variant="secondary"
                              onClick={() => {
                                void onResolveApproval(approval.approval_id, 'approve_once')
                              }}
                              loading={resolvingApprovalKey === `${approval.approval_id}:approve_once`}
                            >
                              Approve once
                            </Button>
                          ) : null}
                          {approval.allowed_decisions.includes('reject') && onResolveApproval ? (
                            <Button
                              type="button"
                              size="sm"
                              variant="destructive"
                              onClick={() => {
                                void onResolveApproval(approval.approval_id, 'reject')
                              }}
                              loading={resolvingApprovalKey === `${approval.approval_id}:reject`}
                            >
                              Reject
                            </Button>
                          ) : null}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="mt-3 rounded-xl border border-dashed border-border px-3 py-4 text-xs text-muted-foreground">
                Approval metadata is present, but the detailed approval payload is not available yet on this surface.
              </div>
            )}
          </div>
        ) : null}
      </div>

      <div className="border-t border-border px-4 py-4">
        <div className="flex flex-wrap items-center gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={onRefresh}
            loading={busyAction === 'status'}
          >
            {busyAction === 'status' ? null : <Loader2 className="h-3.5 w-3.5" />}
            Refresh
          </Button>
          {pauseAvailable ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onPause}
              loading={busyAction === 'pause'}
            >
              {busyAction === 'pause' ? null : <Pause className="h-3.5 w-3.5" />}
              Pause
            </Button>
          ) : null}
          {resumeAvailable ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={onResume}
              loading={busyAction === 'resume'}
            >
              {busyAction === 'resume' ? null : <Play className="h-3.5 w-3.5" />}
              Resume
            </Button>
          ) : null}
          {stopAvailable ? (
            <Button
              type="button"
              variant="destructive"
              size="sm"
              onClick={onStop}
              loading={busyAction === 'stop'}
            >
              {busyAction === 'stop' ? null : <Square className="h-3.5 w-3.5" />}
              Stop
            </Button>
          ) : null}
          <Button type="button" variant="ghost" size="sm" onClick={onOpenConsole}>
            <ExternalLink className="h-3.5 w-3.5" />
            Open console
          </Button>
        </div>
      </div>
    </div>
  )
}
