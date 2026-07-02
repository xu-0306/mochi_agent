'use client'

import {
  AlertCircle,
  CheckCircle2,
  ExternalLink,
  ShieldAlert,
  Target,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import type { ExecutionTranscriptEvent, SubagentTranscriptSummary } from '@/lib/api'
import type { GoalFocusCalloutView } from '@/lib/goal-focus-surface'
import {
  buildGoalCardChromeCopy,
  buildGoalCardExecutionModeLabel,
  buildGoalCardStatusLabel,
  buildGoalDisplayStateLabel,
  buildGoalModelCountLabel,
} from '@/lib/goal-proposal-copy'
import { cn } from '@/lib/utils'
import { ExecutionTimeline } from './ExecutionTimeline'
import {
  type GoalDrawerBlockerView,
  type GoalHeaderChipView,
} from './GoalHeaderChip'
import { SubagentTimelineCard } from './SubagentTimelineCard'

interface GoalFocusPanelProps {
  goal: GoalHeaderChipView
  blocker?: GoalDrawerBlockerView | null
  callout?: GoalFocusCalloutView | null
  timelineEvents?: ExecutionTranscriptEvent[]
  subagents?: SubagentTranscriptSummary[]
  timelineError?: string | null
  activeSubagentId?: string | null
  expandedSubagentIds?: Set<string>
  onExpandedChange?: (subagentId: string, expanded: boolean) => void
  onOpenSubagent?: (subagentId: string) => void
  onReviewApprovals?: () => void
  onOpenDetails?: () => void
  onOpenConsole?: () => void
}

function displayStateVariant(
  state: GoalHeaderChipView['displayState']
): 'primary' | 'warning' | 'success' | 'error' {
  if (state === 'completed') {
    return 'success'
  }
  if (state === 'failed') {
    return 'error'
  }
  if (state === 'blocked') {
    return 'warning'
  }
  return 'primary'
}

function displayStateIcon(state: GoalHeaderChipView['displayState']) {
  if (state === 'completed') {
    return <CheckCircle2 className="h-3.5 w-3.5 text-success" />
  }
  if (state === 'failed') {
    return <AlertCircle className="h-3.5 w-3.5 text-destructive" />
  }
  if (state === 'blocked') {
    return <ShieldAlert className="h-3.5 w-3.5 text-warning" />
  }
  return <Target className="h-3.5 w-3.5 text-primary-400" />
}

function displayStateFrameTone(state: GoalHeaderChipView['displayState']): string {
  if (state === 'completed') {
    return 'border-success/25 bg-success/10'
  }
  if (state === 'failed') {
    return 'border-destructive/25 bg-destructive/10'
  }
  if (state === 'blocked') {
    return 'border-warning/25 bg-warning/10'
  }
  return 'border-primary-400/25 bg-primary-500/10'
}

function calloutTone(
  tone: GoalFocusCalloutView['tone']
): string {
  if (tone === 'error') {
    return 'border-destructive/30 bg-destructive/10 text-destructive'
  }
  if (tone === 'warning') {
    return 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  return 'border-primary-400/30 bg-primary-500/10 text-primary-100'
}

export function GoalFocusPanel({
  goal,
  blocker = null,
  callout = null,
  timelineEvents = [],
  subagents = [],
  timelineError = null,
  activeSubagentId = null,
  expandedSubagentIds,
  onExpandedChange,
  onOpenSubagent,
  onReviewApprovals,
  onOpenDetails,
  onOpenConsole,
}: GoalFocusPanelProps) {
  const copySource =
    goal.copySource ||
    blocker?.summary ||
    blocker?.latestError ||
    goal.title
  const chromeCopy = buildGoalCardChromeCopy(copySource)
  const displayLabel = buildGoalDisplayStateLabel(copySource, goal.displayState)
  const executionLabel = buildGoalCardExecutionModeLabel(copySource, goal.executionMode)
  const statusLabel =
    buildGoalCardStatusLabel(copySource, goal.status) ??
    goal.status.replaceAll('_', ' ')
  const hasTimeline = subagents.length > 0 || timelineEvents.length > 0 || Boolean(timelineError)
  const summaryIntro =
    goal.runtimeMode ||
    (goal.displayState === 'completed'
      ? chromeCopy.recentSummaryIntro
      : goal.displayState === 'failed'
        ? chromeCopy.goalNeedsAttentionBody
      : goal.displayState === 'blocked'
        ? chromeCopy.goalNeedsAttentionBody
        : chromeCopy.activeSummaryIntro)

  return (
    <section className="rounded-[1.75rem] border border-border bg-elevated-layer/95 p-4 shadow-sm sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant={displayStateVariant(goal.displayState)}>{displayLabel}</Badge>
            <Badge variant="outline">{statusLabel}</Badge>
            <Badge variant="outline">{executionLabel}</Badge>
            {goal.modelCount > 0 ? (
              <Badge variant="outline">
                {buildGoalModelCountLabel(copySource, goal.modelCount)}
              </Badge>
            ) : null}
            {goal.protocolId ? <Badge variant="outline">{goal.protocolId}</Badge> : null}
            {goal.pendingApprovalCount > 0 ? (
              <Badge variant="error">{goal.pendingApprovalCount}</Badge>
            ) : null}
          </div>
          <div className="mt-3 flex items-start gap-3">
            <span
              className={cn(
                'mt-0.5 flex h-9 w-9 shrink-0 items-center justify-center rounded-full border',
                displayStateFrameTone(goal.displayState)
              )}
            >
              {displayStateIcon(goal.displayState)}
            </span>
            <div className="min-w-0">
              <h2 className="text-base font-semibold leading-6 text-foreground">{goal.title}</h2>
              <p className="mt-1 text-sm leading-6 text-muted-foreground">
                {summaryIntro}
              </p>
            </div>
          </div>
        </div>
        <div className="flex shrink-0 flex-wrap items-center gap-2">
          {onOpenDetails ? (
            <Button type="button" size="sm" variant="secondary" onClick={onOpenDetails}>
              {chromeCopy.openDetailsLabel}
            </Button>
          ) : null}
          {onOpenConsole ? (
            <Button type="button" size="sm" variant="ghost" onClick={onOpenConsole}>
              <ExternalLink className="h-3.5 w-3.5" />
              {chromeCopy.openConsoleLabel}
            </Button>
          ) : null}
        </div>
      </div>

      {callout ? (
        <div className={cn('mt-4 rounded-2xl border p-3', calloutTone(callout.tone))}>
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2">
                <AlertCircle className="h-4 w-4 shrink-0" />
                <p className="text-[11px] font-semibold uppercase tracking-[0.08em]">
                  {callout.title}
                </p>
              </div>
              <p className="mt-2 text-sm leading-6">{callout.message}</p>
              {callout.detail ? (
                <p className="mt-2 text-xs leading-5 opacity-90">{callout.detail}</p>
              ) : null}
              {callout.meta.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {callout.meta.map((item) => (
                    <Badge
                      key={item}
                      variant="outline"
                      className="max-w-full border-current/20 bg-transparent text-current"
                    >
                      <span className="truncate">{item}</span>
                    </Badge>
                  ))}
                </div>
              ) : null}
            </div>
            {callout.actionLabel ? (
              <Button
                type="button"
                size="sm"
                variant={callout.tone === 'error' ? 'destructive' : 'secondary'}
                onClick={() => {
                  if (callout.action === 'review_approvals') {
                    onReviewApprovals?.()
                    return
                  }
                  if (callout.action === 'open_console') {
                    onOpenConsole?.()
                    return
                  }
                  onOpenDetails?.()
                }}
              >
                {callout.actionLabel}
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}

      <div className="mt-4 grid gap-3 sm:grid-cols-2">
        <div className="rounded-2xl border border-border bg-surface-layer/70 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {chromeCopy.goalSummaryLabel}
          </p>
          <p className="mt-2 whitespace-pre-wrap break-words text-sm leading-6 text-foreground">
            {goal.title}
          </p>
        </div>
        <div className="rounded-2xl border border-border bg-surface-layer/70 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {chromeCopy.goalIdLabel}
          </p>
          <p className="mt-2 break-all font-mono text-xs leading-6 text-foreground">
            {goal.goalId ?? chromeCopy.notStartedLabel}
          </p>
        </div>
      </div>

      {hasTimeline ? (
        <div className="mt-4 space-y-4">
          {timelineError ? (
            <div className="rounded-2xl border border-warning/30 bg-warning/10 px-3 py-2 text-xs text-warning-foreground">
              {timelineError}
            </div>
          ) : null}
          {subagents.length > 0 ? (
            <div className="grid gap-3 sm:grid-cols-2">
              {subagents.map((subagent) => (
                <SubagentTimelineCard
                  key={subagent.subagentId}
                  subagent={subagent}
                  active={subagent.subagentId === activeSubagentId}
                  expanded={expandedSubagentIds?.has(subagent.subagentId)}
                  onExpandedChange={onExpandedChange}
                  onOpen={onOpenSubagent}
                  onOpenThread={onOpenSubagent}
                  recentEvents={timelineEvents.filter((event) => event.subagentId === subagent.subagentId)}
                />
              ))}
            </div>
          ) : null}
          <ExecutionTimeline
            events={timelineEvents}
            subagents={subagents}
            activeSubagentId={activeSubagentId}
            onOpenSubagent={onOpenSubagent}
          />
        </div>
      ) : null}
    </section>
  )
}

export type { GoalFocusPanelProps }
