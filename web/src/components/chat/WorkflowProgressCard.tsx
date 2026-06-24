'use client'

import * as React from 'react'
import { AlertCircle, CheckCircle2, ChevronDown, ExternalLink, Workflow } from 'lucide-react'
import type { WorkflowAgentStatus, WorkflowProgressCardView } from '@/components/workflow/types'
import { Button } from '@/components/ui/button'
import { cn, formatRelativeTime } from '@/lib/utils'

interface WorkflowProgressCardProps {
  card: WorkflowProgressCardView
}

function statusTone(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized === 'complete' || normalized === 'succeeded' || normalized === 'done') {
    return 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  if (normalized === 'error' || normalized === 'failed' || normalized === 'cancelled') {
    return 'border-destructive/30 bg-destructive/10 text-destructive'
  }
  if (normalized === 'partial' || normalized.includes('awaiting') || normalized.includes('blocked')) {
    return 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  return 'border-primary-400/30 bg-primary-500/10 text-primary-200'
}

function roleStatusLabel(status: WorkflowAgentStatus): string {
  if (status === 'running_tool') {
    return 'running tool'
  }
  return status.replaceAll('_', ' ')
}

function resultTitle(card: WorkflowProgressCardView): string {
  if (card.finalResult.status === 'complete') {
    return 'Workflow completed'
  }
  if (card.finalResult.status === 'partial') {
    return 'Workflow finalized with partial results'
  }
  if (card.finalResult.status === 'error') {
    return 'Workflow needs attention'
  }
  return 'Workflow in progress'
}

export function WorkflowProgressCard({ card }: WorkflowProgressCardProps) {
  const [expanded, setExpanded] = React.useState(false)
  const updatedLabel = card.updatedAt ? formatRelativeTime(new Date(card.updatedAt)) : null
  const finalContent = card.finalResult.content?.trim() ?? ''

  return (
    <div className="w-full max-w-[820px] rounded-2xl border border-border bg-elevated-layer/95 p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary-400/30 bg-primary-500/12">
              <Workflow className="h-4 w-4 text-primary-300" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-foreground">{card.title}</p>
              <p className="text-xs text-muted-foreground">
                {card.phaseLabel}
                {updatedLabel ? ` / updated ${updatedLabel}` : ''}
              </p>
            </div>
          </div>
        </div>
        <span className={cn('rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize', statusTone(card.status))}>
          {card.status}
        </span>
      </div>

      <div className="mt-4 rounded-xl border border-border bg-surface-layer/70 p-3">
        <div className="flex items-start gap-2">
          {card.finalResult.status === 'complete' ? (
            <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-300" />
          ) : card.finalResult.status === 'error' ? (
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-destructive" />
          ) : (
            <Workflow className="mt-0.5 h-4 w-4 shrink-0 text-primary-300" />
          )}
          <div className="min-w-0 flex-1">
            <p className="text-sm font-medium text-foreground">{resultTitle(card)}</p>
            <p className="mt-1 text-xs leading-5 text-muted-foreground">
              {finalContent || card.summary}
            </p>
          </div>
        </div>
      </div>

      {card.roles.length > 0 ? (
        <div className="mt-4 grid gap-2 sm:grid-cols-2">
          {card.roles.map((role) => (
            <div key={role.roleId} className="rounded-xl border border-border bg-surface-layer/50 p-3">
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-xs font-semibold text-foreground">{role.label}</p>
                <span className={cn('rounded-full border px-2 py-0.5 text-[10px] capitalize', statusTone(role.status))}>
                  {roleStatusLabel(role.status)}
                </span>
              </div>
              <p className="mt-2 line-clamp-2 text-xs leading-5 text-muted-foreground">
                {role.currentAction}
              </p>
              {role.lastOutputSummary ? (
                <p className="mt-2 line-clamp-2 text-[11px] leading-4 text-muted-foreground/80">
                  {role.lastOutputSummary}
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button asChild type="button" variant="secondary" size="sm">
          <a href={`/agent-runs/${encodeURIComponent(card.runId)}`}>
            Open workflow
            <ExternalLink className="h-3.5 w-3.5" />
          </a>
        </Button>
        {card.transcriptSnippets.length > 0 ? (
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((current) => !current)}
          >
            {expanded ? 'Hide agent notes' : 'Show agent notes'}
            <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', expanded && 'rotate-180')} />
          </Button>
        ) : null}
      </div>

      {expanded && card.transcriptSnippets.length > 0 ? (
        <div className="mt-3 space-y-2 border-t border-border pt-3">
          {card.transcriptSnippets.map((message) => (
            <div key={message.id} className="rounded-lg bg-surface-layer/60 px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <p className="text-[11px] font-semibold text-foreground">{message.label}</p>
                {message.meta ? <span className="text-[10px] text-muted-foreground">{message.meta}</span> : null}
              </div>
              <p className="mt-1 line-clamp-3 text-xs leading-5 text-muted-foreground">{message.content}</p>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
