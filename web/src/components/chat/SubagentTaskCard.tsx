'use client'

import { AlertCircle, Bot, CheckCircle2, ExternalLink, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { DelegatedSubagentCardView } from '@/lib/subagent-tasks'
import { cn, formatRelativeTime } from '@/lib/utils'

interface SubagentTaskCardProps {
  card: DelegatedSubagentCardView
  onOpenTask?: (taskId: string) => void
}

function statusTone(card: DelegatedSubagentCardView): string {
  if (card.state === 'creating') {
    return 'border-sky-400/30 bg-sky-500/10 text-sky-100'
  }
  if (card.state === 'error' && !card.taskId) {
    return 'border-destructive/30 bg-destructive/10 text-destructive'
  }

  const status = card.status
  const normalized = (status ?? '').toLowerCase()
  if (normalized === 'succeeded' || normalized === 'completed' || normalized === 'done') {
    return 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'border-destructive/30 bg-destructive/10 text-destructive'
  }
  if (normalized === 'running' || normalized === 'queued' || normalized === 'resumed' || normalized === 'awaiting_approval') {
    return 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  return 'border-border bg-surface-layer text-muted-foreground'
}

function statusIcon(card: DelegatedSubagentCardView) {
  if (card.state === 'creating') {
    return <Loader2 className="h-4 w-4 animate-spin text-sky-100" />
  }
  if (card.state === 'error' && !card.taskId) {
    return <AlertCircle className="h-4 w-4 text-destructive" />
  }

  const status = card.status
  const normalized = (status ?? '').toLowerCase()
  if (normalized === 'succeeded' || normalized === 'completed' || normalized === 'done') {
    return <CheckCircle2 className="h-4 w-4 text-emerald-300" />
  }
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return <AlertCircle className="h-4 w-4 text-destructive" />
  }
  if (normalized === 'running' || normalized === 'queued' || normalized === 'resumed' || normalized === 'awaiting_approval') {
    return <Loader2 className="h-4 w-4 animate-spin text-warning-foreground" />
  }
  return <Bot className="h-4 w-4 text-primary-300" />
}

export function SubagentTaskCard({ card, onOpenTask }: SubagentTaskCardProps) {
  const updatedLabel = card.updatedAt ? formatRelativeTime(new Date(card.updatedAt)) : null
  const statusLabel =
    card.state === 'creating'
      ? 'creating'
      : card.state === 'error' && !card.taskId
        ? 'error'
        : card.status ?? 'created'

  return (
    <div className="w-full max-w-[760px] rounded-2xl border border-border bg-elevated-layer/95 p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary-400/30 bg-primary-500/12">
              <Bot className="h-4 w-4 text-primary-300" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-foreground">{card.title}</p>
              <p className="text-xs text-muted-foreground">
                Subagent task
                {updatedLabel ? ` / updated ${updatedLabel}` : ''}
              </p>
            </div>
          </div>
        </div>
        <span className={cn('inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize', statusTone(card))}>
          {statusIcon(card)}
          {statusLabel.replaceAll('_', ' ')}
        </span>
      </div>

      {card.instruction ? (
        <div className="mt-4 rounded-xl border border-border bg-surface-layer/70 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Main agent instruction
          </p>
          <p className="mt-1 line-clamp-4 whitespace-pre-wrap text-xs leading-5 text-foreground/90">
            {card.instruction}
          </p>
        </div>
      ) : null}

      {card.state === 'creating' ? (
        <div className="mt-3 rounded-xl border border-sky-400/20 bg-sky-500/10 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-sky-100">
            Creating subagent
          </p>
          <p className="mt-1 whitespace-pre-wrap text-xs leading-5 text-sky-50/90">
            Waiting for the delegated task to be created before the conversation can open.
          </p>
        </div>
      ) : null}

      {card.finalAnswer || card.error ? (
        <div className="mt-3 rounded-xl border border-border bg-surface-layer/50 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {card.error ? 'Issue' : 'Latest result'}
          </p>
          <p className={cn('mt-1 line-clamp-4 whitespace-pre-wrap text-xs leading-5', card.error ? 'text-destructive' : 'text-muted-foreground')}>
            {card.error ?? card.finalAnswer}
          </p>
        </div>
      ) : null}

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button
          type="button"
          variant="secondary"
          size="sm"
          disabled={!card.taskId}
          onClick={() => {
            if (card.taskId) {
              onOpenTask?.(card.taskId)
            }
          }}
        >
          {card.taskId
            ? 'Open subagent conversation'
            : card.state === 'creating'
              ? 'Creating subagent...'
              : 'Subagent unavailable'}
          {card.taskId ? <ExternalLink className="h-3.5 w-3.5" /> : null}
        </Button>
      </div>
    </div>
  )
}
