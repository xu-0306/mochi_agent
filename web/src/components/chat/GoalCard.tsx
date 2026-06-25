'use client'

import {
  AlertTriangle,
  Compass,
  Flag,
  Layers3,
  PlayCircle,
  RefreshCcw,
  Target,
  type LucideIcon,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { GoalCardView } from '@/lib/chat'
import { cn } from '@/lib/utils'

interface GoalCardProps {
  card: GoalCardView
}

function formatEnumLabel(value: string): string {
  return value.replaceAll('_', ' ')
}

function kindLabel(kind: GoalCardView['kind']): string {
  if (kind === 'revised_proposal') {
    return 'Revised proposal'
  }
  if (kind === 'started') {
    return 'Goal started'
  }
  return 'Proposal'
}

function executionModeLabel(mode: GoalCardView['executionMode']): string {
  return mode === 'single_agent' ? 'Single agent' : 'Workflow'
}

function kindTone(kind: GoalCardView['kind'], superseded: boolean): string {
  if (superseded) {
    return 'border-border bg-surface-layer text-muted-foreground'
  }
  if (kind === 'revised_proposal') {
    return 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  if (kind === 'started') {
    return 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  return 'border-primary-400/30 bg-primary-500/10 text-primary-200'
}

function statusTone(status: string | null | undefined): string {
  const normalized = (status ?? '').toLowerCase()
  if (
    normalized === 'completed' ||
    normalized === 'succeeded' ||
    normalized === 'done'
  ) {
    return 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  if (
    normalized === 'running' ||
    normalized === 'active' ||
    normalized === 'started' ||
    normalized === 'in_progress'
  ) {
    return 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200'
  }
  if (
    normalized === 'waiting_approval' ||
    normalized === 'blocked' ||
    normalized === 'paused' ||
    normalized === 'awaiting_approval' ||
    normalized === 'awaiting_resources' ||
    normalized === 'stalled' ||
    normalized === 'partial'
  ) {
    return 'border-warning/30 bg-warning/10 text-warning-foreground'
  }
  if (
    normalized === 'failed' ||
    normalized === 'error' ||
    normalized === 'cancelled'
  ) {
    return 'border-destructive/30 bg-destructive/10 text-destructive'
  }
  return 'border-border bg-surface-layer text-muted-foreground'
}

function kindIcon(kind: GoalCardView['kind']): LucideIcon {
  if (kind === 'revised_proposal') {
    return RefreshCcw
  }
  if (kind === 'started') {
    return PlayCircle
  }
  return Flag
}

function MetaTile({
  icon: Icon,
  label,
  value,
  mono = false,
}: {
  icon: LucideIcon
  label: string
  value: string
  mono?: boolean
}) {
  return (
    <div className="rounded-xl border border-border bg-surface-layer/55 p-3">
      <div className="flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        <Icon className="h-3.5 w-3.5" />
        <span>{label}</span>
      </div>
      <p
        className={cn(
          'mt-2 break-words text-xs leading-5 text-foreground',
          mono && 'font-mono text-[11px]'
        )}
      >
        {value}
      </p>
    </div>
  )
}

export function GoalCard({ card }: GoalCardProps) {
  const KindIcon = kindIcon(card.kind)
  const visibleModels = card.models.slice(0, 3)
  const hiddenModelCount = Math.max(0, card.models.length - visibleModels.length)
  const metadata = [
    {
      icon: Target,
      label: 'Execution',
      value: executionModeLabel(card.executionMode),
      mono: false,
    },
    ...(card.protocolId
      ? [{
          icon: Layers3,
          label: 'Protocol',
          value: card.protocolId,
          mono: true,
        }]
      : []),
    ...(card.runtimeMode
      ? [{
          icon: Compass,
          label: 'Runtime',
          value: card.runtimeMode,
          mono: false,
        }]
      : []),
    ...(card.goalId
      ? [{
          icon: Flag,
          label: 'Goal ID',
          value: card.goalId,
          mono: true,
        }]
      : []),
  ]

  return (
    <div className="w-full max-w-[780px] rounded-2xl border border-border bg-elevated-layer/95 p-4 shadow-sm">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary-400/30 bg-primary-500/12">
              <KindIcon className="h-4 w-4 text-primary-300" />
            </span>
            <div className="min-w-0">
              <p className="truncate text-sm font-semibold text-foreground">{card.label}</p>
              <p className="text-xs text-muted-foreground">{kindLabel(card.kind)}</p>
            </div>
          </div>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <span
            className={cn(
              'rounded-full border px-2.5 py-1 text-[11px] font-medium',
              kindTone(card.kind, Boolean(card.superseded))
            )}
          >
            {kindLabel(card.kind)}
          </span>
          <Badge variant="outline">{executionModeLabel(card.executionMode)}</Badge>
          {card.status ? (
            <span
              className={cn(
                'rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize',
                statusTone(card.status)
              )}
            >
              {formatEnumLabel(card.status)}
            </span>
          ) : null}
          {card.superseded ? <Badge variant="outline">Superseded</Badge> : null}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-border bg-surface-layer/70 p-3">
        <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          Objective
        </p>
        <p className="mt-1 whitespace-pre-wrap text-sm leading-6 text-foreground">
          {card.objective}
        </p>
      </div>

      {metadata.length > 0 ? (
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          {metadata.map((item) => (
            <MetaTile
              key={item.label}
              icon={item.icon}
              label={item.label}
              value={item.value}
              mono={item.mono}
            />
          ))}
        </div>
      ) : null}

      {visibleModels.length > 0 ? (
        <div className="mt-4 rounded-xl border border-border bg-surface-layer/50 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Models
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {visibleModels.map((model) => (
              <Badge key={model} variant="outline" className="max-w-full">
                <span className="truncate">{model}</span>
              </Badge>
            ))}
            {hiddenModelCount > 0 ? (
              <Badge variant="outline">+{hiddenModelCount} more</Badge>
            ) : null}
          </div>
        </div>
      ) : null}

      {card.roleSummary ? (
        <div className="mt-4 rounded-xl border border-border bg-surface-layer/50 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            Role summary
          </p>
          <p className="mt-1 whitespace-pre-wrap text-xs leading-5 text-foreground/90">
            {card.roleSummary}
          </p>
        </div>
      ) : null}

      {card.riskNote ? (
        <div className="mt-4 rounded-xl border border-warning/30 bg-warning/10 p-3">
          <div className="flex items-start gap-2">
            <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-warning-foreground" />
            <div className="min-w-0">
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-warning-foreground">
                Risk note
              </p>
              <p className="mt-1 whitespace-pre-wrap text-xs leading-5 text-warning-foreground/90">
                {card.riskNote}
              </p>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  )
}
