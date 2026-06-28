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
import {
  buildGoalCardChromeCopy,
  buildGoalCardExecutionModeLabel,
  buildGoalCardKindLabel,
  buildGoalCardStatusLabel,
  buildGoalHiddenModelsLabel,
  buildGoalProposalSystemCtaCopy,
} from '@/lib/goal-proposal-copy'
import { cn } from '@/lib/utils'

interface GoalCardProps {
  card: GoalCardView
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
  const copySource =
    card.copySource ||
    card.objective ||
    card.roleSummary ||
    card.runtimeMode ||
    card.label
  const chromeCopy = buildGoalCardChromeCopy(copySource)
  const localizedKindLabel = buildGoalCardKindLabel(copySource, card.kind)
  const localizedExecutionModeLabel = buildGoalCardExecutionModeLabel(
    copySource,
    card.executionMode
  )
  const localizedStatusLabel = buildGoalCardStatusLabel(copySource, card.status)
  const showProposalCta =
    !card.superseded &&
    (card.kind === 'proposal' || card.kind === 'revised_proposal')
  const proposalCtaCopy = buildGoalProposalSystemCtaCopy(
    copySource
  )
  const metadata = [
    {
      icon: Target,
      label: chromeCopy.executionLabel,
      value: localizedExecutionModeLabel,
      mono: false,
    },
    ...(card.protocolId
      ? [{
          icon: Layers3,
          label: chromeCopy.protocolLabel,
          value: card.protocolId,
          mono: true,
        }]
      : []),
    ...(card.runtimeMode
      ? [{
          icon: Compass,
          label: chromeCopy.runtimeLabel,
          value: card.runtimeMode,
          mono: false,
        }]
      : []),
    ...(card.goalId
      ? [{
          icon: Flag,
          label: chromeCopy.goalIdLabel,
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
              <p className="text-xs text-muted-foreground">{localizedKindLabel}</p>
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
            {localizedKindLabel}
          </span>
          <Badge variant="outline">{localizedExecutionModeLabel}</Badge>
          {localizedStatusLabel ? (
            <span
              className={cn(
                'rounded-full border px-2.5 py-1 text-[11px] font-medium capitalize',
                statusTone(card.status)
              )}
            >
              {localizedStatusLabel}
            </span>
          ) : null}
          {card.superseded ? <Badge variant="outline">{chromeCopy.supersededLabel}</Badge> : null}
        </div>
      </div>

      <div className="mt-4 rounded-xl border border-border bg-surface-layer/70 p-3">
        <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          {chromeCopy.objectiveLabel}
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
            {chromeCopy.modelsLabel}
          </p>
          <div className="mt-2 flex flex-wrap gap-2">
            {visibleModels.map((model) => (
              <Badge key={model} variant="outline" className="max-w-full">
                <span className="truncate">{model}</span>
              </Badge>
            ))}
            {hiddenModelCount > 0 ? (
              <Badge variant="outline">{buildGoalHiddenModelsLabel(copySource, hiddenModelCount)}</Badge>
            ) : null}
          </div>
        </div>
      ) : null}

      {showProposalCta ? (
        <div className="mt-4 rounded-xl border border-primary-400/20 bg-primary-500/8 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {proposalCtaCopy.title}
          </p>
          <div className="mt-2 grid gap-2 sm:grid-cols-3">
            <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                <PlayCircle className="h-4 w-4 text-primary-300" />
                <span>{proposalCtaCopy.launchLabel}</span>
              </div>
              <p className="mt-1 text-xs leading-5 text-foreground/80">
                {proposalCtaCopy.launchBody}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                <RefreshCcw className="h-4 w-4 text-primary-300" />
                <span>{proposalCtaCopy.reviseLabel}</span>
              </div>
              <p className="mt-1 text-xs leading-5 text-foreground/80">
                {proposalCtaCopy.reviseBody}
              </p>
            </div>
            <div className="rounded-xl border border-border bg-surface-layer/60 p-3">
              <div className="flex items-center gap-2 text-xs font-semibold text-foreground">
                <Compass className="h-4 w-4 text-primary-300" />
                <span>{proposalCtaCopy.chatLabel}</span>
              </div>
              <p className="mt-1 text-xs leading-5 text-foreground/80">
                {proposalCtaCopy.chatBody}
              </p>
            </div>
          </div>
        </div>
      ) : null}

      {card.roleSummary ? (
        <div className="mt-4 rounded-xl border border-border bg-surface-layer/50 p-3">
          <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
            {chromeCopy.roleSummaryLabel}
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
                {chromeCopy.riskNoteLabel}
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
