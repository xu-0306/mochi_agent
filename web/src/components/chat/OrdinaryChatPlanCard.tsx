'use client'

import type { OrdinaryChatPlanState, OrdinaryChatTurn } from '@/lib/ordinary-chat-plan'
import {
  failureLearningDisplayItems,
  sortOrdinaryChatTurns,
} from '@/lib/ordinary-chat-plan'

interface OrdinaryChatPlanCardProps {
  state: OrdinaryChatPlanState
}

function latestVisibleTurn(state: OrdinaryChatPlanState): OrdinaryChatTurn | null {
  const allTurns = sortOrdinaryChatTurns(state)
  const turns = allTurns.filter(
    (turn) =>
      turn.plan ||
      turn.evidence.receipts.length > 0 ||
      turn.blockers.length > 0 ||
      Object.values(turn.failure_learning).some((count) => count > 0)
  )
  if (turns.length > 0) return turns.at(-1) ?? null
  const hasSessionLearning = failureLearningDisplayItems(state).some(
    (item) => item.value > 0
  )
  return hasSessionLearning || state.costCoverage.expected_turns > 0
    ? (allTurns.at(-1) ?? null)
    : null
}

export function OrdinaryChatPlanCard({ state }: OrdinaryChatPlanCardProps) {
  const turn = latestVisibleTurn(state)
  if (!turn) return null

  const planItems = turn.plan?.items ?? []
  const evidenceCount = turn.evidence.receipts.reduce(
    (count, receipt) => count + receipt.criteria.filter((criterion) => criterion.verdict === 'verified').length,
    0
  )
  const title = turn.plan?.objective || 'Adaptive task progress'
  const learningItems = failureLearningDisplayItems(state)
  const hasFailureLearning = learningItems.some((item) => item.value > 0)

  return (
    <section
      aria-label="Ordinary Chat task plan"
      className="mb-3 rounded-xl border border-border/70 bg-surface-layer/60 px-4 py-3 text-sm"
    >
      <div className="flex items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="truncate font-medium text-foreground">{title}</p>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Task plan · {turn.status.replace('_', ' ')}
          </p>
        </div>
        <span className="shrink-0 rounded-full border border-border px-2 py-0.5 text-[11px] capitalize text-muted-foreground">
          {turn.status.replace('_', ' ')}
        </span>
      </div>

      {planItems.length > 0 ? (
        <ul className="mt-3 space-y-1.5">
          {planItems.slice(0, 6).map((item) => (
            <li key={item.item_id ?? item.title ?? 'plan-item'} className="flex items-start gap-2 text-xs">
              <span
                className={`mt-1 h-1.5 w-1.5 shrink-0 rounded-full ${
                  item.status === 'completed'
                    ? 'bg-success'
                    : item.status === 'blocked'
                      ? 'bg-destructive'
                      : 'bg-muted-foreground'
                }`}
              />
              <span className="min-w-0 flex-1 text-muted-foreground">{item.title ?? 'Untitled step'}</span>
              <span className="shrink-0 capitalize text-muted-foreground/80">{item.status ?? 'pending'}</span>
            </li>
          ))}
        </ul>
      ) : null}

      {evidenceCount > 0 ? (
        <p className="mt-2 text-xs text-success">Verified evidence: {evidenceCount}</p>
      ) : null}

      {state.costCoverage.expected_turns > 0 ? (
        <p
          aria-label="Adaptive runtime cost coverage"
          className="mt-2 text-[11px] text-muted-foreground"
        >
          Cost coverage · {state.costCoverage.coverage} · tokens{' '}
          {state.costCoverage.token_coverage} · wall {state.costCoverage.wall_coverage}
        </p>
      ) : null}

      {hasFailureLearning ? (
        <div
          aria-label="Failure learning diagnostics"
          className="mt-2 rounded-lg border border-border/60 bg-background/40 px-2.5 py-2"
        >
          <p className="text-[11px] font-medium text-muted-foreground">
            Failure learning · {state.failureLearning.coverage}
          </p>
          <dl className="mt-1 grid grid-cols-2 gap-x-3 gap-y-1 text-[11px] text-muted-foreground sm:grid-cols-4">
            {learningItems.map((item) => (
              <div key={item.label} className="flex items-center justify-between gap-2 sm:block">
                <dt>{item.label}</dt>
                <dd className="font-medium tabular-nums text-foreground">{item.value}</dd>
              </div>
            ))}
          </dl>
        </div>
      ) : null}

      {turn.blockers.length > 0 ? (
        <div className="mt-2 rounded-lg border border-warning/30 bg-warning/10 px-2.5 py-2 text-xs text-warning-foreground">
          <span className="font-medium">Blocker:</span> {turn.blockers.slice(0, 2).join('; ')}
        </div>
      ) : null}
    </section>
  )
}
