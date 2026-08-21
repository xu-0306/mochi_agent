'use client'

import * as React from 'react'
import { ExternalLink, Pause, Play, Send, Square, X } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { GoalDrawerModel } from '@/lib/goal-drawer-model'
import type { GoalExecutionRecordTarget } from '@/lib/goal-execution-record-target'

interface GoalComposerDrawerProps {
  model: GoalDrawerModel
  executionRecord: GoalExecutionRecordTarget
  busyAction?: 'status' | 'pause' | 'resume' | 'stop' | 'steer' | null
  onPause: () => void
  onResume: () => void
  onStop: () => void
  onSteer: (instruction: string) => void
  onClear: () => void
  onOpenExecutionRecord: () => void
}

export function GoalComposerDrawer({
  model,
  executionRecord,
  busyAction = null,
  onPause,
  onResume,
  onStop,
  onSteer,
  onClear,
  onOpenExecutionRecord,
}: GoalComposerDrawerProps) {
  const [instruction, setInstruction] = React.useState('')
  const [clearConfirmationOpen, setClearConfirmationOpen] = React.useState(false)
  const isBusy = busyAction !== null

  React.useEffect(() => {
    setClearConfirmationOpen(false)
  }, [model.goalId])

  const handleSteerSubmit = (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault()
    const nextInstruction = instruction.trim()
    if (!nextInstruction || isBusy) {
      return
    }
    onSteer(nextInstruction)
    setInstruction('')
  }

  const handleConfirmClear = () => {
    if (isBusy) {
      return
    }
    setClearConfirmationOpen(false)
    onClear()
  }

  return (
    <section
      aria-busy={isBusy}
      aria-label="Goal controls"
      className="mx-auto w-full max-w-4xl border-x border-t border-border bg-elevated-layer/95 px-4 py-3 shadow-sm sm:rounded-t-xl sm:border-x"
      data-testid="goal-composer-drawer"
    >
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <p className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">Goal</p>
          <p className="truncate text-sm font-semibold text-foreground">{model.title}</p>
          <p
            aria-live="polite"
            className="mt-0.5 text-xs text-muted-foreground"
            data-testid="goal-composer-status"
          >
            {model.status}
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {model.canPause ? (
            <Button
              data-testid="goal-action-pause"
              disabled={isBusy}
              loading={busyAction === 'pause'}
              onClick={onPause}
              size="sm"
              type="button"
              variant="outline"
            >
              <Pause className="h-3.5 w-3.5" />
              Pause
            </Button>
          ) : null}
          {model.canResume ? (
            <Button
              data-testid="goal-action-resume"
              disabled={isBusy}
              loading={busyAction === 'resume'}
              onClick={onResume}
              size="sm"
              type="button"
              variant="outline"
            >
              <Play className="h-3.5 w-3.5" />
              Resume
            </Button>
          ) : null}
          {model.canStop ? (
            <Button
              data-testid="goal-action-stop"
              disabled={isBusy}
              loading={busyAction === 'stop'}
              onClick={onStop}
              size="sm"
              type="button"
              variant="destructive"
            >
              <Square className="h-3.5 w-3.5" />
              Stop
            </Button>
          ) : null}
          {model.canOpenExecutionRecord && executionRecord.kind === 'agent_run' ? (
            <Button
              data-testid="goal-action-execution-record"
              disabled={isBusy}
              onClick={onOpenExecutionRecord}
              size="sm"
              type="button"
              variant="ghost"
            >
              <ExternalLink className="h-3.5 w-3.5" />
              Execution record
            </Button>
          ) : null}
          {model.canClear && !clearConfirmationOpen ? (
            <Button
              data-testid="goal-action-clear"
              disabled={isBusy}
              onClick={() => setClearConfirmationOpen(true)}
              size="sm"
              type="button"
              variant="ghost"
            >
              <X className="h-3.5 w-3.5" />
              Clear
            </Button>
          ) : null}
        </div>
      </div>
      {clearConfirmationOpen ? (
        <div
          aria-live="polite"
          className="mt-3 flex flex-wrap items-center justify-between gap-3 rounded-md border border-border bg-canvas px-3 py-2 text-xs text-muted-foreground"
          data-testid="goal-clear-disclosure"
        >
          <p className="min-w-0">
            This only hides the Goal from this chat. Its execution and durable history, including
            linked Agent Runs, remain available.
          </p>
          <div className="flex shrink-0 gap-2">
            <Button
              data-testid="goal-clear-cancel"
              disabled={isBusy}
              onClick={() => setClearConfirmationOpen(false)}
              size="sm"
              type="button"
              variant="ghost"
            >
              Keep visible
            </Button>
            <Button
              data-testid="goal-clear-confirm"
              disabled={isBusy}
              onClick={handleConfirmClear}
              size="sm"
              type="button"
              variant="outline"
            >
              Hide from chat
            </Button>
          </div>
        </div>
      ) : null}
      {model.canEdit ? (
        <form className="mt-3 flex gap-2" onSubmit={handleSteerSubmit}>
          <input
            aria-label="Steer this goal"
            className="min-w-0 flex-1 rounded-md border border-border bg-canvas px-3 py-2 text-sm"
            data-testid="goal-steer-input"
            disabled={isBusy}
            onChange={(event) => setInstruction(event.target.value)}
            placeholder="Steer this goal"
            value={instruction}
          />
          <Button
            data-testid="goal-action-steer"
            disabled={isBusy || instruction.trim().length === 0}
            loading={busyAction === 'steer'}
            size="sm"
            type="submit"
          >
            <Send className="h-3.5 w-3.5" />
            Steer
          </Button>
        </form>
      ) : null}
    </section>
  )
}
