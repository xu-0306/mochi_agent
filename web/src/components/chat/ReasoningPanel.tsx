'use client'

import * as React from 'react'
import { Brain, ChevronDown, Loader2, Search, TerminalSquare, AlertCircle } from 'lucide-react'
import type { FileChangeSummary } from '@/lib/chat-p2'
import { extractFileChangeFromReasoningStep } from '@/lib/chat-p2'
import { cn } from '@/lib/utils'
import type { ReasoningStep, TokenStats } from '@/lib/chat'
import { FileChangeCard } from './FileChangeCard'
import {
  deriveReasoningPanelSummary,
  getNextReasoningPanelOpen,
  resolveReasoningGenerationTime,
} from './reasoning-panel-state'

interface ReasoningPanelProps {
  steps: ReasoningStep[]
  isStreaming?: boolean
  tokenStats?: TokenStats
  onUndoFileChange?: (change: FileChangeSummary) => Promise<void> | void
}

function StepIcon({ type, status }: { type: ReasoningStep['type']; status?: ReasoningStep['status'] }) {
  if (type === 'thinking') {
    return status === 'running' ? (
      <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-400" />
    ) : (
      <Brain className="h-3.5 w-3.5 text-primary-400" />
    )
  }
  if (type === 'tool_call') {
    return <Search className="h-3.5 w-3.5 text-primary-400" />
  }
  if (type === 'tool_result') {
    return <TerminalSquare className="h-3.5 w-3.5 text-primary-400" />
  }
  return <AlertCircle className="h-3.5 w-3.5 text-error" />
}

function stepLabel(step: ReasoningStep): string {
  if (step.type === 'thinking') {
    return 'Reasoning'
  }
  if (step.type === 'tool_call') {
    return step.toolName ? `Using ${step.toolName}` : 'Using tool'
  }
  if (step.type === 'tool_result') {
    return step.toolName ? `${step.toolName} finished` : 'Tool finished'
  }
  return 'Issue'
}

export function ReasoningPanel({
  steps,
  isStreaming = false,
  tokenStats,
  onUndoFileChange,
}: ReasoningPanelProps) {
  const [open, setOpen] = React.useState(isStreaming)
  const [userInteracted, setUserInteracted] = React.useState(false)
  const previousStreamingRef = React.useRef(isStreaming)

  React.useEffect(() => {
    setOpen((previousOpen) => getNextReasoningPanelOpen({
      previousOpen,
      previousStreaming: previousStreamingRef.current,
      isStreaming,
      userInteracted,
    }))
    previousStreamingRef.current = isStreaming
  }, [isStreaming, userInteracted])

  if (steps.length === 0) {
    return null
  }

  const summary = deriveReasoningPanelSummary({
    steps,
    isStreaming,
    generationTimeMs: resolveReasoningGenerationTime(tokenStats),
  })

  return (
    <div className="mb-3 max-w-[720px] overflow-hidden rounded-2xl border border-border/70 bg-surface-layer/55 shadow-[0_10px_30px_rgba(0,0,0,0.18)]">
      <button
        type="button"
        onClick={() => {
          setUserInteracted(true)
          setOpen((value) => !value)
        }}
        className="flex w-full items-center justify-between gap-3 px-3 py-2.5 text-left transition-colors hover:bg-white/[0.02]"
      >
        <div className="flex min-w-0 items-center gap-2">
          <div className={cn(
            'flex h-6 w-6 shrink-0 items-center justify-center rounded-full border',
            summary.hasError
              ? 'border-error/30 bg-error/10'
              : 'border-primary-500/30 bg-primary-500/10'
          )}>
            {isStreaming ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-400" />
            ) : summary.hasError ? (
              <AlertCircle className="h-3.5 w-3.5 text-error" />
            ) : (
              <Brain className="h-3.5 w-3.5 text-primary-400" />
            )}
          </div>
          <div className="min-w-0">
            <p className="text-sm font-medium text-foreground">
              {summary.title}
            </p>
            <p
              className={cn(
                'truncate text-xs',
                summary.hasError ? 'text-error/90' : 'text-muted-foreground'
              )}
              title={summary.latestIssue ?? undefined}
            >
              {summary.detail}
            </p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {isStreaming ? (
            <span className="rounded-full border border-primary-500/20 bg-primary-500/10 px-2 py-0.5 text-[11px] font-medium text-primary-300">
              live
            </span>
          ) : null}
          {!isStreaming ? (
            <span className="hidden text-[11px] text-muted-foreground sm:inline">
              {open ? 'Hide trace' : 'Show trace'}
            </span>
          ) : null}
          <ChevronDown
            className={cn('h-4 w-4 shrink-0 text-muted-foreground transition-transform', open && 'rotate-180')}
          />
        </div>
      </button>

      {open ? (
        <div className="space-y-3 border-t border-border/60 bg-black/[0.08] px-3 py-3">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-start gap-3">
              <div className="flex flex-col items-center pt-0.5">
                <div className="flex h-6 w-6 items-center justify-center rounded-full border border-border/80 bg-canvas/90">
                  <StepIcon type={step.type} status={step.status} />
                </div>
                {index < steps.length - 1 ? <div className="mt-1 h-6 w-px bg-border" /> : null}
              </div>
              <div className="min-w-0 flex-1 pb-1">
                <p className="text-xs font-medium text-muted-foreground">{stepLabel(step)}</p>
                {step.content ? (
                  <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground/90">
                    {step.content}
                  </p>
                ) : null}
                {step.errorCode ? (
                  <p className="mt-1 break-all text-xs text-error/90">
                    {step.errorCode}
                  </p>
                ) : null}
                {(() => {
                  const fileChange = extractFileChangeFromReasoningStep(step)
                  if (!fileChange) {
                    return null
                  }
                  return (
                    <div className="mt-3">
                      <FileChangeCard change={fileChange} onUndo={onUndoFileChange} />
                    </div>
                  )
                })()}
              </div>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  )
}
