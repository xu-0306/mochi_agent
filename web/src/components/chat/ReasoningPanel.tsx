'use client'

import * as React from 'react'
import { Brain, ChevronDown, Loader2, Search, TerminalSquare, AlertCircle } from 'lucide-react'
import { buildFilesystemFileUrl } from '@/lib/api'
import type { FileChangeSummary } from '@/lib/chat-p2'
import { extractFileChangeGroupFromReasoningStep } from '@/lib/chat-p2'
import { cn } from '@/lib/utils'
import type { ReasoningStep, TokenStats } from '@/lib/chat'
import { FileChangeCard } from './FileChangeCard'
import { getReasoningStepBadge } from './reasoning-badges'
import { ToolCallCard } from './ToolCallCard'
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
  if (type === 'status') {
    return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-400" />
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
  if (step.type === 'status') {
    return 'Progress'
  }
  if (step.type === 'tool_call') {
    return step.toolName ? `Using ${step.toolName}` : 'Using tool'
  }
  if (step.type === 'tool_result') {
    return step.toolName ? `${step.toolName} finished` : 'Tool finished'
  }
  return 'Issue'
}

function hasToolExposure(step: ReasoningStep): boolean {
  return Boolean(
    step.toolExposure &&
      (
        step.toolExposure.exposedTools.length > 0 ||
        step.toolExposure.workspaceBound !== undefined ||
        step.toolExposure.attachmentCount !== undefined
      )
  )
}

function hasTransport(step: ReasoningStep): boolean {
  return Boolean(
    step.transport &&
      (
        step.transport.summaryApplied !== undefined ||
        step.transport.overflowPersisted !== undefined ||
        step.transport.referenceId ||
        step.transport.artifactPath ||
        step.transport.sourcePath
      )
  )
}

function formatBooleanFlag(value: boolean | undefined): string | null {
  if (value === true) {
    return 'yes'
  }
  if (value === false) {
    return 'no'
  }
  return null
}

function diagnosticsSignature(step: ReasoningStep): string | null {
  const parts: string[] = []

  if (step.toolExposure) {
    if (step.toolExposure.exposedTools.length > 0) {
      parts.push(`tools:${step.toolExposure.exposedTools.join(',')}`)
    }
    if (typeof step.toolExposure.workspaceBound === 'boolean') {
      parts.push(`workspace:${step.toolExposure.workspaceBound}`)
    }
    if (typeof step.toolExposure.attachmentCount === 'number') {
      parts.push(`attachments:${step.toolExposure.attachmentCount}`)
    }
  }

  if (step.transport) {
    if (typeof step.transport.summaryApplied === 'boolean') {
      parts.push(`summary:${step.transport.summaryApplied}`)
    }
    if (typeof step.transport.overflowPersisted === 'boolean') {
      parts.push(`overflow:${step.transport.overflowPersisted}`)
    }
    if (step.transport.referenceId) {
      parts.push(`reference:${step.transport.referenceId}`)
    }
    if (step.transport.artifactPath) {
      parts.push(`artifact:${step.transport.artifactPath}`)
    }
    if (step.transport.sourcePath) {
      parts.push(`source:${step.transport.sourcePath}`)
    }
  }

  return parts.length > 0 ? parts.join('|') : null
}

function StepDiagnostics({
  step,
  suppressDuplicate,
}: {
  step: ReasoningStep
  suppressDuplicate?: boolean
}) {
  const showExposure = hasToolExposure(step)
  const showTransport = hasTransport(step)

  if ((!showExposure && !showTransport) || suppressDuplicate) {
    return null
  }

  const exposure = step.toolExposure
  const transport = step.transport
  const artifactHref = transport?.artifactPath
    ? buildFilesystemFileUrl(transport.artifactPath)
    : null

  return (
    <div className="mt-3 space-y-3 rounded-2xl border border-border/70 bg-canvas/45 p-3">
      {showExposure && exposure ? (
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
              Workspace tools
            </p>
            {typeof exposure.workspaceBound === 'boolean' ? (
              <span className="rounded-full border border-border/80 bg-canvas/80 px-2 py-0.5 text-[10px] text-muted-foreground">
                workspace bound: {exposure.workspaceBound ? 'yes' : 'no'}
              </span>
            ) : null}
            {typeof exposure.attachmentCount === 'number' ? (
              <span className="rounded-full border border-border/80 bg-canvas/80 px-2 py-0.5 text-[10px] text-muted-foreground">
                attachments: {exposure.attachmentCount}
              </span>
            ) : null}
          </div>
          {exposure.exposedTools.length > 0 ? (
            <div className="flex flex-wrap gap-1.5">
              {exposure.exposedTools.map((toolName) => (
                <span
                  key={toolName}
                  className="rounded-full border border-primary-500/20 bg-primary-500/10 px-2 py-0.5 text-[11px] text-primary-200"
                >
                  {toolName}
                </span>
              ))}
            </div>
          ) : null}
        </div>
      ) : null}

      {showTransport && transport ? (
        <div className="space-y-2">
          <p className="text-[11px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
            Tool transport
          </p>
          <div className="grid gap-2 sm:grid-cols-2">
            {formatBooleanFlag(transport.summaryApplied) ? (
              <div className="rounded-xl border border-border/70 bg-black/10 px-2.5 py-2">
                <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Summarized</p>
                <p className="mt-1 text-xs text-foreground/90">
                  {formatBooleanFlag(transport.summaryApplied)}
                </p>
              </div>
            ) : null}
            {formatBooleanFlag(transport.overflowPersisted) ? (
              <div className="rounded-xl border border-border/70 bg-black/10 px-2.5 py-2">
                <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Overflow persisted</p>
                <p className="mt-1 text-xs text-foreground/90">
                  {formatBooleanFlag(transport.overflowPersisted)}
                </p>
              </div>
            ) : null}
          </div>
          {transport.referenceId ? (
            <div>
              <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Reference ID</p>
              <p className="mt-1 break-all font-mono text-xs text-foreground/90">
                {transport.referenceId}
              </p>
            </div>
          ) : null}
          {transport.artifactPath ? (
            <div>
              <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Artifact path</p>
              <div className="mt-1 flex flex-wrap items-center gap-2">
                <p className="break-all font-mono text-xs text-foreground/90">
                  {transport.artifactPath}
                </p>
                {artifactHref ? (
                  <a
                    href={artifactHref}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-full border border-primary-500/20 bg-primary-500/10 px-2 py-0.5 text-[11px] text-primary-200 transition-colors hover:bg-primary-500/18"
                  >
                    Open artifact
                  </a>
                ) : null}
              </div>
            </div>
          ) : null}
          {transport.sourcePath ? (
            <div>
              <p className="text-[10px] uppercase tracking-[0.08em] text-muted-foreground">Source path</p>
              <p className="mt-1 break-all font-mono text-xs text-foreground/70">
                {transport.sourcePath}
              </p>
            </div>
          ) : null}
        </div>
      ) : null}
    </div>
  )
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
  const renderedDiagnosticSignatures = new Set<string>()

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
          {steps.map((step, index) => {
            const badge = getReasoningStepBadge(step)
            const signature = diagnosticsSignature(step)
            const suppressDuplicateDiagnostics =
              signature !== null && renderedDiagnosticSignatures.has(signature)
            if (signature !== null && !suppressDuplicateDiagnostics) {
              renderedDiagnosticSignatures.add(signature)
            }
            return (
              <div key={step.id} className="flex items-start gap-3">
                <div className="flex flex-col items-center pt-0.5">
                  <div className="flex h-6 w-6 items-center justify-center rounded-full border border-border/80 bg-canvas/90">
                    <StepIcon type={step.type} status={step.status} />
                  </div>
                  {index < steps.length - 1 ? <div className="mt-1 h-6 w-px bg-border" /> : null}
                </div>
                <div className="min-w-0 flex-1 pb-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-xs font-medium text-muted-foreground">{stepLabel(step)}</p>
                    {badge ? (
                      <span className="rounded-full border border-border/80 bg-canvas/80 px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.08em] text-muted-foreground">
                        {badge}
                      </span>
                    ) : null}
                  </div>
                  {step.type === 'tool_call' || step.type === 'tool_result' ? (
                    <div className="mt-2">
                      <ToolCallCard
                        toolName={step.toolName ?? 'tool'}
                        args={step.toolArgs}
                        result={step.toolResult}
                        metadata={step.toolMeta}
                        callId={step.toolCallId}
                        errorMessage={step.toolError}
                        status={step.status === 'running' ? 'calling' : step.status}
                        type={step.type}
                      />
                    </div>
                  ) : step.content ? (
                    <p className="mt-1 whitespace-pre-wrap break-words text-sm leading-relaxed text-foreground/90">
                      {step.content}
                    </p>
                  ) : null}
                  {step.errorCode ? (
                    <p className="mt-1 break-all text-xs text-error/90">
                      {step.errorCode}
                    </p>
                  ) : null}
                  <StepDiagnostics step={step} suppressDuplicate={suppressDuplicateDiagnostics} />
                  {(() => {
                    const fileChangeGroup = extractFileChangeGroupFromReasoningStep(step)
                    if (!fileChangeGroup) {
                      return null
                    }
                    return (
                      <div className="mt-3">
                        <FileChangeCard group={fileChangeGroup} onUndo={onUndoFileChange} />
                      </div>
                    )
                  })()}
                </div>
              </div>
            )
          })}
        </div>
      ) : null}
    </div>
  )
}
