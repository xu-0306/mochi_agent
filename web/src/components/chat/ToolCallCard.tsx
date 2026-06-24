'use client'

import * as React from 'react'
import {
  AlertTriangle,
  Bot,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  ExternalLink,
  Settings2,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { useI18n } from '@/lib/i18n'
import {
  DELEGATE_SUBAGENT_TOOL_NAME,
  delegatedSubagentTitle,
  resolveDelegatedSubagentView,
} from '@/lib/subagent-tasks'

interface ToolCallCardProps {
  toolName: string
  args?: Record<string, unknown>
  result?: unknown
  metadata?: Record<string, unknown>
  callId?: string
  errorMessage?: string
  status?: 'calling' | 'success' | 'error'
  type: 'tool_call' | 'tool_result' | 'tool_call_request' | 'tool_call_result'
  onOpenTask?: (taskId: string) => void
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function getEvidenceNotice(metadata?: Record<string, unknown>): string | null {
  const evidenceQuality = metadata?.evidence_quality
  if (!isRecord(evidenceQuality)) {
    return null
  }
  if (evidenceQuality.status !== 'insufficient_evidence') {
    return null
  }
  const url = typeof evidenceQuality.url === 'string' && evidenceQuality.url.trim()
    ? evidenceQuality.url.trim()
    : 'the fetched page'
  const chars = typeof evidenceQuality.chars === 'number' ? evidenceQuality.chars : null
  const lines = typeof evidenceQuality.lines === 'number' ? evidenceQuality.lines : null
  const size = chars !== null && lines !== null
    ? ` (${chars} chars, ${lines} non-empty lines)`
    : ''
  return `Insufficient extracted evidence from ${url}${size}. A follow-up retrieval is required before answering.`
}

export function ToolCallCard({
  toolName,
  args,
  result,
  metadata,
  callId,
  errorMessage,
  status = 'success',
  type,
  onOpenTask,
}: ToolCallCardProps) {
  const { t } = useI18n()
  const isResult = type === 'tool_result' || type === 'tool_call_result'
  const isError = status === 'error'
  const evidenceNotice = getEvidenceNotice(metadata)
  const delegatedSubagent =
    toolName === DELEGATE_SUBAGENT_TOOL_NAME && isResult
      ? resolveDelegatedSubagentView({ result, metadata, args })
      : null
  const [open, setOpen] = React.useState(!isResult || isError || Boolean(evidenceNotice))
  const headerLabel = delegatedSubagent ? delegatedSubagentTitle(delegatedSubagent) : toolName

  return (
    <div
      className={cn(
        'max-w-[560px] rounded-lg border text-sm font-mono',
        isResult
          ? isError
            ? 'border-border border-l-[3px] border-l-error bg-error/5'
            : 'border-border border-l-[3px] border-l-success bg-surface-layer'
          : 'border-border bg-secondary/10'
      )}
    >
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left transition-colors duration-150 hover:bg-white/5"
      >
        {isError ? (
          <AlertTriangle className="h-3.5 w-3.5 shrink-0 text-error" />
        ) : isResult ? (
          <CheckCircle2 className="h-3.5 w-3.5 text-success shrink-0" />
        ) : (
          <Settings2 className="h-3.5 w-3.5 text-muted-foreground shrink-0" />
        )}
        <span className="flex-1 truncate text-xs font-medium text-foreground">
          <span className="text-primary-400">{headerLabel}</span>
          {!isResult ? <span className="text-muted-foreground">()</span> : null}
          {status === 'calling' && (
            <span className="ml-2 text-muted-foreground animate-pulse">{t('chat.tool.running')}</span>
          )}
          {isError ? (
            <span className="ml-2 text-error">{t('chat.tool.failed')}</span>
          ) : null}
          {evidenceNotice ? (
            <span className="ml-2 text-amber-300">insufficient evidence</span>
          ) : null}
        </span>
        {open ? (
          <ChevronDown className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        ) : (
          <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
      </button>

      {open && (
        <div className="px-3 pb-3 border-t border-border mt-0 pt-2">
          {delegatedSubagent ? (
            <div className="mb-2 rounded-lg border border-primary-500/20 bg-primary-500/10 p-3 font-sans">
              <div className="flex items-start gap-2">
                <div className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/25 bg-primary-500/15">
                  <Bot className="h-3.5 w-3.5 text-primary-300" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="text-sm font-semibold text-foreground">
                      {delegatedSubagentTitle(delegatedSubagent)}
                    </p>
                    {delegatedSubagent.status ? (
                      <span className="rounded-full border border-white/10 bg-canvas/60 px-2 py-0.5 text-[10px] uppercase tracking-[0.08em] text-muted-foreground">
                        {delegatedSubagent.status}
                      </span>
                    ) : null}
                  </div>
                  {delegatedSubagent.instruction ? (
                    <p className="mt-1 line-clamp-3 text-xs leading-relaxed text-muted-foreground">
                      {delegatedSubagent.instruction}
                    </p>
                  ) : null}
                  {delegatedSubagent.protocol ? (
                    <p className="mt-1 text-[11px] text-muted-foreground/80">
                      Protocol: {delegatedSubagent.protocol}
                    </p>
                  ) : null}
                  {delegatedSubagent.taskId && onOpenTask ? (
                    <button
                      type="button"
                      onClick={() => onOpenTask(delegatedSubagent.taskId as string)}
                      className="mt-2 inline-flex items-center gap-1.5 rounded-full border border-primary-500/30 bg-primary-500/15 px-2.5 py-1 text-xs font-medium text-primary-200 transition-colors hover:bg-primary-500/25"
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                      Open subagent task
                    </button>
                  ) : null}
                </div>
              </div>
            </div>
          ) : null}
          {callId ? (
            <div className="mb-2">
              <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">Call ID</p>
              <p className="break-all rounded bg-canvas p-2 text-xs text-foreground/80">
                {callId}
              </p>
            </div>
          ) : null}
          {!isResult && args && (
            <div>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">{t('chat.tool.args')}</p>
              <pre className="text-xs text-foreground/80 whitespace-pre-wrap break-all overflow-auto max-h-48 bg-canvas rounded p-2">
                {JSON.stringify(args, null, 2)}
              </pre>
            </div>
          )}
          {errorMessage ? (
            <div className={!isResult && args ? 'mt-2' : ''}>
              <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">{t('chat.tool.error')}</p>
              <pre className="max-h-48 overflow-auto whitespace-pre-wrap break-all rounded bg-canvas p-2 text-xs text-error">
                {errorMessage}
              </pre>
            </div>
          ) : null}
          {evidenceNotice ? (
            <div className={(!isResult && args) || errorMessage ? 'mt-2' : ''}>
              <p className="mb-1 text-[10px] uppercase tracking-wide text-muted-foreground">Evidence</p>
              <p className="rounded border border-amber-400/30 bg-amber-400/10 p-2 text-xs text-amber-100">
                {evidenceNotice}
              </p>
            </div>
          ) : null}
          {result !== undefined && (
            <div className={(!isResult && args) || errorMessage || evidenceNotice ? 'mt-2' : ''}>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">{t('chat.tool.result')}</p>
              <pre className="text-xs text-foreground/80 whitespace-pre-wrap break-all overflow-auto max-h-48 bg-canvas rounded p-2">
                {typeof result === 'string' ? result : JSON.stringify(result, null, 2)}
              </pre>
            </div>
          )}
          {metadata && (
            <div className={(result !== undefined) || errorMessage || evidenceNotice ? 'mt-2' : ''}>
              <p className="text-[10px] text-muted-foreground uppercase tracking-wide mb-1">Metadata</p>
              <pre className="text-xs text-foreground/70 whitespace-pre-wrap break-all overflow-auto max-h-48 bg-canvas rounded p-2">
                {JSON.stringify(metadata, null, 2)}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
