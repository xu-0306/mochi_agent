'use client'

import {
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronRight,
  Clock3,
  Loader2,
  Wrench,
} from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { ExecutionTranscriptEvent, SubagentTranscriptSummary } from '@/lib/api'
import { summarizeRuntimeBlocker } from '@/lib/execution-transcript'
import {
  describeSubagentEventMetadata,
  deriveSubagentEventStatus,
  formatSubagentEventKindLabel,
  formatSubagentEventTitle,
  formatSubagentStatusLabel,
  isInformativeSubagentStatus,
  subagentStatusVariant,
} from '@/lib/subagent-protocol-events'
import { cn, formatRelativeTime } from '@/lib/utils'

interface ExecutionTimelineProps {
  events: ExecutionTranscriptEvent[]
  subagents: SubagentTranscriptSummary[]
  activeSubagentId?: string | null
  onOpenSubagent?: (subagentId: string) => void
}

function eventVariant(
  type: string,
  status: string | null | undefined
): 'outline' | 'primary' | 'success' | 'warning' | 'error' {
  const statusVariant = subagentStatusVariant(status)
  if (statusVariant !== 'outline') {
    return statusVariant
  }
  if (type === 'subagent_completed' || type === 'run_completed') {
    return 'success'
  }
  if (type === 'runtime_blocked') {
    return 'warning'
  }
  if (type.includes('cancel') || type.includes('interrupted')) {
    return type.includes('deferred') || type.includes('requested') || type.includes('interrupted')
      ? 'warning'
      : 'error'
  }
  if (type.includes('failed') || type.includes('error')) {
    return 'error'
  }
  if (type.includes('started') || type.includes('tool') || type.includes('thinking') || type.includes('progress')) {
    return 'primary'
  }
  return 'outline'
}

function eventIcon(type: string, status?: string | null) {
  const statusVariant = subagentStatusVariant(status)
  if (statusVariant === 'error') {
    return <AlertCircle className="h-3.5 w-3.5 text-destructive" />
  }
  if (statusVariant === 'warning') {
    return <AlertCircle className="h-3.5 w-3.5 text-warning" />
  }
  if (type === 'runtime_blocked') {
    return <AlertCircle className="h-3.5 w-3.5 text-warning" />
  }
  if (type === 'subagent_completed' || type === 'run_completed') {
    return <CheckCircle2 className="h-3.5 w-3.5 text-success" />
  }
  if (type === 'subagent_tool_call' || type === 'subagent_tool_result') {
    return <Wrench className="h-3.5 w-3.5 text-primary-400" />
  }
  if (type === 'subagent_thinking') {
    return <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-400" />
  }
  if (type === 'subagent_started') {
    return <Bot className="h-3.5 w-3.5 text-primary-400" />
  }
  return <Clock3 className="h-3.5 w-3.5 text-muted-foreground" />
}

function rowSummary(event: ExecutionTranscriptEvent): string {
  if (event.type === 'runtime_blocked') {
    return summarizeRuntimeBlocker(event)
  }
  return event.summary ?? event.content ?? formatSubagentEventTitle(event)
}

export function ExecutionTimeline({
  events,
  subagents,
  activeSubagentId = null,
  onOpenSubagent,
}: ExecutionTimelineProps) {
  if (events.length === 0) {
    return null
  }

  const subagentById = new Map(subagents.map((item) => [item.subagentId, item]))

  return (
    <div className="rounded-lg border border-border bg-surface-layer/50">
      <div className="border-b border-border px-3 py-2">
        <p className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">
          Execution highlights
        </p>
      </div>
      <div className="divide-y divide-border/70">
        {events.map((event, index) => {
          const subagentId = event.subagentId ?? null
          const subagent = subagentId ? subagentById.get(subagentId) ?? null : null
          const interactive = Boolean(subagentId && onOpenSubagent)
          const isActive = activeSubagentId !== null && subagentId === activeSubagentId
          const eventStatus = deriveSubagentEventStatus(event.type, event.status)
          const showEventStatus =
            isInformativeSubagentStatus(eventStatus) && eventStatus !== subagent?.status
          const actorLabel =
            subagent?.title ??
            event.title ??
            event.roleId ??
            (subagentId ? 'Subagent' : 'Runtime')
          const eventTitle = formatSubagentEventTitle(event)
          const eventDetails =
            event.type === 'runtime_blocked' || event.type === 'subagent_tool_result'
              ? describeSubagentEventMetadata(event)
              : []

          return (
            <button
              key={`${event.seq ?? 'seq'}:${event.type}:${subagentId ?? index}`}
              type="button"
              disabled={!interactive}
              onClick={() => {
                if (interactive && subagentId) {
                  onOpenSubagent?.(subagentId)
                }
              }}
              className={cn(
                'flex w-full items-start gap-3 px-3 py-3 text-left',
                interactive ? 'hover:bg-elevated-layer/70' : 'cursor-default',
                isActive ? 'bg-primary-500/5' : ''
              )}
            >
              <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-canvas/70">
                {eventIcon(event.type, eventStatus)}
              </div>
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="truncate text-sm font-medium text-foreground">{eventTitle}</span>
                  <Badge
                    variant={eventVariant(event.type, eventStatus)}
                    className="min-w-[5.5rem] justify-center"
                  >
                    {formatSubagentEventKindLabel(event)}
                  </Badge>
                  <Badge variant="outline" className="max-w-full">
                    <span className="truncate">{actorLabel}</span>
                  </Badge>
                  {showEventStatus ? (
                    <Badge
                      variant={subagentStatusVariant(eventStatus)}
                      className="min-w-[5rem] justify-center"
                    >
                      {formatSubagentStatusLabel(eventStatus as string)}
                    </Badge>
                  ) : null}
                  {subagent?.status ? (
                    <Badge variant="outline" className="min-w-[5rem] justify-center">
                      {formatSubagentStatusLabel(subagent.status)}
                    </Badge>
                  ) : null}
                </div>
                <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
                  {rowSummary(event)}
                </p>
                <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px] text-muted-foreground">
                  {eventDetails.map((item) => (
                    <span key={item} className="break-all">{item}</span>
                  ))}
                  {event.createdAt ? <span>{formatRelativeTime(event.createdAt)}</span> : null}
                </div>
              </div>
              {interactive ? <ChevronRight className="mt-1 h-4 w-4 shrink-0 text-muted-foreground" /> : null}
            </button>
          )
        })}
      </div>
    </div>
  )
}

export type { ExecutionTimelineProps }
