'use client'

import * as React from 'react'
import { Bot, ChevronDown, ChevronRight, ExternalLink, Loader2 } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import type { ExecutionTranscriptEvent, SubagentTranscriptSummary } from '@/lib/api'
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

interface SubagentTimelineCardProps {
  subagent: SubagentTranscriptSummary
  active?: boolean
  expanded?: boolean
  onExpandedChange?: (subagentId: string, expanded: boolean) => void
  onOpen?: (subagentId: string) => void
  onOpenThread?: (subagentId: string) => void
  recentEvents?: ExecutionTranscriptEvent[]
}

function eventSummary(event: ExecutionTranscriptEvent): string {
  return event.summary ?? event.content ?? formatSubagentEventTitle(event)
}

function nonEmpty(value: string | null | undefined): string | null {
  const trimmed = value?.trim()
  return trimmed && trimmed.length > 0 ? trimmed : null
}

function shouldShowApprovalPreview(status: string): boolean {
  const normalized = status.toLowerCase()
  return normalized === 'awaiting_approval' || normalized === 'blocked'
}

export function SubagentTimelineCard({
  subagent,
  active = false,
  expanded,
  onExpandedChange,
  onOpen,
  onOpenThread,
  recentEvents,
}: SubagentTimelineCardProps) {
  const [uncontrolledExpanded, setUncontrolledExpanded] = React.useState(false)
  const contentId = React.useId()
  const isExpanded = expanded ?? uncontrolledExpanded
  const openThread = onOpenThread ?? onOpen
  const updatedLabel = subagent.updatedAt
    ? formatRelativeTime(subagent.updatedAt)
    : subagent.createdAt
      ? formatRelativeTime(subagent.createdAt)
      : null
  const title = subagent.title ?? subagent.roleId ?? subagent.subagentId
  const compactSummary =
    nonEmpty(subagent.summary) ?? nonEmpty(subagent.outputPreview) ?? nonEmpty(subagent.promptPreview)
  const recentTimeline = recentEvents?.slice(-3) ?? []
  const promptPreview = nonEmpty(subagent.promptPreview)
  const outputPreview = nonEmpty(subagent.outputPreview)
  const summaryPreview = nonEmpty(subagent.summary)
  const approvalPreview = shouldShowApprovalPreview(subagent.status)
    ? summaryPreview ?? outputPreview ?? promptPreview
    : null

  const setExpanded = (nextExpanded: boolean) => {
    if (expanded === undefined) {
      setUncontrolledExpanded(nextExpanded)
    }
    onExpandedChange?.(subagent.subagentId, nextExpanded)
  }

  return (
    <article
      className={cn(
        'w-full min-w-0 rounded-lg border transition-colors',
        active
          ? 'border-primary-500/40 bg-primary-500/10'
          : 'border-border bg-surface-layer/60 hover:bg-elevated-layer/80'
      )}
    >
      <div className="flex min-w-0 items-start gap-2 px-3 py-2.5">
        <button
          type="button"
          disabled={!openThread}
          onClick={() => openThread?.(subagent.subagentId)}
          className={cn(
            'flex min-w-0 flex-1 items-start gap-2 rounded-md text-left outline-none focus-visible:ring-2 focus-visible:ring-primary-500/50',
            openThread ? 'cursor-pointer' : 'cursor-default'
          )}
        >
          <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-primary-500/20 bg-primary-500/10">
            {subagent.status.toLowerCase() === 'running' ? (
              <Loader2 className="h-3.5 w-3.5 animate-spin text-primary-400" />
            ) : (
              <Bot className="h-3.5 w-3.5 text-primary-400" />
            )}
          </span>
          <div className="min-w-0 flex-1">
            <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
              <p className="min-w-0 break-words text-sm font-medium leading-5 text-foreground">
                {title}
              </p>
              <Badge variant={subagentStatusVariant(subagent.status)} className="shrink-0">
                {formatSubagentStatusLabel(subagent.status)}
              </Badge>
              <Badge variant="outline" className="shrink-0">
                {subagent.eventCount} events
              </Badge>
            </div>
            <p className="mt-0.5 break-all text-xs leading-5 text-muted-foreground">
              {subagent.modelId ?? 'model pending'}
              {updatedLabel ? ` / updated ${updatedLabel}` : ''}
            </p>
            {compactSummary ? (
              <p className="mt-1.5 line-clamp-2 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
                {compactSummary}
              </p>
            ) : null}
          </div>
        </button>
        <div className="flex shrink-0 items-center gap-1">
          {openThread ? (
            <button
              type="button"
              onClick={() => openThread(subagent.subagentId)}
              className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-elevated-layer focus-visible:ring-2 focus-visible:ring-primary-500/50"
              aria-label={`Open ${title}`}
              title="Open thread"
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </button>
          ) : null}
          <button
            type="button"
            onClick={() => setExpanded(!isExpanded)}
            className="inline-flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground outline-none transition-colors hover:bg-elevated-layer focus-visible:ring-2 focus-visible:ring-primary-500/50"
            aria-expanded={isExpanded}
            aria-controls={contentId}
            aria-label={`${isExpanded ? 'Collapse' : 'Expand'} ${title}`}
            title={isExpanded ? 'Collapse' : 'Expand'}
          >
            {isExpanded ? (
              <ChevronDown className="h-4 w-4" />
            ) : (
              <ChevronRight className="h-4 w-4" />
            )}
          </button>
        </div>
      </div>
      {isExpanded ? (
        <div id={contentId} className="space-y-3 border-t border-border/70 px-3 py-3">
          {recentTimeline.length > 0 ? (
            <div>
              <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
                Highlights
              </p>
              <div className="mt-2 space-y-2">
                {recentTimeline.map((event, index) => (
                  <RecentTimelineEvent
                    key={`${event.seq ?? index}:${event.type}:${event.createdAt ?? 'event'}`}
                    event={event}
                  />
                ))}
              </div>
            </div>
          ) : null}
          {approvalPreview ? (
            <PreviewSection label="Approval" value={approvalPreview} />
          ) : null}
          {summaryPreview && summaryPreview !== approvalPreview ? (
            <PreviewSection label="Summary" value={summaryPreview} />
          ) : null}
          {promptPreview ? <PreviewSection label="Prompt" value={promptPreview} /> : null}
          {outputPreview ? <PreviewSection label="Output" value={outputPreview} /> : null}
        </div>
      ) : null}
    </article>
  )
}

function RecentTimelineEvent({ event }: { event: ExecutionTranscriptEvent }) {
  const status = deriveSubagentEventStatus(event.type, event.status)
  const details =
    event.type === 'runtime_blocked' || event.type === 'subagent_tool_result'
      ? describeSubagentEventMetadata(event)
      : []

  return (
    <div className="flex min-w-0 items-start gap-2 text-xs leading-5">
      <span className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-primary-400/80" />
      <div className="min-w-0 flex-1">
        <div className="flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1">
          <span className="font-medium text-foreground">{formatSubagentEventTitle(event)}</span>
          <Badge variant="outline" className="h-5 px-1.5 text-[10px]">
            {formatSubagentEventKindLabel(event)}
          </Badge>
          {isInformativeSubagentStatus(status) ? (
            <Badge variant={subagentStatusVariant(status)} className="h-5 px-1.5 text-[10px]">
              {formatSubagentStatusLabel(status as string)}
            </Badge>
          ) : null}
          {event.createdAt ? (
            <span className="text-[11px] text-muted-foreground">
              {formatRelativeTime(event.createdAt)}
            </span>
          ) : null}
        </div>
        <p className="line-clamp-2 whitespace-pre-wrap break-words text-muted-foreground">
          {eventSummary(event)}
        </p>
        {details.length > 0 ? (
          <div className="mt-1 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            {details.map((item) => (
              <span key={item} className="break-all">{item}</span>
            ))}
          </div>
        ) : null}
      </div>
    </div>
  )
}

function PreviewSection({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0">
      <p className="text-[11px] font-semibold uppercase tracking-[0.08em] text-muted-foreground">
        {label}
      </p>
      <p className="mt-1 line-clamp-3 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
        {value}
      </p>
    </div>
  )
}

export type { SubagentTimelineCardProps }
