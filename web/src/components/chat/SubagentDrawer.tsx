'use client'

import * as React from 'react'
import {
  AlertCircle,
  Bot,
  CheckCircle2,
  ChevronDown,
  Clock3,
  MessageSquareText,
  RotateCcw,
  Send,
  XCircle,
  Wrench,
} from 'lucide-react'
import { CopyButton } from '@/components/chat/CopyButton'
import { Button } from '@/components/ui/button'
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from '@/components/ui/sheet'
import { Textarea } from '@/components/ui/textarea'
import { Badge } from '@/components/ui/badge'
import type { ExecutionTranscriptEvent, SubagentTranscriptDetail } from '@/lib/api'
import { summarizeRuntimeBlocker } from '@/lib/execution-transcript'
import {
  describeSubagentEventMetadata,
  deriveSubagentEventStatus,
  deriveSubagentMessageDeliveryStatus,
  formatSubagentEventTitle,
  formatSubagentStatusLabel,
  isInformativeSubagentStatus,
  subagentStatusVariant,
} from '@/lib/subagent-protocol-events'
import { cn, formatRelativeTime } from '@/lib/utils'

interface SubagentDrawerProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  subagent: SubagentTranscriptDetail | null
  loading?: boolean
  error?: string | null
  guidanceValue: string
  onGuidanceChange: (value: string) => void
  onSendGuidance: () => void | Promise<void>
  onCancel?: () => void | Promise<void>
  onResume?: () => void | Promise<void>
  onOpenApprovals?: (approvalIds: string[]) => void
  sendingGuidance?: boolean
  canceling?: boolean
  resuming?: boolean
}

interface ApprovalDetails {
  event: ExecutionTranscriptEvent
  approvalIds: string[]
  approvalToolName: string | null
  approvalReason: string | null
  approvalScope: string | null
  approvalKind: string | null
  securityDecision: string | null
  policySource: string | null
  replaySafe: boolean | null
  allowedDecisions: string[]
  approvalWaitStartedAt: string | null
}

function normalizeStatus(status: string | null | undefined): string {
  return (status ?? '').trim().toLowerCase()
}

function isSubagentCancelableStatus(status: string | null | undefined): boolean {
  return [
    'active',
    'created',
    'in_progress',
    'pending',
    'processing',
    'queued',
    'resumed',
    'running',
    'started',
    'working',
  ].includes(normalizeStatus(status))
}

function isSubagentResumableStatus(status: string | null | undefined): boolean {
  return [
    'approval_required',
    'awaiting_approval',
    'blocked',
    'cancelled',
    'canceled',
    'error',
    'failed',
    'interrupted',
    'paused',
    'stalled',
    'waiting',
    'waiting_approval',
  ].includes(normalizeStatus(status))
}

function summarizeEvent(event: ExecutionTranscriptEvent): string {
  if (event.type === 'runtime_blocked') {
    return summarizeRuntimeBlocker(event)
  }
  return event.summary ?? event.content ?? formatSubagentEventTitle(event)
}

function getString(value: unknown): string | null {
  return typeof value === 'string' && value.trim().length > 0 ? value.trim() : null
}

function getStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) {
    return []
  }
  return value
    .map((item) => (typeof item === 'string' ? item.trim() : ''))
    .filter((item) => item.length > 0)
}

function uniqueStrings(values: Array<string | null | undefined>): string[] {
  const seen = new Set<string>()
  const result: string[] = []
  for (const value of values) {
    const text = getString(value)
    if (!text || seen.has(text)) {
      continue
    }
    seen.add(text)
    result.push(text)
  }
  return result
}

function getPendingApprovals(event: ExecutionTranscriptEvent): Array<Record<string, unknown>> {
  const pending = event.metadata.pending_approvals
  if (!Array.isArray(pending)) {
    return []
  }
  return pending.filter((item): item is Record<string, unknown> => typeof item === 'object' && item !== null)
}

function findLatestApprovalEvent(events: ExecutionTranscriptEvent[]): ExecutionTranscriptEvent | null {
  for (let index = events.length - 1; index >= 0; index -= 1) {
    const event = events[index]
    if (
      event.type === 'runtime_blocked' ||
      (event.type === 'subagent_tool_result' && event.status === 'approval_required')
    ) {
      return event
    }
  }
  return null
}

function getApprovalDetails(event: ExecutionTranscriptEvent | null): ApprovalDetails | null {
  if (!event || typeof event.metadata !== 'object' || event.metadata === null) {
    return null
  }

  const approvalMetadata = event.metadata
  const pendingApprovals = getPendingApprovals(event)
  const primaryPendingApproval = pendingApprovals[0] ?? null
  const replaySafeCandidate = primaryPendingApproval?.replay_safe ?? approvalMetadata.replay_safe

  return {
    event,
    approvalIds: uniqueStrings([
      ...getStringArray(approvalMetadata.approval_ids),
      getString(approvalMetadata.approval_id),
      ...pendingApprovals.map((item) => getString(item.approval_id)),
    ]),
    approvalToolName:
      getString(primaryPendingApproval?.tool_name) ??
      getStringArray(approvalMetadata.tool_names)[0] ??
      getString(approvalMetadata.tool_name),
    approvalReason: getString(primaryPendingApproval?.reason) ?? getString(approvalMetadata.reason),
    approvalScope: getString(primaryPendingApproval?.approval_scope) ?? getString(approvalMetadata.approval_scope),
    approvalKind: getString(primaryPendingApproval?.approval_kind) ?? getString(approvalMetadata.approval_kind),
    securityDecision:
      getString(primaryPendingApproval?.security_decision) ?? getString(approvalMetadata.security_decision),
    policySource: getString(primaryPendingApproval?.policy_source) ?? getString(approvalMetadata.policy_source),
    replaySafe: typeof replaySafeCandidate === 'boolean' ? replaySafeCandidate : null,
    allowedDecisions: getStringArray(primaryPendingApproval?.allowed_decisions ?? approvalMetadata.allowed_decisions),
    approvalWaitStartedAt:
      getString(primaryPendingApproval?.approval_wait_started_at) ??
      getString(approvalMetadata.approval_wait_started_at),
  }
}

function deliveryStatusFromEventType(type: string): string | null {
  return deriveSubagentMessageDeliveryStatus(type)
}

function getDeliveryStatus(event: ExecutionTranscriptEvent): string | null {
  return (
    getString(event.deliveryStatus) ??
    getString(event.metadata.delivery_status) ??
    getString(event.metadata.deliveryStatus) ??
    deliveryStatusFromEventType(event.type) ??
    (event.type.startsWith('subagent_message_') ? getString(event.status) : null)
  )
}

function getDeliveryMessageId(event: ExecutionTranscriptEvent): string | null {
  return getString(event.messageId) ?? getString(event.metadata.message_id) ?? getString(event.metadata.messageId)
}

function isSubagentMessageEvent(event: ExecutionTranscriptEvent): boolean {
  return event.type === 'subagent_message' || event.type.startsWith('subagent_message_')
}

function getSubagentMessageContent(event: ExecutionTranscriptEvent): string {
  return (
    event.content ??
    event.summary ??
    getString(event.metadata.content) ??
    getString(event.metadata.message) ??
    ''
  )
}

function collapseSubagentMessageEvents(events: ExecutionTranscriptEvent[]): ExecutionTranscriptEvent[] {
  const latestIndexByMessageId = new Map<string, number>()
  const contentByMessageId = new Map<string, string>()

  events.forEach((event, index) => {
    if (!isSubagentMessageEvent(event)) {
      return
    }
    const messageId = getDeliveryMessageId(event)
    if (!messageId) {
      return
    }
    latestIndexByMessageId.set(messageId, index)
    const content = getSubagentMessageContent(event)
    if (content && !contentByMessageId.has(messageId)) {
      contentByMessageId.set(messageId, content)
    }
  })

  return events
    .map((event, index) => {
      if (!isSubagentMessageEvent(event)) {
        return event
      }
      const messageId = getDeliveryMessageId(event)
      if (!messageId || latestIndexByMessageId.get(messageId) === index) {
        const content = messageId ? contentByMessageId.get(messageId) : null
        return content && !event.content ? { ...event, content } : event
      }
      return null
    })
    .filter((event): event is ExecutionTranscriptEvent => event !== null)
}

function deliveryStatusVariant(status: string): 'primary' | 'success' | 'warning' | 'error' | 'outline' {
  return subagentStatusVariant(status)
}

function ThreadIcon({ event }: { event: ExecutionTranscriptEvent }) {
  const className = 'h-4 w-4'
  const status = deriveSubagentEventStatus(event.type, event.status)
  if (
    event.type === 'runtime_blocked' ||
    status === 'approval_required' ||
    status === 'cancel_deferred' ||
    status === 'cancel_requested' ||
    status === 'interrupted'
  ) {
    return <AlertCircle className={cn(className, 'text-warning')} />
  }
  if (status === 'cancelled' || status === 'canceled') {
    return <XCircle className={cn(className, 'text-destructive')} />
  }
  if (event.type === 'subagent_tool_call' || event.type === 'subagent_tool_result') {
    return <Wrench className={cn(className, 'text-primary-400')} />
  }
  if (event.type === 'subagent_prompt') {
    return <MessageSquareText className={cn(className, 'text-primary-400')} />
  }
  if (event.type === 'subagent_completed') {
    return <CheckCircle2 className={cn(className, event.status === 'failed' ? 'text-destructive' : 'text-success')} />
  }
  if (event.type === 'subagent_progress' || event.type === 'subagent_thinking') {
    return <Clock3 className={cn(className, 'text-muted-foreground')} />
  }
  return <Bot className={cn(className, 'text-primary-400')} />
}

function CollapsibleThreadCard({
  title,
  description,
  copyValue,
  copyLabel,
  defaultOpen = false,
  children,
}: {
  title: string
  description?: string
  copyValue?: string
  copyLabel?: string
  defaultOpen?: boolean
  children: React.ReactNode
}) {
  const [expanded, setExpanded] = React.useState(defaultOpen)
  const contentId = React.useId()

  return (
    <section className="rounded-lg border border-border bg-canvas/60">
      <div className="flex items-center gap-1 px-3 py-2">
        <button
          type="button"
          className="flex min-w-0 flex-1 items-center gap-2 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
          aria-expanded={expanded}
          aria-controls={contentId}
          onClick={() => setExpanded((value) => !value)}
        >
          <ChevronDown
            className={cn('h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform', !expanded && '-rotate-90')}
          />
          <div className="min-w-0 flex-1">
            <p className="truncate text-xs font-semibold text-foreground">{title}</p>
            {description ? <p className="truncate text-[11px] text-muted-foreground">{description}</p> : null}
          </div>
        </button>
        {copyValue ? <CopyButton text={copyValue} label={copyLabel ?? `Copy ${title}`} /> : null}
      </div>
      {expanded ? (
        <div id={contentId} className="border-t border-border px-3 py-3">
          {children}
        </div>
      ) : null}
    </section>
  )
}

function PromptCards({ systemPrompt, userPrompt }: { systemPrompt: string; userPrompt: string }) {
  const combinedPrompt = [systemPrompt ? `System:\n${systemPrompt}` : null, userPrompt ? `User:\n${userPrompt}` : null]
    .filter(Boolean)
    .join('\n\n')

  return (
    <CollapsibleThreadCard
      title="Prompt"
      description={userPrompt ? 'System and user instructions' : 'Prompt details'}
      copyValue={combinedPrompt || undefined}
      copyLabel="Copy prompt"
      defaultOpen={false}
    >
      <div className="space-y-3">
        <div>
          <div className="mb-1 flex items-center justify-between gap-2">
            <p className="text-[11px] font-semibold uppercase text-muted-foreground">System</p>
            {systemPrompt ? <CopyButton text={systemPrompt} label="Copy system prompt" /> : null}
          </div>
          <p className="whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
            {systemPrompt || 'Not available.'}
          </p>
        </div>
        <div>
          <div className="mb-1 flex items-center justify-between gap-2">
            <p className="text-[11px] font-semibold uppercase text-muted-foreground">User</p>
            {userPrompt ? <CopyButton text={userPrompt} label="Copy user prompt" /> : null}
          </div>
          <p className="whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
            {userPrompt || 'Not available.'}
          </p>
        </div>
      </div>
    </CollapsibleThreadCard>
  )
}

function OutputCard({ outputText, completed }: { outputText: string; completed: boolean }) {
  return (
    <CollapsibleThreadCard
      title="Output"
      description={outputText ? (completed ? 'Final result' : 'Latest preview') : 'No output yet'}
      copyValue={outputText || undefined}
      copyLabel="Copy output"
      defaultOpen={Boolean(outputText)}
    >
      <p className="whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
        {outputText || 'No output yet.'}
      </p>
    </CollapsibleThreadCard>
  )
}

function ApprovalBlockerCard({
  details,
  onOpenApprovals,
}: {
  details: ApprovalDetails
  onOpenApprovals?: (approvalIds: string[]) => void
}) {
  const {
    event,
    approvalIds,
    approvalToolName,
    approvalReason,
    approvalScope,
    approvalKind,
    securityDecision,
    policySource,
    replaySafe,
    allowedDecisions,
    approvalWaitStartedAt,
  } = details

  return (
    <section className="rounded-lg border border-warning/40 bg-warning/10 px-3 py-3">
      <div className="flex items-start gap-3">
        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-warning/30 bg-warning/15">
          <AlertCircle className="h-4 w-4 text-warning" />
        </span>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="text-sm font-semibold text-foreground">Waiting for approval</p>
            <Badge variant="warning">blocked</Badge>
          </div>
          <p className="mt-1 whitespace-pre-wrap break-words text-xs leading-5 text-muted-foreground">
            {summarizeEvent(event)}
          </p>
          <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            {approvalIds.length > 0 ? <span>ID {approvalIds.join(', ')}</span> : null}
            {approvalToolName ? <span>Tool {approvalToolName}</span> : null}
            {approvalScope ? <span>Scope {approvalScope}</span> : null}
            {approvalKind ? <span>Kind {approvalKind}</span> : null}
            {approvalWaitStartedAt ? <span>Since {formatRelativeTime(approvalWaitStartedAt)}</span> : null}
            {replaySafe !== null ? <span>Replay {replaySafe ? 'safe' : 'unsafe'}</span> : null}
          </div>
          {(approvalReason || securityDecision || policySource || allowedDecisions.length > 0) ? (
            <div className="mt-3 space-y-1 text-[11px] leading-5 text-muted-foreground">
              {approvalReason ? <p>Reason: {approvalReason}</p> : null}
              {securityDecision ? <p>Decision: {securityDecision}</p> : null}
              {policySource ? <p>Policy: {policySource}</p> : null}
              {allowedDecisions.length > 0 ? <p>Available actions: {allowedDecisions.join(', ')}</p> : null}
            </div>
          ) : null}
          {approvalIds.length > 0 && onOpenApprovals ? (
            <Button
              type="button"
              size="sm"
              variant="outline"
              className="mt-3 h-8 rounded-full px-3 text-xs"
              onClick={() => onOpenApprovals(approvalIds)}
            >
              Review approval{approvalIds.length > 1 ? 's' : ''}
            </Button>
          ) : null}
        </div>
      </div>
    </section>
  )
}

function ThreadEventItem({
  event,
  isLatestApprovalEvent,
  approvalDetails,
  onOpenApprovals,
}: {
  event: ExecutionTranscriptEvent
  isLatestApprovalEvent: boolean
  approvalDetails: ApprovalDetails | null
  onOpenApprovals?: (approvalIds: string[]) => void
}) {
  if (isSubagentMessageEvent(event)) {
    return <SubagentMessageEventItem event={event} />
  }

  if (isLatestApprovalEvent && approvalDetails) {
    return <ApprovalBlockerCard details={approvalDetails} onOpenApprovals={onOpenApprovals} />
  }

  const details = describeSubagentEventMetadata(event)
  const summary = summarizeEvent(event)
  const status = deriveSubagentEventStatus(event.type, event.status)

  return (
    <article className="flex gap-3">
      <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-full border border-border bg-surface-layer">
        <ThreadIcon event={event} />
      </span>
      <div className="min-w-0 flex-1 rounded-lg border border-border bg-canvas/60 px-3 py-3">
        <div className="flex min-w-0 flex-wrap items-center gap-2">
          <p className="min-w-0 break-words text-sm font-medium text-foreground">{formatSubagentEventTitle(event)}</p>
          {isInformativeSubagentStatus(status) ? (
            <Badge variant={subagentStatusVariant(status)}>{formatSubagentStatusLabel(status as string)}</Badge>
          ) : null}
          {event.createdAt ? (
            <span className="text-[11px] text-muted-foreground">{formatRelativeTime(event.createdAt)}</span>
          ) : null}
        </div>
        <p className="mt-2 whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
          {summary}
        </p>
        {details.length > 0 ? (
          <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-muted-foreground">
            {details.map((item) => (
              <span key={item} className="break-all">{item}</span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  )
}

function SubagentMessageEventItem({ event }: { event: ExecutionTranscriptEvent }) {
  const status = getDeliveryStatus(event)
  const content = getSubagentMessageContent(event)
  const details = describeSubagentEventMetadata(event)

  return (
    <article className="flex justify-end">
      <div className="max-w-[88%] min-w-0 rounded-lg border border-primary-500/20 bg-primary-500/10 px-3 py-2.5">
        <div className="mb-1 flex min-w-0 items-center justify-end gap-2">
          {event.createdAt ? (
            <span className="text-[11px] text-muted-foreground">{formatRelativeTime(event.createdAt)}</span>
          ) : null}
          {status ? (
            <Badge variant={deliveryStatusVariant(status)} className="h-5 px-1.5 text-[10px]">
              {formatSubagentStatusLabel(status)}
            </Badge>
          ) : null}
        </div>
        <p className="whitespace-pre-wrap break-words text-xs leading-5 text-foreground">
          {content || 'Follow-up guidance'}
        </p>
        {details.length > 0 ? (
          <div className="mt-2 flex flex-wrap justify-end gap-2 text-[11px] text-muted-foreground">
            {details.map((item) => (
              <span key={item} className="break-all">{item}</span>
            ))}
          </div>
        ) : null}
      </div>
    </article>
  )
}

function EmptyState({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-lg border border-border bg-canvas/60 px-3 py-4 text-sm text-muted-foreground">
      {children}
    </div>
  )
}

export function SubagentDrawer({
  open,
  onOpenChange,
  subagent,
  loading = false,
  error = null,
  guidanceValue,
  onGuidanceChange,
  onSendGuidance,
  onCancel,
  onResume,
  onOpenApprovals,
  sendingGuidance = false,
  canceling = false,
  resuming = false,
}: SubagentDrawerProps) {
  const promptSystem = subagent?.systemPrompt ?? ''
  const promptUser = subagent?.userPrompt ?? subagent?.promptPreview ?? ''
  const outputText = subagent?.outputPreview ?? subagent?.summary ?? ''
  const latestApprovalEvent = subagent ? findLatestApprovalEvent(subagent.events) : null
  const approvalDetails = getApprovalDetails(latestApprovalEvent)
  const threadEvents = React.useMemo(
    () => (subagent ? collapseSubagentMessageEvents(subagent.events) : []),
    [subagent]
  )
  const completed = ['completed', 'succeeded', 'done'].includes(normalizeStatus(subagent?.status))
  const showCancel = Boolean(subagent && onCancel && isSubagentCancelableStatus(subagent.status))
  const showResume = Boolean(subagent && onResume && isSubagentResumableStatus(subagent.status))
  const actionBusy = canceling || resuming

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent
        side="right"
        className="h-full w-screen max-w-[100vw] overflow-hidden p-0 sm:w-[34rem] sm:max-w-[34rem]"
      >
        <div className="flex h-full min-w-0 flex-col bg-surface-layer/95">
          <SheetHeader className="mb-0 border-b border-border px-4 py-4 pr-12">
            <div className="flex min-w-0 items-start gap-3">
              <span className="mt-0.5 flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary-500/20 bg-primary-500/10">
                <Bot className="h-4 w-4 text-primary-400" />
              </span>
              <div className="min-w-0 flex-1">
                <SheetTitle className="truncate text-sm">
                  {subagent?.title ?? subagent?.roleId ?? 'Subagent'}
                </SheetTitle>
                <SheetDescription className="mt-1 flex min-w-0 flex-wrap items-center gap-x-2 gap-y-1 text-xs">
                  <span className="min-w-0 max-w-full truncate">{subagent?.modelId ?? 'model pending'}</span>
                  {subagent?.updatedAt ? <span>updated {formatRelativeTime(subagent.updatedAt)}</span> : null}
                </SheetDescription>
              </div>
            </div>
            {subagent ? (
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex min-w-0 flex-wrap items-center gap-2">
                  <Badge variant={subagentStatusVariant(subagent.status)} className="min-w-[6.5rem] justify-center">
                    {formatSubagentStatusLabel(subagent.status)}
                  </Badge>
                  <Badge variant="outline">{subagent.eventCount} events</Badge>
                </div>
                {(showCancel || showResume) ? (
                  <div className="flex shrink-0 items-center gap-1">
                    {showResume ? (
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        className="h-7 px-2"
                        title="Resume subagent"
                        aria-label="Resume subagent"
                        onClick={() => {
                          void onResume?.()
                        }}
                        loading={resuming}
                        disabled={actionBusy}
                      >
                        {!resuming ? <RotateCcw className="h-3.5 w-3.5" /> : null}
                        <span>Resume</span>
                      </Button>
                    ) : null}
                    {showCancel ? (
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        className="h-7 px-2 text-muted-foreground hover:text-destructive"
                        title="Cancel subagent"
                        aria-label="Cancel subagent"
                        onClick={() => {
                          void onCancel?.()
                        }}
                        loading={canceling}
                        disabled={actionBusy}
                      >
                        {!canceling ? <XCircle className="h-3.5 w-3.5" /> : null}
                        <span>Cancel</span>
                      </Button>
                    ) : null}
                  </div>
                ) : null}
              </div>
            ) : null}
          </SheetHeader>

          <main className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            <div className="space-y-3">
              {error ? (
                <div className="rounded-lg border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {error}
                </div>
              ) : null}
              {loading ? <EmptyState>Loading subagent transcript...</EmptyState> : null}
              {!loading && !subagent && !error ? <EmptyState>No subagent selected.</EmptyState> : null}
              {subagent ? (
                <>
                  <PromptCards systemPrompt={promptSystem} userPrompt={promptUser} />
                  {threadEvents.length > 0 ? (
                    <div className="space-y-3" aria-label="Subagent conversation">
                      {threadEvents.map((event, index) => (
                        <ThreadEventItem
                          key={`${event.seq ?? index}:${event.type}`}
                          event={event}
                          isLatestApprovalEvent={event === latestApprovalEvent}
                          approvalDetails={approvalDetails}
                          onOpenApprovals={onOpenApprovals}
                        />
                      ))}
                    </div>
                  ) : (
                    <EmptyState>No transcript events yet.</EmptyState>
                  )}
                  <OutputCard outputText={outputText} completed={completed} />
                </>
              ) : null}
            </div>
          </main>

          {subagent ? (
            <footer className="shrink-0 border-t border-border bg-elevated-layer/95 px-3 py-3">
              <div className="flex items-end gap-2">
                <Textarea
                  value={guidanceValue}
                  onChange={(event) => onGuidanceChange(event.target.value)}
                  autoResize
                  minRows={1}
                  maxRows={6}
                  placeholder="Ask for follow-up changes"
                  className="max-h-40 min-h-10 resize-none"
                  aria-label="Ask this subagent for follow-up changes"
                />
                <Button
                  type="button"
                  size="icon"
                  title="Send follow-up guidance"
                  aria-label="Send follow-up guidance"
                  onClick={() => {
                    void onSendGuidance()
                  }}
                  loading={sendingGuidance}
                  disabled={guidanceValue.trim().length === 0 || actionBusy}
                >
                  {!sendingGuidance ? <Send className="h-4 w-4" /> : null}
                </Button>
              </div>
            </footer>
          ) : null}
        </div>
      </SheetContent>
    </Sheet>
  )
}

export type { SubagentDrawerProps }
