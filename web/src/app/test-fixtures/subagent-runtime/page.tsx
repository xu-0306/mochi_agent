'use client'

import * as React from 'react'
import { ExecutionTimeline } from '@/components/chat/ExecutionTimeline'
import { SubagentDrawer } from '@/components/chat/SubagentDrawer'
import { SubagentTimelineCard } from '@/components/chat/SubagentTimelineCard'
import { TaskPanel } from '@/components/chat/TaskPanel'
import { Badge } from '@/components/ui/badge'
import type {
  ExecutionTranscriptEvent,
  SubagentTranscriptDetail,
  SubagentTranscriptSummary,
} from '@/lib/api'

const now = Date.parse('2026-06-30T04:00:00.000Z')

function iso(minutesAgo: number): string {
  return new Date(now - minutesAgo * 60_000).toISOString()
}

const longSystemPrompt = [
  'You are a delegated research subagent operating inside Mochi.',
  'Preserve approval metadata exactly, never execute unsafe commands without the existing approval surface, and keep the user-facing transcript readable across narrow mobile layouts.',
  'When prompts are long, the prompt tab must scroll instead of pushing the drawer controls outside the viewport.',
  'Repeat critical constraints: approval_id, approval_ids, pending_approvals, replay safety, policy source, and tool names must remain inspectable without horizontal overflow.',
].join('\n\n')

const longUserPrompt = [
  'Investigate the chat and goal runtime parity work. Produce a detailed status update with evidence from transcript events, then pause for approval before running any shell command that could mutate the workspace.',
  'This paragraph intentionally contains a very long token-like value to prove wrapping: approval-fixture-super-long-identifier-that-should-wrap-without-breaking-the-drawer-layout-20260630-abcdefghijklmnopqrstuvwxyz.',
  'After approval is resolved, continue with focused guidance only and do not create synthetic prompt-visible assistant messages for display-only transcript cards.',
].join('\n\n')

const outputPreview = [
  'Collected runtime evidence from transcript APIs, normalized session subagent events, and the drawer approval banner.',
  'The subagent is blocked on an exec approval and has not run the unsafe action. Output remains available as a preview while waiting.',
  'Long output sentinel: output-preview-super-long-token-that-must-wrap-inside-the-output-tab-without-horizontal-overflow-20260630-abcdefghijklmnopqrstuvwxyz.',
].join('\n\n')

const approvalMetadata = {
  blocker_type: 'approval',
  approval_id: 'exec-approval-fixture-primary',
  approval_ids: ['exec-approval-fixture-primary', 'exec-approval-fixture-secondary'],
  tool_names: ['exec_command'],
  call_id: 'fixture-approval-tool-call',
  recommended_action: 'resolve_approval',
  approval_kind: 'exec',
  approval_scope: 'workspace',
  approval_status: 'pending',
  approval_wait_started_at: iso(18),
  replay_safe: false,
  reason: 'Exec command requires approval before replaying the delegated subagent action.',
  security_decision: 'require_approval',
  policy_source: 'workspace_policy',
  allowed_decisions: ['approve_once', 'reject'],
  pending_approvals: [
    {
      approval_id: 'exec-approval-fixture-primary',
      tool_name: 'exec_command',
      reason: 'Run a bounded shell smoke command after explicit approval.',
      approval_kind: 'exec',
      approval_scope: 'workspace',
      security_decision: 'require_approval',
      policy_source: 'workspace_policy',
      replay_safe: false,
      allowed_decisions: ['approve_once', 'reject'],
      approval_wait_started_at: iso(18),
    },
  ],
}

function event(
  seq: number,
  type: string,
  subagentId: string | null,
  summary: string,
  status = 'running',
  metadata: Record<string, unknown> = {}
): ExecutionTranscriptEvent {
  return {
    type,
    seq,
    parentType: 'chat_turn',
    parentId: 'fixture-turn-1',
    subagentId,
    roleId: subagentId ? 'researcher' : null,
    title: subagentId === 'fixture-subagent-approval' ? 'Approval parity researcher' : 'Runtime coordinator',
    modelId: 'qwen3-coder-fixture',
    status,
    summary,
    content: summary,
    metadata,
    createdAt: iso(Math.max(1, 31 - seq)),
  }
}

const approvalEvents: ExecutionTranscriptEvent[] = [
  event(1, 'subagent_started', 'fixture-subagent-approval', 'Subagent started with a long prompt and approval-aware runtime metadata.'),
  event(2, 'subagent_prompt', 'fixture-subagent-approval', longUserPrompt, 'running', {
    system_prompt: longSystemPrompt,
    user_prompt: longUserPrompt,
  }),
  ...Array.from({ length: 18 }, (_, index) =>
    event(
      index + 3,
      index % 3 === 0 ? 'subagent_thinking' : 'subagent_progress',
      'fixture-subagent-approval',
      `Collected evidence chunk ${index + 1}. This event is intentionally verbose so the timeline row wraps cleanly on mobile without clipping or shifting neighboring controls.`
    )
  ),
  event(21, 'subagent_tool_call', 'fixture-subagent-approval', 'Requested exec_command for a bounded runtime parity smoke check.', 'running', {
    tool_name: 'exec_command',
    call_id: 'fixture-approval-tool-call',
    command: 'rg -n "subagent" web/src/components/chat',
  }),
  event(22, 'subagent_tool_result', 'fixture-subagent-approval', 'Approval required before exec_command can run.', 'approval_required', approvalMetadata),
  event(23, 'runtime_blocked', 'fixture-subagent-approval', 'Runtime blocked on operator approval for exec_command.', 'blocked', approvalMetadata),
  event(24, 'subagent_progress', 'fixture-subagent-approval', 'Waiting for user guidance while preserving approval metadata in the drawer.'),
  event(25, 'subagent_message_queued', 'fixture-subagent-approval', 'Prioritize the transcript delivery state before output polish.', 'queued', {
    message_id: 'fixture-message-applied',
    delivery_mode: 'inject_now',
    delivery_status: 'queued',
  }),
  event(26, 'subagent_message_applied', 'fixture-subagent-approval', 'Prioritize the transcript delivery state before output polish.', 'applied', {
    message_id: 'fixture-message-applied',
    delivery_mode: 'inject_now',
    delivery_status: 'applied',
  }),
  event(27, 'subagent_message_queued', 'fixture-subagent-approval', 'Keep this note pending for the next safe checkpoint.', 'queued', {
    message_id: 'fixture-message-queued',
    delivery_mode: 'next_checkpoint',
    delivery_status: 'queued',
  }),
  event(28, 'subagent_message_deferred', 'fixture-subagent-approval', 'Do not cancel the current approval-blocked tool call.', 'deferred', {
    message_id: 'fixture-message-deferred',
    delivery_mode: 'after_current_tool',
    delivery_status: 'deferred',
    delivery_reason: 'tool_in_progress',
  }),
]

const secondSubagentEvents: ExecutionTranscriptEvent[] = [
  event(29, 'subagent_started', 'fixture-subagent-observer', 'Observer subagent started to compare reload behavior.'),
  event(30, 'subagent_progress', 'fixture-subagent-observer', 'Observer verified session transcript summaries stay display-only.'),
  event(31, 'subagent_completed', 'fixture-subagent-observer', 'Observer completed with no approval required.', 'completed'),
]

const cancellationEvents: ExecutionTranscriptEvent[] = [
  event(32, 'subagent_started', 'fixture-subagent-cancel', 'Cancellation fixture subagent started a long-running tool check.'),
  event(33, 'subagent_tool_call', 'fixture-subagent-cancel', 'Started a long-running tool call that can be cancelled.', 'running', {
    tool_name: 'exec_command',
    call_id: 'fixture-cancel-tool-call',
    command: 'git status --short web/src/components/chat',
  }),
  event(
    34,
    'subagent_tool_cancel_requested',
    'fixture-subagent-cancel',
    'Operator requested cancellation for the active tool call.',
    'cancel_requested',
    {
      tool_name: 'exec_command',
      call_id: 'fixture-cancel-tool-call',
    }
  ),
  event(
    35,
    'subagent_tool_cancel_deferred',
    'fixture-subagent-cancel',
    'Tool cancellation was deferred until the current safe checkpoint.',
    'cancel_deferred',
    {
      tool_name: 'exec_command',
      call_id: 'fixture-cancel-tool-call',
      delivery_reason: 'tool_checkpoint_pending',
    }
  ),
  event(
    36,
    'subagent_tool_cancelled',
    'fixture-subagent-cancel',
    'The active tool call was cancelled before producing a result.',
    'cancelled',
    {
      tool_name: 'exec_command',
      call_id: 'fixture-cancel-tool-call',
    }
  ),
  event(
    37,
    'subagent_interrupted',
    'fixture-subagent-cancel',
    'Subagent was interrupted after the tool cancellation completed.',
    'interrupted'
  ),
  event(38, 'subagent_message_queued', 'fixture-subagent-cancel', 'Cancel the current tool and pause for operator review.', 'queued', {
    message_id: 'fixture-message-cancelled',
    delivery_mode: 'inject_now',
    delivery_status: 'queued',
  }),
  event(39, 'subagent_message_cancelled', 'fixture-subagent-cancel', 'Cancel the current tool and pause for operator review.', 'cancelled', {
    message_id: 'fixture-message-cancelled',
    delivery_mode: 'inject_now',
    delivery_status: 'cancelled',
    delivery_reason: 'operator_cancelled',
  }),
]

const runLevelBlocked = event(
  40,
  'runtime_blocked',
  null,
  'Run-level approval blocker without a subagent id should stay in the timeline but must not create a fake drawer card.',
  'blocked',
  {
    blocker_type: 'approval',
    approval_ids: ['run-level-approval-only'],
    recommended_action: 'resolve_approval',
  }
)

const transcriptEvents = [...approvalEvents, ...secondSubagentEvents, ...cancellationEvents, runLevelBlocked]

const subagents: SubagentTranscriptSummary[] = [
  {
    subagentId: 'fixture-subagent-approval',
    parentType: 'chat_turn',
    parentId: 'fixture-turn-1',
    sessionId: 'fixture-session',
    agentRunId: 'fixture-run',
    roleId: 'researcher',
    title: 'Approval parity researcher',
    modelId: 'qwen3-coder-fixture',
    status: 'awaiting_approval',
    promptPreview: longUserPrompt,
    summary: 'Blocked on exec approval with long prompt, output preview, and pending guidance.',
    outputPreview,
    eventCount: approvalEvents.length,
    createdAt: iso(30),
    updatedAt: iso(5),
  },
  {
    subagentId: 'fixture-subagent-observer',
    parentType: 'chat_turn',
    parentId: 'fixture-turn-1',
    sessionId: 'fixture-session',
    agentRunId: 'fixture-run',
    roleId: 'observer',
    title: 'Reload observer',
    modelId: 'qwen3-coder-fixture',
    status: 'completed',
    promptPreview: 'Check reload persistence and summary merging.',
    summary: 'Completed reload-oriented observation.',
    outputPreview: 'No approval required.',
    eventCount: secondSubagentEvents.length,
    createdAt: iso(10),
    updatedAt: iso(2),
  },
  {
    subagentId: 'fixture-subagent-cancel',
    parentType: 'chat_turn',
    parentId: 'fixture-turn-1',
    sessionId: 'fixture-session',
    agentRunId: 'fixture-run',
    roleId: 'operator_cancel',
    title: 'Cancellation observer',
    modelId: 'qwen3-coder-fixture',
    status: 'interrupted',
    promptPreview: 'Exercise tool cancellation and interruption protocol states.',
    summary: 'Interrupted after a deferred tool cancellation request resolved.',
    outputPreview: 'No final output because the subagent was interrupted.',
    eventCount: cancellationEvents.length,
    createdAt: iso(9),
    updatedAt: iso(1),
  },
]

const detailsById: Record<string, SubagentTranscriptDetail> = {
  'fixture-subagent-approval': {
    ...subagents[0],
    systemPrompt: longSystemPrompt,
    userPrompt: longUserPrompt,
    events: approvalEvents,
  },
  'fixture-subagent-observer': {
    ...subagents[1],
    systemPrompt: 'Observe reload persistence without mutating runtime state.',
    userPrompt: 'Confirm session subagent cards can be restored after reload.',
    events: secondSubagentEvents,
  },
  'fixture-subagent-cancel': {
    ...subagents[2],
    systemPrompt: 'Exercise interruption and cancellation protocol display states.',
    userPrompt: 'Start a cancellable tool, request cancellation, defer once, then interrupt the subagent.',
    events: cancellationEvents,
  },
}

export default function SubagentRuntimeFixturePage() {
  const [activeSubagentId, setActiveSubagentId] = React.useState('fixture-subagent-approval')
  const [expandedSubagentIds, setExpandedSubagentIds] = React.useState<Set<string>>(
    () => new Set(['fixture-subagent-approval', 'fixture-subagent-cancel'])
  )
  const [drawerOpen, setDrawerOpen] = React.useState(true)
  const [taskPanelOpen, setTaskPanelOpen] = React.useState(false)
  const [focusedApprovalIds, setFocusedApprovalIds] = React.useState<string[] | null>(null)
  const [guidance, setGuidance] = React.useState('')
  const [guidanceLog, setGuidanceLog] = React.useState<string[]>([])

  const activeSubagent = detailsById[activeSubagentId] ?? null
  const subagentEventsById = React.useMemo(() => {
    const grouped = new Map<string, ExecutionTranscriptEvent[]>()
    for (const subagent of subagents) {
      grouped.set(
        subagent.subagentId,
        transcriptEvents.filter((event) => event.subagentId === subagent.subagentId)
      )
    }
    return grouped
  }, [])

  const openSubagent = React.useCallback((subagentId: string) => {
    setActiveSubagentId(subagentId)
    setDrawerOpen(true)
  }, [])

  return (
    <div className="h-full overflow-y-auto bg-canvas">
      <div className="mx-auto flex w-full max-w-6xl flex-col gap-4 px-4 py-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div className="min-w-0">
            <h1 className="text-lg font-semibold text-foreground">Subagent runtime fixture</h1>
            <p className="mt-1 max-w-3xl text-sm text-muted-foreground">
              Deterministic production-component fixture for timeline, drawer, approval metadata, prompt scrolling,
              output wrapping, and guidance input.
            </p>
          </div>
          <Badge variant="warning">approval required</Badge>
        </div>

        <section className="grid gap-3 md:grid-cols-2" aria-label="Subagent cards">
          {subagents.map((subagent) => (
            <SubagentTimelineCard
              key={subagent.subagentId}
              subagent={subagent}
              active={activeSubagentId === subagent.subagentId}
              expanded={expandedSubagentIds.has(subagent.subagentId)}
              onExpandedChange={(subagentId, expanded) => {
                setExpandedSubagentIds((current) => {
                  const next = new Set(current)
                  if (expanded) {
                    next.add(subagentId)
                  } else {
                    next.delete(subagentId)
                  }
                  return next
                })
              }}
              onOpen={openSubagent}
              onOpenThread={openSubagent}
              recentEvents={subagentEventsById.get(subagent.subagentId) ?? []}
            />
          ))}
        </section>

        <ExecutionTimeline
          events={transcriptEvents}
          subagents={subagents}
          activeSubagentId={activeSubagentId}
          onOpenSubagent={openSubagent}
        />

        {guidanceLog.length > 0 ? (
          <section className="rounded-lg border border-border bg-surface-layer/60 p-3" aria-label="Guidance log">
            <p className="text-xs font-semibold uppercase tracking-[0.08em] text-muted-foreground">Guidance log</p>
            <ul className="mt-2 space-y-1 text-sm text-foreground">
              {guidanceLog.map((item, index) => (
                <li key={`${index}:${item}`}>{item}</li>
              ))}
            </ul>
          </section>
        ) : null}
      </div>

      <SubagentDrawer
        open={drawerOpen}
        onOpenChange={setDrawerOpen}
        subagent={activeSubagent}
        guidanceValue={guidance}
        onGuidanceChange={setGuidance}
        onSendGuidance={() => {
          const trimmed = guidance.trim()
          if (!trimmed) {
            return
          }
          setGuidanceLog((items) => [...items, trimmed])
          setGuidance('')
        }}
        onOpenApprovals={(approvalIds) => {
          setFocusedApprovalIds(approvalIds)
          setDrawerOpen(false)
          window.requestAnimationFrame(() => setTaskPanelOpen(true))
        }}
      />
      <TaskPanel
        open={taskPanelOpen}
        onOpenChange={setTaskPanelOpen}
        focusedApprovalIds={focusedApprovalIds}
      />
    </div>
  )
}
