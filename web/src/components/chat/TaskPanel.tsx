'use client'

import * as React from 'react'
import { Activity, ExternalLink, FilePenLine, PanelRightClose, RotateCcw, SendHorizontal, Square, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Textarea } from '@/components/ui/textarea'
import { FloatingPanelShell } from '@/components/chat/FloatingPanelShell'
import { FileChangeCard } from '@/components/chat/FileChangeCard'
import { PanelSectionCard } from '@/components/chat/PanelSectionCard'
import * as api from '@/lib/api'
import type { AgentRunDetail, AgentRunHealthSummary, ApprovalSummary, TaskDetail, TaskSummary } from '@/lib/api'
import type { FileChangeGroupSummary, PatchPreviewResult } from '@/lib/file-change-preview'
import { buildDelegatedSubagentTranscript, delegatedSubagentTitle, resolveDelegatedSubagentView } from '@/lib/subagent-tasks'
import { useTaskStore } from '@/lib/stores/task-store'
import { cn } from '@/lib/utils'

function statusVariant(status: string): 'neutral' | 'warning' | 'success' | 'error' | 'outline' {
  const normalized = status.toLowerCase()
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'error'
  }
  if (normalized === 'running' || normalized === 'pending' || normalized === 'waiting_approval' || normalized === 'awaiting_approval' || normalized === 'resumed') {
    return 'warning'
  }
  if (normalized === 'completed' || normalized === 'done' || normalized === 'succeeded') {
    return 'success'
  }
  return 'outline'
}

function formatTime(value: string | null): string {
  if (!value) {
    return '-'
  }
  const parsed = Date.parse(value)
  if (Number.isNaN(parsed)) {
    return value
  }
  return new Date(parsed).toLocaleString()
}

function previewText(value: string, max = 80): string {
  const trimmed = value.trim()
  if (trimmed.length <= max) {
    return trimmed
  }
  return `${trimmed.slice(0, max)}...`
}

function formatMetadataLabel(value: string): string {
  return value
    .split('_')
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ')
}

function formatRulePreview(rule: api.CommandRule | null): string {
  if (!rule) {
    return '-'
  }
  const target = rule.tokens.length > 0 ? rule.tokens.join(' ') : 'the requested command'
  const matchLabel = rule.match === 'exact' ? 'exactly matches' : 'starts with'
  const shells = rule.shells.length > 0 ? ` in ${rule.shells.join(', ')}` : ''
  return `Allow commands that ${matchLabel} "${target}"${shells}.`
}

function formatExecOutputTail(value: string, maxLines = 8, maxChars = 1000): string | null {
  const trimmed = value.trim()
  if (!trimmed) {
    return null
  }
  const lines = trimmed.split(/\r?\n/).slice(-maxLines)
  const tail = lines.join('\n')
  if (tail.length <= maxChars) {
    return tail
  }
  return `...${tail.slice(-(maxChars - 3))}`
}

function formatExecutionResultPreview(value: unknown): string | null {
  if (typeof value === 'string') {
    return formatExecOutputTail(value, 6, 800)
  }
  if (value == null) {
    return null
  }
  try {
    return previewText(JSON.stringify(value), 240)
  } catch {
    return previewText(String(value), 240)
  }
}

function isTerminalExecStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'completed' ||
    normalized === 'failed' ||
    normalized === 'stopped' ||
    normalized === 'cancelled' ||
    normalized === 'exited'
  )
}

function isTerminalTaskStatus(status: string | null | undefined): boolean {
  const normalized = (status ?? '').toLowerCase()
  return (
    normalized === 'completed' ||
    normalized === 'done' ||
    normalized === 'succeeded' ||
    normalized === 'failed' ||
    normalized === 'error' ||
    normalized === 'cancelled'
  )
}

function isWorkflowAttentionStatus(
  run: AgentRunDetail | null,
  health: AgentRunHealthSummary | null
): boolean {
  const runStatus = (run?.status ?? '').toLowerCase()
  return (
    runStatus === 'failed' ||
    runStatus === 'error' ||
    runStatus === 'stalled' ||
    runStatus === 'awaiting_resources' ||
    runStatus === 'paused' ||
    Boolean(run?.latest_error) ||
    Boolean(health?.degraded) ||
    Object.keys(health?.detached_exec_jobs ?? {}).length > 0
  )
}

export type TaskPanelMode = 'default' | 'subagent'

interface ApprovalReviewState {
  originalPatchText: string | null
  patchText: string | null
  isEditingPatch: boolean
  isPreviewLoading: boolean
  preview: PatchPreviewResult | null
  previewError: string | null
  lastPreviewedPatchText: string | null
}

function buildPreviewGroup(
  approval: ApprovalSummary,
  preview: PatchPreviewResult,
  patchText: string | null
): FileChangeGroupSummary | null {
  if (preview.files.length === 0) {
    return null
  }

  return {
    id: `${approval.approval_id}:preview`,
    sourceTool: approval.tool_name,
    title: preview.valid ? 'Edited patch preview' : 'Patch preview',
    patchText,
    files: preview.files,
  }
}

function ApprovalReviewCard({
  approval,
  execState,
  reviewState,
  isResolvingApprovalKey,
  onResolve,
  onReject,
  refreshApprovalExecSession,
  stopApprovalExecSession,
  setApprovalPatchEditing,
  setApprovalPatchText,
  previewApprovalPatch,
  resetApprovalPatch,
}: {
  approval: ApprovalSummary
  execState: {
    data: api.ApprovalExecSessionPayload | null
    isLoading: boolean
    isStopping: boolean
    error: string | null
  } | undefined
  reviewState: ApprovalReviewState
  isResolvingApprovalKey: string | null
  onResolve: (
    approvalId: string,
    decision: 'approve_once' | 'approve_and_save_rule',
    options?: {
      rule?: api.CommandRule
      replayOverride?: {
        patchText?: string | null
      }
    }
  ) => Promise<void>
  onReject: (approvalId: string) => Promise<void>
  refreshApprovalExecSession: (approvalId: string, yieldTimeMs?: number) => Promise<void>
  stopApprovalExecSession: (approvalId: string) => Promise<void>
  setApprovalPatchEditing: (approvalId: string, editing: boolean) => void
  setApprovalPatchText: (approvalId: string, patchText: string) => void
  previewApprovalPatch: (approvalId: string) => Promise<void>
  resetApprovalPatch: (approvalId: string) => void
}) {
  const deferredPatchText = React.useDeferredValue(reviewState.patchText ?? '')
  const patchTextChanged = (reviewState.patchText ?? '') !== (reviewState.originalPatchText ?? '')
  const canEditPatch = approval.patch_validation_supported && reviewState.patchText !== null
  const execSession = execState?.data?.session ?? null
  const execStatus = execSession?.status ?? execState?.data?.exec_status ?? approval.exec_status
  const stdoutTail = formatExecOutputTail(execSession?.stdout ?? '')
  const stderrTail = formatExecOutputTail(execSession?.stderr ?? '')
  const resultPreview = formatExecutionResultPreview(
    execState?.data?.execution_result ?? approval.execution_result
  )
  const canStopSession = Boolean(approval.exec_session_id && !isTerminalExecStatus(execStatus))
  const previewGroup = buildPreviewGroup(approval, reviewState.preview ?? {
    valid: false,
    summary: null,
    errors: [],
    warnings: [],
    patchText: reviewState.patchText,
    files: [],
  }, reviewState.patchText)
  const reviewGroups = reviewState.isEditingPatch && previewGroup
    ? [previewGroup]
    : approval.file_change_groups
  const patchValidationBlocked = reviewState.isEditingPatch && patchTextChanged && (
    reviewState.isPreviewLoading ||
    reviewState.preview == null ||
    reviewState.lastPreviewedPatchText !== reviewState.patchText ||
    reviewState.preview.valid === false ||
    reviewState.previewError !== null
  )

  React.useEffect(() => {
    if (!canEditPatch || !reviewState.isEditingPatch) {
      return
    }
    if (reviewState.patchText == null) {
      return
    }
    if (reviewState.lastPreviewedPatchText === reviewState.patchText && reviewState.preview) {
      return
    }

    const timer = window.setTimeout(() => {
      void previewApprovalPatch(approval.approval_id)
    }, 250)

    return () => window.clearTimeout(timer)
  }, [
    approval.approval_id,
    canEditPatch,
    deferredPatchText,
    previewApprovalPatch,
    reviewState.isEditingPatch,
    reviewState.lastPreviewedPatchText,
    reviewState.patchText,
    reviewState.preview,
  ])

  return (
    <div className="rounded-[1.35rem] border border-white/8 bg-[linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] px-3 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
      <div className="mb-2 flex items-center justify-between gap-2">
        <span className="truncate text-xs font-semibold uppercase tracking-[0.16em] text-foreground/90">
          {approval.tool_name}
        </span>
        <Badge variant={statusVariant(approval.status)}>{approval.status}</Badge>
      </div>
      <p className="mb-3 text-[11px] text-muted-foreground">
        Task: {approval.task_id ?? 'Standalone approval'}
      </p>

      <div className="mb-3 grid gap-2 text-[11px] text-muted-foreground">
        {approval.command ? (
          <div className="space-y-1">
            <p className="font-medium text-foreground">Command</p>
            <pre className="overflow-x-auto rounded-lg border border-white/8 bg-canvas/75 px-2 py-2 font-mono text-[10px] leading-relaxed text-foreground break-all whitespace-pre-wrap">
              {approval.command}
            </pre>
          </div>
        ) : null}
        <div className="grid gap-1">
          {approval.shell ? (
            <p>
              Shell: <span className="text-foreground">{approval.shell}</span>
            </p>
          ) : null}
          {approval.workdir ? (
            <p className="break-all">
              Workdir: <span className="text-foreground">{approval.workdir}</span>
            </p>
          ) : null}
          {approval.source ? (
            <p>
              Source: <span className="text-foreground">{formatMetadataLabel(approval.source)}</span>
            </p>
          ) : null}
          <p>
            Decision: <span className="text-foreground">{approval.security_decision ? formatMetadataLabel(approval.security_decision) : '-'}</span>
          </p>
          <p>
            Kind: <span className="text-foreground">{formatMetadataLabel(approval.approval_kind)}</span>
          </p>
          <p>
            Scope: <span className="text-foreground">{formatMetadataLabel(approval.approval_scope)}</span>
          </p>
          <p>
            Requires approval: <span className="text-foreground">{approval.requires_approval ? 'Yes' : 'No'}</span>
          </p>
          <p>
            Replay safe: <span className="text-foreground">{approval.replay_safe ? 'Yes' : 'No'}</span>
          </p>
          {approval.policy_source ? (
            <p>
              Policy source: <span className="text-foreground">{formatMetadataLabel(approval.policy_source)}</span>
            </p>
          ) : null}
        </div>
        {approval.policy_reason ? (
          <div className="rounded-lg border border-amber-400/20 bg-amber-400/10 px-2.5 py-2 text-[11px] leading-relaxed text-amber-100">
            <span className="font-medium text-amber-50">Policy reason:</span> {approval.policy_reason}
          </div>
        ) : null}
        {approval.reason ? (
          <p>
            User reason: <span className="text-foreground">{approval.reason}</span>
          </p>
        ) : null}
        {approval.suggested_rule ? (
          <div className="rounded-lg border border-white/8 bg-canvas/75 px-2.5 py-2">
            <p className="mb-1 font-medium text-foreground">Suggested rule</p>
            <p className="leading-relaxed text-muted-foreground">
              {formatRulePreview(approval.suggested_rule)}
            </p>
          </div>
        ) : null}
      </div>

      {reviewGroups.length > 0 ? (
        <div className="mb-3 space-y-3">
          {reviewGroups.map((group, index) => (
            <FileChangeCard
              key={group.id}
              group={group}
              actions={index === 0 && canEditPatch ? (
                <Button
                  type="button"
                  size="sm"
                  variant={reviewState.isEditingPatch ? 'secondary' : 'outline'}
                  className="rounded-full px-3 text-xs"
                  onClick={() => setApprovalPatchEditing(approval.approval_id, !reviewState.isEditingPatch)}
                >
                  <FilePenLine className="h-3.5 w-3.5" />
                  {reviewState.isEditingPatch ? 'Hide patch' : 'Edit patch'}
                </Button>
              ) : null}
            />
          ))}
        </div>
      ) : canEditPatch ? (
        <div className="mb-3">
          <Button
            type="button"
            size="sm"
            variant={reviewState.isEditingPatch ? 'secondary' : 'outline'}
            className="rounded-full px-3 text-xs"
            onClick={() => setApprovalPatchEditing(approval.approval_id, !reviewState.isEditingPatch)}
          >
            <FilePenLine className="h-3.5 w-3.5" />
            {reviewState.isEditingPatch ? 'Hide patch' : 'Edit patch'}
          </Button>
        </div>
      ) : null}

      {reviewState.isEditingPatch ? (
        <div className="mb-3 rounded-[1.1rem] border border-white/8 bg-canvas/55 px-3 py-3">
          <div className="flex flex-wrap items-start justify-between gap-3">
            <div>
              <p className="text-sm font-semibold text-foreground">Editable patch</p>
              <p className="mt-1 text-xs text-muted-foreground">
                Update the raw patch and Mochi will live-preview it before approval.
              </p>
            </div>
            <div className="flex items-center gap-2">
              <Button
                type="button"
                size="sm"
                variant="ghost"
                className="rounded-full px-3 text-xs"
                onClick={() => resetApprovalPatch(approval.approval_id)}
              >
                <RotateCcw className="h-3.5 w-3.5" />
                Reset
              </Button>
            </div>
          </div>
          <Textarea
            value={reviewState.patchText ?? ''}
            onChange={(event) => setApprovalPatchText(approval.approval_id, event.target.value)}
            className="mt-3 min-h-[14rem] rounded-2xl border-white/10 bg-[#090d19] font-mono text-[12px] leading-6 text-slate-100"
            spellCheck={false}
          />
          <div className="mt-3 space-y-2 text-xs">
            <div className="flex flex-wrap items-center gap-2">
              <span className={cn(
                'rounded-full border px-2.5 py-1 text-[11px] font-medium',
                reviewState.isPreviewLoading
                  ? 'border-sky-400/20 bg-sky-400/10 text-sky-100'
                  : reviewState.preview?.valid
                    ? 'border-emerald-400/20 bg-emerald-400/10 text-emerald-100'
                    : 'border-amber-400/20 bg-amber-400/10 text-amber-100'
              )}>
                {reviewState.isPreviewLoading
                  ? 'Validating patch...'
                  : reviewState.preview?.valid
                    ? 'Patch valid'
                    : patchTextChanged
                      ? 'Patch needs fixes'
                      : 'Original patch'}
              </span>
              {reviewState.preview?.summary ? (
                <span className="text-muted-foreground">{reviewState.preview.summary}</span>
              ) : null}
            </div>
            {reviewState.preview?.errors.length ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive">
                {reviewState.preview.errors.join(' ')}
              </div>
            ) : null}
            {reviewState.preview?.warnings.length ? (
              <div className="rounded-xl border border-amber-400/20 bg-amber-400/10 px-3 py-2 text-amber-100">
                {reviewState.preview.warnings.join(' ')}
              </div>
            ) : null}
            {reviewState.previewError ? (
              <div className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive">
                {reviewState.previewError}
              </div>
            ) : null}
          </div>
        </div>
      ) : null}

      {approval.exec_session_id ? (
        <div className="mb-3 space-y-2 rounded-lg border border-white/8 bg-canvas/65 px-2.5 py-2">
          <div className="flex items-start justify-between gap-2">
            <div className="min-w-0">
              <p className="font-medium text-foreground">Exec replay</p>
              <p className="break-all">
                Session: <span className="text-foreground">{approval.exec_session_id}</span>
              </p>
              {approval.exec_approval_id ? (
                <p className="break-all">
                  Exec approval: <span className="text-foreground">{approval.exec_approval_id}</span>
                </p>
              ) : null}
              <p>
                Status: <span className="text-foreground">{execStatus ?? '-'}</span>
              </p>
            </div>
            <Badge variant={statusVariant(execStatus ?? 'pending')}>{execStatus ?? 'pending'}</Badge>
          </div>
          <div className="flex flex-wrap gap-1.5">
            <Button
              size="sm"
              variant="outline"
              className="h-7 rounded-full px-3 text-xs"
              loading={execState?.isLoading ?? false}
              onClick={() => void refreshApprovalExecSession(approval.approval_id)}
            >
              Refresh output
            </Button>
            <Button
              size="sm"
              variant="outline"
              className="h-7 rounded-full px-3 text-xs"
              disabled={!canStopSession}
              loading={execState?.isStopping ?? false}
              onClick={() => void stopApprovalExecSession(approval.approval_id)}
            >
              <Square className="h-3.5 w-3.5" />
              Stop session
            </Button>
          </div>
          {stdoutTail ? (
            <div className="space-y-1">
              <p className="font-medium text-foreground">Stdout tail</p>
              <pre className="overflow-x-auto rounded-lg border border-white/8 bg-black/30 px-2 py-2 font-mono text-[10px] leading-relaxed text-emerald-100 whitespace-pre-wrap">
                {stdoutTail}
              </pre>
            </div>
          ) : null}
          {stderrTail ? (
            <div className="space-y-1">
              <p className="font-medium text-foreground">Stderr tail</p>
              <pre className="overflow-x-auto rounded-lg border border-white/8 bg-black/30 px-2 py-2 font-mono text-[10px] leading-relaxed text-rose-100 whitespace-pre-wrap">
                {stderrTail}
              </pre>
            </div>
          ) : null}
          {!stdoutTail && !stderrTail && resultPreview ? (
            <div className="space-y-1">
              <p className="font-medium text-foreground">Execution result</p>
              <pre className="overflow-x-auto rounded-lg border border-white/8 bg-black/30 px-2 py-2 font-mono text-[10px] leading-relaxed text-slate-100 whitespace-pre-wrap break-all">
                {resultPreview}
              </pre>
            </div>
          ) : null}
          {execState?.error ? (
            <p className="text-destructive">{execState.error}</p>
          ) : null}
        </div>
      ) : null}

      {patchValidationBlocked ? (
        <p className="mb-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
          Approval is blocked until the edited patch validates successfully.
        </p>
      ) : null}

      <div className="flex flex-wrap gap-1.5">
        <Button
          size="sm"
          variant="secondary"
          className="h-7 rounded-full px-3 text-xs"
          disabled={patchValidationBlocked}
          loading={isResolvingApprovalKey === `${approval.approval_id}:approve_once`}
          onClick={() => void onResolve(approval.approval_id, 'approve_once', {
            replayOverride: reviewState.isEditingPatch && patchTextChanged
              ? { patchText: reviewState.patchText }
              : undefined,
          })}
        >
          Approve once
        </Button>
        {approval.allowed_decisions.includes('approve_and_save_rule') ? (
          <Button
            size="sm"
            variant="secondary"
            className="h-7 rounded-full px-3 text-xs"
            disabled={patchValidationBlocked}
            loading={isResolvingApprovalKey === `${approval.approval_id}:approve_and_save_rule`}
            onClick={() =>
              void onResolve(
                approval.approval_id,
                'approve_and_save_rule',
                {
                  rule: approval.suggested_rule ?? undefined,
                  replayOverride: reviewState.isEditingPatch && patchTextChanged
                    ? { patchText: reviewState.patchText }
                    : undefined,
                }
              )
            }
          >
            Approve + save rule
          </Button>
        ) : null}
        <Button
          size="sm"
          variant="outline"
          className="h-7 rounded-full px-3 text-xs"
          loading={isResolvingApprovalKey === `${approval.approval_id}:reject`}
          onClick={() => void onReject(approval.approval_id)}
        >
          Reject
        </Button>
      </div>
    </div>
  )
}

function transcriptBubbleClass(role: 'user' | 'assistant' | 'system'): string {
  if (role === 'user') {
    return 'border-primary-500/30 bg-primary-500/12'
  }
  if (role === 'assistant') {
    return 'border-white/10 bg-canvas/70'
  }
  return 'border-amber-400/20 bg-amber-400/10'
}

function transcriptMetaClass(role: 'user' | 'assistant' | 'system'): string {
  if (role === 'user') {
    return 'text-primary-200/80'
  }
  if (role === 'assistant') {
    return 'text-muted-foreground'
  }
  return 'text-amber-100/80'
}

function transcriptAlignmentClass(role: 'user' | 'assistant' | 'system'): string {
  if (role === 'user') {
    return 'justify-end'
  }
  if (role === 'system') {
    return 'justify-center'
  }
  return 'justify-start'
}

interface TaskPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  mode?: TaskPanelMode
  focusedTaskId?: string | null
  workflowRunId?: string | null
  onOpenWorkflowRun?: (runId: string) => void
}

function TaskPanelBody({
  isLoading,
  approvals,
  approvalExecSessions,
  approvalReviewStates,
  tasks,
  selectedTaskId,
  selectedTaskDetail,
  isLoadingDetail,
  isResolvingApprovalKey,
  isMutatingTaskId,
  error,
  load,
  selectTask,
  resolveApprovalDecision,
  setApprovalPatchEditing,
  setApprovalPatchText,
  previewApprovalPatch,
  resetApprovalPatch,
  refreshApprovalExecSession,
  reject,
  resume,
  cancel,
  stopApprovalExecSession,
  workflowRun,
  workflowHealth,
  workflowLoading,
  mode,
  onOpenWorkflowRun,
  onClose,
}: {
  isLoading: boolean
  approvals: ApprovalSummary[]
  approvalExecSessions: Record<
    string,
    {
      data: api.ApprovalExecSessionPayload | null
      isLoading: boolean
      isStopping: boolean
      error: string | null
    }
  >
  approvalReviewStates: Record<string, ApprovalReviewState>
  tasks: TaskSummary[]
  selectedTaskId: string | null
  selectedTaskDetail: TaskDetail | null
  isLoadingDetail: boolean
  isResolvingApprovalKey: string | null
  isMutatingTaskId: string | null
  error: string | null
  load: () => Promise<void>
  selectTask: (taskId: string) => Promise<void>
  resolveApprovalDecision: (
    approvalId: string,
    decision: 'approve_once' | 'approve_and_save_rule' | 'reject',
    options?: {
      rule?: api.CommandRule
      replayOverride?: {
        patchText?: string | null
      }
    }
  ) => Promise<void>
  setApprovalPatchEditing: (approvalId: string, editing: boolean) => void
  setApprovalPatchText: (approvalId: string, patchText: string) => void
  previewApprovalPatch: (approvalId: string) => Promise<void>
  resetApprovalPatch: (approvalId: string) => void
  refreshApprovalExecSession: (approvalId: string, yieldTimeMs?: number) => Promise<void>
  reject: (approvalId: string) => Promise<void>
  resume: (taskId: string) => Promise<void>
  cancel: (taskId: string) => Promise<void>
  stopApprovalExecSession: (approvalId: string) => Promise<void>
  workflowRun: AgentRunDetail | null
  workflowHealth: AgentRunHealthSummary | null
  workflowLoading: boolean
  mode: TaskPanelMode
  onOpenWorkflowRun?: (runId: string) => void
  onClose?: () => void
}) {
  const pendingApprovals = approvals.filter((item) => item.status === 'pending')
  const subagentFocusedMode = mode === 'subagent'
  const delegatedTaskCount = tasks.filter((task) => resolveDelegatedSubagentView({ task }) !== null).length
  const selectedSubagentView = selectedTaskDetail
    ? resolveDelegatedSubagentView({ task: selectedTaskDetail })
    : null
  const selectedConversationTitle = selectedSubagentView
    ? delegatedSubagentTitle(selectedSubagentView)
    : selectedTaskDetail?.task_id ?? 'Task detail'
  const subagentTranscript = React.useMemo(
    () => (selectedTaskDetail ? buildDelegatedSubagentTranscript(selectedTaskDetail) : []),
    [selectedTaskDetail]
  )
  const [guidanceText, setGuidanceText] = React.useState('')
  const [guidancePending, setGuidancePending] = React.useState(false)
  const [guidanceError, setGuidanceError] = React.useState<string | null>(null)
  const [guidanceSuccess, setGuidanceSuccess] = React.useState<string | null>(null)
  const workflowNeedsAttention = isWorkflowAttentionStatus(workflowRun, workflowHealth)
  const showWorkflowSection = !subagentFocusedMode || workflowNeedsAttention || workflowLoading
  const showPendingApprovalsSection = !subagentFocusedMode || pendingApprovals.length > 0

  React.useEffect(() => {
    setGuidanceText('')
    setGuidancePending(false)
    setGuidanceError(null)
    setGuidanceSuccess(null)
  }, [selectedTaskId])

  const handleQueueSubagentGuidance = async (): Promise<void> => {
    if (!selectedTaskDetail || !selectedSubagentView) {
      return
    }

    const content = guidanceText.trim()
    if (!content) {
      return
    }

    setGuidancePending(true)
    setGuidanceError(null)
    setGuidanceSuccess(null)

    try {
      await api.appendTaskMessage(selectedTaskDetail.task_id, {
        content,
        metadata: {
          channel: 'delegated_subagent_guidance',
          delivery: 'guidance_only',
          source: 'task_panel',
          ...(selectedSubagentView.displayName ? { display_name: selectedSubagentView.displayName } : {}),
          ...(selectedSubagentView.role ? { role: selectedSubagentView.role } : {}),
        },
      })
      setGuidanceText('')
      setGuidanceSuccess('Guidance queued for the next applicable turn.')
      await selectTask(selectedTaskDetail.task_id)
    } catch (error) {
      if (error instanceof api.ApiError && error.status === 404) {
        setGuidanceError('Guidance could not be queued yet. This backend does not expose POST /v1/tasks/{task_id}/messages.')
      } else {
        setGuidanceError(error instanceof Error ? error.message : 'Failed to queue guidance')
      }
    } finally {
      setGuidancePending(false)
    }
  }

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.14),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
              <Activity className="h-3.5 w-3.5" />
              Agent workspace
            </div>
            <h2 className="text-base font-semibold text-foreground">
              {subagentFocusedMode
                ? 'Subagent conversation'
                : delegatedTaskCount > 0
                  ? 'Subagent conversations'
                  : 'Background activity'}
            </h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              {subagentFocusedMode
                ? 'Focused view for one delegated subagent while keeping blocking approvals and workflow issues reachable.'
                : delegatedTaskCount > 0
                ? 'Switch between delegated subagents, read their transcript, and send follow-up guidance without leaving the chat.'
                : 'Review active tasks, pending approvals, and recoverable runs without moving the chat.'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              className="rounded-full border border-white/8 bg-canvas/55"
              onClick={() => void load()}
              loading={isLoading}
            >
              Refresh
            </Button>
            {onClose ? (
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                onClick={onClose}
                title="Hide tasks"
                aria-label="Hide tasks"
                className="hidden rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground lg:inline-flex"
              >
                <PanelRightClose className="h-4 w-4" />
              </Button>
            ) : null}
          </div>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-4">
        <div className="flex flex-col gap-4">
          {showWorkflowSection ? (
            <div className={cn(subagentFocusedMode ? 'order-4' : 'order-1')}>
              <PanelSectionCard
                title="Workflow run"
                description={
                  subagentFocusedMode
                    ? 'Only shown here when the workflow needs attention or is still loading.'
                    : 'Bound workflow run state for this chat session, including recovery and detached exec visibility.'
                }
              >
            {workflowLoading ? (
              <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                Loading workflow run...
              </p>
            ) : workflowRun ? (
              <div className="space-y-3 rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex min-w-0 items-center gap-2">
                    <Workflow className="h-4 w-4 shrink-0 text-primary-300" />
                    <span className="truncate font-semibold text-foreground">
                      {workflowRun.title || workflowRun.run_id}
                    </span>
                  </div>
                  <Badge variant={statusVariant(workflowRun.status)}>{workflowRun.status}</Badge>
                </div>
                <div className="grid gap-1 text-muted-foreground">
                  <p>Run ID: {workflowRun.run_id}</p>
                  <p>Updated: {formatTime(workflowRun.updated_at)}</p>
                  <p>Events: {workflowRun.events.length}</p>
                  {workflowHealth?.schedule_health_status ? (
                    <p>Recovery: {workflowHealth.schedule_health_status}</p>
                  ) : null}
                  {workflowHealth?.detached_exec_jobs ? (
                    <p>Detached exec jobs: {Object.keys(workflowHealth.detached_exec_jobs).length}</p>
                  ) : null}
                </div>
                {workflowRun.latest_error ? (
                  <p className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive">
                    {workflowRun.latest_error}
                  </p>
                ) : null}
                {onOpenWorkflowRun ? (
                  <div className="flex gap-1.5 pt-1">
                    <Button
                      size="sm"
                      variant="outline"
                      className="h-7 rounded-full px-3 text-xs"
                      onClick={() => onOpenWorkflowRun(workflowRun.run_id)}
                    >
                      <ExternalLink className="h-3.5 w-3.5" />
                      Open workflow run
                    </Button>
                  </div>
                ) : null}
              </div>
            ) : (
              <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                No workflow run is currently bound to this chat.
              </p>
            )}
              </PanelSectionCard>
            </div>
          ) : null}

          {showPendingApprovalsSection ? (
            <div className={cn(subagentFocusedMode ? 'order-3' : 'order-2')}>
              <PanelSectionCard
                title="Pending approvals"
                description={
                  subagentFocusedMode
                    ? 'Kept visible here only when approvals are still blocking progress.'
                    : 'Decisions that need confirmation before a task can continue.'
                }
              >
            <div className="max-h-[42rem] space-y-3 overflow-y-auto pr-1">
              {pendingApprovals.length === 0 ? (
                <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                  No pending approvals.
                </p>
              ) : (
                pendingApprovals.map((approval) => (
                  <ApprovalReviewCard
                    key={approval.approval_id}
                    approval={approval}
                    execState={approvalExecSessions[approval.approval_id]}
                    reviewState={approvalReviewStates[approval.approval_id] ?? {
                      originalPatchText: approval.editable_patch_text,
                      patchText: approval.editable_patch_text,
                      isEditingPatch: false,
                      isPreviewLoading: false,
                      preview: null,
                      previewError: null,
                      lastPreviewedPatchText: null,
                    }}
                    isResolvingApprovalKey={isResolvingApprovalKey}
                    onResolve={async (approvalId, decision, options) => {
                      await resolveApprovalDecision(approvalId, decision, options)
                    }}
                    onReject={reject}
                    refreshApprovalExecSession={refreshApprovalExecSession}
                    stopApprovalExecSession={stopApprovalExecSession}
                    setApprovalPatchEditing={setApprovalPatchEditing}
                    setApprovalPatchText={setApprovalPatchText}
                    previewApprovalPatch={previewApprovalPatch}
                    resetApprovalPatch={resetApprovalPatch}
                  />
                ))
              )}
            </div>
              </PanelSectionCard>
            </div>
          ) : null}

          <div className={cn(subagentFocusedMode ? 'order-2' : 'order-3')}>
            <PanelSectionCard
              title={delegatedTaskCount > 0 ? 'Subagents' : 'Recent tasks'}
              description={
                subagentFocusedMode
                  ? 'Switch to another delegated subagent without leaving the focused conversation flow.'
                  : delegatedTaskCount > 0
                    ? 'Open a delegated subagent by name. Non-subagent background tasks are listed here too.'
                    : 'Pick a run to inspect status, output summary, and control actions.'
              }
            >
            <div className="space-y-2">
              {tasks.length === 0 ? (
                <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                  No tasks yet.
                </p>
              ) : (
                tasks.map((task) => {
                  const subagentView = resolveDelegatedSubagentView({ task })
                  return (
                    <button
                      key={task.task_id}
                      type="button"
                      onClick={() => void selectTask(task.task_id)}
                      className={cn(
                        'w-full rounded-xl border px-3 py-3 text-left transition-all duration-150',
                        selectedTaskId === task.task_id
                          ? 'border-primary-500/35 bg-primary-500/10 shadow-[0_10px_24px_rgba(94,106,210,0.16)]'
                          : 'border-white/8 bg-surface-layer/70 hover:border-primary-500/20 hover:bg-elevated-layer/70'
                      )}
                    >
                      <div className="mb-1 flex items-center justify-between gap-2">
                        <span className="truncate text-xs font-semibold text-foreground">
                          {subagentView ? delegatedSubagentTitle(subagentView) : task.task_id}
                        </span>
                        <Badge variant={statusVariant(task.status)}>{task.status}</Badge>
                      </div>
                      {task.task_type ? (
                        <p className="mb-1 text-[11px] text-muted-foreground">
                          {subagentView ? 'Subagent conversation' : formatMetadataLabel(task.task_type)}
                        </p>
                      ) : null}
                      <p className="text-[11px] leading-relaxed text-muted-foreground">
                        {previewText(subagentView?.instruction ?? task.input_message)}
                      </p>
                    </button>
                  )
                })
              )}
            </div>
            </PanelSectionCard>
          </div>

          <div className={cn(subagentFocusedMode ? 'order-1' : 'order-4')}>
            <PanelSectionCard
              title={selectedSubagentView ? 'Subagent conversation' : 'Task detail'}
              description={
                selectedSubagentView
                  ? 'Conversation-style view of the main-agent handoff, subagent outputs, and your follow-up guidance.'
                  : 'Expanded state, final answer preview, and mutation actions for the selected run.'
              }
            >
            {isLoadingDetail ? (
              <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                Loading detail...
              </p>
            ) : selectedTaskDetail ? (
              <div className="space-y-3 rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                <div className="flex items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="truncate text-sm font-semibold text-foreground">
                      {selectedConversationTitle}
                    </p>
                    <p className="mt-0.5 truncate font-mono text-[10px] text-muted-foreground">
                      {selectedTaskDetail.task_id}
                    </p>
                  </div>
                  <Badge variant={statusVariant(selectedTaskDetail.status)}>{selectedTaskDetail.status}</Badge>
                </div>
                <div className="grid gap-1 text-muted-foreground">
                  <p>Created {formatTime(selectedTaskDetail.created_at)}</p>
                  <p>Updated {formatTime(selectedTaskDetail.updated_at)}</p>
                  <p>{selectedTaskDetail.events.length} recorded event{selectedTaskDetail.events.length === 1 ? '' : 's'}</p>
                </div>
                {selectedSubagentView ? (
                  <div className="rounded-xl border border-primary-500/20 bg-primary-500/10 px-3 py-2">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-primary-300">
                      Main agent handoff
                    </p>
                    <p className="mt-1 text-sm font-semibold text-foreground">
                      {delegatedSubagentTitle(selectedSubagentView)}
                    </p>
                    {selectedSubagentView.instruction ? (
                      <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-muted-foreground">
                        {selectedSubagentView.instruction}
                      </p>
                    ) : null}
                  </div>
                ) : null}
                {selectedSubagentView ? (
                  <div className="rounded-xl border border-white/8 bg-canvas/55 px-3 py-3">
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-foreground/85">
                          Conversation
                        </p>
                        <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                          Main-agent instructions, your guidance, subagent outputs, artifacts, and final answer events.
                        </p>
                      </div>
                      <Badge variant="outline">{subagentTranscript.length} entries</Badge>
                    </div>

                    <div className="mt-3 space-y-2">
                      {subagentTranscript.length === 0 ? (
                        <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                          No subagent transcript entries have been recorded yet.
                        </p>
                      ) : (
                        subagentTranscript.map((entry) => (
                          <div
                            key={entry.id}
                            className={cn('flex', transcriptAlignmentClass(entry.role))}
                          >
                            <div
                              className={cn(
                                'max-w-[92%] rounded-[1.15rem] border px-3 py-2 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]',
                                transcriptBubbleClass(entry.role)
                              )}
                            >
                              <div className="flex flex-wrap items-center justify-between gap-2">
                                <span className="text-[11px] font-semibold text-foreground/90">
                                  {entry.label}
                                </span>
                                {entry.timestamp ? (
                                  <span className={cn('text-[10px]', transcriptMetaClass(entry.role))}>
                                    {formatTime(entry.timestamp)}
                                  </span>
                                ) : null}
                              </div>
                              <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-foreground">
                                {entry.content}
                              </p>
                              {entry.meta ? (
                                <p className={cn('mt-2 text-[10px]', transcriptMetaClass(entry.role))}>
                                  {entry.meta}
                                </p>
                              ) : null}
                            </div>
                          </div>
                        ))
                      )}
                    </div>

                    <div className="mt-3 rounded-xl border border-white/8 bg-surface-layer/65 px-3 py-3">
                      <div className="flex flex-wrap items-start justify-between gap-2">
                        <div>
                          <p className="text-sm font-semibold text-foreground">Message this subagent</p>
                          <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">
                            Your message is queued as additional context for the next applicable subagent turn.
                          </p>
                        </div>
                        <Badge variant="outline">Guidance</Badge>
                      </div>
                      <Textarea
                        value={guidanceText}
                        onChange={(event) => setGuidanceText(event.target.value)}
                        placeholder={
                          isTerminalTaskStatus(selectedTaskDetail.status)
                            ? 'Task is finished. Guidance queue is disabled.'
                            : `Send guidance to ${selectedSubagentView.displayName ?? 'the subagent'}...`
                        }
                        minRows={3}
                        autoResize
                        disabled={guidancePending || isTerminalTaskStatus(selectedTaskDetail.status)}
                        className="mt-3 rounded-2xl border-white/10 bg-[#090d19] text-[12px] leading-6 text-slate-100 placeholder:text-slate-400"
                      />
                      {guidanceError ? (
                        <p className="mt-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-[11px] text-destructive">
                          {guidanceError}
                        </p>
                      ) : null}
                      {guidanceSuccess ? (
                        <p className="mt-3 rounded-xl border border-emerald-400/25 bg-emerald-400/10 px-3 py-2 text-[11px] text-emerald-100">
                          {guidanceSuccess}
                        </p>
                      ) : null}
                      <div className="mt-3 flex justify-end">
                        <Button
                          size="sm"
                          variant="secondary"
                          className="h-8 rounded-full px-3 text-xs"
                          loading={guidancePending}
                          disabled={
                            guidancePending ||
                            isTerminalTaskStatus(selectedTaskDetail.status) ||
                            guidanceText.trim().length === 0
                          }
                          onClick={() => void handleQueueSubagentGuidance()}
                        >
                          <SendHorizontal className="h-3.5 w-3.5" />
                          Send guidance
                        </Button>
                      </div>
                    </div>
                  </div>
                ) : null}
                {selectedTaskDetail.error ? (
                  <p className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive">
                    {selectedTaskDetail.error}
                  </p>
                ) : null}
                {selectedTaskDetail.final_answer ? (
                  <div className="rounded-xl border border-white/8 bg-canvas/75 px-3 py-2">
                    <p className="text-[11px] font-semibold uppercase tracking-[0.12em] text-muted-foreground">
                      {selectedSubagentView ? 'Subagent final answer' : 'Final answer'}
                    </p>
                    <p className="mt-1 leading-relaxed text-foreground">
                      {previewText(selectedTaskDetail.final_answer, 200)}
                    </p>
                  </div>
                ) : null}
                <div className="flex gap-1.5 pt-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="h-7 rounded-full px-3 text-xs"
                    loading={isMutatingTaskId === selectedTaskDetail.task_id}
                    onClick={() => void resume(selectedTaskDetail.task_id)}
                  >
                    {selectedSubagentView ? 'Resume subagent' : 'Resume'}
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 rounded-full px-3 text-xs"
                    loading={isMutatingTaskId === selectedTaskDetail.task_id}
                    onClick={() => void cancel(selectedTaskDetail.task_id)}
                  >
                    {selectedSubagentView ? 'Cancel subagent' : 'Cancel'}
                  </Button>
                </div>
              </div>
            ) : (
              <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                Select a task to preview detail.
              </p>
            )}
            {error ? <p className="mt-2 text-xs text-destructive">{error}</p> : null}
            </PanelSectionCard>
          </div>
        </div>
      </div>
    </div>
  )
}

export function TaskPanel({
  open,
  onOpenChange,
  mode = 'default',
  focusedTaskId = null,
  workflowRunId,
  onOpenWorkflowRun,
}: TaskPanelProps) {
  const {
    tasks,
    approvals,
    approvalExecSessions,
    approvalReviewStates,
    selectedTaskId,
    selectedTaskDetail,
    isLoading,
    isLoadingDetail,
    isResolvingApprovalKey,
    isMutatingTaskId,
    error,
    load,
    selectTask,
    resolveApprovalDecision,
    setApprovalPatchEditing,
    setApprovalPatchText,
    previewApprovalPatch,
    resetApprovalPatch,
    refreshApprovalExecSession,
    reject,
    resume,
    cancel,
    stopApprovalExecSession,
  } = useTaskStore()
  const [workflowRun, setWorkflowRun] = React.useState<AgentRunDetail | null>(null)
  const [workflowHealth, setWorkflowHealth] = React.useState<AgentRunHealthSummary | null>(null)
  const [workflowLoading, setWorkflowLoading] = React.useState(false)

  React.useEffect(() => {
    if (!open || !focusedTaskId || selectedTaskId === focusedTaskId) {
      return
    }
    void selectTask(focusedTaskId)
  }, [focusedTaskId, open, selectTask, selectedTaskId])

  React.useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 8000)
    return () => window.clearInterval(timer)
  }, [load])

  React.useEffect(() => {
    if (!open) {
      return
    }
    approvals
      .filter((approval) => approval.status === 'pending' && approval.exec_session_id)
      .forEach((approval) => {
        const execState = approvalExecSessions[approval.approval_id]
        if (!execState?.data && !execState?.isLoading) {
          void refreshApprovalExecSession(approval.approval_id, 150)
        }
      })
  }, [approvalExecSessions, approvals, open, refreshApprovalExecSession])

  React.useEffect(() => {
    if (!workflowRunId) {
      setWorkflowRun(null)
      setWorkflowHealth(null)
      setWorkflowLoading(false)
      return
    }

    let cancelled = false
    const loadWorkflow = async () => {
      setWorkflowLoading(true)
      try {
        const [run, health] = await Promise.all([
          api.fetchAgentRun(workflowRunId),
          api.fetchAgentRunHealth(workflowRunId).catch(() => null),
        ])
        if (!cancelled) {
          setWorkflowRun(run)
          setWorkflowHealth(health)
        }
      } catch {
        if (!cancelled) {
          setWorkflowRun(null)
          setWorkflowHealth(null)
        }
      } finally {
        if (!cancelled) {
          setWorkflowLoading(false)
        }
      }
    }

    void loadWorkflow()
    const timer = window.setInterval(() => {
      void loadWorkflow()
    }, 8000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [workflowRunId])

  return (
    <FloatingPanelShell
      open={open}
      onOpenChange={onOpenChange}
      desktopSide="right"
      desktopWidthClass="w-[23rem]"
      desktopBreakpoint="lg"
    >
      <TaskPanelBody
        isLoading={isLoading}
        approvals={approvals}
        approvalExecSessions={approvalExecSessions}
        approvalReviewStates={approvalReviewStates}
        tasks={tasks}
        selectedTaskId={selectedTaskId}
        selectedTaskDetail={selectedTaskDetail}
        isLoadingDetail={isLoadingDetail}
        isResolvingApprovalKey={isResolvingApprovalKey}
        isMutatingTaskId={isMutatingTaskId}
        error={error}
        load={load}
        selectTask={selectTask}
        resolveApprovalDecision={resolveApprovalDecision}
        setApprovalPatchEditing={setApprovalPatchEditing}
        setApprovalPatchText={setApprovalPatchText}
        previewApprovalPatch={previewApprovalPatch}
        resetApprovalPatch={resetApprovalPatch}
        refreshApprovalExecSession={refreshApprovalExecSession}
        reject={reject}
        resume={resume}
        cancel={cancel}
        stopApprovalExecSession={stopApprovalExecSession}
        workflowRun={workflowRun}
        workflowHealth={workflowHealth}
        workflowLoading={workflowLoading}
        mode={mode}
        onOpenWorkflowRun={onOpenWorkflowRun}
        onClose={() => onOpenChange(false)}
      />
    </FloatingPanelShell>
  )
}
