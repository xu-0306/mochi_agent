'use client'

import * as React from 'react'
import { Activity, ExternalLink, PanelRightClose, Workflow } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { FloatingPanelShell } from '@/components/chat/FloatingPanelShell'
import { PanelSectionCard } from '@/components/chat/PanelSectionCard'
import * as api from '@/lib/api'
import type { AgentRunDetail, AgentRunHealthSummary, ApprovalSummary, TaskDetail, TaskSummary } from '@/lib/api'
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

interface TaskPanelProps {
  open: boolean
  onOpenChange: (open: boolean) => void
  workflowRunId?: string | null
  onOpenWorkflowRun?: (runId: string) => void
}

function TaskPanelBody({
  isLoading,
  approvals,
  tasks,
  selectedTaskId,
  selectedTaskDetail,
  isLoadingDetail,
  isResolvingApprovalId,
  isMutatingTaskId,
  error,
  load,
  selectTask,
  approve,
  reject,
  resume,
  cancel,
  workflowRun,
  workflowHealth,
  workflowLoading,
  onOpenWorkflowRun,
  onClose,
}: {
  isLoading: boolean
  approvals: ApprovalSummary[]
  tasks: TaskSummary[]
  selectedTaskId: string | null
  selectedTaskDetail: TaskDetail | null
  isLoadingDetail: boolean
  isResolvingApprovalId: string | null
  isMutatingTaskId: string | null
  error: string | null
  load: () => Promise<void>
  selectTask: (taskId: string) => Promise<void>
  approve: (approvalId: string) => Promise<void>
  reject: (approvalId: string) => Promise<void>
  resume: (taskId: string) => Promise<void>
  cancel: (taskId: string) => Promise<void>
  workflowRun: AgentRunDetail | null
  workflowHealth: AgentRunHealthSummary | null
  workflowLoading: boolean
  onOpenWorkflowRun?: (runId: string) => void
  onClose?: () => void
}) {
  const pendingApprovals = approvals.filter((item) => item.status === 'pending')

  return (
    <div className="flex h-full flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.14),transparent_34%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-primary-500/20 bg-primary-500/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-primary-300">
              <Activity className="h-3.5 w-3.5" />
              Task Monitor
            </div>
            <h2 className="text-base font-semibold text-foreground">Background activity</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Review active tasks, pending approvals, and recoverable runs without moving the chat.
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
        <div className="space-y-4">
          <PanelSectionCard
            title="Workflow run"
            description="Bound workflow run state for this chat session, including recovery and detached exec visibility."
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

          <PanelSectionCard
            title="Pending approvals"
            description="Decisions that need confirmation before a task can continue."
          >
            <div className="max-h-56 space-y-2 overflow-y-auto">
              {pendingApprovals.length === 0 ? (
                <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                  No pending approvals.
                </p>
              ) : (
                pendingApprovals.map((approval) => (
                  <div key={approval.approval_id} className="rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                    <div className="mb-2 flex items-center justify-between gap-2">
                      <span className="truncate text-xs font-semibold text-foreground">{approval.tool_name}</span>
                      <Badge variant={statusVariant(approval.status)}>{approval.status}</Badge>
                    </div>
                    <p className="mb-2 text-[11px] text-muted-foreground">Task: {approval.task_id}</p>
                    <div className="mb-3 grid gap-1 text-[11px] text-muted-foreground">
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
                      {approval.reason ? (
                        <p>
                          User reason: <span className="text-foreground">{approval.reason}</span>
                        </p>
                      ) : null}
                      {approval.policy_reason ? (
                        <p>
                          Policy reason: <span className="text-foreground">{approval.policy_reason}</span>
                        </p>
                      ) : null}
                    </div>
                    <div className="flex gap-1.5">
                      <Button
                        size="sm"
                        variant="secondary"
                        className="h-7 rounded-full px-3 text-xs"
                        loading={isResolvingApprovalId === approval.approval_id}
                        onClick={() => void approve(approval.approval_id)}
                      >
                        Approve
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        className="h-7 rounded-full px-3 text-xs"
                        loading={isResolvingApprovalId === approval.approval_id}
                        onClick={() => void reject(approval.approval_id)}
                      >
                        Reject
                      </Button>
                    </div>
                  </div>
                ))
              )}
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Recent tasks"
            description="Pick a run to inspect status, output summary, and control actions."
          >
            <div className="space-y-2">
              {tasks.length === 0 ? (
                <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                  No tasks yet.
                </p>
              ) : (
                tasks.map((task) => (
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
                      <span className="truncate text-xs font-semibold text-foreground">{task.task_id}</span>
                      <Badge variant={statusVariant(task.status)}>{task.status}</Badge>
                    </div>
                    {task.task_type ? (
                      <p className="mb-1 text-[11px] text-muted-foreground">{formatMetadataLabel(task.task_type)}</p>
                    ) : null}
                    <p className="text-[11px] leading-relaxed text-muted-foreground">{previewText(task.input_message)}</p>
                  </button>
                ))
              )}
            </div>
          </PanelSectionCard>

          <PanelSectionCard
            title="Task detail"
            description="Expanded state, final answer preview, and mutation actions for the selected run."
          >
            {isLoadingDetail ? (
              <p className="rounded-xl border border-dashed border-white/10 bg-surface-layer/50 px-3 py-4 text-center text-xs text-muted-foreground">
                Loading detail...
              </p>
            ) : selectedTaskDetail ? (
              <div className="space-y-3 rounded-xl border border-white/8 bg-surface-layer/70 px-3 py-3 text-xs shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]">
                <div className="flex items-center justify-between gap-2">
                  <span className="font-semibold text-foreground">{selectedTaskDetail.task_id}</span>
                  <Badge variant={statusVariant(selectedTaskDetail.status)}>{selectedTaskDetail.status}</Badge>
                </div>
                <div className="grid gap-1 text-muted-foreground">
                  <p>Created: {formatTime(selectedTaskDetail.created_at)}</p>
                  <p>Updated: {formatTime(selectedTaskDetail.updated_at)}</p>
                  <p>Events: {selectedTaskDetail.events.length}</p>
                </div>
                {selectedTaskDetail.error ? (
                  <p className="rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-destructive">
                    {selectedTaskDetail.error}
                  </p>
                ) : null}
                {selectedTaskDetail.final_answer ? (
                  <p className="rounded-xl border border-white/8 bg-canvas/75 px-3 py-2 leading-relaxed text-foreground">
                    {previewText(selectedTaskDetail.final_answer, 200)}
                  </p>
                ) : null}
                <div className="flex gap-1.5 pt-1">
                  <Button
                    size="sm"
                    variant="secondary"
                    className="h-7 rounded-full px-3 text-xs"
                    loading={isMutatingTaskId === selectedTaskDetail.task_id}
                    onClick={() => void resume(selectedTaskDetail.task_id)}
                  >
                    Resume
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 rounded-full px-3 text-xs"
                    loading={isMutatingTaskId === selectedTaskDetail.task_id}
                    onClick={() => void cancel(selectedTaskDetail.task_id)}
                  >
                    Cancel
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
  )
}

export function TaskPanel({ open, onOpenChange, workflowRunId, onOpenWorkflowRun }: TaskPanelProps) {
  const {
    tasks,
    approvals,
    selectedTaskId,
    selectedTaskDetail,
    isLoading,
    isLoadingDetail,
    isResolvingApprovalId,
    isMutatingTaskId,
    error,
    load,
    selectTask,
    approve,
    reject,
    resume,
    cancel,
  } = useTaskStore()
  const [workflowRun, setWorkflowRun] = React.useState<AgentRunDetail | null>(null)
  const [workflowHealth, setWorkflowHealth] = React.useState<AgentRunHealthSummary | null>(null)
  const [workflowLoading, setWorkflowLoading] = React.useState(false)

  React.useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 8000)
    return () => window.clearInterval(timer)
  }, [load])

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
        tasks={tasks}
        selectedTaskId={selectedTaskId}
        selectedTaskDetail={selectedTaskDetail}
        isLoadingDetail={isLoadingDetail}
        isResolvingApprovalId={isResolvingApprovalId}
        isMutatingTaskId={isMutatingTaskId}
        error={error}
        load={load}
        selectTask={selectTask}
        approve={approve}
        reject={reject}
        resume={resume}
        cancel={cancel}
        workflowRun={workflowRun}
        workflowHealth={workflowHealth}
        workflowLoading={workflowLoading}
        onOpenWorkflowRun={onOpenWorkflowRun}
        onClose={() => onOpenChange(false)}
      />
    </FloatingPanelShell>
  )
}
