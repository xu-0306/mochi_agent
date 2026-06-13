'use client'

import * as React from 'react'
import { Activity, AlertCircle, CheckCircle2, ExternalLink, Loader2, ShieldAlert, TerminalSquare, Workflow } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import * as api from '@/lib/api'
import type { AgentRunDetail, AgentRunHealthSummary, ApprovalSummary, TaskDetail, TaskSummary } from '@/lib/api'
import { useTaskStore } from '@/lib/stores/task-store'
import { cn } from '@/lib/utils'

interface ChatActivityStripProps {
  currentSessionId?: string | null
  projectId?: string | null
  workflowRunId?: string | null
  onOpenTaskPanel: () => void
  onOpenWorkflowRun?: (runId: string) => void
}

function statusVariant(status: string): 'neutral' | 'warning' | 'success' | 'error' | 'outline' {
  const normalized = status.toLowerCase()
  if (normalized === 'failed' || normalized === 'error' || normalized === 'cancelled') {
    return 'error'
  }
  if (normalized === 'completed' || normalized === 'done' || normalized === 'succeeded') {
    return 'success'
  }
  if (normalized === 'running' || normalized === 'queued' || normalized === 'resumed' || normalized === 'awaiting_approval') {
    return 'warning'
  }
  return 'outline'
}

function previewText(value: string | null | undefined, max = 140): string | null {
  const normalized = (value ?? '').trim()
  if (!normalized) {
    return null
  }
  if (normalized.length <= max) {
    return normalized
  }
  return `${normalized.slice(0, max)}...`
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function matchesCurrentContext(
  task: TaskSummary,
  currentSessionId: string | null | undefined,
  projectId: string | null | undefined
): boolean {
  if (currentSessionId) {
    return task.session_id === currentSessionId
  }
  if (projectId) {
    return task.project_id === projectId
  }
  return false
}

function extractRecentExecSummary(taskDetail: TaskDetail | null): {
  stream: 'stdout' | 'stderr' | 'status'
  text: string
} | null {
  if (!taskDetail) {
    return null
  }

  for (const event of [...taskDetail.events].reverse()) {
    if (!isRecord(event) || event.type !== 'tool_call_result') {
      continue
    }
    const result = isRecord(event.result) ? event.result : {}
    const stdout = typeof result.stdout === 'string' ? result.stdout.trim() : ''
    if (stdout) {
      return { stream: 'stdout', text: previewText(stdout, 200) ?? stdout }
    }
    const stderr = typeof result.stderr === 'string' ? result.stderr.trim() : ''
    if (stderr) {
      return { stream: 'stderr', text: previewText(stderr, 200) ?? stderr }
    }
    const sessionId = typeof result.session_id === 'string' ? result.session_id.trim() : ''
    const status = typeof result.status === 'string' ? result.status.trim() : ''
    if (sessionId && status === 'running') {
      return { stream: 'status', text: `Background exec session active: ${sessionId}` }
    }
  }

  return null
}

function ActivityCard({
  title,
  icon,
  children,
  className,
}: {
  title: string
  icon: React.ReactNode
  children: React.ReactNode
  className?: string
}) {
  return (
    <div className={cn('rounded-2xl border border-border/80 bg-surface-layer/75 px-4 py-3 shadow-[0_10px_26px_rgba(0,0,0,0.12)]', className)}>
      <div className="mb-2 flex items-center gap-2 text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
        {icon}
        <span>{title}</span>
      </div>
      {children}
    </div>
  )
}

export function ChatActivityStrip({
  currentSessionId,
  projectId,
  workflowRunId,
  onOpenTaskPanel,
  onOpenWorkflowRun,
}: ChatActivityStripProps) {
  const {
    tasks,
    approvals,
    selectedTaskId,
    selectedTaskDetail,
    isLoading,
    isResolvingApprovalId,
    approve,
    reject,
    selectTask,
  } = useTaskStore()
  const [workflowRun, setWorkflowRun] = React.useState<AgentRunDetail | null>(null)
  const [workflowHealth, setWorkflowHealth] = React.useState<AgentRunHealthSummary | null>(null)
  const [workflowLoading, setWorkflowLoading] = React.useState(false)

  React.useEffect(() => {
    if (!workflowRunId) {
      setWorkflowRun(null)
      setWorkflowHealth(null)
      setWorkflowLoading(false)
      return
    }

    let cancelled = false
    const load = async () => {
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

    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 8000)
    return () => {
      cancelled = true
      window.clearInterval(timer)
    }
  }, [workflowRunId])

  const contextualTasks = React.useMemo(
    () => tasks.filter((task) => matchesCurrentContext(task, currentSessionId, projectId)),
    [currentSessionId, projectId, tasks]
  )

  const activeTasks = React.useMemo(
    () => contextualTasks.filter((task) => ['queued', 'running', 'resumed', 'awaiting_approval'].includes(task.status)),
    [contextualTasks]
  )

  const pendingApprovals = React.useMemo(() => {
    const taskIds = new Set(contextualTasks.map((task) => task.task_id))
    if (taskIds.size === 0) {
      return []
    }
    return approvals.filter((approval) => approval.status === 'pending' && taskIds.has(approval.task_id))
  }, [approvals, contextualTasks])

  const focusTask = activeTasks[0] ?? null

  React.useEffect(() => {
    if (!focusTask || selectedTaskId === focusTask.task_id) {
      return
    }
    void selectTask(focusTask.task_id)
  }, [focusTask, selectTask, selectedTaskId])

  const recentExecSummary = React.useMemo(
    () =>
      selectedTaskDetail && focusTask && selectedTaskDetail.task_id === focusTask.task_id
        ? extractRecentExecSummary(selectedTaskDetail)
        : null,
    [focusTask, selectedTaskDetail]
  )

  const detachedExecJobsCount = React.useMemo(() => {
    const jobs = workflowHealth?.detached_exec_jobs
    return jobs ? Object.keys(jobs).length : 0
  }, [workflowHealth])

  if (
    pendingApprovals.length === 0 &&
    activeTasks.length === 0 &&
    !workflowRun &&
    !workflowLoading &&
    !isLoading
  ) {
    return null
  }

  const firstApproval = pendingApprovals[0] ?? null

  return (
    <div className="border-t border-border bg-canvas/95 py-3 backdrop-blur">
      <div className="mx-auto w-full max-w-[960px] px-4">
        <div className="mb-3 flex items-center justify-between gap-3">
          <div className="flex min-w-0 items-center gap-2">
            <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full border border-primary-500/30 bg-primary-500/10">
              {isLoading ? (
                <Loader2 className="h-4 w-4 animate-spin text-primary-400" />
              ) : (
                <Activity className="h-4 w-4 text-primary-400" />
              )}
            </div>
            <div className="min-w-0">
                <p className="text-sm font-medium text-foreground">Live task activity</p>
                <p className="truncate text-xs text-muted-foreground">
                {pendingApprovals.length > 0
                  ? `${pendingApprovals.length} approval${pendingApprovals.length > 1 ? 's' : ''} waiting`
                  : workflowRun
                    ? `Workflow run ${workflowRun.status}`
                  : activeTasks.length > 0
                    ? `${activeTasks.length} active task${activeTasks.length > 1 ? 's' : ''}`
                    : 'Refreshing background state'}
              </p>
            </div>
          </div>
          <Button type="button" variant="ghost" size="sm" onClick={onOpenTaskPanel}>
            <ExternalLink className="h-3.5 w-3.5" />
            Open task panel
          </Button>
        </div>

        <div className="grid gap-3 lg:grid-cols-3">
          {firstApproval ? (
            <ActivityCard title="Pending approval" icon={<ShieldAlert className="h-3.5 w-3.5" />}>
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium text-foreground">{firstApproval.tool_name}</p>
                <Badge variant={statusVariant(firstApproval.status)}>{firstApproval.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {firstApproval.reason || firstApproval.policy_reason || 'This tool call needs a decision before work can continue.'}
              </p>
              <div className="mt-3 flex gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="secondary"
                  loading={isResolvingApprovalId === firstApproval.approval_id}
                  onClick={() => void approve(firstApproval.approval_id)}
                >
                  Approve
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  loading={isResolvingApprovalId === firstApproval.approval_id}
                  onClick={() => void reject(firstApproval.approval_id)}
                >
                  Reject
                </Button>
              </div>
            </ActivityCard>
          ) : null}

          {workflowRun ? (
            <ActivityCard title="Workflow run" icon={<Workflow className="h-3.5 w-3.5" />}>
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium text-foreground">
                  {workflowRun.title || workflowRun.run_id}
                </p>
                <Badge variant={statusVariant(workflowRun.status)}>{workflowRun.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {previewText(workflowRun.topic, 180) ?? 'Workflow run is active in this chat session.'}
              </p>
              {workflowRun.latest_error ? (
                <p className="mt-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {previewText(workflowRun.latest_error, 200)}
                </p>
              ) : null}
              {!workflowRun.latest_error && detachedExecJobsCount > 0 ? (
                <p className="mt-3 rounded-xl border border-border/70 bg-canvas/70 px-3 py-2 text-xs text-foreground">
                  {detachedExecJobsCount} detached exec job{detachedExecJobsCount > 1 ? 's' : ''} tracked for this run.
                </p>
              ) : null}
              {!workflowRun.latest_error && !detachedExecJobsCount && workflowHealth?.schedule_health_status ? (
                <p className="mt-3 rounded-xl border border-border/70 bg-canvas/70 px-3 py-2 text-xs text-foreground">
                  Recovery state: {workflowHealth.schedule_health_status}
                </p>
              ) : null}
              {onOpenWorkflowRun ? (
                <div className="mt-3">
                  <Button
                    type="button"
                    size="sm"
                    variant="outline"
                    onClick={() => onOpenWorkflowRun(workflowRun.run_id)}
                  >
                    <ExternalLink className="h-3.5 w-3.5" />
                    Open workflow run
                  </Button>
                </div>
              ) : null}
            </ActivityCard>
          ) : null}

          {focusTask ? (
            <ActivityCard title="Active task" icon={<TerminalSquare className="h-3.5 w-3.5" />}>
              <div className="flex items-center justify-between gap-2">
                <p className="truncate text-sm font-medium text-foreground">{focusTask.task_id}</p>
                <Badge variant={statusVariant(focusTask.status)}>{focusTask.status}</Badge>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                {previewText(focusTask.input_message) ?? 'Task is running.'}
              </p>
              {recentExecSummary ? (
                <div className="mt-3 rounded-xl border border-border/70 bg-canvas/70 px-3 py-2">
                  <div className="mb-1 flex items-center gap-2 text-[11px] font-medium uppercase tracking-[0.12em] text-muted-foreground">
                    {recentExecSummary.stream === 'stderr' ? (
                      <AlertCircle className="h-3.5 w-3.5 text-rose-300" />
                    ) : recentExecSummary.stream === 'stdout' ? (
                      <CheckCircle2 className="h-3.5 w-3.5 text-emerald-300" />
                    ) : (
                      <TerminalSquare className="h-3.5 w-3.5 text-primary-300" />
                    )}
                    <span>{recentExecSummary.stream}</span>
                  </div>
                  <p className="whitespace-pre-wrap break-words text-xs text-foreground">
                    {recentExecSummary.text}
                  </p>
                </div>
              ) : null}
              {!recentExecSummary && focusTask.final_answer ? (
                <p className="mt-3 rounded-xl border border-border/70 bg-canvas/70 px-3 py-2 text-xs text-foreground">
                  {previewText(focusTask.final_answer, 200)}
                </p>
              ) : null}
              {!recentExecSummary && focusTask.error ? (
                <p className="mt-3 rounded-xl border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {previewText(focusTask.error, 200)}
                </p>
              ) : null}
            </ActivityCard>
          ) : null}
        </div>
      </div>
    </div>
  )
}
