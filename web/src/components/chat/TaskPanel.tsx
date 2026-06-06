'use client'

import * as React from 'react'
import { PanelRightClose } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import {
  Sheet,
  SheetContent,
  SheetDescription,
  SheetHeader,
  SheetTitle,
} from '@/components/ui/sheet'
import type { ApprovalSummary, TaskDetail, TaskSummary } from '@/lib/api'
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
  onClose?: () => void
}) {
  const pendingApprovals = approvals.filter((item) => item.status === 'pending')

  return (
    <div className="flex h-full flex-col bg-surface-layer">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Tasks</h2>
          <p className="text-xs text-muted-foreground">Background runs & approvals</p>
        </div>
        <div className="flex items-center gap-1.5">
          <Button size="sm" variant="ghost" onClick={() => void load()} loading={isLoading}>
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
              className="hidden lg:inline-flex"
            >
              <PanelRightClose className="h-4 w-4" />
            </Button>
          ) : null}
        </div>
      </div>

      <div className="border-b border-border px-4 py-3">
        <p className="mb-2 text-xs font-medium text-muted-foreground">Pending approvals</p>
        <div className="max-h-44 space-y-2 overflow-y-auto">
          {pendingApprovals.length === 0 ? (
            <p className="text-xs text-muted-foreground">No pending approvals.</p>
          ) : (
            pendingApprovals.map((approval) => (
              <div key={approval.approval_id} className="rounded-md border border-border bg-canvas px-2.5 py-2">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <span className="truncate text-xs font-medium text-foreground">{approval.tool_name}</span>
                  <Badge variant={statusVariant(approval.status)}>{approval.status}</Badge>
                </div>
                <p className="mb-2 text-[11px] text-muted-foreground">Task: {approval.task_id}</p>
                <div className="mb-2 space-y-1 text-[11px] text-muted-foreground">
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
                    className="h-7 px-2 text-xs"
                    loading={isResolvingApprovalId === approval.approval_id}
                    onClick={() => void approve(approval.approval_id)}
                  >
                    Approve
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    className="h-7 px-2 text-xs"
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
      </div>

      <div className="grid min-h-0 flex-1 grid-rows-[minmax(0,1fr)_minmax(0,1fr)]">
        <div className="min-h-0 border-b border-border px-4 py-3">
          <p className="mb-2 text-xs font-medium text-muted-foreground">Recent tasks</p>
          <div className="h-full space-y-2 overflow-y-auto">
            {tasks.length === 0 ? (
              <p className="text-xs text-muted-foreground">No tasks yet.</p>
            ) : (
              tasks.map((task) => (
                <button
                  key={task.task_id}
                  type="button"
                  onClick={() => void selectTask(task.task_id)}
                  className={cn(
                    'w-full rounded-md border px-2.5 py-2 text-left',
                    selectedTaskId === task.task_id
                      ? 'border-primary/40 bg-primary/5'
                      : 'border-border bg-canvas hover:bg-muted/40'
                  )}
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="truncate text-xs font-medium text-foreground">{task.task_id}</span>
                    <Badge variant={statusVariant(task.status)}>{task.status}</Badge>
                  </div>
                  {task.task_type ? (
                    <p className="mb-1 text-[11px] text-muted-foreground">{formatMetadataLabel(task.task_type)}</p>
                  ) : null}
                  <p className="text-[11px] text-muted-foreground">{previewText(task.input_message)}</p>
                </button>
              ))
            )}
          </div>
        </div>

        <div className="min-h-0 px-4 py-3">
          <p className="mb-2 text-xs font-medium text-muted-foreground">Task detail</p>
          {isLoadingDetail ? (
            <p className="text-xs text-muted-foreground">Loading detail...</p>
          ) : selectedTaskDetail ? (
            <div className="h-full space-y-2 overflow-y-auto rounded-md border border-border bg-canvas px-2.5 py-2 text-xs">
              <div className="flex items-center justify-between gap-2">
                <span className="font-medium text-foreground">{selectedTaskDetail.task_id}</span>
                <Badge variant={statusVariant(selectedTaskDetail.status)}>{selectedTaskDetail.status}</Badge>
              </div>
              <p className="text-muted-foreground">Created: {formatTime(selectedTaskDetail.created_at)}</p>
              <p className="text-muted-foreground">Updated: {formatTime(selectedTaskDetail.updated_at)}</p>
              <p className="text-muted-foreground">Events: {selectedTaskDetail.events.length}</p>
              {selectedTaskDetail.error ? (
                <p className="rounded border border-destructive/30 bg-destructive/10 px-2 py-1 text-destructive">
                  {selectedTaskDetail.error}
                </p>
              ) : null}
              {selectedTaskDetail.final_answer ? (
                <p className="text-foreground">{previewText(selectedTaskDetail.final_answer, 200)}</p>
              ) : null}
              <div className="flex gap-1.5 pt-1">
                <Button
                  size="sm"
                  variant="secondary"
                  className="h-7 px-2 text-xs"
                  loading={isMutatingTaskId === selectedTaskDetail.task_id}
                  onClick={() => void resume(selectedTaskDetail.task_id)}
                >
                  Resume
                </Button>
                <Button
                  size="sm"
                  variant="outline"
                  className="h-7 px-2 text-xs"
                  loading={isMutatingTaskId === selectedTaskDetail.task_id}
                  onClick={() => void cancel(selectedTaskDetail.task_id)}
                >
                  Cancel
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-xs text-muted-foreground">Select a task to preview detail.</p>
          )}
          {error ? <p className="mt-2 text-xs text-destructive">{error}</p> : null}
        </div>
      </div>
    </div>
  )
}

export function TaskPanel({ open, onOpenChange }: TaskPanelProps) {
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
  const [isDesktop, setIsDesktop] = React.useState(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return false
    }
    return window.matchMedia('(min-width: 1024px)').matches
  })

  React.useEffect(() => {
    if (typeof window === 'undefined' || typeof window.matchMedia !== 'function') {
      return
    }
    const mediaQuery = window.matchMedia('(min-width: 1024px)')
    const syncDesktopState = () => {
      setIsDesktop(mediaQuery.matches)
    }
    syncDesktopState()
    mediaQuery.addEventListener('change', syncDesktopState)
    return () => mediaQuery.removeEventListener('change', syncDesktopState)
  }, [])

  React.useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 8000)
    return () => window.clearInterval(timer)
  }, [load])

  return (
    <>
      <aside
        className={cn(
          'hidden border-l border-border bg-surface-layer lg:flex lg:h-full lg:flex-col',
          open ? 'lg:w-80' : 'lg:w-0 lg:overflow-hidden lg:border-l-0',
          'transition-all duration-200'
        )}
      >
        {open ? (
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
            onClose={() => onOpenChange(false)}
          />
        ) : null}
      </aside>

      {!isDesktop ? (
        <Sheet open={open} onOpenChange={onOpenChange}>
          <SheetContent side="right" className="w-full max-w-md p-0 lg:hidden">
            <SheetHeader className="sr-only">
              <SheetTitle>Tasks</SheetTitle>
              <SheetDescription>Background runs and pending approvals.</SheetDescription>
            </SheetHeader>
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
            />
          </SheetContent>
        </Sheet>
      ) : null}
    </>
  )
}
