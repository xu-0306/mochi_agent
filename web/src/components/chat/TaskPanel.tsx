'use client'

import * as React from 'react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { useTaskStore } from '@/lib/stores/task-store'

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

export function TaskPanel() {
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

  React.useEffect(() => {
    void load()
    const timer = window.setInterval(() => {
      void load()
    }, 8000)
    return () => window.clearInterval(timer)
  }, [load])

  return (
    <aside className="hidden w-80 border-l border-border bg-surface-layer lg:flex lg:h-full lg:flex-col">
      <div className="flex items-center justify-between border-b border-border px-4 py-3">
        <div>
          <h2 className="text-sm font-semibold text-foreground">Tasks</h2>
          <p className="text-xs text-muted-foreground">Background runs & approvals</p>
        </div>
        <Button size="sm" variant="ghost" onClick={() => void load()} loading={isLoading}>
          Refresh
        </Button>
      </div>

      <div className="border-b border-border px-4 py-3">
        <p className="mb-2 text-xs font-medium text-muted-foreground">Pending approvals</p>
        <div className="max-h-44 space-y-2 overflow-y-auto">
          {approvals.filter((item) => item.status === 'pending').length === 0 ? (
            <p className="text-xs text-muted-foreground">No pending approvals.</p>
          ) : (
            approvals
              .filter((item) => item.status === 'pending')
              .map((approval) => (
                <div key={approval.approval_id} className="rounded-md border border-border bg-canvas px-2.5 py-2">
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="truncate text-xs font-medium text-foreground">{approval.tool_name}</span>
                    <Badge variant={statusVariant(approval.status)}>{approval.status}</Badge>
                  </div>
                  <p className="mb-2 text-[11px] text-muted-foreground">Task: {approval.task_id}</p>
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
                  className={[
                    'w-full rounded-md border px-2.5 py-2 text-left',
                    selectedTaskId === task.task_id
                      ? 'border-primary/40 bg-primary/5'
                      : 'border-border bg-canvas hover:bg-muted/40',
                  ].join(' ')}
                >
                  <div className="mb-1 flex items-center justify-between gap-2">
                    <span className="truncate text-xs font-medium text-foreground">{task.task_id}</span>
                    <Badge variant={statusVariant(task.status)}>{task.status}</Badge>
                  </div>
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
    </aside>
  )
}
