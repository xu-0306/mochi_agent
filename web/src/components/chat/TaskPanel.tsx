'use client'

import * as React from 'react'
import { Activity, PanelRightClose } from 'lucide-react'
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

function SectionCard({
  title,
  description,
  children,
}: {
  title: string
  description?: string
  children: React.ReactNode
}) {
  return (
    <section className="rounded-2xl border border-white/8 bg-canvas/70 p-4 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] backdrop-blur-sm">
      <div className="mb-3">
        <h3 className="text-sm font-semibold text-foreground">{title}</h3>
        {description ? (
          <p className="mt-1 text-[11px] leading-relaxed text-muted-foreground">{description}</p>
        ) : null}
      </div>
      {children}
    </section>
  )
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
          <SectionCard
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
          </SectionCard>

          <SectionCard
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
          </SectionCard>

          <SectionCard
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
          </SectionCard>
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
          'absolute right-3 top-3 bottom-3 z-30 hidden w-[23rem] overflow-hidden rounded-[28px] border border-white/10 bg-surface-layer/92 shadow-[0_28px_80px_rgba(0,0,0,0.45)] backdrop-blur-xl transition-all duration-300 ease-out-smooth lg:flex lg:flex-col',
          open
            ? 'pointer-events-auto translate-x-0 opacity-100'
            : 'pointer-events-none translate-x-8 opacity-0'
        )}
        aria-hidden={!open}
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
