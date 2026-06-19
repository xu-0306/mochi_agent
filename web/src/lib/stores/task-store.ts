'use client'

import { create } from 'zustand'
import {
  cancelTask,
  type CommandRule,
  createTask,
  fetchApprovalExecSession,
  fetchApprovals,
  fetchTask,
  fetchTasks,
  previewWorkspacePatch,
  type PatchPreviewResult,
  resolveApproval,
  resumeTask,
  stopApprovalExecSession,
  type ApprovalExecSessionPayload,
  type ApprovalSummary,
  type CreateTaskInput,
  type TaskDetail,
  type TaskSummary,
} from '@/lib/api'

interface ApprovalExecSessionState {
  data: ApprovalExecSessionPayload | null
  isLoading: boolean
  isStopping: boolean
  error: string | null
}

interface ApprovalReviewState {
  originalPatchText: string | null
  patchText: string | null
  isEditingPatch: boolean
  isPreviewLoading: boolean
  preview: PatchPreviewResult | null
  previewError: string | null
  lastPreviewedPatchText: string | null
}

interface TaskStore {
  tasks: TaskSummary[]
  approvals: ApprovalSummary[]
  approvalExecSessions: Record<string, ApprovalExecSessionState>
  approvalReviewStates: Record<string, ApprovalReviewState>
  selectedTaskId: string | null
  selectedTaskDetail: TaskDetail | null
  isLoading: boolean
  isLoadingDetail: boolean
  isResolvingApprovalKey: string | null
  isMutatingTaskId: string | null
  error: string | null
  load: () => Promise<void>
  selectTask: (taskId: string) => Promise<void>
  create: (input: CreateTaskInput) => Promise<TaskSummary>
  resume: (taskId: string) => Promise<void>
  cancel: (taskId: string) => Promise<void>
  resolveApprovalDecision: (
    approvalId: string,
    decision: 'approve_once' | 'approve_and_save_rule' | 'reject',
    options?: {
      rule?: CommandRule
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
  stopApprovalExecSession: (approvalId: string) => Promise<void>
  reject: (approvalId: string) => Promise<void>
}

function upsertTask(tasks: TaskSummary[], next: TaskSummary): TaskSummary[] {
  const index = tasks.findIndex((item) => item.task_id === next.task_id)
  if (index === -1) {
    return [next, ...tasks]
  }
  return tasks.map((item, itemIndex) => (itemIndex === index ? next : item))
}

function reconcileApprovalExecSessions(
  current: Record<string, ApprovalExecSessionState>,
  approvals: ApprovalSummary[]
): Record<string, ApprovalExecSessionState> {
  const next: Record<string, ApprovalExecSessionState> = {}
  for (const approval of approvals) {
    if (!approval.exec_session_id) {
      continue
    }
    next[approval.approval_id] = current[approval.approval_id] ?? {
      data: null,
      isLoading: false,
      isStopping: false,
      error: null,
    }
  }
  return next
}

function buildApprovalReviewState(approval: ApprovalSummary): ApprovalReviewState {
  return {
    originalPatchText: approval.editable_patch_text,
    patchText: approval.editable_patch_text,
    isEditingPatch: false,
    isPreviewLoading: false,
    preview: null,
    previewError: null,
    lastPreviewedPatchText: null,
  }
}

function reconcileApprovalReviewStates(
  current: Record<string, ApprovalReviewState>,
  approvals: ApprovalSummary[]
): Record<string, ApprovalReviewState> {
  const next: Record<string, ApprovalReviewState> = {}
  for (const approval of approvals) {
    const existing = current[approval.approval_id]
    if (!existing) {
      next[approval.approval_id] = buildApprovalReviewState(approval)
      continue
    }

    next[approval.approval_id] = {
      ...existing,
      originalPatchText: approval.editable_patch_text,
      patchText:
        existing.patchText == null || existing.patchText === existing.originalPatchText
          ? approval.editable_patch_text
          : existing.patchText,
    }
  }
  return next
}

export const useTaskStore = create<TaskStore>((set, get) => ({
  tasks: [],
  approvals: [],
  approvalExecSessions: {},
  approvalReviewStates: {},
  selectedTaskId: null,
  selectedTaskDetail: null,
  isLoading: false,
  isLoadingDetail: false,
  isResolvingApprovalKey: null,
  isMutatingTaskId: null,
  error: null,

  load: async () => {
    if (get().isLoading) {
      return
    }
    set({ isLoading: true, error: null })
    try {
      const [tasks, approvals] = await Promise.all([fetchTasks(), fetchApprovals()])
      const selectedTaskId = get().selectedTaskId ?? tasks[0]?.task_id ?? null
      set({
        tasks,
        approvals,
        approvalExecSessions: reconcileApprovalExecSessions(get().approvalExecSessions, approvals),
        approvalReviewStates: reconcileApprovalReviewStates(get().approvalReviewStates, approvals),
        selectedTaskId,
        isLoading: false,
      })
      if (selectedTaskId) {
        await get().selectTask(selectedTaskId)
      }
    } catch (error) {
      set({
        isLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load tasks',
      })
    }
  },

  selectTask: async (taskId) => {
    set({ selectedTaskId: taskId, isLoadingDetail: true, error: null })
    try {
      const detail = await fetchTask(taskId)
      if (get().selectedTaskId !== taskId) {
        return
      }
      set({ selectedTaskDetail: detail, isLoadingDetail: false })
    } catch (error) {
      if (get().selectedTaskId !== taskId) {
        return
      }
      set({
        selectedTaskDetail: null,
        isLoadingDetail: false,
        error: error instanceof Error ? error.message : 'Failed to load task detail',
      })
    }
  },

  create: async (input) => {
    const created = await createTask(input)
    set((state) => ({
      tasks: upsertTask(state.tasks, created),
      selectedTaskId: created.task_id,
      error: null,
    }))
    await get().selectTask(created.task_id)
    return created
  },

  resume: async (taskId) => {
    set({ isMutatingTaskId: taskId, error: null })
    try {
      const updated = await resumeTask(taskId)
      set((state) => ({
        tasks: upsertTask(state.tasks, updated),
        isMutatingTaskId: null,
      }))
      if (get().selectedTaskId === taskId) {
        await get().selectTask(taskId)
      }
    } catch (error) {
      set({
        isMutatingTaskId: null,
        error: error instanceof Error ? error.message : 'Failed to resume task',
      })
    }
  },

  cancel: async (taskId) => {
    set({ isMutatingTaskId: taskId, error: null })
    try {
      const updated = await cancelTask(taskId)
      set((state) => ({
        tasks: upsertTask(state.tasks, updated),
        isMutatingTaskId: null,
      }))
      if (get().selectedTaskId === taskId) {
        await get().selectTask(taskId)
      }
    } catch (error) {
      set({
        isMutatingTaskId: null,
        error: error instanceof Error ? error.message : 'Failed to cancel task',
      })
    }
  },

  resolveApprovalDecision: async (approvalId, decision, options) => {
    const resolveKey = `${approvalId}:${decision}`
    set({ isResolvingApprovalKey: resolveKey, error: null })
    try {
      await resolveApproval(approvalId, {
        decision,
        ...(options?.rule ? { rule: options.rule } : {}),
        ...(options?.replayOverride ? { replayOverride: options.replayOverride } : {}),
      })
      const [tasks, approvals] = await Promise.all([fetchTasks(), fetchApprovals()])
      set({
        tasks,
        approvals,
        approvalExecSessions: reconcileApprovalExecSessions(get().approvalExecSessions, approvals),
        approvalReviewStates: reconcileApprovalReviewStates(get().approvalReviewStates, approvals),
        isResolvingApprovalKey: null,
      })
      const selectedTaskId = get().selectedTaskId
      if (selectedTaskId) {
        await get().selectTask(selectedTaskId)
      }
    } catch (error) {
      set({
        isResolvingApprovalKey: null,
        error: error instanceof Error ? error.message : 'Failed to resolve approval request',
      })
    }
  },

  reject: async (approvalId) => {
    await get().resolveApprovalDecision(approvalId, 'reject')
  },

  setApprovalPatchEditing: (approvalId, editing) => {
    set((state) => ({
      approvalReviewStates: {
        ...state.approvalReviewStates,
        [approvalId]: {
          ...(state.approvalReviewStates[approvalId] ?? {
            originalPatchText: null,
            patchText: null,
            isEditingPatch: false,
            isPreviewLoading: false,
            preview: null,
            previewError: null,
            lastPreviewedPatchText: null,
          }),
          isEditingPatch: editing,
        },
      },
    }))
  },

  setApprovalPatchText: (approvalId, patchText) => {
    set((state) => ({
      approvalReviewStates: {
        ...state.approvalReviewStates,
        [approvalId]: {
          ...(state.approvalReviewStates[approvalId] ?? {
            originalPatchText: null,
            patchText: null,
            isEditingPatch: true,
            isPreviewLoading: false,
            preview: null,
            previewError: null,
            lastPreviewedPatchText: null,
          }),
          patchText,
        },
      },
    }))
  },

  previewApprovalPatch: async (approvalId) => {
    const approval = get().approvals.find((item) => item.approval_id === approvalId)
    const reviewState = get().approvalReviewStates[approvalId]
    const patchText = reviewState?.patchText
    if (!approval || patchText == null) {
      return
    }

    set((state) => ({
      approvalReviewStates: {
        ...state.approvalReviewStates,
        [approvalId]: {
          ...(state.approvalReviewStates[approvalId] ?? buildApprovalReviewState(approval)),
          isPreviewLoading: true,
          previewError: null,
        },
      },
      error: null,
    }))

    try {
      const preview = await previewWorkspacePatch({
        approvalId,
        patchText,
      })
      if (get().approvalReviewStates[approvalId]?.patchText !== patchText) {
        return
      }
      set((state) => ({
        approvalReviewStates: {
          ...state.approvalReviewStates,
          [approvalId]: {
            ...(state.approvalReviewStates[approvalId] ?? buildApprovalReviewState(approval)),
            isPreviewLoading: false,
            preview,
            previewError: null,
            lastPreviewedPatchText: patchText,
          },
        },
      }))
    } catch (error) {
      if (get().approvalReviewStates[approvalId]?.patchText !== patchText) {
        return
      }
      set((state) => ({
        approvalReviewStates: {
          ...state.approvalReviewStates,
          [approvalId]: {
            ...(state.approvalReviewStates[approvalId] ?? buildApprovalReviewState(approval)),
            isPreviewLoading: false,
            previewError: error instanceof Error ? error.message : 'Failed to preview patch',
            lastPreviewedPatchText: patchText,
          },
        },
      }))
    }
  },

  resetApprovalPatch: (approvalId) => {
    const approval = get().approvals.find((item) => item.approval_id === approvalId)
    if (!approval) {
      return
    }
    set((state) => ({
      approvalReviewStates: {
        ...state.approvalReviewStates,
        [approvalId]: buildApprovalReviewState(approval),
      },
    }))
  },

  refreshApprovalExecSession: async (approvalId, yieldTimeMs = 250) => {
    set((state) => ({
      approvalExecSessions: {
        ...state.approvalExecSessions,
        [approvalId]: {
          data: state.approvalExecSessions[approvalId]?.data ?? null,
          isLoading: true,
          isStopping: state.approvalExecSessions[approvalId]?.isStopping ?? false,
          error: null,
        },
      },
      error: null,
    }))
    try {
      const payload = await fetchApprovalExecSession(approvalId, { yield_time_ms: yieldTimeMs })
      set((state) => ({
        approvalExecSessions: {
          ...state.approvalExecSessions,
          [approvalId]: {
            data: payload,
            isLoading: false,
            isStopping: false,
            error: null,
          },
        },
      }))
    } catch (error) {
      set((state) => ({
        approvalExecSessions: {
          ...state.approvalExecSessions,
          [approvalId]: {
            data: state.approvalExecSessions[approvalId]?.data ?? null,
            isLoading: false,
            isStopping: state.approvalExecSessions[approvalId]?.isStopping ?? false,
            error: error instanceof Error ? error.message : 'Failed to refresh exec session',
          },
        },
      }))
    }
  },

  stopApprovalExecSession: async (approvalId) => {
    set((state) => ({
      approvalExecSessions: {
        ...state.approvalExecSessions,
        [approvalId]: {
          data: state.approvalExecSessions[approvalId]?.data ?? null,
          isLoading: state.approvalExecSessions[approvalId]?.isLoading ?? false,
          isStopping: true,
          error: null,
        },
      },
      error: null,
    }))
    try {
      const payload = await stopApprovalExecSession(approvalId)
      set((state) => ({
        approvalExecSessions: {
          ...state.approvalExecSessions,
          [approvalId]: {
            data: payload,
            isLoading: false,
            isStopping: false,
            error: null,
          },
        },
      }))
    } catch (error) {
      set((state) => ({
        approvalExecSessions: {
          ...state.approvalExecSessions,
          [approvalId]: {
            data: state.approvalExecSessions[approvalId]?.data ?? null,
            isLoading: state.approvalExecSessions[approvalId]?.isLoading ?? false,
            isStopping: false,
            error: error instanceof Error ? error.message : 'Failed to stop exec session',
          },
        },
      }))
    }
  },
}))
