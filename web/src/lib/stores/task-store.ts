'use client'

import { create } from 'zustand'
import {
  cancelTask,
  createTask,
  fetchApprovals,
  fetchTask,
  fetchTasks,
  resolveApproval,
  resumeTask,
  type ApprovalSummary,
  type CreateTaskInput,
  type TaskDetail,
  type TaskSummary,
} from '@/lib/api'

interface TaskStore {
  tasks: TaskSummary[]
  approvals: ApprovalSummary[]
  selectedTaskId: string | null
  selectedTaskDetail: TaskDetail | null
  isLoading: boolean
  isLoadingDetail: boolean
  isResolvingApprovalId: string | null
  isMutatingTaskId: string | null
  error: string | null
  load: () => Promise<void>
  selectTask: (taskId: string) => Promise<void>
  create: (input: CreateTaskInput) => Promise<TaskSummary>
  resume: (taskId: string) => Promise<void>
  cancel: (taskId: string) => Promise<void>
  approve: (approvalId: string) => Promise<void>
  reject: (approvalId: string) => Promise<void>
}

function upsertTask(tasks: TaskSummary[], next: TaskSummary): TaskSummary[] {
  const index = tasks.findIndex((item) => item.task_id === next.task_id)
  if (index === -1) {
    return [next, ...tasks]
  }
  return tasks.map((item, itemIndex) => (itemIndex === index ? next : item))
}

export const useTaskStore = create<TaskStore>((set, get) => ({
  tasks: [],
  approvals: [],
  selectedTaskId: null,
  selectedTaskDetail: null,
  isLoading: false,
  isLoadingDetail: false,
  isResolvingApprovalId: null,
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

  approve: async (approvalId) => {
    set({ isResolvingApprovalId: approvalId, error: null })
    try {
      await resolveApproval(approvalId, { decision: 'approve' })
      const [tasks, approvals] = await Promise.all([fetchTasks(), fetchApprovals()])
      set({
        tasks,
        approvals,
        isResolvingApprovalId: null,
      })
      const selectedTaskId = get().selectedTaskId
      if (selectedTaskId) {
        await get().selectTask(selectedTaskId)
      }
    } catch (error) {
      set({
        isResolvingApprovalId: null,
        error: error instanceof Error ? error.message : 'Failed to approve request',
      })
    }
  },

  reject: async (approvalId) => {
    set({ isResolvingApprovalId: approvalId, error: null })
    try {
      await resolveApproval(approvalId, { decision: 'reject' })
      const [tasks, approvals] = await Promise.all([fetchTasks(), fetchApprovals()])
      set({
        tasks,
        approvals,
        isResolvingApprovalId: null,
      })
      const selectedTaskId = get().selectedTaskId
      if (selectedTaskId) {
        await get().selectTask(selectedTaskId)
      }
    } catch (error) {
      set({
        isResolvingApprovalId: null,
        error: error instanceof Error ? error.message : 'Failed to reject request',
      })
    }
  },
}))
