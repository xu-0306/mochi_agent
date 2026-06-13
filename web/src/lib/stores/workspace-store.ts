'use client'

import { create } from 'zustand'
import {
  fetchWorkspaceChanges,
  fetchWorkspaceDiff,
  fetchWorkspacePreview,
  fetchWorkspaceTree,
  type WorkspaceChange,
  type WorkspaceDiffResult,
  type WorkspacePreviewResult,
  type WorkspaceTreeItem,
} from '@/lib/api'

interface WorkspaceContext {
  sessionId: string | null
  projectId: string | null
}

interface WorkspaceStore extends WorkspaceContext {
  workspaceDir: string | null
  currentPath: string | null
  currentRelativePath: string
  parentPath: string | null
  selectedPath: string | null
  selectedFilePath: string | null
  items: WorkspaceTreeItem[]
  changes: WorkspaceChange[]
  preview: WorkspacePreviewResult | null
  diff: WorkspaceDiffResult | null
  isTreeLoading: boolean
  isPreviewLoading: boolean
  isChangesLoading: boolean
  isDiffLoading: boolean
  error: string | null
  setContext: (context: WorkspaceContext) => void
  loadTree: (path?: string | null) => Promise<void>
  previewFile: (path: string) => Promise<void>
  loadChanges: (path?: string | null) => Promise<void>
  loadDiff: (path: string) => Promise<void>
  clearPreview: () => void
}

function sameContext(a: WorkspaceContext, b: WorkspaceContext): boolean {
  return a.sessionId === b.sessionId && a.projectId === b.projectId
}

export const useWorkspaceStore = create<WorkspaceStore>((set, get) => ({
  sessionId: null,
  projectId: null,
  workspaceDir: null,
  currentPath: null,
  currentRelativePath: '.',
  parentPath: null,
  selectedPath: null,
  selectedFilePath: null,
  items: [],
  changes: [],
  preview: null,
  diff: null,
  isTreeLoading: false,
  isPreviewLoading: false,
  isChangesLoading: false,
  isDiffLoading: false,
  error: null,

  setContext: (context) => {
    if (sameContext(get(), context)) {
      return
    }
    set({
      sessionId: context.sessionId,
      projectId: context.projectId,
      workspaceDir: null,
      currentPath: null,
      currentRelativePath: '.',
      parentPath: null,
      selectedPath: null,
      selectedFilePath: null,
      items: [],
      changes: [],
      preview: null,
      diff: null,
      error: null,
    })
  },

  loadTree: async (path) => {
    const { sessionId, projectId } = get()
    set({ isTreeLoading: true, error: null })
    try {
      const result = await fetchWorkspaceTree({
        sessionId,
        projectId,
        path: path ?? undefined,
      })
      set({
        workspaceDir: result.workspaceDir,
        currentPath: result.path,
        currentRelativePath: result.relativePath,
        parentPath: result.parent,
        selectedPath: result.selectedPath,
        items: result.items,
        isTreeLoading: false,
      })
    } catch (error) {
      set({
        isTreeLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load workspace tree.',
      })
    }
  },

  previewFile: async (path) => {
    const { sessionId, projectId } = get()
    set({ isPreviewLoading: true, error: null, diff: null, selectedFilePath: path })
    try {
      const result = await fetchWorkspacePreview(path, { sessionId, projectId })
      set({
        workspaceDir: result.workspaceDir,
        preview: result,
        isPreviewLoading: false,
      })
    } catch (error) {
      set({
        isPreviewLoading: false,
        preview: null,
        selectedFilePath: null,
        error: error instanceof Error ? error.message : 'Failed to preview workspace file.',
      })
    }
  },

  loadChanges: async (path) => {
    const { sessionId, projectId } = get()
    set({ isChangesLoading: true, error: null })
    try {
      const result = await fetchWorkspaceChanges({
        sessionId,
        projectId,
        path: path ?? undefined,
      })
      set({
        workspaceDir: result.workspaceDir,
        changes: result.items,
        isChangesLoading: false,
      })
    } catch (error) {
      set({
        isChangesLoading: false,
        error: error instanceof Error ? error.message : 'Failed to load workspace changes.',
      })
    }
  },

  loadDiff: async (path) => {
    const { sessionId, projectId } = get()
    set({ isDiffLoading: true, error: null, preview: null, selectedFilePath: path })
    try {
      const result = await fetchWorkspaceDiff(path, { sessionId, projectId })
      set({
        diff: result,
        isDiffLoading: false,
      })
    } catch (error) {
      set({
        isDiffLoading: false,
        diff: null,
        selectedFilePath: null,
        error: error instanceof Error ? error.message : 'Failed to load workspace diff.',
      })
    }
  },

  clearPreview: () => set({ preview: null, diff: null, selectedFilePath: null }),
}))
