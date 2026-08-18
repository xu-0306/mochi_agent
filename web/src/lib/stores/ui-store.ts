'use client'

import { create } from 'zustand'

interface UIStore {
  sidebarCollapsed: boolean
  workspacePanelOpen: boolean
  setSidebarCollapsed: (collapsed: boolean) => void
  setWorkspacePanelOpen: (open: boolean) => void
  toggleSidebar: () => void
  toggleWorkspacePanel: () => void
}

export const useUIStore = create<UIStore>((set) => ({
  sidebarCollapsed: false,
  // Keep the chat surface calm on first load; the workspace remains one click away.
  workspacePanelOpen: false,
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setWorkspacePanelOpen: (open) => set({ workspacePanelOpen: open }),
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  toggleWorkspacePanel: () =>
    set((state) => ({ workspacePanelOpen: !state.workspacePanelOpen })),
}))
