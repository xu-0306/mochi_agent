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
  workspacePanelOpen: true,
  setSidebarCollapsed: (collapsed) => set({ sidebarCollapsed: collapsed }),
  setWorkspacePanelOpen: (open) => set({ workspacePanelOpen: open }),
  toggleSidebar: () =>
    set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
  toggleWorkspacePanel: () =>
    set((state) => ({ workspacePanelOpen: !state.workspacePanelOpen })),
}))
