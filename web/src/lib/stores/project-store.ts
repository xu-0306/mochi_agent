'use client'

import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import {
  createProject as createProjectApi,
  deleteProject as deleteProjectApi,
  fetchProjects,
  type ProjectSummary,
  updateProject as updateProjectApi,
} from '@/lib/api'

interface ProjectStore {
  projects: ProjectSummary[]
  activeProjectId: string | null
  expandedProjectIds: string[]
  isLoadingProjects: boolean
  hasLoadedProjects: boolean
  error: string | null
  loadProjects: () => Promise<void>
  setActiveProjectId: (projectId: string | null) => void
  toggleProjectExpanded: (projectId: string) => void
  createProject: (input: { name: string; workspaceDir: string }) => Promise<ProjectSummary>
  updateProject: (
    projectId: string,
    input: { name?: string; workspaceDir?: string }
  ) => Promise<ProjectSummary>
  deleteProject: (projectId: string) => Promise<void>
}

export const useProjectStore = create<ProjectStore>()(
  persist(
    (set, get) => ({
      projects: [],
      activeProjectId: null,
      expandedProjectIds: [],
      isLoadingProjects: false,
      hasLoadedProjects: false,
      error: null,

      loadProjects: async () => {
        if (get().isLoadingProjects) {
          return
        }

        set({ isLoadingProjects: true, error: null })
        try {
          const projects = await fetchProjects()
          set((state) => ({
            projects,
            activeProjectId:
              state.activeProjectId && projects.some((project) => project.id === state.activeProjectId)
                ? state.activeProjectId
                : null,
            expandedProjectIds: state.expandedProjectIds.filter((projectId) =>
              projects.some((project) => project.id === projectId)
            ),
            isLoadingProjects: false,
            hasLoadedProjects: true,
            error: null,
          }))
        } catch (error: unknown) {
          set({
            isLoadingProjects: false,
            hasLoadedProjects: true,
            error: error instanceof Error ? error.message : 'Failed to load projects',
          })
        }
      },

      setActiveProjectId: (projectId) => set({ activeProjectId: projectId }),

      toggleProjectExpanded: (projectId) =>
        set((state) => ({
          expandedProjectIds: state.expandedProjectIds.includes(projectId)
            ? state.expandedProjectIds.filter((id) => id !== projectId)
            : [...state.expandedProjectIds, projectId],
        })),

      createProject: async (input) => {
        const project = await createProjectApi(input)
        set((state) => ({
          projects: [project, ...state.projects],
          activeProjectId: project.id,
          expandedProjectIds: state.expandedProjectIds.includes(project.id)
            ? state.expandedProjectIds
            : [project.id, ...state.expandedProjectIds],
          error: null,
        }))
        return project
      },

      updateProject: async (projectId, input) => {
        const project = await updateProjectApi(projectId, input)
        set((state) => ({
          projects: state.projects.map((item) => (item.id === projectId ? project : item)),
          error: null,
        }))
        return project
      },

      deleteProject: async (projectId) => {
        await deleteProjectApi(projectId)
        set((state) => ({
          projects: state.projects.filter((project) => project.id !== projectId),
          activeProjectId: state.activeProjectId === projectId ? null : state.activeProjectId,
          expandedProjectIds: state.expandedProjectIds.filter((id) => id !== projectId),
          error: null,
        }))
      },
    }),
    {
      name: 'mochi.project-ui.v1',
      partialize: (state) => ({
        activeProjectId: state.activeProjectId,
        expandedProjectIds: state.expandedProjectIds,
      }),
    }
  )
)
