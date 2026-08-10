'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import {
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Cloud,
  FolderPlus,
  FolderOpen,
  Library,
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  Trash2,
  UserRound,
  Waypoints,
  Workflow,
  Zap,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Input } from '@/components/ui/input'
import * as api from '@/lib/api'
import { useI18n } from '@/lib/i18n'
import { useProjectStore } from '@/lib/stores/project-store'
import {
  type Session,
  useSessionStore,
} from '@/lib/stores/session-store'
import { useUIStore } from '@/lib/stores/ui-store'
import { SessionItem } from './SessionItem'

interface ProjectDialogState {
  mode: 'create' | 'edit'
  projectId: string | null
  name: string
  workspaceDir: string
}

function basename(path: string): string {
  const normalized = path.replace(/[\\/]+$/, '')
  if (!normalized) {
    return path
  }
  const parts = normalized.split(/[\\/]/)
  return parts[parts.length - 1] ?? path
}

function groupSessionsByProject(sessions: Session[]) {
  const draftSessions = sessions.filter((session) => session.isDraft)
  const persistedSessions = sessions.filter((session) => !session.isDraft)

  return {
    drafts: draftSessions,
    unassigned: persistedSessions.filter((session) => session.projectId === null),
    byProject: persistedSessions.reduce<Record<string, Session[]>>((acc, session) => {
      if (!session.projectId) {
        return acc
      }
      acc[session.projectId] = [...(acc[session.projectId] ?? []), session]
      return acc
    }, {}),
  }
}

function sortSessions(sessions: Session[]): Session[] {
  return [...sessions].sort((a, b) => b.lastMessageAt.getTime() - a.lastMessageAt.getTime())
}

export function Sidebar() {
  const router = useRouter()
  const { t } = useI18n()
  const collapsed = useUIStore((state) => state.sidebarCollapsed)
  const setSidebarCollapsed = useUIStore((state) => state.setSidebarCollapsed)
  const [search, setSearch] = React.useState('')
  const [selectionMode, setSelectionMode] = React.useState(false)
  const [selectedSessionIds, setSelectedSessionIds] = React.useState<string[]>([])
  const [pendingBulkDeleteIds, setPendingBulkDeleteIds] = React.useState<string[]>([])
  const [isBulkDeleting, setIsBulkDeleting] = React.useState(false)
  const [pendingDeleteSession, setPendingDeleteSession] = React.useState<Session | null>(null)
  const [pendingDeleteProjectId, setPendingDeleteProjectId] = React.useState<string | null>(null)
  const [projectDialog, setProjectDialog] = React.useState<ProjectDialogState | null>(null)
  const [projectDirectoryError, setProjectDirectoryError] = React.useState<string | null>(null)
  const [isSelectingProjectDirectory, setIsSelectingProjectDirectory] = React.useState(false)
  const [mobileNavOpen, setMobileNavOpen] = React.useState(false)
  const [mobileNavTarget, setMobileNavTarget] = React.useState<'search' | 'projects'>('search')
  const mobileNavDialogRef = React.useRef<HTMLElement>(null)
  const mobileNavCloseRef = React.useRef<HTMLButtonElement>(null)
  const mobileNavReturnFocusRef = React.useRef<HTMLElement | null>(null)

  const {
    sessions,
    currentSessionId,
    hasLoaded,
    loadSessions,
    setCurrentSession,
    createDraftSession,
    renameSession,
    deleteSession,
    moveSessionToProject,
  } = useSessionStore()
  const {
    projects,
    activeProjectId,
    expandedProjectIds,
    hasLoadedProjects,
    isLoadingProjects,
    loadProjects,
    setActiveProjectId,
    toggleProjectExpanded,
    createProject,
    updateProject,
    deleteProject,
  } = useProjectStore()

  React.useEffect(() => {
    if (!hasLoaded) {
      void loadSessions()
    }
  }, [hasLoaded, loadSessions])

  React.useEffect(() => {
    if (!hasLoadedProjects && !isLoadingProjects) {
      void loadProjects()
    }
  }, [hasLoadedProjects, isLoadingProjects, loadProjects])

  React.useEffect(() => {
    const handleMobileNavigation = (event: Event) => {
      const target = (event as CustomEvent<'search' | 'projects'>).detail
      if (target !== 'search' && target !== 'projects') {
        return
      }
      mobileNavReturnFocusRef.current = document.activeElement instanceof HTMLElement
        ? document.activeElement
        : null
      setMobileNavTarget(target)
      setMobileNavOpen(true)
    }
    window.addEventListener('mochi:open-mobile-navigation', handleMobileNavigation)
    return () => window.removeEventListener('mochi:open-mobile-navigation', handleMobileNavigation)
  }, [])

  const closeMobileNavigation = React.useCallback(() => {
    setMobileNavOpen(false)
    window.requestAnimationFrame(() => {
      const returnTarget = mobileNavReturnFocusRef.current
      const fallbackTarget = document.querySelector<HTMLElement>('[aria-controls="mobile-workspace-tools"]')
      ;(returnTarget?.isConnected ? returnTarget : fallbackTarget)?.focus()
    })
  }, [])

  React.useEffect(() => {
    if (!mobileNavOpen) {
      return
    }

    mobileNavCloseRef.current?.focus()
    const handleMobileNavigationKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault()
        closeMobileNavigation()
        return
      }
      if (event.key !== 'Tab') {
        return
      }

      const dialog = mobileNavDialogRef.current
      const focusable = dialog
        ? Array.from(dialog.querySelectorAll<HTMLElement>('button:not([disabled]), input:not([disabled]), [href], [tabindex]:not([tabindex="-1"])'))
        : []
      if (focusable.length === 0) {
        return
      }

      const first = focusable[0]
      const last = focusable[focusable.length - 1]
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault()
        last?.focus()
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault()
        first?.focus()
      }
    }

    document.addEventListener('keydown', handleMobileNavigationKeyDown)
    return () => document.removeEventListener('keydown', handleMobileNavigationKeyDown)
  }, [closeMobileNavigation, mobileNavOpen])

  const normalizedSearch = search.trim().toLowerCase()
  const visibleSessions = normalizedSearch
    ? sessions.filter((session) => {
        const haystacks = [session.title, session.lastMessage]
        return haystacks.some((value) => value.toLowerCase().includes(normalizedSearch))
      })
    : sessions

  const { drafts, unassigned, byProject } = React.useMemo(
    () => groupSessionsByProject(visibleSessions),
    [visibleSessions]
  )
  const visibleSessionIds = React.useMemo(() => {
    const ids: string[] = []
    ids.push(...drafts.map((session) => session.id))
    ids.push(...unassigned.map((session) => session.id))
    for (const projectId of expandedProjectIds) {
      ids.push(...(byProject[projectId] ?? []).map((session) => session.id))
    }
    return ids
  }, [byProject, drafts, expandedProjectIds, unassigned])
  const selectedVisibleCount = React.useMemo(
    () => visibleSessionIds.filter((id) => selectedSessionIds.includes(id)).length,
    [selectedSessionIds, visibleSessionIds]
  )
  const allVisibleSelected =
    visibleSessionIds.length > 0 && selectedVisibleCount === visibleSessionIds.length

  React.useEffect(() => {
    if (collapsed && selectionMode) {
      setSelectionMode(false)
      setSelectedSessionIds([])
    }
  }, [collapsed, selectionMode])

  const handleNewSession = () => {
    const draftId = createDraftSession(activeProjectId)
    void draftId
    router.push('/')
  }

  const handleSelectSession = (id: string) => {
    setCurrentSession(id)
    router.push('/')
  }

  const handleToggleSelectionMode = () => {
    setSelectionMode((current) => {
      if (current) {
        setSelectedSessionIds([])
      }
      return !current
    })
  }

  const handleToggleSessionSelected = (id: string) => {
    setSelectedSessionIds((current) =>
      current.includes(id)
        ? current.filter((item) => item !== id)
        : [...current, id]
    )
  }

  const handleToggleSelectAllVisible = () => {
    if (allVisibleSelected) {
      setSelectedSessionIds((current) => current.filter((id) => !visibleSessionIds.includes(id)))
      return
    }

    setSelectedSessionIds((current) => {
      const next = new Set(current)
      for (const id of visibleSessionIds) {
        next.add(id)
      }
      return [...next]
    })
  }

  const handleConfirmDeleteSession = () => {
    if (!pendingDeleteSession) {
      return
    }
    const sessionId = pendingDeleteSession.id
    setPendingDeleteSession(null)
    void deleteSession(sessionId)
  }

  const handleConfirmBulkDelete = async () => {
    if (pendingBulkDeleteIds.length === 0) {
      return
    }

    const ids = [...pendingBulkDeleteIds]
    setPendingBulkDeleteIds([])
    setIsBulkDeleting(true)
    try {
      for (const id of ids) {
        await deleteSession(id)
      }
      await loadSessions()
      setSelectedSessionIds([])
      setSelectionMode(false)
    } finally {
      setIsBulkDeleting(false)
    }
  }

  const handleConfirmDeleteProject = () => {
    if (!pendingDeleteProjectId) {
      return
    }
    const projectId = pendingDeleteProjectId
    setPendingDeleteProjectId(null)
    void deleteProject(projectId)
  }

  const openCreateProject = () => {
    setProjectDirectoryError(null)
    setProjectDialog({
      mode: 'create',
      projectId: null,
      name: '',
      workspaceDir: '',
    })
  }

  const openEditProject = (projectId: string) => {
    const project = projects.find((item) => item.id === projectId)
    if (!project) {
      return
    }
    setProjectDirectoryError(null)
    setProjectDialog({
      mode: 'edit',
      projectId,
      name: project.name,
      workspaceDir: project.workspaceDir,
    })
  }

  const updateProjectWorkspaceDir = (workspaceDir: string, options?: { inferName?: boolean }) => {
    setProjectDirectoryError(null)
    setProjectDialog((state) => {
      if (!state) {
        return state
      }
      const inferredName = options?.inferName && state.name.trim() === ''
        ? basename(workspaceDir)
        : state.name
      return {
        ...state,
        name: inferredName,
        workspaceDir,
      }
    })
  }

  const handleSelectProjectDirectory = async () => {
    const initialPath = projectDialog?.workspaceDir.trim() || undefined
    setProjectDirectoryError(null)
    setIsSelectingProjectDirectory(true)
    try {
      const result = await api.selectFilesystemDirectory({
        initialPath,
        title: 'Select Project Root',
      })
      if (result.selected && result.path) {
        updateProjectWorkspaceDir(result.path, { inferName: true })
      }
    } catch (error) {
      setProjectDirectoryError(
        error instanceof Error
          ? error.message
          : t('sidebar.projectDirectoryPickerFailed')
      )
    } finally {
      setIsSelectingProjectDirectory(false)
    }
  }

  const submitProjectDialog = async () => {
    const dialog = projectDialog
    if (!dialog) {
      return
    }
    const name = dialog.name.trim()
    const workspaceDir = dialog.workspaceDir.trim()
    if (!name || !workspaceDir) {
      return
    }

    if (dialog.mode === 'create') {
      await createProject({ name, workspaceDir })
    } else if (dialog.projectId) {
      await updateProject(dialog.projectId, { name, workspaceDir })
    }
    setProjectDialog(null)
  }

  const renderSession = (session: Session) => (
    <SessionItem
      key={session.id}
      session={session}
      isActive={session.id === currentSessionId}
      isCollapsed={collapsed}
      selectionMode={selectionMode}
      selected={selectedSessionIds.includes(session.id)}
      projects={projects}
      onClick={() => handleSelectSession(session.id)}
      onToggleSelected={() => handleToggleSessionSelected(session.id)}
      onRename={(title) => void renameSession(session.id, title)}
      onMoveToProject={(projectId) => void moveSessionToProject(session.id, projectId)}
      onDelete={() => setPendingDeleteSession(session)}
    />
  )

  return (
    <>
      <aside
        data-mochi-sidebar
        className={cn(
          'flex h-full shrink-0 flex-col overflow-hidden border-r border-border bg-sidebar-layer',
          'transition-[width] duration-300 ease-out-smooth',
          collapsed ? 'w-16' : 'w-[248px]',
          'max-md:w-16'
        )}
      >
        <div className={cn('flex h-14 items-center border-b border-border/80 px-3 shrink-0 max-md:justify-center max-md:px-0', collapsed && 'justify-center px-0')}>
          {!collapsed ? (
            <div className="flex min-w-0 flex-1 items-center gap-2.5 max-md:flex-none">
              <span data-mochi-sidebar-mark className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-foreground text-background shadow-sm">
                <Zap className="h-3.5 w-3.5" />
              </span>
              <div className="sidebar-wide min-w-0">
                <span className="block truncate text-[13px] font-semibold text-foreground">Mochi</span>
                <span className="block truncate text-[10px] text-muted-foreground">Personal workspace</span>
              </div>
            </div>
          ) : (
            <span className="flex h-8 w-8 items-center justify-center rounded-lg bg-foreground text-background shadow-sm">
              <Zap className="h-4 w-4" />
            </span>
          )}
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => setSidebarCollapsed(!collapsed)}
            aria-label={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
            className="shrink-0"
          >
            {collapsed ? <PanelLeftOpen className="h-4 w-4" /> : <PanelLeftClose className="h-4 w-4" />}
          </Button>
        </div>

        <div className={cn('flex flex-col gap-2 px-3 pb-2 pt-3', collapsed && 'items-center px-2')}>
          <Button
            variant="outline"
            size={collapsed ? 'icon' : 'md'}
            className={cn(
              'w-full border-border/80 bg-surface-layer/80 shadow-[0_1px_2px_rgba(20,20,24,0.04)] hover:border-primary-500/35 hover:bg-surface-layer',
              collapsed && 'w-9'
            )}
            onClick={handleNewSession}
            title={t('sidebar.newChatShortcut')}
          >
            <Plus className="h-4 w-4" />
            {!collapsed && <span className="sidebar-wide max-md:hidden">{t('sidebar.newChat')}</span>}
          </Button>

          {!collapsed ? (
            <>
              <div className="sidebar-wide relative">
                <Input
                  id="sidebar-search-input"
                  placeholder={t('sidebar.searchPlaceholder')}
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  leftIcon={<Search className="h-3.5 w-3.5" />}
                  size="sm"
                  className="pl-8"
                />
              </div>
              <Button variant="ghost" size="sm" className="sidebar-wide w-full justify-start text-muted-foreground hover:text-foreground" onClick={openCreateProject}>
                <FolderPlus className="h-4 w-4" />
                <span>New Project</span>
              </Button>
              <div className="flex items-center gap-2">
                <Button
                  variant={selectionMode ? 'secondary' : 'ghost'}
                  size="sm"
                  onClick={handleToggleSelectionMode}
                  title={t('sidebar.bulkDelete')}
                  className="justify-start"
                >
                  <Trash2 className="h-4 w-4" />
                  <span className="sidebar-wide max-md:hidden">{selectionMode ? t('sidebar.bulkCancel') : t('sidebar.bulkDelete')}</span>
                </Button>
                {selectionMode ? (
                  <span className="text-xs font-medium text-muted-foreground">
                    {t('sidebar.bulkSelectedCount', { count: selectedSessionIds.length })}
                  </span>
                ) : null}
              </div>
            </>
          ) : null}
        </div>

        <nav className="flex-1 space-y-3 overflow-y-auto px-2 py-2">
          <div className="space-y-0.5">
            <SidebarRouteButton
              collapsed={collapsed}
              icon={<Waypoints className="h-4 w-4" />}
              title="Goals"
              onClick={() => router.push('/goals')}
            />
            <SidebarRouteButton
              collapsed={collapsed}
              icon={<Workflow className="h-4 w-4" />}
              title="Workflows"
              onClick={() => router.push('/agent-runs')}
            />
            <SidebarRouteButton
              collapsed={collapsed}
              icon={<Library className="h-4 w-4" />}
              title="Skills"
              onClick={() => router.push('/skills')}
            />
          </div>
          {selectionMode && !collapsed ? (
            <div className="mx-1 rounded-xl border border-primary-500/20 bg-primary-500/8 p-2.5">
              <p className="text-xs font-semibold uppercase tracking-[0.14em] text-primary-300">
                {t('sidebar.bulkSelectedCount', { count: selectedSessionIds.length })}
              </p>
              <div className="mt-2 flex flex-wrap gap-2">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={handleToggleSelectAllVisible}
                  disabled={visibleSessionIds.length === 0}
                >
                  <Check className="h-4 w-4" />
                  <span>{allVisibleSelected ? t('sidebar.bulkClearAll') : t('sidebar.bulkSelectAll')}</span>
                </Button>
                <Button
                  variant="destructive"
                  size="sm"
                  onClick={() => setPendingBulkDeleteIds(selectedSessionIds)}
                  disabled={selectedSessionIds.length === 0 || isBulkDeleting}
                  loading={isBulkDeleting}
                >
                  <Trash2 className="h-4 w-4" />
                  <span>{t('sidebar.bulkDeleteSelected')}</span>
                </Button>
              </div>
            </div>
          ) : null}

          {drafts.length > 0 && !collapsed ? (
            <SidebarSection title="Draft Chat">
              {sortSessions(drafts).map(renderSession)}
            </SidebarSection>
          ) : null}

          <SidebarSection title="Projects" collapsed={collapsed} mobileHidden>
            {projects.map((project) => {
              const expanded = expandedProjectIds.includes(project.id)
              const projectSessions = sortSessions(byProject[project.id] ?? [])
              const isActiveProject = project.id === activeProjectId

              return (
                <div
                  key={project.id}
                  className={cn(
                    'rounded-lg border border-transparent',
                    isActiveProject && 'border-primary-500/25 bg-primary-500/8'
                  )}
                >
                  <button
                    type="button"
                    className="flex w-full items-center gap-2 px-2 py-2 text-left"
                    onClick={() => {
                      setActiveProjectId(project.id)
                      toggleProjectExpanded(project.id)
                    }}
                  >
                    {expanded ? <ChevronDown className="h-4 w-4 shrink-0" /> : <ChevronRight className="h-4 w-4 shrink-0" />}
                    <div className="min-w-0 flex-1">
                      <div className="truncate text-sm font-medium text-foreground">{project.name}</div>
                      {!collapsed ? (
                        <div className="truncate text-[11px] text-muted-foreground">
                          {basename(project.workspaceDir)}
                        </div>
                      ) : null}
                    </div>
                  </button>

                  {!collapsed ? (
                    <div className="flex gap-1 border-t border-border/50 px-2 pb-2 pt-1">
                      <Button variant="ghost" size="sm" onClick={() => setActiveProjectId(project.id)}>
                        Use
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => openEditProject(project.id)}>
                        Edit
                      </Button>
                      <Button variant="ghost" size="sm" onClick={() => setPendingDeleteProjectId(project.id)}>
                        Delete
                      </Button>
                    </div>
                  ) : null}

                  {expanded && !collapsed ? (
                    <div className="space-y-2 border-t border-border/50 px-2 py-2">
                      {projectSessions.length > 0 ? (
                        projectSessions.map(renderSession)
                      ) : (
                        <p className="px-2 py-2 text-xs text-muted-foreground">
                          No chats yet.
                        </p>
                      )}
                    </div>
                  ) : null}
                </div>
              )
            })}
            {projects.length === 0 && !collapsed ? (
              <p className="px-2 py-2 text-xs text-muted-foreground">No projects yet.</p>
            ) : null}
          </SidebarSection>

          <SidebarSection title="Unassigned Chats" collapsed={collapsed} mobileHidden>
            {sortSessions(unassigned).map(renderSession)}
            {unassigned.length === 0 && !collapsed ? (
              <p className="px-2 py-2 text-xs text-muted-foreground">No unassigned chats.</p>
            ) : null}
          </SidebarSection>
        </nav>

        <div className={cn('shrink-0 border-t border-border/80 p-2', collapsed ? 'space-y-1' : 'space-y-0.5')}>
          <div
            className={cn('flex items-center gap-2 px-2 py-1.5 text-[11px] text-muted-foreground', collapsed && 'justify-center px-0')}
            title="Local context synced"
          >
            <Cloud className="h-3.5 w-3.5 shrink-0 text-success" />
            {!collapsed ? <span className="sidebar-wide">Local context synced</span> : null}
          </div>
          <Button
            variant="ghost"
            size={collapsed ? 'icon' : 'sm'}
            className={collapsed ? 'mx-auto w-9' : 'w-full justify-start text-muted-foreground'}
            title="Help"
            aria-label="Help"
            onClick={() => router.push('/settings')}
          >
            <CircleHelp className="h-4 w-4" />
            {!collapsed ? <span className="sidebar-wide">Help</span> : null}
          </Button>
          <Button
            variant="ghost"
            size={collapsed ? 'icon' : 'sm'}
            className={collapsed ? 'mx-auto w-9' : 'w-full justify-start'}
            title="Profile"
            aria-label="Profile"
            onClick={() => router.push('/settings')}
          >
            <span className="flex h-6 w-6 items-center justify-center rounded-full border border-border bg-foreground text-background">
              <UserRound className="h-3.5 w-3.5" />
            </span>
            {!collapsed ? <span className="sidebar-wide">Profile</span> : null}
          </Button>
        </div>
      </aside>

      {mobileNavOpen ? (
        <div
          className="fixed inset-0 z-[90] bg-black/35 md:hidden"
          role="presentation"
          onClick={closeMobileNavigation}
        >
          <section
            ref={mobileNavDialogRef}
            role="dialog"
            aria-modal="true"
            aria-labelledby="mobile-navigation-title"
            className="absolute inset-x-3 bottom-3 top-16 flex flex-col overflow-hidden rounded-xl border border-border bg-surface-layer shadow-2xl"
            onClick={(event) => event.stopPropagation()}
          >
            <div className="flex items-center justify-between border-b border-border/80 px-4 py-3">
              <div>
                <p className="text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Workspace</p>
                <h2 id="mobile-navigation-title" className="text-base font-semibold text-foreground">
                  {mobileNavTarget === 'search' ? 'Search chats' : 'Projects and sessions'}
                </h2>
              </div>
              <Button ref={mobileNavCloseRef} type="button" variant="ghost" size="icon-sm" title="Close navigation" aria-label="Close navigation" onClick={closeMobileNavigation}>
                <ChevronRight className="h-4 w-4 rotate-90" />
              </Button>
            </div>
            <div className="flex-1 overflow-y-auto p-4">
              {mobileNavTarget === 'search' ? (
                <>
                  <Input
                    id="mobile-sidebar-search-input"
                    autoFocus
                    placeholder={t('sidebar.searchPlaceholder')}
                    value={search}
                    onChange={(event) => setSearch(event.target.value)}
                    leftIcon={<Search className="h-3.5 w-3.5" />}
                    size="sm"
                    className="pl-8"
                  />
                  <div className="mt-3 space-y-1">
                    {visibleSessions.length > 0 ? visibleSessions.map((session) => (
                      <button
                        key={session.id}
                        type="button"
                        className="flex w-full items-center justify-between gap-3 rounded-md border border-transparent px-3 py-2 text-left text-sm hover:border-border hover:bg-muted/45"
                        onClick={() => { closeMobileNavigation(); handleSelectSession(session.id) }}
                      >
                        <span className="min-w-0 truncate">{session.title}</span>
                        <span className="shrink-0 text-[11px] text-muted-foreground">{session.lastMessageAt.toLocaleDateString()}</span>
                      </button>
                    )) : <p className="py-6 text-center text-sm text-muted-foreground">No matching chats.</p>}
                  </div>
                </>
              ) : (
                <div className="space-y-2">
                  <Button type="button" variant="outline" size="sm" className="w-full justify-start" onClick={() => { closeMobileNavigation(); openCreateProject() }}>
                    <FolderPlus className="h-4 w-4" />
                    New Project
                  </Button>
                  {projects.length > 0 ? projects.map((project) => {
                    const projectSessions = sortSessions(byProject[project.id] ?? [])
                    return (
                      <div key={project.id} className="border-y border-border/70 py-2">
                        <button
                          type="button"
                          className="flex w-full items-center gap-2 px-2 py-1 text-left"
                          onClick={() => { setActiveProjectId(project.id); toggleProjectExpanded(project.id) }}
                        >
                          <FolderOpen className="h-4 w-4 shrink-0" />
                          <span className="min-w-0 flex-1 truncate text-sm font-medium">{project.name}</span>
                          <span className="text-[11px] text-muted-foreground">{projectSessions.length}</span>
                        </button>
                        {expandedProjectIds.includes(project.id) ? projectSessions.map((session) => (
                          <button
                            key={session.id}
                            type="button"
                            className="block w-full truncate px-8 py-1.5 text-left text-xs text-muted-foreground hover:bg-muted/45 hover:text-foreground"
                            onClick={() => { closeMobileNavigation(); handleSelectSession(session.id) }}
                          >
                            {session.title}
                          </button>
                        )) : null}
                      </div>
                    )
                  }) : <p className="py-6 text-center text-sm text-muted-foreground">No projects yet.</p>}
                  {unassigned.length > 0 ? (
                    <div className="border-y border-border/70 py-2">
                      <p className="px-2 py-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">Unassigned chats</p>
                      {unassigned.map((session) => (
                        <button key={session.id} type="button" className="block w-full truncate px-2 py-1.5 text-left text-xs text-muted-foreground hover:bg-muted/45 hover:text-foreground" onClick={() => { closeMobileNavigation(); handleSelectSession(session.id) }}>
                          {session.title}
                        </button>
                      ))}
                    </div>
                  ) : null}
                </div>
              )}
            </div>
          </section>
        </div>
      ) : null}

      <Dialog
        open={pendingDeleteSession !== null}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDeleteSession(null)
          }
        }}
      >
        <DialogContent className="w-[calc(100vw-2rem)] max-w-[430px] rounded-xl border-border/80 p-0 shadow-2xl">
          <DialogHeader className="mb-0 px-5 pb-3 pt-5">
            <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-error/12 text-error">
              <Trash2 className="h-4 w-4" />
            </div>
            <DialogTitle className="text-lg">{t('sidebar.deleteDialogTitle')}</DialogTitle>
            <DialogDescription className="leading-6">
              {t('sidebar.deleteDialogDescription', {
                title: pendingDeleteSession?.title ?? t('sidebar.newChat'),
              })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-0 gap-2 border-t border-border/70 px-5 py-4 sm:flex-row sm:justify-end">
            <Button type="button" variant="ghost" size="md" onClick={() => setPendingDeleteSession(null)}>
              {t('common.cancel')}
            </Button>
            <Button type="button" variant="destructive" size="md" onClick={handleConfirmDeleteSession}>
              {t('sidebar.deleteDialogAction')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={pendingBulkDeleteIds.length > 0}
        onOpenChange={(open) => {
          if (!open) {
            setPendingBulkDeleteIds([])
          }
        }}
      >
        <DialogContent className="w-[calc(100vw-2rem)] max-w-[430px] rounded-xl border-border/80 p-0 shadow-2xl">
          <DialogHeader className="mb-0 px-5 pb-3 pt-5">
            <div className="mb-2 flex h-9 w-9 items-center justify-center rounded-lg bg-error/12 text-error">
              <Trash2 className="h-4 w-4" />
            </div>
            <DialogTitle className="text-lg">{t('sidebar.bulkDeleteDialogTitle')}</DialogTitle>
            <DialogDescription className="leading-6">
              {t('sidebar.bulkDeleteDialogDescription', {
                count: pendingBulkDeleteIds.length,
              })}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-0 gap-2 border-t border-border/70 px-5 py-4 sm:flex-row sm:justify-end">
            <Button type="button" variant="ghost" size="md" onClick={() => setPendingBulkDeleteIds([])}>
              {t('common.cancel')}
            </Button>
            <Button
              type="button"
              variant="destructive"
              size="md"
              onClick={() => void handleConfirmBulkDelete()}
              loading={isBulkDeleting}
            >
              {t('sidebar.bulkDeleteSelected')}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={pendingDeleteProjectId !== null}
        onOpenChange={(open) => {
          if (!open) {
            setPendingDeleteProjectId(null)
          }
        }}
      >
        <DialogContent className="w-[calc(100vw-2rem)] max-w-[430px] rounded-xl border-border/80 p-0 shadow-2xl">
          <DialogHeader className="mb-0 px-5 pb-3 pt-5">
            <DialogTitle className="text-lg">Delete project</DialogTitle>
            <DialogDescription className="leading-6">
              Assigned chats will become unassigned.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="mt-0 gap-2 border-t border-border/70 px-5 py-4 sm:flex-row sm:justify-end">
            <Button type="button" variant="ghost" size="md" onClick={() => setPendingDeleteProjectId(null)}>
              {t('common.cancel')}
            </Button>
            <Button type="button" variant="destructive" size="md" onClick={handleConfirmDeleteProject}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog
        open={projectDialog !== null}
        onOpenChange={(open) => {
          if (!open) {
            setProjectDialog(null)
          }
        }}
      >
        <DialogContent className="w-[calc(100vw-2rem)] max-w-[560px] rounded-xl border-border/80 p-0 shadow-2xl">
          <DialogHeader className="mb-0 px-5 pb-3 pt-5">
            <DialogTitle className="text-lg">
              {projectDialog?.mode === 'edit' ? 'Edit Project' : 'New Project'}
            </DialogTitle>
            <DialogDescription>
              Bind chats under one workspace directory.
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4 px-5 py-4">
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Project name</label>
              <Input
                value={projectDialog?.name ?? ''}
                onChange={(event) =>
                  setProjectDialog((state) =>
                    state ? { ...state, name: event.target.value } : state
                  )
                }
                placeholder="My Project"
              />
            </div>
            <div className="space-y-2">
              <label className="text-sm font-medium text-foreground">Workspace directory</label>
              <div className="flex min-w-0 items-center gap-2">
                <Input
                  value={projectDialog?.workspaceDir ?? ''}
                  onChange={(event) => updateProjectWorkspaceDir(event.target.value)}
                  placeholder="G:\\_python\\STT&TTS"
                />
                <Button
                  type="button"
                  variant="secondary"
                  size="md"
                  className="shrink-0"
                  onClick={() => void handleSelectProjectDirectory()}
                  loading={isSelectingProjectDirectory}
                >
                  <FolderOpen className="h-4 w-4" />
                  {t('sidebar.browseFolder')}
                </Button>
              </div>
              {projectDirectoryError ? (
                <div className="rounded-md border border-destructive/30 bg-destructive/10 px-3 py-2 text-xs text-destructive">
                  {projectDirectoryError}
                </div>
              ) : null}
            </div>
          </div>
          <DialogFooter className="mt-0 gap-2 border-t border-border/70 px-5 py-4 sm:flex-row sm:justify-end">
            <Button type="button" variant="ghost" size="md" onClick={() => setProjectDialog(null)}>
              {t('common.cancel')}
            </Button>
            <Button type="button" variant="primary" size="md" onClick={() => void submitProjectDialog()}>
              Save
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}

function SidebarSection({
  title,
  collapsed = false,
  mobileHidden = false,
  children,
}: {
  title: string
  collapsed?: boolean
  mobileHidden?: boolean
  children: React.ReactNode
}) {
  return (
    <div className={cn('space-y-1', mobileHidden && 'max-md:hidden')}>
      {!collapsed ? (
        <div className="sidebar-wide px-2 py-1 text-[11px] font-semibold uppercase tracking-wide text-muted-foreground">
          {title}
        </div>
      ) : null}
      <div className="space-y-0.5">{children}</div>
    </div>
  )
}

function SidebarRouteButton({
  collapsed,
  icon,
  title,
  onClick,
}: {
  collapsed: boolean
  icon: React.ReactNode
  title: string
  onClick: () => void
}) {
  return (
    <Button
      type="button"
      variant="ghost"
      size={collapsed ? 'icon' : 'sm'}
      title={title}
      aria-label={title}
      onClick={onClick}
      className={cn(
        collapsed ? 'mx-auto w-9' : 'h-auto w-full justify-start rounded-lg px-3 py-2',
        'text-muted-foreground hover:bg-surface-layer/75 hover:text-foreground'
      )}
    >
      {icon}
      {!collapsed ? <span className="sidebar-wide min-w-0 truncate text-left text-sm font-medium">{title}</span> : null}
    </Button>
  )
}
