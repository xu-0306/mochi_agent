'use client'

import * as React from 'react'
import { useRouter } from 'next/navigation'
import {
  PanelLeftClose,
  PanelLeftOpen,
  Plus,
  Search,
  Pin,
  Zap,
  Settings,
  Library,
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { useI18n } from '@/lib/i18n'
import { useUIStore } from '@/lib/stores/ui-store'
import { SessionItem } from './SessionItem'
import {
  useSessionStore,
  getPinnedSessions,
  type Session,
} from '@/lib/stores/session-store'

function getDateKey(date: Date, timeZone: string | undefined): string {
  const formatter = new Intl.DateTimeFormat('en-US', {
    ...(timeZone ? { timeZone } : {}),
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  })
  const parts = formatter.formatToParts(date)
  const year = parts.find((part) => part.type === 'year')?.value ?? '0000'
  const month = parts.find((part) => part.type === 'month')?.value ?? '00'
  const day = parts.find((part) => part.type === 'day')?.value ?? '00'
  return `${year}-${month}-${day}`
}

function groupSessionsByDisplayDate(sessions: Session[], timeZone: string | undefined) {
  const now = new Date()
  const todayKey = getDateKey(now, timeZone)
  const weekAgoKey = getDateKey(new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000), timeZone)
  const unpinned = sessions.filter((session) => !session.isPinned)

  return {
    pinned: getPinnedSessions(sessions),
    today: unpinned.filter((session) => getDateKey(session.lastMessageAt, timeZone) >= todayKey),
    thisWeek: unpinned.filter((session) => {
      const key = getDateKey(session.lastMessageAt, timeZone)
      return key < todayKey && key >= weekAgoKey
    }),
    older: unpinned.filter((session) => getDateKey(session.lastMessageAt, timeZone) < weekAgoKey),
  }
}

export function Sidebar() {
  const router = useRouter()
  const { resolvedTimeZone, t } = useI18n()
  const collapsed = useUIStore((state) => state.sidebarCollapsed)
  const setSidebarCollapsed = useUIStore((state) => state.setSidebarCollapsed)
  const [search, setSearch] = React.useState('')

  const {
    sessions,
    currentSessionId,
    hasLoaded,
    loadSessions,
    setCurrentSession,
    createSession,
    renameSession,
    deleteSession,
  } = useSessionStore()

  React.useEffect(() => {
    if (!hasLoaded) {
      void loadSessions()
    }
  }, [hasLoaded, loadSessions])

  const filtered = search.trim()
    ? sessions.filter(
        (s) =>
          s.title.toLowerCase().includes(search.toLowerCase()) ||
          s.lastMessage.toLowerCase().includes(search.toLowerCase())
      )
    : sessions

  const { pinned, today, thisWeek, older } = React.useMemo(
    () => groupSessionsByDisplayDate(filtered, resolvedTimeZone),
    [filtered, resolvedTimeZone]
  )

  const handleNewSession = () => {
    const id = createSession()
    void id
    router.push('/')
  }

  const handleSelectSession = (id: string) => {
    setCurrentSession(id)
    router.push('/')
  }

  const renderSession = (s: (typeof sessions)[number]) => (
    <SessionItem
      key={s.id}
      session={s}
      isActive={s.id === currentSessionId}
      isCollapsed={collapsed}
      onClick={() => handleSelectSession(s.id)}
      onRename={(title) => void renameSession(s.id, title)}
      onDelete={() => void deleteSession(s.id)}
    />
  )

  return (
    <aside
      className={cn(
        'flex flex-col h-full bg-sidebar-layer border-r border-border',
        'transition-[width] duration-300 ease-out-smooth overflow-hidden shrink-0',
        collapsed ? 'w-16' : 'w-[260px]'
      )}
    >
      {/* Header */}
      <div className={cn('flex items-center h-12 border-b border-border px-3 shrink-0', collapsed && 'justify-center px-0')}>
        {!collapsed && (
          <div className="flex items-center gap-2 flex-1 min-w-0">
            <Zap className="h-5 w-5 text-primary-500 shrink-0" />
            <span className="font-semibold text-sm text-foreground truncate">Mochi</span>
          </div>
        )}
        <Button
          variant="ghost"
          size="icon-sm"
          onClick={() => setSidebarCollapsed(!collapsed)}
          aria-label={collapsed ? t('sidebar.expand') : t('sidebar.collapse')}
          className="shrink-0"
        >
          {collapsed ? (
            <PanelLeftOpen className="h-4 w-4" />
          ) : (
            <PanelLeftClose className="h-4 w-4" />
          )}
        </Button>
      </div>

      {/* Actions */}
      <div className={cn('px-3 pt-3 pb-2 flex flex-col gap-2', collapsed && 'items-center px-2')}>
        <Button
          variant="primary"
          size={collapsed ? 'icon' : 'md'}
          className={cn('w-full', collapsed && 'w-9')}
          onClick={handleNewSession}
          title={t('sidebar.newChatShortcut')}
        >
          <Plus className="h-4 w-4" />
          {!collapsed && <span>{t('sidebar.newChat')}</span>}
        </Button>

        {!collapsed && (
          <div className="relative">
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
        )}

        <div className={cn('flex gap-1', collapsed ? 'flex-col' : 'grid grid-cols-2')}>
          <Button
            variant="ghost"
            size={collapsed ? 'icon' : 'sm'}
            onClick={() => router.push('/skills')}
            title={t('sidebar.skills')}
            className={collapsed ? 'w-9' : 'justify-start'}
          >
            <Library className="h-4 w-4" />
            {!collapsed && <span>Skills</span>}
          </Button>
          <Button
            variant="ghost"
            size={collapsed ? 'icon' : 'sm'}
            onClick={() => router.push('/settings')}
            title={t('sidebar.settings')}
            className={collapsed ? 'w-9' : 'justify-start'}
          >
            <Settings className="h-4 w-4" />
            {!collapsed && <span>{t('sidebar.settings')}</span>}
          </Button>
        </div>
      </div>

      {/* Session list */}
      <nav className="flex-1 overflow-y-auto px-2 py-1 space-y-0.5">
        {/* Pinned */}
        {pinned.length > 0 && (
          <SessionGroup
            label={t('sidebar.pinned')}
            icon={<Pin className="h-3 w-3" />}
            collapsed={collapsed}
          >
            {pinned.map(renderSession)}
          </SessionGroup>
        )}

        {/* Today */}
        {today.length > 0 && (
          <SessionGroup label={t('sidebar.today')} collapsed={collapsed}>
            {today.map(renderSession)}
          </SessionGroup>
        )}

        {/* This week */}
        {thisWeek.length > 0 && (
          <SessionGroup label={t('sidebar.thisWeek')} collapsed={collapsed}>
            {thisWeek.map(renderSession)}
          </SessionGroup>
        )}

        {/* Older */}
        {older.length > 0 && (
          <SessionGroup label={t('sidebar.older')} collapsed={collapsed}>
            {older.map(renderSession)}
          </SessionGroup>
        )}

        {filtered.length === 0 && (
          <p className="text-xs text-muted-foreground text-center py-6">
            {search ? t('sidebar.noResults') : t('sidebar.noSessions')}
          </p>
        )}
      </nav>
    </aside>
  )
}

interface SessionGroupProps {
  label: string
  icon?: React.ReactNode
  collapsed: boolean
  children: React.ReactNode
}

function SessionGroup({ label, icon, collapsed, children }: SessionGroupProps) {
  return (
    <div className="mb-2">
      {!collapsed && (
        <div className="flex items-center gap-1.5 px-2 py-1 mb-0.5">
          {icon && <span className="text-muted-foreground">{icon}</span>}
          <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wide">
            {label}
          </span>
        </div>
      )}
      <div className="space-y-0.5">{children}</div>
    </div>
  )
}
