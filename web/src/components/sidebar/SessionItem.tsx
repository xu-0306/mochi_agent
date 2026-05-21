'use client'

import * as React from 'react'
import { Check, Pencil, Trash2, X } from 'lucide-react'
import { cn, truncate, formatRelativeTime } from '@/lib/utils'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useI18n } from '@/lib/i18n'
import type { ProjectSummary } from '@/lib/api'
import type { Session, ChannelSource } from '@/lib/stores/session-store'

interface SessionItemProps {
  session: Session
  isActive: boolean
  isCollapsed?: boolean
  projects?: ProjectSummary[]
  onClick: () => void
  onRename?: (title: string) => void
  onMoveToProject?: (projectId: string | null) => void
  onDelete?: () => void
}

const sourceLabel: Record<ChannelSource, string> = {
  web: 'Web',
  cli: 'CLI',
  discord: 'Discord',
  telegram: 'TG',
}

const sourceBadgeVariant: Record<ChannelSource, 'web' | 'cli' | 'discord' | 'telegram'> = {
  web: 'web',
  cli: 'cli',
  discord: 'discord',
  telegram: 'telegram',
}

function displaySessionTitle(title: string, fallback: string): string {
  return title === '\u65b0\u5c0d\u8a71' || title === 'New chat' ? fallback : title
}

export function SessionItem({
  session,
  isActive,
  isCollapsed = false,
  projects = [],
  onClick,
  onRename,
  onMoveToProject,
  onDelete,
}: SessionItemProps) {
  const { locale, resolvedTimeZone, t } = useI18n()
  const [isEditing, setIsEditing] = React.useState(false)
  const [draftTitle, setDraftTitle] = React.useState(session.title)
  const visibleTitle = displaySessionTitle(session.title, t('sidebar.newChat'))

  React.useEffect(() => {
    setDraftTitle(session.title)
  }, [session.title])

  const submitRename = () => {
    const nextTitle = draftTitle.trim()
    if (nextTitle && nextTitle !== session.title) {
      onRename?.(nextTitle)
    }
    setIsEditing(false)
  }

  const cancelRename = () => {
    setDraftTitle(session.title)
    setIsEditing(false)
  }

  if (!isCollapsed && isEditing) {
    return (
      <div
        className={cn(
          'rounded-md px-1.5 py-1.5',
          isActive ? 'bg-primary-500/12' : 'bg-muted'
        )}
      >
        <form
          className="flex h-10 items-center gap-1"
          onSubmit={(event) => {
            event.preventDefault()
            submitRename()
          }}
        >
          <Input
            value={draftTitle}
            onChange={(event) => setDraftTitle(event.target.value)}
            autoFocus
            size="sm"
            className="h-7 min-w-0 flex-1 px-2 text-xs"
            onKeyDown={(event) => {
              if (event.key === 'Escape') {
                event.preventDefault()
                cancelRename()
              }
            }}
          />
          <Button type="submit" variant="ghost" size="icon-sm" aria-label={t('sidebar.renameSave')}>
            <Check className="h-3.5 w-3.5" />
          </Button>
          <Button type="button" variant="ghost" size="icon-sm" aria-label={t('sidebar.renameCancel')} onClick={cancelRename}>
            <X className="h-3.5 w-3.5" />
          </Button>
        </form>

        <div className="mt-2">
          <Select
            value={session.projectId ?? '__unassigned__'}
            onValueChange={(value) => {
              onMoveToProject?.(value === '__unassigned__' ? null : value)
            }}
          >
            <SelectTrigger className="h-8 text-xs">
              <SelectValue placeholder="Move to..." />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="__unassigned__">Unassigned</SelectItem>
              {projects.map((project) => (
                <SelectItem key={project.id} value={project.id}>
                  {project.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>
    )
  }

  return (
    <div
      className={cn(
        'relative group flex w-full items-center rounded-md transition-all duration-150',
        isActive
          ? 'bg-primary-500/12 text-foreground'
          : 'text-muted-foreground hover:bg-muted hover:text-foreground',
        isCollapsed && 'justify-center px-0'
      )}
    >
      {/* Active indicator bar */}
      {isActive && (
        <span className="absolute left-0 top-1.5 bottom-1.5 w-0.5 rounded-r-full bg-primary-500" />
      )}

      <button
        type="button"
        onClick={onClick}
        aria-current={isActive ? 'true' : undefined}
        className={cn(
          'flex h-9 min-w-0 flex-1 items-center gap-2 rounded-md px-2 text-left',
          isCollapsed && 'h-9 justify-center px-0'
        )}
      >
        {isCollapsed ? (
        <span className="text-xs font-medium w-8 h-8 flex items-center justify-center rounded-md bg-muted/50">
          {visibleTitle.charAt(0).toUpperCase()}
        </span>
      ) : (
        <div className="flex flex-col min-w-0 flex-1 gap-0.5">
          <div className="flex items-center justify-between gap-1">
            <span className="text-sm font-medium truncate">{visibleTitle}</span>
            <span className="text-[10px] text-muted-foreground shrink-0">
              {formatRelativeTime(session.lastMessageAt, {
                locale,
                timeZone: resolvedTimeZone,
              })}
            </span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="text-xs text-muted-foreground truncate flex-1">
              {truncate(session.lastMessage, 40)}
            </span>
            {session.source !== 'web' && (
              <Badge variant={sourceBadgeVariant[session.source]} className="shrink-0 text-[9px] h-4 px-1">
                {sourceLabel[session.source]}
              </Badge>
            )}
          </div>
        </div>
      )}
      </button>

      {!isCollapsed ? (
        <div className="mr-1 hidden shrink-0 items-center gap-0.5 group-hover:flex group-focus-within:flex">
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={t('sidebar.rename')}
            onClick={(event) => {
              event.stopPropagation()
              setIsEditing(true)
            }}
          >
            <Pencil className="h-3.5 w-3.5" />
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon-sm"
            aria-label={t('sidebar.deleteConversation')}
            onClick={(event) => {
              event.stopPropagation()
              onDelete?.()
            }}
          >
            <Trash2 className="h-3.5 w-3.5 text-error" />
          </Button>
        </div>
      ) : null}
    </div>
  )
}
