'use client'

import * as React from 'react'
import { Brain, Command, Mic, Search, Settings, Sparkles } from 'lucide-react'
import { cn } from '@/lib/utils'

export type CommandPaletteAction =
  | { kind: 'builtin'; id: 'clear' | 'settings' | 'voice' | 'model' | 'export' | 'workflow' | 'chat'; label: string; description: string }
  | { kind: 'skill'; id: string; label: string; description: string }

interface CommandPaletteProps {
  open: boolean
  query: string
  actions: CommandPaletteAction[]
  selectedIndex: number
  onSelectedIndexChange: (index: number) => void
  onSelect: (action: CommandPaletteAction) => void
  loading?: boolean
}

export function buildBuiltinActions(): CommandPaletteAction[] {
  return [
    { kind: 'builtin', id: 'clear', label: '/clear', description: 'Clear the current chat view' },
    { kind: 'builtin', id: 'settings', label: '/settings', description: 'Open settings' },
    { kind: 'builtin', id: 'voice', label: '/voice', description: 'Open voice overlay' },
    { kind: 'builtin', id: 'model', label: '/model', description: 'Focus model selector' },
    { kind: 'builtin', id: 'export', label: '/export', description: 'Export current conversation' },
    { kind: 'builtin', id: 'workflow', label: '/workflow', description: 'Enable workflow mode for this conversation' },
    { kind: 'builtin', id: 'chat', label: '/chat', description: 'Return this conversation to normal chat mode' },
  ]
}

function iconForAction(action: CommandPaletteAction) {
  if (action.kind === 'skill') {
    return <Brain className="h-3.5 w-3.5 text-primary-400" />
  }
  if (action.id === 'settings') {
    return <Settings className="h-3.5 w-3.5 text-primary-400" />
  }
  if (action.id === 'voice') {
    return <Mic className="h-3.5 w-3.5 text-primary-400" />
  }
  if (action.id === 'model' || action.id === 'export' || action.id === 'workflow' || action.id === 'chat') {
    return <Sparkles className="h-3.5 w-3.5 text-primary-400" />
  }
  return <Command className="h-3.5 w-3.5 text-primary-400" />
}

export function CommandPalette({
  open,
  query,
  actions,
  selectedIndex,
  onSelectedIndexChange,
  onSelect,
  loading = false,
}: CommandPaletteProps) {
  const filtered = actions
  const builtins = filtered.filter((action) => action.kind === 'builtin')
  const skillActions = filtered.filter((action) => action.kind === 'skill')

  React.useEffect(() => {
    if (selectedIndex >= filtered.length && filtered.length > 0) {
      onSelectedIndexChange(0)
    }
  }, [filtered.length, onSelectedIndexChange, selectedIndex])

  if (!open || (filtered.length === 0 && !loading)) {
    return null
  }

  return (
    <div className="absolute bottom-full left-0 right-0 z-50 mb-2 rounded-xl border border-border bg-elevated-layer shadow-xl">
      <div className="flex items-center gap-2 border-b border-border px-3 py-2 text-xs text-muted-foreground">
        <Search className="h-3.5 w-3.5" />
        <span>Command palette</span>
        {query ? <span className="truncate">({query})</span> : null}
      </div>
      <div className="max-h-80 overflow-y-auto py-1">
        {loading ? (
          <div className="px-3 py-2 text-xs text-muted-foreground">Searching skills...</div>
        ) : null}
        {builtins.length > 0 ? (
          <div className="px-3 pb-1 pt-1 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Built-in
          </div>
        ) : null}
        {filtered.map((action, index) =>
          action.kind === 'builtin' ? (
            <button
              key={`${action.kind}:${action.id}`}
              type="button"
              onMouseEnter={() => onSelectedIndexChange(index)}
              onClick={() => onSelect(action)}
              className={cn(
                'flex w-full items-start gap-3 px-3 py-2 text-left transition-colors',
                index === selectedIndex ? 'bg-muted' : 'hover:bg-muted/70'
              )}
            >
              <div className="mt-0.5">{iconForAction(action)}</div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{action.label}</span>
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Built-in</span>
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">{action.description}</p>
              </div>
            </button>
          ) : null
        )}
        {skillActions.length > 0 ? (
          <div className="px-3 pb-1 pt-2 text-[10px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Skills
          </div>
        ) : null}
        {filtered.map((action, index) =>
          action.kind === 'skill' ? (
            <button
              key={`${action.kind}:${action.id}`}
              type="button"
              onMouseEnter={() => onSelectedIndexChange(index)}
              onClick={() => onSelect(action)}
              className={cn(
                'flex w-full items-start gap-3 px-3 py-2 text-left transition-colors',
                index === selectedIndex ? 'bg-muted' : 'hover:bg-muted/70'
              )}
            >
              <div className="mt-0.5">{iconForAction(action)}</div>
              <div className="min-w-0 flex-1">
                <div className="flex items-center gap-2">
                  <span className="text-sm font-medium text-foreground">{action.label}</span>
                  <span className="text-[10px] uppercase tracking-wide text-muted-foreground">Skill</span>
                </div>
                <p className="mt-0.5 text-xs text-muted-foreground">{action.description}</p>
              </div>
            </button>
          ) : null
        )}
      </div>
    </div>
  )
}
