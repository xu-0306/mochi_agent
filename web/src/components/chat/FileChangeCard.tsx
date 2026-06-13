'use client'

import * as React from 'react'
import { ChevronDown, FileText, RotateCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { FileChangeSummary } from '@/lib/chat-p2'
import { summarizeDiffStats } from '@/lib/chat-p2'
import { cn } from '@/lib/utils'

interface FileChangeCardProps {
  change: FileChangeSummary
  onUndo?: (change: FileChangeSummary) => Promise<void> | void
}

export function FileChangeCard({ change, onUndo }: FileChangeCardProps) {
  const [expanded, setExpanded] = React.useState(false)
  const [undoing, setUndoing] = React.useState(false)
  const [undone, setUndone] = React.useState(false)
  const stats = summarizeDiffStats(change.diff)
  const fileName = React.useMemo(() => {
    const segments = change.filePath.split(/[\\/]/)
    return segments[segments.length - 1] || change.filePath
  }, [change.filePath])

  const handleUndo = React.useCallback(async () => {
    if (!onUndo) {
      return
    }
    setUndoing(true)
    try {
      await onUndo(change)
      setUndone(true)
    } finally {
      setUndoing(false)
    }
  }, [change, onUndo])

  return (
    <div className="rounded-xl border border-border/80 bg-surface-layer/80 shadow-[0_10px_24px_rgba(0,0,0,0.16)]">
      <div className="flex flex-wrap items-start gap-3 px-3 py-3">
        <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-border bg-canvas">
          <FileText className="h-4 w-4 text-primary-400" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <p className="truncate text-sm font-medium text-foreground">{fileName}</p>
            <span className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300">
              +{stats.additions}
            </span>
            <span className="rounded-full bg-rose-500/10 px-2 py-0.5 text-[10px] font-medium text-rose-300">
              -{stats.deletions}
            </span>
            {change.undoAvailable && !undone ? (
              <span className="rounded-full bg-primary-500/10 px-2 py-0.5 text-[10px] font-medium text-primary-300">
                Undo ready
              </span>
            ) : null}
            {undone ? (
              <span className="rounded-full bg-slate-500/10 px-2 py-0.5 text-[10px] font-medium text-slate-300">
                Undone
              </span>
            ) : null}
          </div>
          <p className="mt-1 break-all text-xs text-muted-foreground">{change.filePath}</p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            type="button"
            variant="ghost"
            size="sm"
            onClick={() => setExpanded((value) => !value)}
          >
            <ChevronDown className={cn('h-3.5 w-3.5 transition-transform', expanded && 'rotate-180')} />
            {expanded ? 'Hide diff' : 'Show diff'}
          </Button>
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => void handleUndo()}
            disabled={!change.undoAvailable || undone || !onUndo}
            loading={undoing}
          >
            <RotateCcw className="h-3.5 w-3.5" />
            Undo
          </Button>
        </div>
      </div>

      {expanded ? (
        <div className="border-t border-border/70 px-3 py-3">
          <pre
            className={cn(
              'max-h-[28rem] overflow-auto rounded-lg border border-border bg-canvas p-4 text-xs text-foreground',
              'whitespace-pre-wrap break-all font-mono'
            )}
          >
            {change.diff ?? 'No diff available.'}
          </pre>
        </div>
      ) : null}
    </div>
  )
}
