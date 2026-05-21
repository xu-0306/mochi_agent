'use client'

import * as React from 'react'
import { FileText, RotateCcw, Search } from 'lucide-react'
import { Button } from '@/components/ui/button'
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import type { FileChangeSummary } from '@/lib/chat-p2'
import { summarizeDiffStats } from '@/lib/chat-p2'
import { cn } from '@/lib/utils'

interface FileChangeCardProps {
  change: FileChangeSummary
  onUndo?: (change: FileChangeSummary) => Promise<void> | void
}

export function FileChangeCard({ change, onUndo }: FileChangeCardProps) {
  const [reviewOpen, setReviewOpen] = React.useState(false)
  const [undoing, setUndoing] = React.useState(false)
  const [undone, setUndone] = React.useState(false)
  const stats = summarizeDiffStats(change.diff)

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
    <>
      <div className="rounded-lg border border-border bg-surface-layer">
        <div className="flex flex-wrap items-center gap-3 px-3 py-3">
          <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md border border-border bg-canvas">
            <FileText className="h-4 w-4 text-primary-400" />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium text-foreground">
              Edited {change.filePath}
            </p>
            <p className="mt-0.5 text-xs text-muted-foreground">
              {stats.additions > 0 ? `+${stats.additions}` : '+0'} {stats.deletions > 0 ? `-${stats.deletions}` : '-0'}
              {undone ? '  Undone' : ''}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={() => setReviewOpen(true)}
            >
              <Search className="h-3.5 w-3.5" />
              Review
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
      </div>

      <Dialog open={reviewOpen} onOpenChange={setReviewOpen}>
        <DialogContent className="max-w-3xl">
          <DialogHeader>
            <DialogTitle>Review file change</DialogTitle>
            <DialogDescription>{change.filePath}</DialogDescription>
          </DialogHeader>
          <pre
            className={cn(
              'max-h-[60vh] overflow-auto rounded-lg border border-border bg-canvas p-4 text-xs text-foreground',
              'whitespace-pre-wrap break-all font-mono'
            )}
          >
            {change.diff ?? 'No diff available.'}
          </pre>
          <DialogFooter>
            <Button type="button" variant="ghost" onClick={() => setReviewOpen(false)}>
              Close
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </>
  )
}
