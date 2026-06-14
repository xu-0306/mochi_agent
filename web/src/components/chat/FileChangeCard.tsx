'use client'

import * as React from 'react'
import {
  ChevronDown,
  ChevronUp,
  FileCode2,
  Files,
  GitCommitHorizontal,
  PencilRuler,
  RotateCcw,
  WandSparkles,
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { CopyButton } from '@/components/chat/CopyButton'
import { DiffViewer } from '@/components/chat/DiffViewer'
import { getFileName, type FileChangeGroupSummary, type FileChangeSummary } from '@/lib/file-change-preview'
import { cn } from '@/lib/utils'

interface FileChangeCardProps {
  group: FileChangeGroupSummary
  onUndo?: (change: FileChangeSummary) => Promise<void> | void
  actions?: React.ReactNode
}

function groupIcon(sourceTool: string) {
  if (sourceTool === 'apply_patch') {
    return WandSparkles
  }
  if (sourceTool === 'file_edit') {
    return PencilRuler
  }
  if (sourceTool === 'file_write') {
    return GitCommitHorizontal
  }
  return Files
}

function statusTone(status: string): string {
  const normalized = status.toLowerCase()
  if (normalized === 'added') {
    return 'border-emerald-400/20 bg-emerald-400/10 text-emerald-200'
  }
  if (normalized === 'deleted') {
    return 'border-rose-400/20 bg-rose-400/10 text-rose-200'
  }
  if (normalized === 'renamed') {
    return 'border-sky-400/20 bg-sky-400/10 text-sky-200'
  }
  return 'border-amber-300/20 bg-amber-300/10 text-amber-100'
}

export function FileChangeCard({ group, onUndo, actions }: FileChangeCardProps) {
  const [expandedPaths, setExpandedPaths] = React.useState<Record<string, boolean>>(() => (
    Object.fromEntries(group.files.map((file, index) => [file.filePath, group.files.length === 1 && index === 0]))
  ))
  const [undoingPaths, setUndoingPaths] = React.useState<Record<string, boolean>>({})
  const [undonePaths, setUndonePaths] = React.useState<Record<string, boolean>>({})
  const Icon = groupIcon(group.sourceTool)

  React.useEffect(() => {
    setExpandedPaths((current) => {
      const next = { ...current }
      for (const file of group.files) {
        if (!(file.filePath in next)) {
          next[file.filePath] = group.files.length === 1
        }
      }
      return next
    })
  }, [group.files])

  const totals = React.useMemo(
    () => group.files.reduce(
      (summary, file) => ({
        additions: summary.additions + file.additions,
        deletions: summary.deletions + file.deletions,
      }),
      { additions: 0, deletions: 0 }
    ),
    [group.files]
  )

  const toggleExpanded = React.useCallback((filePath: string) => {
    setExpandedPaths((current) => ({
      ...current,
      [filePath]: !current[filePath],
    }))
  }, [])

  const handleUndo = React.useCallback(async (change: FileChangeSummary) => {
    if (!onUndo) {
      return
    }

    setUndoingPaths((current) => ({ ...current, [change.filePath]: true }))
    try {
      await onUndo(change)
      setUndonePaths((current) => ({ ...current, [change.filePath]: true }))
    } finally {
      setUndoingPaths((current) => ({ ...current, [change.filePath]: false }))
    }
  }, [onUndo])

  return (
    <section className="overflow-hidden rounded-[1.35rem] border border-white/8 bg-[radial-gradient(circle_at_top_right,rgba(94,106,210,0.16),transparent_38%),linear-gradient(180deg,rgba(255,255,255,0.04),rgba(255,255,255,0.02))] shadow-[0_16px_40px_rgba(0,0,0,0.22)]">
      <div className="flex flex-wrap items-start justify-between gap-3 border-b border-white/8 px-4 py-3">
        <div className="flex min-w-0 items-start gap-3">
          <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-2xl border border-white/10 bg-canvas/70">
            <Icon className="h-4 w-4 text-primary-300" />
          </div>
          <div className="min-w-0">
            <div className="flex flex-wrap items-center gap-2">
              <p className="text-sm font-semibold text-foreground">{group.title}</p>
              <span className="rounded-full border border-white/10 bg-white/[0.04] px-2 py-0.5 text-[10px] uppercase tracking-[0.16em] text-muted-foreground">
                {group.files.length} file{group.files.length === 1 ? '' : 's'}
              </span>
              <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-medium text-emerald-200">
                +{totals.additions}
              </span>
              <span className="rounded-full border border-rose-400/20 bg-rose-400/10 px-2 py-0.5 text-[10px] font-medium text-rose-200">
                -{totals.deletions}
              </span>
            </div>
            <p className="mt-1 text-xs text-muted-foreground">
              {group.sourceTool === 'apply_patch'
                ? 'Patch-derived preview'
                : group.sourceTool === 'file_edit'
                  ? 'Inline file edit'
                  : group.sourceTool === 'file_write'
                    ? 'Direct file write'
                    : 'Workspace mutation preview'}
            </p>
          </div>
        </div>
        {actions ? <div className="flex shrink-0 items-center gap-2">{actions}</div> : null}
      </div>

      <div className="space-y-3 px-3 py-3">
        {group.files.map((file) => {
          const isExpanded = Boolean(expandedPaths[file.filePath])
          const canUndo = file.undoAvailable && Boolean(onUndo) && !undonePaths[file.filePath]

          return (
            <div
              key={`${group.id}:${file.filePath}`}
              className="overflow-hidden rounded-[1.1rem] border border-white/8 bg-black/[0.14] shadow-[inset_0_1px_0_rgba(255,255,255,0.03)]"
            >
              <div className="flex flex-wrap items-start gap-3 px-3 py-3">
                <div className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl border border-white/10 bg-canvas/80">
                  <FileCode2 className="h-4 w-4 text-primary-300" />
                </div>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <p className="truncate text-sm font-medium text-foreground">
                      {getFileName(file.filePath)}
                    </p>
                    <span className={cn(
                      'rounded-full border px-2 py-0.5 text-[10px] font-medium uppercase tracking-[0.14em]',
                      statusTone(file.status)
                    )}>
                      {file.status}
                    </span>
                    <span className="rounded-full border border-emerald-400/20 bg-emerald-400/10 px-2 py-0.5 text-[10px] font-medium text-emerald-200">
                      +{file.additions}
                    </span>
                    <span className="rounded-full border border-rose-400/20 bg-rose-400/10 px-2 py-0.5 text-[10px] font-medium text-rose-200">
                      -{file.deletions}
                    </span>
                    {file.undoAvailable && !undonePaths[file.filePath] ? (
                      <span className="rounded-full border border-primary-400/20 bg-primary-500/10 px-2 py-0.5 text-[10px] font-medium text-primary-200">
                        Undo ready
                      </span>
                    ) : null}
                    {undonePaths[file.filePath] ? (
                      <span className="rounded-full border border-white/10 bg-white/[0.05] px-2 py-0.5 text-[10px] font-medium text-slate-300">
                        Undone
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-1 break-all font-mono text-[11px] text-muted-foreground">
                    {file.displayPath}
                  </p>
                </div>
                <div className="flex items-center gap-1.5">
                  <CopyButton
                    text={file.filePath}
                    label="Copy path"
                    className="h-8 w-8 rounded-full border border-white/10 bg-canvas/70 text-slate-300 hover:bg-elevated-layer hover:text-white"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="rounded-full border border-white/10 bg-canvas/70 px-3 text-xs"
                    onClick={() => toggleExpanded(file.filePath)}
                  >
                    {isExpanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
                    {isExpanded ? 'Hide diff' : 'Show diff'}
                  </Button>
                  <Button
                    type="button"
                    variant="secondary"
                    size="sm"
                    className="rounded-full px-3 text-xs"
                    disabled={!canUndo}
                    loading={undoingPaths[file.filePath]}
                    onClick={() => void handleUndo(file)}
                  >
                    <RotateCcw className="h-3.5 w-3.5" />
                    Undo
                  </Button>
                </div>
              </div>

              {isExpanded ? (
                <div className="border-t border-white/8 px-3 pb-3">
                  <DiffViewer
                    diff={file.diff}
                    filePath={file.relativePath ?? file.filePath}
                    oldValue={file.originalContent}
                    newValue={file.newContent}
                    className="mt-3"
                    maxHeightClassName="max-h-[26rem]"
                  />
                </div>
              ) : null}
            </div>
          )
        })}
      </div>
    </section>
  )
}
