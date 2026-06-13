'use client'

import * as React from 'react'
import { ChevronLeft, FileCode2, FileText, Folder, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import type { WorkspaceTreeItem } from '@/lib/api'
import { cn } from '@/lib/utils'

interface WorkspaceTreeProps {
  currentRelativePath: string
  parentPath: string | null
  items: WorkspaceTreeItem[]
  selectedFilePath: string | null
  isLoading: boolean
  onOpenDirectory: (path: string) => void
  onOpenFile: (path: string) => void
}

function iconForItem(item: WorkspaceTreeItem) {
  if (item.isDir) {
    return <Folder className="h-4 w-4 text-sky-300" />
  }
  if (/\.(py|ts|tsx|js|jsx|json|ya?ml|toml|html|css|scss|sql|md)$/i.test(item.name)) {
    return <FileCode2 className="h-4 w-4 text-emerald-300" />
  }
  return <FileText className="h-4 w-4 text-slate-300" />
}

export function WorkspaceTree({
  currentRelativePath,
  parentPath,
  items,
  selectedFilePath,
  isLoading,
  onOpenDirectory,
  onOpenFile,
}: WorkspaceTreeProps) {
  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-2 border-b border-border/70 px-3 py-2">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Workspace
          </p>
          <p className="mt-1 truncate text-sm font-medium text-foreground">{currentRelativePath}</p>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="icon-sm"
          disabled={!parentPath || isLoading}
          onClick={() => {
            if (parentPath) {
              onOpenDirectory(parentPath)
            }
          }}
          title="Go to parent directory"
        >
          <ChevronLeft className="h-4 w-4" />
        </Button>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto">
        {isLoading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : items.length === 0 ? (
          <div className="px-3 py-5 text-sm text-muted-foreground">This folder is empty.</div>
        ) : (
          <div className="space-y-1 p-2">
            {items.map((item) => {
              const active = selectedFilePath === item.path
              return (
                <button
                  key={item.path}
                  type="button"
                  onClick={() => (item.isDir ? onOpenDirectory(item.path) : onOpenFile(item.path))}
                  className={cn(
                    'flex w-full items-center gap-2 rounded-lg border px-3 py-2 text-left transition-colors',
                    active
                      ? 'border-primary-500/50 bg-primary-500/10 text-foreground'
                      : 'border-transparent bg-surface-layer/60 text-muted-foreground hover:border-border hover:bg-surface-layer hover:text-foreground'
                  )}
                >
                  {iconForItem(item)}
                  <div className="min-w-0 flex-1">
                    <p className="truncate text-sm font-medium">{item.name}</p>
                    <p className="truncate text-[11px] text-muted-foreground">
                      {item.relativePath}
                    </p>
                  </div>
                </button>
              )
            })}
          </div>
        )}
      </div>
    </div>
  )
}
