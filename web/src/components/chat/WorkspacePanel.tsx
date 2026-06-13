'use client'

import * as React from 'react'
import { FolderTree, GitCompareArrows, Loader2, PanelLeftClose, RefreshCcw } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import type { ChatAttachment } from '@/lib/chat'
import { useWorkspaceStore } from '@/lib/stores/workspace-store'
import { WorkspacePreview } from './WorkspacePreview'
import { WorkspaceTree } from './WorkspaceTree'

interface WorkspacePanelProps {
  onAttachAttachment: (attachment: ChatAttachment) => void
  onClose?: () => void
}

export function WorkspacePanel({ onAttachAttachment, onClose }: WorkspacePanelProps) {
  const {
    currentRelativePath,
    parentPath,
    items,
    changes,
    selectedFilePath,
    preview,
    diff,
    isTreeLoading,
    isPreviewLoading,
    isChangesLoading,
    isDiffLoading,
    error,
    loadTree,
    previewFile,
    loadChanges,
    loadDiff,
  } = useWorkspaceStore()
  const [activeTab, setActiveTab] = React.useState<'files' | 'changes'>('files')

  return (
    <div className="flex h-full min-h-0 flex-col overflow-hidden rounded-[inherit] bg-[radial-gradient(circle_at_top_right,rgba(72,163,255,0.16),transparent_36%),linear-gradient(180deg,rgba(255,255,255,0.04),transparent_24%)]">
      <div className="border-b border-white/8 bg-canvas/40 px-4 py-4 backdrop-blur-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="mb-2 inline-flex items-center gap-2 rounded-full border border-sky-400/20 bg-sky-400/10 px-2.5 py-1 text-[11px] font-medium uppercase tracking-[0.18em] text-sky-200">
              <FolderTree className="h-3.5 w-3.5" />
              Workbench
            </div>
            <h2 className="text-base font-semibold text-foreground">Workspace cockpit</h2>
            <p className="mt-1 text-xs leading-relaxed text-muted-foreground">
              Browse files, inspect diffs, and queue structured references without pushing the chat layout around.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="ghost"
              size="icon-sm"
              title="Refresh workspace"
              onClick={() => {
                void loadTree()
                void loadChanges()
              }}
              className="rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground"
            >
              <RefreshCcw className="h-4 w-4" />
            </Button>
            {onClose ? (
              <Button
                type="button"
                size="icon-sm"
                variant="ghost"
                onClick={onClose}
                title="Hide workspace cockpit"
                aria-label="Hide workspace cockpit"
                className="rounded-full border border-white/8 bg-canvas/55 text-muted-foreground hover:bg-elevated-layer hover:text-foreground"
              >
                <PanelLeftClose className="h-4 w-4" />
              </Button>
            ) : null}
          </div>
        </div>
      </div>

      <div className="border-b border-white/8 px-4 py-3">
        <Tabs
          value={activeTab}
          onValueChange={(value) => setActiveTab(value as 'files' | 'changes')}
        >
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="files" className="gap-1.5">
              <FolderTree className="h-3.5 w-3.5" />
              Files
            </TabsTrigger>
            <TabsTrigger value="changes" className="gap-1.5">
              <GitCompareArrows className="h-3.5 w-3.5" />
              Changes
            </TabsTrigger>
          </TabsList>
        </Tabs>
      </div>

      {error ? (
        <div className="border-b border-white/8 bg-destructive/10 px-4 py-2 text-xs text-destructive">
          {error}
        </div>
      ) : null}

      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-[minmax(280px,0.9fr)_minmax(360px,1.1fr)]">
        <div className="min-h-0 border-b border-white/8 lg:border-r lg:border-b-0">
          {activeTab === 'files' ? (
            <WorkspaceTree
              currentRelativePath={currentRelativePath}
              parentPath={parentPath}
              items={items}
              selectedFilePath={selectedFilePath}
              isLoading={isTreeLoading}
              onOpenDirectory={(path) => void loadTree(path)}
              onOpenFile={(path) => void previewFile(path)}
            />
          ) : (
            <div className="flex h-full flex-col">
              <div className="flex items-center justify-between gap-3 border-b border-white/8 px-3 py-2">
                <div>
                  <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
                    Changed Files
                  </p>
                  <p className="mt-1 text-sm font-medium text-foreground">{changes.length} entries</p>
                </div>
                {isChangesLoading ? <Loader2 className="h-4 w-4 animate-spin text-muted-foreground" /> : null}
              </div>
              <div className="min-h-0 flex-1 overflow-y-auto p-2">
                {changes.length === 0 && !isChangesLoading ? (
                  <div className="rounded-lg border border-dashed border-border bg-surface-layer/40 px-3 py-5 text-sm text-muted-foreground">
                    No git-backed changes detected inside this workspace.
                  </div>
                ) : (
                  <div className="space-y-2">
                    {changes.map((change) => (
                      <button
                        key={`${change.path}:${change.status}`}
                        type="button"
                        onClick={() => void loadDiff(change.path)}
                        className="w-full rounded-lg border border-transparent bg-surface-layer/60 px-3 py-2 text-left transition-colors hover:border-border hover:bg-surface-layer"
                      >
                        <div className="flex items-center justify-between gap-3">
                          <p className="truncate text-sm font-medium text-foreground">
                            {change.relativePath}
                          </p>
                          <span className="shrink-0 rounded-full border border-border/70 px-2 py-0.5 text-[10px] uppercase tracking-[0.12em] text-muted-foreground">
                            {change.status}
                          </span>
                        </div>
                        <p className="mt-1 text-[11px] text-muted-foreground">
                          +{change.addedLines} / -{change.deletedLines}
                        </p>
                      </button>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="min-h-0">
          <WorkspacePreview
            preview={preview}
            diff={diff}
            loading={isPreviewLoading || isDiffLoading}
            onAttachFile={onAttachAttachment}
            onQuoteSelection={onAttachAttachment}
          />
        </div>
      </div>
    </div>
  )
}
