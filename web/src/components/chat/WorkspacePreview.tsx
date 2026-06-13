'use client'

import * as React from 'react'
import { FileCode2, GitCompareArrows, Loader2, Paperclip, Quote } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import type { WorkspaceDiffResult, WorkspacePreviewResult } from '@/lib/api'
import type { ChatAttachment } from '@/lib/chat'

interface WorkspacePreviewProps {
  preview: WorkspacePreviewResult | null
  diff: WorkspaceDiffResult | null
  loading: boolean
  onAttachFile: (attachment: ChatAttachment) => void
  onQuoteSelection: (attachment: ChatAttachment) => void
}

function clampLineNumber(value: string, fallback: number): number {
  const parsed = Number.parseInt(value, 10)
  return Number.isFinite(parsed) && parsed > 0 ? parsed : fallback
}

export function WorkspacePreview({
  preview,
  diff,
  loading,
  onAttachFile,
  onQuoteSelection,
}: WorkspacePreviewProps) {
  const [lineStart, setLineStart] = React.useState('1')
  const [lineEnd, setLineEnd] = React.useState('1')

  React.useEffect(() => {
    setLineStart('1')
    setLineEnd('1')
  }, [preview?.path])

  const lineCount = React.useMemo(() => {
    if (!preview?.text) {
      return 1
    }
    return Math.max(1, preview.text.split('\n').length)
  }, [preview?.text])

  const previewAttachment = React.useMemo<ChatAttachment | null>(() => {
    if (!preview) {
      return null
    }
    return {
      id: `workspace-file:${preview.path}`,
      name: preview.name,
      path: preview.path,
      source: 'workspace_file',
    }
  }, [preview])

  const quoteAttachment = React.useMemo<ChatAttachment | null>(() => {
    if (!preview) {
      return null
    }
    const start = Math.min(clampLineNumber(lineStart, 1), lineCount)
    const end = Math.min(Math.max(clampLineNumber(lineEnd, start), start), lineCount)
    const lines = preview.text.split('\n')
    const quote = lines.slice(start - 1, end).join('\n').trim()
    return {
      id: `workspace-selection:${preview.path}:${start}:${end}`,
      name: preview.name,
      path: preview.path,
      source: 'workspace_selection',
      lineStart: start,
      lineEnd: end,
      quote: quote.length > 400 ? `${quote.slice(0, 397)}...` : quote,
    }
  }, [lineCount, lineEnd, lineStart, preview])

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between gap-3 border-b border-border/70 px-4 py-3">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-muted-foreground">
            Preview
          </p>
          <p className="mt-1 truncate text-sm font-medium text-foreground">
            {preview?.relativePath ?? diff?.relativePath ?? 'Select a file'}
          </p>
        </div>
        {previewAttachment ? (
          <Button
            type="button"
            variant="secondary"
            size="sm"
            onClick={() => onAttachFile(previewAttachment)}
          >
            <Paperclip className="h-3.5 w-3.5" />
            Attach file
          </Button>
        ) : null}
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        {loading ? (
          <div className="flex h-full items-center justify-center text-muted-foreground">
            <Loader2 className="h-4 w-4 animate-spin" />
          </div>
        ) : preview ? (
          <div className="space-y-4">
            <div className="rounded-xl border border-border bg-[#0c1224] p-4 text-slate-100 shadow-inner">
              <div className="mb-3 flex items-center gap-2 text-xs text-slate-400">
                <FileCode2 className="h-3.5 w-3.5" />
                <span>{preview.mediaType}</span>
                {preview.truncated ? <span>Preview truncated</span> : null}
              </div>
              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[13px] leading-6">
                {preview.text || 'No preview text available.'}
              </pre>
            </div>

            <div className="rounded-xl border border-border bg-surface-layer px-4 py-3">
              <div className="flex items-center gap-2">
                <Quote className="h-4 w-4 text-primary-300" />
                <p className="text-sm font-medium text-foreground">Quote selection</p>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">
                Add a structured `workspace_selection` attachment to the composer without leaving chat.
              </p>
              <div className="mt-3 grid grid-cols-[1fr_1fr_auto] gap-2">
                <Input
                  value={lineStart}
                  onChange={(event) => setLineStart(event.target.value)}
                  placeholder="Start line"
                />
                <Input
                  value={lineEnd}
                  onChange={(event) => setLineEnd(event.target.value)}
                  placeholder="End line"
                />
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => {
                    if (quoteAttachment) {
                      onQuoteSelection(quoteAttachment)
                    }
                  }}
                  disabled={!quoteAttachment?.quote}
                >
                  <Quote className="h-3.5 w-3.5" />
                  Quote
                </Button>
              </div>
            </div>
          </div>
        ) : diff ? (
          <div className="space-y-4">
            <div className="flex items-center gap-2 text-sm text-foreground">
              <GitCompareArrows className="h-4 w-4 text-primary-300" />
              <span>{diff.addedLines} additions</span>
              <span>{diff.deletedLines} deletions</span>
            </div>
            <div className="rounded-xl border border-border bg-[#0c1224] p-4 text-slate-100 shadow-inner">
              <pre className="overflow-x-auto whitespace-pre-wrap break-words font-mono text-[13px] leading-6">
                {diff.diff || 'No diff text available.'}
              </pre>
            </div>
          </div>
        ) : (
          <div className="flex h-full items-center justify-center rounded-xl border border-dashed border-border bg-surface-layer/40 px-6 text-center text-sm text-muted-foreground">
            Pick a file or changed entry to inspect it here.
          </div>
        )}
      </div>
    </div>
  )
}
